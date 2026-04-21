"""Thin wrapper that launches IsaacSim's SimulationApp with the project's custom kit.

Usage (must be called BEFORE any `omni.*` / `isaacsim.*` imports):

    from utils_tasks.sim_launcher import launch_sim
    simulation_app = launch_sim()

    # ... now safe to import omni, isaacsim.core.api, etc.
"""
from __future__ import annotations
import os
import argparse
from typing import Optional, Callable, Any
import json
from collections import deque

import cv2
import imageio

import threading
import time
import numpy as np

from utils_tasks.basic_utils import draw_box_with_text, PlanningInput, PlanningOutput
from utils_tasks.tracking_utils import MPC_Controller

"""Common argparse definitions shared by all pure_isaacsim_eval_*_wheeled.py scripts."""

def make_common_parser(description: str,
                       default_scene_dir: str = "/workspace/FLUX/assets/dynbench/isaacsim_scene",
                       ) -> argparse.ArgumentParser:
    """Return a parser pre-populated with args common to every task script.

    Task scripts can add their own args on top before calling parse_args().
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--scene_dir",      type=str,   default=default_scene_dir)
    p.add_argument("--scene_index",    type=int,   default=0)
    p.add_argument("--scene_scale",    type=float, default=1.0)
    p.add_argument("--stop_threshold", type=float, default=-3.0,
                   help="Algorithm critic stop threshold")
    p.add_argument("--num_episodes",   type=int,   default=100)
    p.add_argument("--speed",          type=float, default=0.5,
                   help="Desired linear speed (m/s)")
    p.add_argument("--port",           type=int,   default=9999,
                   help="Algorithm server port")
    p.add_argument("--gpu_id",         type=int,   default=0)
    p.add_argument("--max_steps",      type=int,   default=1200,
                   help="Max steps per episode (120s @ 10Hz)")
    return p

def launch_sim(headless: bool = False,
               renderer: str = "RayTracedLighting",
               enable_cameras: bool = True,
               experience_kit: str = "isaacsim.exp.action_and_event_data_generation.base.kit"):
    """Start SimulationApp with the project's custom experience kit.

    Returns the SimulationApp instance. All Omni/IsaacSim imports in the caller
    must happen AFTER this call.
    """
    from isaacsim import SimulationApp  # imported lazily on purpose

    exp_path = os.environ.get("EXP_PATH")
    if exp_path is None:
        raise RuntimeError("EXP_PATH environment variable is not set.")

    custom_app_path = os.path.join(exp_path, experience_kit)

    simulation_app = SimulationApp(
        launch_config={
            "renderer": renderer,
            "headless": headless,
            "enable_cameras": enable_cameras,
            "crash_reporter/enabled": False,
            "crash_reporter/skip_old_dump_upload": True,
        },
        experience=custom_app_path,
    )
    return simulation_app

"""
Isaac Sim evaluation script utilities.

This module intentionally keeps behavior identical to the eval_*.py scripts:
- cleanup order matches existing scripts to avoid camera/replicator shutdown issues
- lighting setup is opt-in (call setup_all_lighting when needed)
- signal handlers call cleanup and exit
"""

def cleanup_simulation(
    *,
    env: Optional[Any] = None,
    simulation_app: Optional[Any] = None,
    stop_event: Optional[Any] = None,
    planning_thread_obj: Optional[Any] = None,
    fps_writer: Optional[list] = None,
) -> None:
    """Best-effort cleanup for Isaac Sim + env + cameras + writers."""
    print("[INFO] Starting cleanup...")
    try:
        # 1) stop planning thread
        try:
            if stop_event is not None:
                stop_event.set()
            if planning_thread_obj is not None and getattr(planning_thread_obj, "is_alive", lambda: False)():
                print("  Stopping planning thread...")
                planning_thread_obj.join(timeout=3)
        except Exception as e:
            print(f"  Warning: planning thread cleanup failed: {e}")

        # 2) close video writers
        if fps_writer is not None:
            print("  Closing video writers...")
            for writer in fps_writer:
                try:
                    writer.close()
                except Exception:
                    pass

        # 3) detach camera sensors annotators to avoid __del__ issues
        if env is not None:
            print("  Detaching camera sensors...")
            try:
                unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else env
                if hasattr(unwrapped_env, "scene") and hasattr(unwrapped_env.scene, "sensors"):
                    camera_sensor = unwrapped_env.scene.sensors.get("camera_sensor")
                    if camera_sensor is not None:
                        if hasattr(camera_sensor, "_annotators"):
                            for annotator in camera_sensor._annotators:
                                try:
                                    annotator.detach()
                                except Exception:
                                    pass
                        camera_sensor._annotators = []
                        camera_sensor._sensor_prims = []
            except Exception as e:
                print(f"  Warning: Camera cleanup failed: {e}")

        # 4) close env
        if env is not None:
            print("  Closing environment...")
            try:
                unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else env
                if hasattr(unwrapped_env, "scene"):
                    try:
                        unwrapped_env.scene.reset()
                    except Exception:
                        pass
                env.close()
            except Exception as e:
                print(f"  Warning: Error closing env: {e}")

        # 5) clear simulation context
        print("  Clearing simulation context...")
        try:
            from isaaclab.sim import SimulationContext

            sim_context = SimulationContext.instance()
            if sim_context is not None:
                sim_context.clear_all_callbacks()
                SimulationContext.clear_instance()
        except Exception as e:
            print(f"  Warning: SimulationContext cleanup failed: {e}")

        # 6) clean Isaac Sim components
        if simulation_app is not None:
            print("  Cleaning Isaac Sim components...")
            try:
                import omni.replicator.core as rep

                rep.orchestrator.stop()
            except Exception as e:
                print(f"  Warning: Replicator cleanup failed: {e}")

            try:
                import carb

                settings = carb.settings.get_settings()
                settings.set("/exts/omni.syntheticdata/enabled", False)
                simulation_app.update()
            except Exception as e:
                print(f"  Warning: SyntheticData disable failed: {e}")

            try:
                import omni.usd

                context = omni.usd.get_context()
                if context:
                    context.close_stage()
                    for _ in range(3):
                        simulation_app.update()
            except Exception as e:
                print(f"  Warning: Stage cleanup failed: {e}")

            import gc

            gc.collect()

            print("  Closing simulation app...")
            try:
                simulation_app.close()
            except Exception as e:
                print(f"  Warning: Simulation close error: {e}")

        # 7) clear CUDA cache
        try:
            import torch

            if torch.cuda.is_available():
                print("  Clearing CUDA cache...")
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
        except Exception as e:
            print(f"  Warning: CUDA cleanup failed: {e}")

        print("[INFO] Cleanup complete")
    except Exception as e:
        print(f"[ERROR] Cleanup failed: {e}")
        import traceback

        traceback.print_exc()


def register_signal_handlers(
    *,
    get_env: Callable[[], Optional[Any]],
    get_simulation_app: Callable[[], Optional[Any]],
    stop_event: Optional[Any] = None,
    get_planning_thread_obj: Optional[Callable[[], Optional[Any]]] = None,
    get_fps_writer: Optional[Callable[[], Optional[list]]] = None,
) -> None:
    """Register SIGINT/SIGTERM handlers that run cleanup then exit."""
    import signal
    import sys

    def _handler(sig, frame):
        print("\n" + "=" * 50)
        print("Received interrupt signal, shutting down...")
        print("=" * 50)
        cleanup_simulation(
            env=get_env(),
            simulation_app=get_simulation_app(),
            stop_event=stop_event,
            planning_thread_obj=get_planning_thread_obj() if get_planning_thread_obj else None,
            fps_writer=get_fps_writer() if get_fps_writer else None,
        )
        print("Exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def setup_all_lighting(env: Any) -> None:
    import omni.usd
    import carb
    from pxr import UsdLux, Gf

    settings = carb.settings.get_settings()
    stage = omni.usd.get_context().get_stage()

    # 1) Viewport Camera Light
    settings.set("/rtx/useViewLightingMode", True)
    settings.set("/rtx/viewLightingMode", 1)

    # 2) ambient and exposure
    settings.set("/rtx/sceneDb/ambientLightIntensity", 3.0)
    settings.set("/rtx/post/tonemap/enable", True)
    settings.set("/rtx/post/tonemap/exposure", 1.0)

    # 3) add per-robot camera light
    camera_sensor = env.unwrapped.scene.sensors["camera_sensor"]
    for i, camera_prim in enumerate(camera_sensor._sensor_prims):
        camera_path = str(camera_prim.GetPath())
        parent_path = "/".join(camera_path.split("/")[:-1])
        light_path = f"{parent_path}/CameraLight"

        if not stage.GetPrimAtPath(light_path):
            sphere_light = UsdLux.SphereLight.Define(stage, light_path)
            sphere_light.CreateIntensityAttr(1000.0)
            sphere_light.CreateRadiusAttr(0.1)
            sphere_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
            print(f"[INFO] ✓ Added light to robot camera {i}")

    print("[INFO] ✓ All lighting setup complete")

"""EpisodeRunner: reusable skeleton of a single evaluation episode.

Absorbs everything that was duplicated across the 6 task scripts:
  - load_episode helpers
  - observation collection (cam pose / rgb / depth / vel / x0)
  - MPC tick
  - physics step + traj_length accumulation
  - termination checks (success / timeout / stuck)
  - video writer + visualization wrapper

Task scripts subclass or just instantiate this and override the 3-4 hooks
that actually differ per task:
  - compute_goal_payload(obs)  -> anything that goes into push_plan_input as `goal`
  - check_success(obs)         -> (is_success, dist_to_goal_or_None)
  - extra_vis_kwargs(obs)      -> dict for visualize_trajectory_global_with_people
  - build_metrics(...)         -> final per-episode dict
"""

# ═══════════════════════════════════════════════════════════════════════════
# Small free helpers shared across scripts
# ═══════════════════════════════════════════════════════════════════════════
def is_stuck(position_history: deque, threshold: float = 0.1) -> bool:
    if len(position_history) < position_history.maxlen:
        return False
    cur = position_history[-1]
    return all(np.linalg.norm(cur - p) < threshold
               for p in list(position_history)[:-1])


def load_episode(episode_path: str):
    with open(episode_path, "r") as f:
        data = json.load(f)
    ep    = data["episode"]
    robot = ep["robot"]
    return (np.array(robot["start_pos"][:2]),
            float(robot["start_orientation"]),
            np.array(robot["goal_pos"][:2]))


# ═══════════════════════════════════════════════════════════════════════════
# Observation struct
# ═══════════════════════════════════════════════════════════════════════════
class Obs:
    """Per-step observation bundle."""
    __slots__ = ("cam_pos", "cam_rot", "rgb", "depth",
                 "robot_vel", "robot_ang", "x0", "step_count")

    def __init__(self, cam_pos, cam_rot, rgb, depth,
                 robot_vel, robot_ang, x0, step_count):
        self.cam_pos    = cam_pos
        self.cam_rot    = cam_rot
        self.rgb        = rgb                  # (1, H, W, 3)
        self.depth      = depth                # (1, H, W)
        self.robot_vel  = robot_vel
        self.robot_ang  = robot_ang
        self.x0         = x0                   # (1, 5)
        self.step_count = step_count


# ═══════════════════════════════════════════════════════════════════════════
# EpisodeRunner
# ═══════════════════════════════════════════════════════════════════════════
class EpisodeRunner:
    """Per-episode driver. One instance per episode (cheap to construct)."""

    def __init__(self,
                 evaluator,
                 vis_manager,
                 diff_ctrl,
                 max_steps: int,
                 save_dir: str,
                 ep_idx: int,
                 algo: str,
                 task_name: str,
                 scene_name: str,
                 goal_required: bool = True,
                 stuck_window: int = 30):
        self.evaluator    = evaluator
        self.vis_manager  = vis_manager
        self.diff_ctrl    = diff_ctrl
        self.max_steps    = max_steps
        self.ep_idx       = ep_idx
        self.algo         = algo
        self.task_name    = task_name
        self.scene_name   = scene_name
        self.goal_required = goal_required

        # Episode state
        self.step_count      = 0
        self.traj_length     = 0.0
        self.success         = False
        self.done            = False
        self.position_history = deque(maxlen=stuck_window)

        # Cached last obs (used by build_metrics)
        self.last_cam_pos = None

        # Video writer
        self._fps_writer = imageio.get_writer(
            f"{save_dir}fps_{ep_idx}.mp4", fps=10, macro_block_size=1,
        )

    # ── setup ────────────────────────────────────────────────────────────
    def prepare_planning(self):
        reset_planning_state(with_goal=self.goal_required)

    def reset_vis_from_current_pose(self):
        cam_pos, cam_rot = self.evaluator.get_camera_pose()
        initial_pose = np.array([
            cam_pos[0], cam_pos[1],
            np.arctan2(cam_rot[1, 0], cam_rot[0, 0]),
        ])
        self.vis_manager.reset(initial_robot_pose=initial_pose)

    # ── per-step ─────────────────────────────────────────────────────────
    def observe(self) -> Obs:
        cam_pos, cam_rot = self.evaluator.get_camera_pose()
        rgb   = self.evaluator.get_rgb()[None]
        depth = self.evaluator.get_depth()[None]
        robot_vel, robot_ang = self.evaluator.get_robot_velocity()
        x0 = np.array([
            cam_pos[0], cam_pos[1],
            np.arctan2(cam_rot[1, 0], cam_rot[0, 0]),
            robot_vel, robot_ang,
        ])[None]   # (1, 5)
        self.last_cam_pos = cam_pos
        return Obs(cam_pos, cam_rot, rgb, depth,
                   robot_vel, robot_ang, x0, self.step_count)

    def push_plan(self, obs: Obs, goal_payload):
        push_plan_input(goal_payload, obs.rgb, obs.depth,
                        obs.cam_pos[None], obs.cam_rot[None])

    def pop_plan(self):
        return pop_plan_output()

    def tick_mpc(self, traj_w, obs: Obs):
        """Return (joint_vels, v, w) for this step.
        If planner hasn't produced a plan yet, returns zeros.
        """
        mpc = get_current_mpc()
        if traj_w is None or mpc is None:
            return np.zeros(2, dtype=np.float32), 0.0, 0.0
        opt_u, _ = mpc.solve(obs.x0[0, :3])
        v, w = float(opt_u[1, 0]), float(opt_u[1, 1])
        jv = self.diff_ctrl.forward(np.array([v, w]))
        return (np.array(jv.joint_velocities, dtype=np.float32), v, w)

    def step_world(self, joint_vels, has_plan: bool):
        """Apply joint velocities, step physics, update counters."""
        self.evaluator.set_joint_velocities(joint_vels)
        self.evaluator.step()
        self.step_count += 1
        if has_plan:
            post_vel, _ = self.evaluator.get_robot_velocity()
            self.traj_length += post_vel * self.evaluator.step_dt

    def record_position(self, obs: Obs):
        self.position_history.append(obs.cam_pos[:2].copy())

    # ── termination ──────────────────────────────────────────────────────
    def check_termination(self,
                          dist_to_target: float | None,
                          success_dist: float = 1.0,
                          success_vel: float = 0.5,
                          slow_success: bool = True):
        """Standard nav termination: SUCCESS (if dist_to_target given),
        TIMEOUT, STUCK.

        For exploration tasks pass dist_to_target=None — success is never
        triggered, only TIMEOUT / STUCK.
        """
        # success
        if dist_to_target is not None and dist_to_target < success_dist:
            vel_ok = (self.evaluator.get_robot_velocity()[0] < success_vel
                      if slow_success else True)
            if vel_ok:
                self.success = True
                self.done    = True
                print(f"[Episode {self.ep_idx}] SUCCESS "
                      f"dist={dist_to_target:.3f} m")
                return

        # timeout
        if self.step_count >= self.max_steps:
            self.done = True
            extra = (f"  dist={dist_to_target:.3f} m"
                     if dist_to_target is not None else "")
            print(f"[Episode {self.ep_idx}] TIMEOUT{extra}")
            return

        # stuck
        if is_stuck(self.position_history):
            self.done = True
            print(f"[Episode {self.ep_idx}] STUCK")
            return

    # ── visualization ────────────────────────────────────────────────────
    def write_vis_frame(self,
                        obs: Obs,
                        traj_w,
                        all_traj_w,
                        all_vals,
                        v: float,
                        w: float,
                        goal_position=None,
                        dist_label_value: float | None = None,
                        dist_label_text: str = "target dist",
                        dist_label_unit: str = "m",
                        rgb_override=None,
                        extra_vis_kwargs: dict | None = None,
                        extra_overlay_lines: list[tuple[int, str]] | None = None):
        """Render one frame and append to the video."""
        extra_vis_kwargs = extra_vis_kwargs or {}
        rgb_show = rgb_override if rgb_override is not None else obs.rgb[0]

        vis_img = self.vis_manager.visualize_trajectory_global_with_people(
            rgb_show, obs.depth[0, :, :, None],
            self.evaluator.cam_intrinsic,
            traj_w[0], robot_pose=obs.x0[0],
            goal_position=goal_position,
            all_trajectories_points=all_traj_w[0],
            all_trajectories_values=all_vals[0],
            **extra_vis_kwargs,
        )
        vis_img = draw_box_with_text(vis_img, 0, 0, 430, 50,
                                     f"cmd lin:{v:.2f} ang:{w:.2f}")
        vis_img = draw_box_with_text(vis_img, 0, 50, 430, 50,
                                     f"actual lin:{obs.robot_vel:.2f} "
                                     f"ang:{obs.robot_ang:.2f}")
        if extra_overlay_lines:
            for (ypx, text) in extra_overlay_lines:
                vis_img = draw_box_with_text(vis_img, 0, ypx, 430, 50, text)
        if dist_label_value is not None:
            vis_img = draw_box_with_text(
                vis_img, 0, 820, 430, 50,
                f"{dist_label_text}:{dist_label_value:.2f} {dist_label_unit}")

        self._fps_writer.append_data(vis_img)
        cv2.imwrite(
            f"frame_{self.algo}_{self.task_name}_{self.scene_name}.png",
            cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR),
        )

    # ── shutdown ─────────────────────────────────────────────────────────
    def close_video(self):
        self._fps_writer.close()

"""Shared planning-thread infrastructure used by all task scripts.

A single background thread pulls planning_input, calls the task-specific
`plan_fn` (one of pointgoal_step / nogoal_step / imagegoal_step), converts
trajectories from camera frame → world frame, rebuilds the MPC, and writes
into planning_output.

Task scripts never touch the raw locks — they call:
    push_plan_input(goal, rgb, depth, cam_pos, cam_rot)
    pop_plan_output()

Task scripts start the thread with:
    from utils_tasks.planning_runtime import start_planning_thread
    from utils_tasks.client_utils import pointgoal_step
    thread = start_planning_thread(plan_fn=pointgoal_step,
                                   port=args.port, speed=args.speed,
                                   goal_required=True)
"""

# ═══════════════════════════════════════════════════════════════════════════
# Shared state (module-global, single-producer / single-consumer)
# ═══════════════════════════════════════════════════════════════════════════
planning_input  = PlanningInput()
planning_output = PlanningOutput()
_input_lock  = threading.Lock()
_output_lock = threading.Lock()
stop_event   = threading.Event()

# Most recent MPC, rebuilt each planner step.
mpc: MPC_Controller | None = None


# ═══════════════════════════════════════════════════════════════════════════
# Helpers for task scripts (replace manual locking)
# ═══════════════════════════════════════════════════════════════════════════
def reset_planning_state(with_goal: bool = True):
    """Clear planning_input/output at the start of each episode."""
    with _input_lock:
        if with_goal:
            planning_input.current_goal = None
        planning_input.current_image = None
        planning_input.current_depth = None
        planning_input.camera_pos    = None
        planning_input.camera_rot    = None
    with _output_lock:
        planning_output.trajectory_points_world = None
        planning_output.all_trajectories_world  = None
        planning_output.all_values_camera       = None
        planning_output.is_planning             = False


def push_plan_input(goal, rgb, depth, cam_pos, cam_rot):
    """Write a fresh observation into planning_input.

    goal may be None (for exploration / nogoal tasks).
    """
    with _input_lock:
        planning_input.current_goal  = goal
        planning_input.current_image = rgb
        planning_input.current_depth = depth
        planning_input.camera_pos    = cam_pos
        planning_input.camera_rot    = cam_rot


def pop_plan_output():
    """Return (traj_w, all_traj_w, all_vals) or (None, None, None) if no plan yet."""
    with _output_lock:
        if planning_output.trajectory_points_world is None:
            return None, None, None
        return (planning_output.trajectory_points_world.copy(),
                planning_output.all_trajectories_world.copy(),
                planning_output.all_values_camera.copy())


def get_current_mpc() -> MPC_Controller | None:
    return mpc


# ═══════════════════════════════════════════════════════════════════════════
# The planning thread
# ═══════════════════════════════════════════════════════════════════════════
def _snapshot_input(goal_required: bool):
    """Atomically snapshot planning_input. Returns None if not ready."""
    with _input_lock:
        required = [planning_input.current_image, planning_input.current_depth,
                    planning_input.camera_pos, planning_input.camera_rot]
        if goal_required:
            required.append(planning_input.current_goal)
        if any(x is None for x in required):
            return None
        snap = dict(
            image      = planning_input.current_image.copy(),
            depth      = planning_input.current_depth.copy(),
            camera_pos = planning_input.camera_pos.copy(),
            camera_rot = planning_input.camera_rot.copy(),
            goal       = (planning_input.current_goal.copy()
                          if planning_input.current_goal is not None else None),
        )
    return snap


def _cam_to_world_traj(traj_cam: np.ndarray,
                       camera_pos: np.ndarray,
                       camera_rot: np.ndarray) -> np.ndarray:
    """Convert a single trajectory from camera frame to world frame (xy only)."""
    out = []
    for pt in traj_cam:
        pt_local = np.array([pt[0], pt[1], 0.0])
        pt_world = camera_pos + camera_rot @ pt_local
        out.append(pt_world[:2])
    return np.array(out)


def _run_planner_once(plan_fn,
                      snap: dict,
                      goal_required: bool,
                      port: int,
                      speed: float):
    """One iteration: call plan_fn, convert to world, update MPC, write output."""
    global mpc

    # 1) Call the task-specific plan_fn
    if goal_required:
        traj_cam, all_traj_cam, all_vals_cam = plan_fn(
            snap["goal"], snap["image"], snap["depth"], port=port
        )
    else:
        traj_cam, all_traj_cam, all_vals_cam = plan_fn(
            snap["image"], snap["depth"], port=port
        )

    # 2) Convert camera-frame trajectories to world frame
    batch_opt_world, batch_all_world = [], []
    for idx in range(traj_cam.shape[0]):
        traj_w = _cam_to_world_traj(traj_cam[idx],
                                    snap["camera_pos"][idx],
                                    snap["camera_rot"][idx])
        batch_opt_world.append(traj_w)

        # Rebuild MPC from the latest optimal trajectory.
        mpc = MPC_Controller(traj_w,
                             desired_v=speed,
                             v_max=speed,
                             w_max=speed)

        batch_all = []
        for traj in all_traj_cam[idx]:
            batch_all.append(
                _cam_to_world_traj(traj, snap["camera_pos"][idx],
                                   snap["camera_rot"][idx])
            )
        batch_all_world.append(batch_all)

    # 3) Publish
    with _output_lock:
        planning_output.trajectory_points_world = np.array(batch_opt_world)
        planning_output.all_trajectories_world  = np.array(batch_all_world)
        planning_output.all_values_camera       = all_vals_cam
        planning_output.is_planning             = False
        planning_output.planning_error          = None


def _planning_thread_fn(plan_fn, port: int, speed: float, goal_required: bool):
    while not stop_event.is_set():
        try:
            snap = _snapshot_input(goal_required)
            if snap is None:
                time.sleep(0.01)
                continue

            with _output_lock:
                planning_output.is_planning = True

            _run_planner_once(plan_fn, snap, goal_required, port, speed)

        except Exception as e:
            with _output_lock:
                planning_output.is_planning    = False
                planning_output.planning_error = str(e)
            print(f"[Planning] error: {e}")
        time.sleep(0.1)


def start_planning_thread(plan_fn, port: int, speed: float,
                          goal_required: bool = True) -> threading.Thread:
    """Spawn the background planning thread.

    plan_fn: one of client_utils.{pointgoal_step, nogoal_step, imagegoal_step}
    goal_required: whether plan_fn needs `current_goal` (False for nogoal/exploration)
    """
    t = threading.Thread(
        target=_planning_thread_fn,
        args=(plan_fn, port, speed, goal_required),
        daemon=True,
    )
    t.start()
    return t

"""
For Exploration
"""
def update_occupancy(global_pcd, camera_int, current_pos, current_rot, robot_rgb, robot_depth):
    """Update voxelized global pointcloud + compute explore area.

    This is copied from the eval_*_nogoal scripts to keep behavior identical.
    """
    from isaaclab.sensors.camera.utils import create_pointcloud_from_rgbd
    import torch

    from utils_tasks.basic_utils import cpu_pointcloud_from_array

    filter_rgb = robot_rgb.clone() if torch.is_tensor(robot_rgb) \
                else torch.as_tensor(robot_rgb)
    filter_depth = robot_depth.clone() if torch.is_tensor(robot_depth) \
                else torch.as_tensor(robot_depth)
    filter_depth[filter_depth > 5.0] = 0
    points, colors = create_pointcloud_from_rgbd(
        camera_int,
        filter_depth,
        filter_rgb,
        position=current_pos,
        orientation=current_rot,
    )
    current_pcd = cpu_pointcloud_from_array(points.cpu().numpy(), colors.cpu().numpy())
    global_pcd = (global_pcd + current_pcd).voxel_down_sample(0.05)
    point_values = np.array(global_pcd.points)
    navigable_pcd = global_pcd.select_by_index(
        np.where(point_values[:, 2] < np.quantile(point_values[:, 2], 0.25) + 0.1)[0]
    )
    navigable_values = np.array(navigable_pcd.points)
    occupancy_dimension = np.ceil((navigable_values.max(axis=0) - navigable_values.min(axis=0)) / 0.1).astype(np.int32)
    occupancy_dimension[0] = max(occupancy_dimension[0], 1)
    occupancy_dimension[1] = max(occupancy_dimension[1], 1)
    occupancy_dimension[2] = max(occupancy_dimension[2], 1)
    occupancy_grid = np.zeros(occupancy_dimension)
    occupancy_index = np.floor((navigable_values - navigable_values.min(axis=0)) / 0.1).astype(np.int32)
    occupancy_grid[occupancy_index[:, 0], occupancy_index[:, 1], occupancy_index[:, 2]] = 1
    explore_area = occupancy_grid.sum() * 0.01
    return global_pcd, navigable_pcd, explore_area

"""Build the per-episode metrics dict. Two flavors:
  - navigation:   point/image/social/dynpoint (success/SPL based)
  - exploration:  nogoal/dynnogoal            (area/time based)

Both optionally take a `soc` dict from SocialMetricsTracker.get_metrics().
"""

def _merge_social(d: dict, soc: dict | None):
    if soc is None:
        return d
    d.update({
        "collision":       soc["collision"],
        "collision_count": soc["collision_count"],
        "min_distance":    soc["min_distance"],
        "avg_distance":    soc["avg_distance"],
        "psi_count":       soc["psi_count"],
        "psi_time":        soc["psi_time"],
        "PSC":             soc["PSC"],
        "SC":              soc["SC"],
    })
    return d


def build_navigation_metrics(ep_idx: int,
                             success: bool,
                             initial_dist: float,
                             final_dist: float,
                             step_count: int,
                             step_dt: float,
                             traj_length: float,
                             soc: dict | None = None) -> dict:
    spl = float(
        (initial_dist or 0.0) / max(traj_length, 1e-3)
    )
    spl = max(0.0, min(1.0, spl)) * float(success)

    d = {
        "episode":           ep_idx,
        "success":           int(success),
        "spl":               spl,
        "time_to_goal":      step_count * step_dt,
        "initial_distance":  initial_dist or 0.0,
        "final_distance":    final_dist,
        "trajectory_length": traj_length,
    }
    return _merge_social(d, soc)


def build_exploration_metrics(ep_idx: int,
                              step_count: int,
                              step_dt: float,
                              explore_area: float,
                              traj_length: float,
                              soc: dict | None = None) -> dict:
    d = {
        "episode":           ep_idx,
        "time":              step_count * step_dt,
        "area":              float(explore_area),
        "trajectory_length": traj_length,
    }
    if soc is not None:
        d["total_time"] = soc.get("total_time", step_count * step_dt)
    return _merge_social(d, soc)


def print_episode_metrics(ep_idx: int, metrics: dict):
    print(f"\n=== Metrics Episode {ep_idx} ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

def load_episode_from_npy(npy_path: str, idx: int):
    """Load one episode from a .npy sample file.

    Column layout (matches imagenav_reset):
        [start_x, start_y, goal_x, goal_y, yaw, ...]

    Returns:
        start_pos  : np.ndarray shape (2,)
        start_yaw  : float  (radians)
        goal_world : np.ndarray shape (2,)
    """
    samples = np.load(npy_path)
    row = samples[idx % len(samples)]
    start_pos  = np.array([row[0], row[1]], dtype=np.float64)
    goal_world = np.array([row[2], row[3]], dtype=np.float64)
    start_yaw  = float(row[4])
    return start_pos, start_yaw, goal_world