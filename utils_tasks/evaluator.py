"""IsaacSimEvaluator: single class that wraps World + Robot + policy camera, with
optional 'enable_*' methods for task-specific extras (metric cam, goal cam).

IMPORTANT: this module imports omni/isaacsim at module load time, so it MUST
only be imported AFTER SimulationApp has been launched
(see utils_tasks.sim_launcher.launch_sim).
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import omni
import omni.usd
import omni.timeline
import omni.replicator.core as rep

import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera as IsaacCamera
from isaacsim.core.prims import SingleArticulation as ArticulationView
from isaacsim.core.utils.types import ArticulationAction

from utils_tasks.basic_utils import adjust_usd_scale


# ═══════════════════════════════════════════════════════════════════════════
# Constants (moved here from every task script)
# ═══════════════════════════════════════════════════════════════════════════
# Robot
ROBOT_USD_PATH    = "/workspace/FLUX/assets/robots/dingo_fixed.usd"
ROBOT_PRIM_PATH   = "/World/Robot"
CAMERA_PRIM_PATH  = "/World/Robot/base_link/front_cam"
ROBOT_JOINT_NAMES = ["left_wheel_joint", "right_wheel_joint"]

# Policy camera (mirrors DINGO_CameraCfg)
CAM_HEIGHT, CAM_WIDTH = 360, 640
SIM_STEP_DT   = 0.01   # physics dt
DECIMATION    = 15     # physics steps per control step  → ~10 Hz

# Metric (exploration) camera
METRIC_CAMERA_PRIM_PATH             = "/World/Robot/base_link/narritor_cam"  # spelling intentional
METRIC_CAM_HEIGHT, METRIC_CAM_WIDTH = 90, 160

# Goal camera (image-goal task)
GOAL_CAMERA_PRIM_PATH           = "/World/GoalCam"
GOAL_CAM_HEIGHT, GOAL_CAM_WIDTH = 360, 640


class IsaacSimEvaluator:
    """Owns the stage, physics World, robot articulation, and policy camera.

    Task-specific extras are enabled through explicit `enable_*` calls before
    `setup()`:
        evaluator.enable_metric_camera()    # for exploration tasks
        evaluator.enable_goal_camera()      # for image-goal tasks
    """

    def __init__(self,
                 simulation_app,
                 usd_path: str,
                 scene_scale: float = 1.0):
        self._sim_app      = simulation_app
        self._usd_path     = usd_path
        self._scene_scale  = scene_scale
        self._world: World | None            = None
        self._robot: ArticulationView | None = None
        self._camera: IsaacCamera | None     = None
        self._cam_intrinsic: np.ndarray | None = None

        self._step_dt   = SIM_STEP_DT
        self._decimation = DECIMATION
        self.step_dt    = SIM_STEP_DT * DECIMATION   # control-loop dt (~0.15s)

        # Optional extras (enabled via enable_*)
        self._use_metric_cam = False
        self._metric_camera: IsaacCamera | None = None
        self._metric_cam_intrinsic: np.ndarray | None = None

        self._use_goal_cam = False
        self._goal_camera: IsaacCamera | None = None

    # ── public properties ────────────────────────────────────────────────
    @property
    def simulation_app(self):
        return self._sim_app

    @property
    def world(self):
        return self._world

    @property
    def robot(self):
        return self._robot

    @property
    def camera(self):
        return self._camera

    @property
    def cam_intrinsic(self) -> np.ndarray:
        return self._cam_intrinsic

    # ── enablers (call BEFORE setup) ─────────────────────────────────────
    def enable_metric_camera(self):
        """Add a low-res downward-looking camera for occupancy/exploration."""
        self._use_metric_cam = True

    def enable_goal_camera(self):
        """Add a standalone goal camera (teleported per-episode for image-goal task)."""
        self._use_goal_cam = True

    # ── setup ────────────────────────────────────────────────────────────
    def setup(self):
        """Open stage, initialize World, spawn robot + cameras, mount annotators."""
        print(f"[Evaluator] Opening scene: {self._usd_path}")
        omni.usd.get_context().open_stage(self._usd_path)
        for _ in range(30):
            self._sim_app.update()
        while omni.usd.get_context().get_stage_loading_status()[1] > 0:
            self._sim_app.update()

        if self._scene_scale != 1.0:
            adjust_usd_scale(scale=self._scene_scale)

        # World
        self._world = World(physics_dt=SIM_STEP_DT,
                            rendering_dt=SIM_STEP_DT * DECIMATION)
        self._world.initialize_physics()

        # Robot USD
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(ROBOT_PRIM_PATH).IsValid():
            print(f"[Evaluator] Adding robot USD: {ROBOT_USD_PATH} → {ROBOT_PRIM_PATH}")
            add_reference_to_stage(ROBOT_USD_PATH, ROBOT_PRIM_PATH)
            for _ in range(10):
                self._sim_app.update()

        # Cameras
        self._spawn_policy_camera()
        if self._use_metric_cam:
            self._spawn_metric_camera()
        if self._use_goal_cam:
            self._spawn_goal_camera()

        # Sanity check
        if not stage.GetPrimAtPath(CAMERA_PRIM_PATH).IsValid():
            print("[DEBUG] Prims under /World/Robot:")
            from pxr import Usd
            for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot")):
                print(" ", prim.GetPath())
            raise RuntimeError(f"Camera prim not found: {CAMERA_PRIM_PATH}")

        # Articulation
        self._robot = ArticulationView(name="dingo_view", prim_path=ROBOT_PRIM_PATH)
        self._world.scene.add(self._robot)
        self._world.reset()

        # Intrinsics
        self._collect_intrinsics()

        print("[Evaluator] Setup complete.")
        omni.timeline.get_timeline_interface().play()

    # ── camera spawn helpers ─────────────────────────────────────────────
    def _spawn_policy_camera(self):
        cam_rot_wxyz_ros = np.array([-0.5, 0.5, -0.5, 0.5], dtype=np.float64)
        cam_trans_ros    = np.array([0.0, 0.0, 0.3],       dtype=np.float64)

        self._camera = IsaacCamera(
            prim_path=CAMERA_PRIM_PATH,
            resolution=(CAM_WIDTH, CAM_HEIGHT),
            translation=cam_trans_ros,
        )
        for _ in range(5):
            self._sim_app.update()

        self._camera.initialize()
        self._camera.set_local_pose(
            translation=cam_trans_ros,
            orientation=cam_rot_wxyz_ros,
            camera_axes="ros",
        )
        self._camera.set_focal_length(1.4)
        self._camera.set_focus_distance(0.205)
        self._camera.set_horizontal_aperture(1.88)
        self._camera.set_clipping_range(0.01, 100.0)
        self._camera.add_distance_to_image_plane_to_frame()

        for _ in range(5):
            self._sim_app.update()
        print(f"[Evaluator] Policy camera initialized at {CAMERA_PRIM_PATH}")

    def _spawn_metric_camera(self):
        # Downward-looking top-view for occupancy
        cam_rot_wxyz_ros = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float64)
        cam_trans_ros    = np.array([0.0, 0.0, 1.0],        dtype=np.float64)

        self._metric_camera = IsaacCamera(
            prim_path=METRIC_CAMERA_PRIM_PATH,
            resolution=(METRIC_CAM_WIDTH, METRIC_CAM_HEIGHT),
            translation=cam_trans_ros,
        )
        for _ in range(5):
            self._sim_app.update()

        self._metric_camera.initialize()
        self._metric_camera.set_local_pose(
            translation=cam_trans_ros,
            orientation=cam_rot_wxyz_ros,
            camera_axes="ros",
        )
        self._metric_camera.set_focal_length(1.4)
        self._metric_camera.set_focus_distance(0.205)
        self._metric_camera.set_horizontal_aperture(1.88)
        self._metric_camera.set_clipping_range(0.01, 100.0)
        self._metric_camera.add_distance_to_image_plane_to_frame()

        for _ in range(5):
            self._sim_app.update()
        print(f"[Evaluator] Metric camera at {METRIC_CAMERA_PRIM_PATH} "
              f"({METRIC_CAM_WIDTH}x{METRIC_CAM_HEIGHT})")

    def _spawn_goal_camera(self):
        """Standalone camera; teleported to the goal pose each episode."""
        self._goal_camera = IsaacCamera(
            prim_path=GOAL_CAMERA_PRIM_PATH,
            resolution=(GOAL_CAM_WIDTH, GOAL_CAM_HEIGHT),
            translation=np.array([0.0, 0.0, 0.3], dtype=np.float64),
        )
        for _ in range(5):
            self._sim_app.update()
        self._goal_camera.initialize()
        self._goal_camera.set_focal_length(1.4)
        self._goal_camera.set_focus_distance(0.205)
        self._goal_camera.set_horizontal_aperture(1.88)
        self._goal_camera.set_clipping_range(0.01, 100.0)
        for _ in range(5):
            self._sim_app.update()
        print(f"[Evaluator] Goal camera at {GOAL_CAMERA_PRIM_PATH}")

    def _collect_intrinsics(self):
        self._cam_intrinsic = np.array(self._camera.get_intrinsics_matrix(),
                                       dtype=np.float32)
        print(f"[Evaluator] Policy camera intrinsic:\n{self._cam_intrinsic}")
        if self._use_metric_cam:
            self._metric_cam_intrinsic = np.array(
                self._metric_camera.get_intrinsics_matrix(), dtype=np.float32)
            print(f"[Evaluator] Metric camera intrinsic:\n{self._metric_cam_intrinsic}")

    # ── per-episode robot reset ──────────────────────────────────────────
    def reset_robot(self, start_pos: np.ndarray, start_yaw: float,
                    height_offset: float = 0.1):
        pos = np.array([start_pos[0], start_pos[1], height_offset],
                       dtype=np.float32)
        quat = rot_utils.euler_angles_to_quats(
            np.array([[0.0, 0.0, start_yaw]], dtype=np.float32)
        )[0]
        self._robot.set_world_pose(position=pos, orientation=quat)
        self._robot.set_joint_velocities(np.zeros(len(ROBOT_JOINT_NAMES)))
        self._robot.set_linear_velocity(np.zeros(3))
        self._robot.set_angular_velocity(np.zeros(3))
        for _ in range(5):
            self._world.step(render=False)

    # ── observations ─────────────────────────────────────────────────────
    def get_camera_pose(self):
        """Camera pose in world-convention (forward=+X, up=+Z) — matches IsaacLab."""
        pos, quat_wxyz = self._camera.get_world_pose(camera_axes="world")
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2],
                              quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
        rot = R.from_quat(quat_xyzw).as_matrix()
        return np.asarray(pos, dtype=np.float64), rot

    def get_rgb(self) -> np.ndarray:
        data = self._camera.get_rgb()
        if data is None:
            return np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
        return np.asarray(data)

    def get_depth(self) -> np.ndarray:
        data = self._camera.get_depth()
        if data is None:
            return np.zeros((CAM_HEIGHT, CAM_WIDTH), dtype=np.float32)
        d = np.asarray(data).astype(np.float32)
        # IsaacSim 对未命中像素返回 inf；planner / 可视化都按 0 处理更安全
        return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    def get_robot_velocity(self):
        lin_vel = self._robot.get_linear_velocity()
        ang_vel = self._robot.get_angular_velocity()
        return float(np.linalg.norm(lin_vel[:2])), float(ang_vel[2])

    # ── metric-camera accessors (torch, for occupancy utils) ─────────────
    def get_metric_camera_pose_torch(self, device: str = "cuda:0"):
        pos_np, quat_wxyz = self._metric_camera.get_world_pose(camera_axes="ros")
        pos_t  = torch.as_tensor(pos_np,    dtype=torch.float32, device=device)
        quat_t = torch.as_tensor(quat_wxyz, dtype=torch.float32, device=device)
        return pos_t, quat_t

    def get_metric_rgb_torch(self, device: str = "cuda:0"):
        data = self._metric_camera.get_rgb()
        if data is None:
            data = np.zeros((METRIC_CAM_HEIGHT, METRIC_CAM_WIDTH, 3),
                            dtype=np.uint8)
        return torch.as_tensor(np.asarray(data), dtype=torch.float32, device=device)

    def get_metric_depth_torch(self, device: str = "cuda:0"):
        data = self._metric_camera.get_depth()
        if data is None:
            data = np.zeros((METRIC_CAM_HEIGHT, METRIC_CAM_WIDTH),
                            dtype=np.float32)
        return torch.as_tensor(np.asarray(data), dtype=torch.float32, device=device)

    def get_metric_intrinsic_torch(self, device: str = "cuda:0"):
        return torch.as_tensor(self._metric_cam_intrinsic,
                               dtype=torch.float32, device=device)

    # ── goal-camera accessors (image-goal task) ──────────────────────────
    def place_goal_camera(self, goal_xy: np.ndarray, yaw: float,
                          height: float = 0.3):
        """Teleport goal camera to (goal_xy, yaw)."""
        pos = np.array([goal_xy[0], goal_xy[1], height], dtype=np.float64)
        # scipy returns xyzw, IsaacCamera wants wxyz
        quat_xyzw = R.from_euler('z', yaw).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0],
                              quat_xyzw[1], quat_xyzw[2]])
        self._goal_camera.set_world_pose(
            position=pos, orientation=quat_wxyz, camera_axes="world",
        )
        for _ in range(5):
            self._world.step(render=False)
            self._sim_app.update()

    def get_goal_rgb(self) -> np.ndarray:
        data = self._goal_camera.get_rgb()
        if data is None:
            return np.zeros((GOAL_CAM_HEIGHT, GOAL_CAM_WIDTH, 3),
                            dtype=np.uint8)
        return np.asarray(data)

    # ── control ──────────────────────────────────────────────────────────
    def set_joint_velocities(self, joint_vels: np.ndarray):
        self._robot.apply_action(
            ArticulationAction(joint_velocities=joint_vels)
        )

    # ── physics step ─────────────────────────────────────────────────────
    def step(self, render: bool = True):
        """Advance one control step (DECIMATION physics steps)."""
        for _ in range(self._decimation):
            self._world.step(render=False)
        if render:
            self._sim_app.update()

    # ── shutdown ─────────────────────────────────────────────────────────
    def close(self):
        print("[Evaluator] Closing...")
        try:
            rep.orchestrator.stop()
        except Exception:
            pass
        try:
            if self._world:
                self._world.stop()
        except Exception:
            pass
        try:
            omni.usd.get_context().close_stage()
        except Exception:
            pass
        try:
            self._sim_app.close()
        except Exception:
            pass
        print("[Evaluator] Closed.")