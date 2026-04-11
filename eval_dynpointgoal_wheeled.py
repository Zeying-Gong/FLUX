"""
Dynamic point-goal navigation toward a moving pedestrian in Isaac Sim.

Uses live target position from the character manager, NavDP point-goal planning,
MPC control, and social metrics across episode JSON files.
"""
import argparse

parser = argparse.ArgumentParser(description="Dynamic point-goal navigation with moving pedestrians")
parser.add_argument(
    "--scene_dir", type=str, default="/workspace/FLUX/assets/dynbench/isaacsim_scene",
    help="Directory containing scene folders",
)
parser.add_argument("--scene_index", type=int, default=5, help="Scene index within scene_dir")
parser.add_argument("--scene_scale", type=float, default=1.0, help="Scene scale factor")
parser.add_argument(
    "--stop_threshold", type=float, default=-3.0,
    help="Stop threshold for exploration turn (critic value)",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episode JSON files / rollouts")
parser.add_argument("--speed", type=float, default=0.5, help="Desired linear speed (m/s)")
parser.add_argument("--port", type=int, default=9999, help="NavDP server port")
parser.add_argument("--gpu_id", type=int, default=0, help="CUDA device id when not using multi-GPU")
args_cli = parser.parse_args()

import os
import sys

print(f"GPU {args_cli.gpu_id}, Scene {args_cli.scene_index}")
print(f"OMNI_USER_DATA_DIR: {os.environ.get('OMNI_USER_DATA_DIR', 'NOT SET')}")
print(f"CARB_APP_DATA_DIR: {os.environ.get('CARB_APP_DATA_DIR', 'NOT SET')}")

from isaaclab.app import AppLauncher

HEADLESS = True
MULTI_GPU = False
NUM_GPUS = 3

CUSTOM_APP_PATH = "/workspace/IsaacLab/apps/isaacsim_4_5/isaaclab.python.dyn.kit"

launcher_kwargs = {
    "headless": HEADLESS,
    "enable_cameras": True,
    "experience": CUSTOM_APP_PATH,
}

if MULTI_GPU:
    launcher_kwargs["multi_gpu"] = True
else:
    launcher_kwargs["device"] = f"cuda:{args_cli.gpu_id}"

app_launcher = AppLauncher(**launcher_kwargs)
simulation_app = app_launcher.app

if MULTI_GPU:
    import carb
    settings = carb.settings.get_settings()
    settings.set("/renderer/multiGpu/enabled", True)
    settings.set("/renderer/multiGpu/maxGpuCount", NUM_GPUS)
    settings.set("/renderer/multiGpu/autoEnable", True)
    print(f"[INFO] Multi-GPU enabled with {NUM_GPUS} GPUs")
else:
    print(f"[INFO] Single GPU mode: GPU {args_cli.gpu_id}")

import omni
import cv2
import carb
import numpy as np
import imageio
import csv
import torch
import open3d as o3d
import asyncio
from scipy.spatial.transform import Rotation as R
from pxr import Usd, Sdf
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from wheeled_robots.controllers.differential_controller import DifferentialController
import time
import threading

from utils_tasks.basic_utils import PlanningInput, PlanningOutput, find_usd_path, write_metrics, draw_box_with_text, adjust_usd_scale
from configs.robots import *
from configs.scenes import *
from configs.tasks import *
from utils_tasks.client_utils import navigator_reset, pointgoal_step
from utils_tasks.visualization_utils import VisualizationManager
from utils_tasks.tracking_utils import MPC_Controller
from utils_tasks.sim_utils import cleanup_simulation, register_signal_handlers, setup_all_lighting
from socialnav_metrics import SocialMetricsTracker, get_people_positions

planning_input = PlanningInput()
planning_output = PlanningOutput()
input_lock = threading.Lock()
output_lock = threading.Lock()
stop_event = threading.Event()
vis_manager = [VisualizationManager(history_size=5) for i in range(args_cli.num_envs)]
mpc = None

register_signal_handlers(
    get_env=lambda: globals().get("env", None),
    get_simulation_app=lambda: globals().get("simulation_app", None),
    stop_event=stop_event,
    get_planning_thread_obj=lambda: globals().get("planning_thread_obj", None),
    get_fps_writer=lambda: globals().get("fps_writer", None),
)


def planning_thread(env, camera_intrinsic):
    """Background thread: NavDP point-goal planning, world-frame trajectories, MPC state."""
    global mpc
    while not stop_event.is_set():
        try:
            with input_lock:
                if planning_input.current_goal is None or planning_input.current_image is None or planning_input.current_depth is None or planning_input.camera_pos is None or planning_input.camera_rot is None:
                    time.sleep(0.01)
                    continue
                goal = planning_input.current_goal.copy()
                image = planning_input.current_image.copy()
                depth = planning_input.current_depth.copy()
                camera_pos = planning_input.camera_pos.copy()
                camera_rot = planning_input.camera_rot.copy()
            with output_lock:
                planning_output.is_planning = True

            trajectory_points_camera, all_trajectories_camera, all_values_camera = pointgoal_step(
                goal, image, depth, port=args_cli.port
            )

            batch_optimal_points_world = []
            for idx in range(trajectory_points_camera.shape[0]):
                trajectory_points_world = []
                for i, point in enumerate(trajectory_points_camera[idx]):
                    if i < 0:
                        continue
                    point_local = np.array([point[0], point[1], 0.0])
                    point_world = camera_pos[idx] + camera_rot[idx] @ point_local
                    trajectory_points_world.append(point_world[:2])
                trajectory_points_world = np.array(trajectory_points_world)
                batch_optimal_points_world.append(trajectory_points_world)
                mpc = MPC_Controller(
                    trajectory_points_world,
                    desired_v=args_cli.speed,
                    v_max=args_cli.speed,
                    w_max=args_cli.speed
                )
            batch_optimal_points_world = np.array(batch_optimal_points_world)

            batch_all_points_world = []
            for idx in range(all_trajectories_camera.shape[0]):
                all_trajectories_world = []
                for traj_camera in all_trajectories_camera[idx]:
                    traj_world = []
                    for point in traj_camera:
                        point_local = np.array([point[0], point[1], 0.0])
                        point_world = camera_pos[idx] + camera_rot[idx] @ point_local
                        traj_world.append(point_world[:2])
                    all_trajectories_world.append(np.array(traj_world))
                batch_all_points_world.append(all_trajectories_world)
            batch_all_points_world = np.array(batch_all_points_world)

            with output_lock:
                planning_output.trajectory_points_world = batch_optimal_points_world
                planning_output.all_trajectories_world = batch_all_points_world
                planning_output.all_values_camera = all_values_camera
                planning_output.is_planning = False
                planning_output.planning_error = None

        except Exception as e:
            print(f"Planning error: {e}")
            with output_lock:
                planning_output.is_planning = False
                planning_output.planning_error = str(e)
        time.sleep(0.1)


def get_target_person_position_from_env(env):
    """Return the first managed character world position [x,y,z], or None."""
    try:
        from omni.anim.people.scripts.global_character_position_manager import GlobalCharacterPositionManager

        char_manager = GlobalCharacterPositionManager.get_instance()
        all_chars = char_manager.get_all_managed_characters()

        if len(all_chars) == 0:
            return None

        target_path = list(all_chars)[0]
        pos = char_manager.get_character_current_pos(target_path)
        return np.array([float(pos[0]), float(pos[1]), float(pos[2])])

    except Exception as e:
        print(f"Error getting target person position: {e}")
        return None


def main():
    scene_list = os.listdir(args_cli.scene_dir)
    scene_list.sort()

    scene_name = scene_list[args_cli.scene_index]
    scene_path = os.path.join(args_cli.scene_dir, scene_name) + "/"
    usd_path, init_path = find_usd_path(scene_path, 'pointgoal')

    print(f"[INFO] Validating episode files in {scene_path}")
    episode_json_files = []
    for episode_id in range(args_cli.num_episodes):
        episode_json_path = os.path.join(scene_path, f"episode_{episode_id}.json")
        if not os.path.exists(episode_json_path):
            raise RuntimeError(
                f"Missing episode file: {episode_json_path}\n"
                f"Expected {args_cli.num_episodes} episodes (0 to {args_cli.num_episodes - 1})"
            )
        episode_json_files.append(episode_json_path)

    print(f"[INFO] Found all {args_cli.num_episodes} episode files OK")

    first_episode_path = episode_json_files[0]

    scene_config = DynPointGoalSceneCfg()
    scene_config.num_envs = args_cli.num_envs
    scene_config.env_spacing = 0.0
    scene_config.terrain = BENCH_TERRAIN_CFG
    scene_config.terrain.usd_path = usd_path
    scene_config.goal = GOAL_CFG
    scene_config.robot = DINGO_CFG
    scene_config.camera_sensor = DINGO_CameraCfg
    scene_config.contact_sensor = DINGO_ContactCfg

    scene_config.people_simulation = True
    scene_config.episode_json_path = first_episode_path

    env_config = DingoDynPointGoalCfg()
    env_config.scene = scene_config
    env_config.events.reset_pose.params = {
        "episode_json_dir": scene_path,
        "num_episodes": args_cli.num_episodes,
        'height_offset': 0.1,
        'robot_visible': False,
        'light_enabled': False
    }
    print(f"[DEBUG] Episode JSON dir: {scene_path}")
    print(f"[DEBUG] First episode path: {first_episode_path}")
    env = ManagerBasedRLEnv(env_config)
    env = RslRlVecEnvWrapper(env)
    adjust_usd_scale(scale=args_cli.scene_scale)

    episode_steps = np.zeros((scene_config.num_envs,), dtype=np.int64)

    if scene_name in ['Hospital', 'Jetracer', 'Office']:
        setup_all_lighting(env)
        for _ in range(5):
            simulation_app.update()

    PREHEAT_STEPS = 10
    for _ in range(PREHEAT_STEPS):
        action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
        obs, rewards, dones, infos = env.step(action)

    camera_intrinsic = env.unwrapped.scene.sensors['camera_sensor'].data.intrinsic_matrices[0]

    planning_thread_obj = threading.Thread(target=planning_thread, args=(env, camera_intrinsic))
    planning_thread_obj.daemon = True
    planning_thread_obj.start()

    controller = DifferentialController(
        name="simple_control",
        wheel_radius=DINGO_WHEEL_RADIUS,
        wheel_base=DINGO_WHEEL_BASE
    )
    algo = navigator_reset(
        camera_intrinsic.cpu().numpy(),
        batch_size=scene_config.num_envs,
        stop_threshold=args_cli.stop_threshold,
        port=args_cli.port
    )

    if algo == "fallback_algo":
        print("[ERROR] Navigator server connection failed!")
        print(f"[INFO] Please start the server: python server.py --port {args_cli.port}")
        cleanup_simulation(env, simulation_app)
        sys.exit(1)

    print(f"[INFO] Connected to navigator server, algorithm: {algo}")

    episode_num = 0
    evaluation_metrics = []
    current_episode_idx = 0
    save_dir = "./metrics/dynpointgoal_%s_%s/%s/" % (algo, args_cli.scene_dir.split("/")[-1], scene_path.split("/")[-2])
    os.makedirs(save_dir, exist_ok=True)

    initial_distance_to_target = None
    fps_writer = [imageio.get_writer(save_dir + "fps_%d.mp4" % i, fps=10) for i in range(scene_config.num_envs)]

    trajectory_length = np.zeros((scene_config.num_envs))

    social_metrics_trackers = [SocialMetricsTracker() for _ in range(scene_config.num_envs)]

    camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
    camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
    camera_rot_quat = camera_rot_quat[:, [1, 2, 3, 0]]
    camera_rot = R.from_quat(camera_rot_quat).as_matrix()

    for i in range(scene_config.num_envs):
        initial_pose = np.array([
            camera_pos[i, 0],
            camera_pos[i, 1],
            np.arctan2(camera_rot[i, 1, 0], camera_rot[i, 0, 0])
        ])
        vis_manager[i].reset(initial_robot_pose=initial_pose)

    if scene_config.people_simulation:
        print("Waiting for NavMesh to be ready...")
        wait_count = 0
        while not env.unwrapped.scene.navmesh_ready and wait_count < 100:
            simulation_app.update()
            wait_count += 1
            if wait_count % 20 == 0:
                print(f"  Waiting... ({wait_count}/100)")

        if env.unwrapped.scene.navmesh_ready:
            print("NavMesh ready.")
        else:
            print("Warning: NavMesh not ready, continuing without people")

    print("[INFO] Waiting for people to respawn...")
    wait_count = 0
    max_wait = 500
    while env.unwrapped.scene._people_setup_in_progress and wait_count < max_wait:
        simulation_app.update()
        wait_count += 1
        if wait_count % 50 == 0:
            print(f"  Waiting... ({wait_count}/{max_wait})")

    if hasattr(env.env, '_recent_positions'):
        env.env._recent_positions.clear()

    frame_count = 0

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                target_person_pos = get_target_person_position_from_env(env)

                if target_person_pos is None:
                    action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
                    obs, rewards, dones, infos = env.step(action)
                    print("Warning: Target person not found")
                    continue

                camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
                camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
                camera_rot_quat = camera_rot_quat[:, [1, 2, 3, 0]]
                camera_rot = R.from_quat(camera_rot_quat).as_matrix()

                if initial_distance_to_target is None:
                    initial_distance_to_target = np.zeros(args_cli.num_envs)
                    print(f"Robot camera position (world): {camera_pos[0]}")
                    print(f"Target person position (world): {target_person_pos}")

                    env_origin = env.unwrapped.scene.env_origins[0].cpu().numpy()
                    robot_local = camera_pos[0] - env_origin
                    person_local = target_person_pos - env_origin
                    print(f"Robot position (scene local): {robot_local}")
                    print(f"Target person position (scene local): {person_local}")

                    for i in range(args_cli.num_envs):
                        initial_distance_to_target[i] = np.linalg.norm(
                            camera_pos[i, :2] - target_person_pos[:2]
                        )
                    print(f"[INFO] Initial distance to target: {initial_distance_to_target[0]:.2f} m")

                goals = np.zeros((args_cli.num_envs, 2))
                for i in range(args_cli.num_envs):
                    rel_vec = target_person_pos[:3] - camera_pos[i]
                    rel_vec_robot = camera_rot[i].T @ rel_vec
                    goals[i] = rel_vec_robot[:2]

                images = infos['observations']['rgb'].cpu().numpy()[:, :, :, 0:3]
                depths = infos['observations']['depth'].cpu().numpy()[:, :, :]

                with input_lock:
                    planning_input.current_goal = goals.copy()
                    planning_input.current_image = images.copy()
                    planning_input.current_depth = depths.copy()
                    planning_input.camera_pos = camera_pos.copy()
                    planning_input.camera_rot = camera_rot.copy()

                robot_vel = env.unwrapped.scene.articulations['robot'].data.root_lin_vel_w[0, :2].norm().cpu().numpy()
                robot_ang_vel = env.unwrapped.scene.articulations['robot'].data.root_ang_vel_w[0, 2].cpu().numpy()

                x0 = np.stack([
                    camera_pos[:, 0], camera_pos[:, 1],
                    np.arctan2(camera_rot[:, 1, 0], camera_rot[:, 0, 0]),
                    [robot_vel], [robot_ang_vel]
                ], axis=-1)

                current_trajectory = None
                current_all_trajectories = None
                current_all_values = None
                with output_lock:
                    if planning_output.trajectory_points_world is not None:
                        current_trajectory = planning_output.trajectory_points_world.copy()
                        current_all_trajectories = planning_output.all_trajectories_world.copy()
                        current_all_values = planning_output.all_values_camera.copy()

                if current_trajectory is not None:
                    action_list = []
                    for i in range(args_cli.num_envs):
                        people_positions, people_char_paths, pos_get_flag = get_people_positions(env)

                        if pos_get_flag:
                            social_metrics_trackers[i].update(camera_pos[i], people_positions, env.unwrapped.step_dt)

                            vis_image = vis_manager[i].visualize_trajectory_global_with_people(
                                images[i], depths[i][:, :, None], camera_intrinsic.cpu().numpy(),
                                current_trajectory[i],
                                robot_pose=x0[i],
                                goal_position=target_person_pos[:2],
                                all_trajectories_points=current_all_trajectories[i],
                                all_trajectories_values=current_all_values[i],
                                people_positions=people_positions,
                                people_positions_dict=people_char_paths,
                            )
                        else:
                            vis_image = vis_manager[i].visualize_trajectory_global(
                                images[i], depths[i][:, :, None], camera_intrinsic.cpu().numpy(),
                                current_trajectory[i],
                                robot_pose=x0[i],
                                goal_position=target_person_pos[:2],
                                all_trajectories_points=current_all_trajectories[i],
                                all_trajectories_values=current_all_values[i]
                            )

                        if mpc is None:
                            continue

                        opt_u_controls, opt_x_states = mpc.solve(x0[i, :3])
                        v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]
                        action = torch.tensor([v, w], device="cuda:0")
                        action_cpu = action.cpu().numpy()
                        joint_velocities = controller.forward(action_cpu).joint_velocities
                        action_list.append(joint_velocities)

                        try:
                            vis_image = draw_box_with_text(
                                vis_image, 0, 0, 430, 50,
                                "cmd lin.:%.2f ang.:%.2f" % (v, w)
                            )
                            vis_image = draw_box_with_text(
                                vis_image, 0, 50, 430, 50,
                                "actual lin.:%.2f ang.:%.2f" % (robot_vel, robot_ang_vel)
                            )
                            if current_all_values is not None:
                                vis_image = draw_box_with_text(
                                    vis_image, 0, 770, 430, 50,
                                    "critic max:%.2f min:%.2f" % (np.max(current_all_values[i]), np.min(current_all_values[i]))
                                )

                            dist_to_target = np.linalg.norm(camera_pos[i, :2] - target_person_pos[:2])
                            vis_image = draw_box_with_text(
                                vis_image, 0, 820, 430, 50,
                                f"target dist:{dist_to_target:.2f} m"
                            )

                            if frame_count > 0:
                                cv2.imwrite(
                                    f"frame_{algo}_dynpointgoal_{scene_name}.png",
                                    cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                                )
                                fps_writer[i].append_data(vis_image)
                            frame_count += 1
                        except Exception:
                            pass

                    action = torch.as_tensor(np.stack(action_list, axis=0), device="cuda:0")
                    obs, rewards, dones, infos = env.step(action)
                    episode_steps += 1
                    trajectory_length += (infos['observations']['policy'][:, 0] * env.unwrapped.step_dt).cpu().numpy()
                else:
                    action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
                    obs, rewards, dones, infos = env.step(action)
                    episode_steps += 1
                    print("Trajectory not ready; zero action")

                for i in range(args_cli.num_envs):
                    if dones[i] == True:
                        episode_num += 1
                        navigator_reset(env_id=i, port=args_cli.port)

                        final_target_pos = get_target_person_position_from_env(env)
                        if final_target_pos is not None:
                            final_distance = np.linalg.norm(camera_pos[i, :2] - final_target_pos[:2])
                        else:
                            final_distance = float('inf')

                        # Success from env log (avoids off-by-one frame vs raw distance)
                        success_flag = infos['log']['Episode_Termination/arrive_goal']
                        spl = np.clip(initial_distance_to_target[i] / max(trajectory_length[i], 0.01), 0, 1) * success_flag

                        social_metrics = social_metrics_trackers[i].get_metrics()

                        evaluation_metrics.append({
                            'episode': current_episode_idx,
                            'success': success_flag,
                            'time_to_goal': episode_steps[i] * env.env.step_dt,
                            'initial_distance': initial_distance_to_target[i],
                            'final_distance': min(1.0, final_distance) if success_flag else final_distance,
                            'trajectory_length': trajectory_length[i],
                            'collision': social_metrics['collision'],
                            'collision_count': social_metrics['collision_count'],
                            'min_distance': social_metrics['min_distance'],
                            'avg_distance': social_metrics['avg_distance'],
                            'psi_count': social_metrics['psi_count'],
                            'psi_time': social_metrics['psi_time'],
                            'PSC': social_metrics['PSC'],
                            'SC': social_metrics['SC'],
                        })

                        print(f"\n=== Metrics of Episode {current_episode_idx} in Scene {scene_name} ===")
                        for key, value in evaluation_metrics[-1].items():
                            if isinstance(value, float):
                                print(f"  {key}: {value:.3f}")
                            else:
                                print(f"  {key}: {value}")

                        fps_writer[i].close()
                        current_episode_idx += 1
                        write_metrics(evaluation_metrics, save_dir + "metric.csv")

                        if current_episode_idx >= args_cli.num_episodes:
                            print(f"\n[INFO] All {args_cli.num_episodes} episodes completed!")
                            print(f"[INFO] Final metrics saved to {save_dir}metric.csv")

                            print("[INFO] Pre-cleanup camera sensors...")
                            try:
                                camera_sensor = env.unwrapped.scene.sensors['camera_sensor']
                                if hasattr(camera_sensor, '_annotators'):
                                    for annotator in camera_sensor._annotators:
                                        try:
                                            annotator.detach()
                                        except Exception:
                                            pass
                                    camera_sensor._annotators = []
                            except Exception as e:
                                print(f"Warning: Camera pre-cleanup failed: {e}")

                            cleanup_simulation(env, simulation_app)
                            return

                        if hasattr(env.env, '_recent_positions'):
                            env.env._recent_positions.clear()
                        with output_lock:
                            planning_output.trajectory_points_world = None
                            planning_output.all_trajectories_world = None
                            planning_output.all_values_camera = None
                            planning_output.is_planning = False
                            planning_output.planning_error = None

                        new_episode_path = os.path.join(scene_path, f"episode_{current_episode_idx}.json")
                        env.unwrapped.scene.cfg.episode_json_path = new_episode_path

                        if env.unwrapped.scene.people is not None or env.unwrapped.scene._people_setup_in_progress:
                            while env.unwrapped.scene._people_setup_in_progress:
                                simulation_app.update()

                            asyncio.ensure_future(env.unwrapped.scene._reset_people_for_episode(new_episode_path))

                            wait_count = 0
                            while not env.unwrapped.scene._people_setup_in_progress and wait_count < 10:
                                simulation_app.update()
                                wait_count += 1

                            print("[INFO] Waiting for people to respawn...")
                            wait_count = 0
                            max_wait = 500
                            while env.unwrapped.scene._people_setup_in_progress and wait_count < max_wait:
                                simulation_app.update()
                                wait_count += 1
                                if wait_count % 50 == 0:
                                    print(f"  Waiting... ({wait_count}/{max_wait})")

                        initial_distance_to_target = None

                        fps_writer[i] = imageio.get_writer(save_dir + "fps_%d.mp4" % current_episode_idx, fps=10)
                        trajectory_length[i] = 0.0

                        camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
                        camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
                        camera_rot_quat = camera_rot_quat[:, [1, 2, 3, 0]]
                        camera_rot = R.from_quat(camera_rot_quat).as_matrix()
                        initial_pose = np.array([
                            camera_pos[i, 0], camera_pos[i, 1],
                            np.arctan2(camera_rot[i, 1, 0], camera_rot[i, 0, 0])
                        ])
                        vis_manager[i].reset(initial_robot_pose=initial_pose)
                        frame_count = 0
                        episode_steps[i] = 0
                        social_metrics_trackers[i].reset()

                        break

                if episode_num > args_cli.num_episodes:
                    break
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected!")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_simulation(
            env=env if 'env' in locals() else None,
            simulation_app=simulation_app if 'simulation_app' in locals() else None
        )


if __name__ == "__main__":
    main()
