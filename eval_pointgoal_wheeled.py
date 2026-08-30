"""
Point-goal navigation evaluation (client).

Runs Isaac Sim with an async planning thread that calls the NavDP server, then
MPC tracking and differential-drive control. Records success rate, SPL, and distance.

Example:
    python eval_pointgoal_wheeled.py \\
        --port 9999 \\
        --scene_dir /path/to/scenes \\
        --scene_index 0 \\
        --scene_scale 0.01
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Point-goal navigation evaluation")
parser.add_argument(
    "--scene_dir", type=str, default="/workspace/FLUX/assets/n1_eval_scenes/cluttered_easy",
    help="Directory containing scene folders",
)
parser.add_argument(
    "--scene_index", type=int, default=0,
    help="Scene index within scene_dir",
)
parser.add_argument(
    "--scene_scale", type=float, default=1.0,
    help="Scene scale factor (e.g. 0.01 shrinks by 100x)",
)
parser.add_argument(
    "--stop_threshold", type=float, default=-3.0,
    help="Stop threshold for exploration turn (critic value)",
)
parser.add_argument(
    "--num_envs", type=int, default=1,
    help="Number of parallel environments",
)
parser.add_argument(
    "--num_episodes", type=int, default=100,
    help="Number of evaluation episodes",
)
parser.add_argument(
    "--speed", type=float, default=0.5,
    help="Desired linear speed (m/s)",
)
parser.add_argument(
    "--port", type=int, default=9999,
    help="NavDP server port",
)
parser.add_argument(
    "--builtin_pointgoal_steps", type=int, default=0,
    help="Run a checkpoint-free reactive PointGoal smoke test for N simulation steps",
)
parser.add_argument(
    "--builtin_video", type=str, default="pointgoal_builtin.mp4",
    help="Output video for --builtin_pointgoal_steps",
)
parser.add_argument(
    "--max_steps", type=int, default=0,
    help="Stop model evaluation after N simulation steps (0 means episode-controlled)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Isaac Sim 5.0: let the installed IsaacLab select its matching experience.
# In particular, do not reuse the old isaacsim_4_5 custom .kit path.
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after AppLauncher (Isaac Lab requirement)
import omni
import cv2
import carb
import numpy as np
import imageio
import os
import csv
import torch
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
if args_cli.builtin_pointgoal_steps <= 0:
    from utils_tasks.tracking_utils import MPC_Controller
else:
    MPC_Controller = None

planning_input = PlanningInput()
planning_output = PlanningOutput()
input_lock = threading.Lock()
output_lock = threading.Lock()
stop_event = threading.Event()
vis_manager = [VisualizationManager(history_size=5) for i in range(args_cli.num_envs)]
mpc = None


def planning_thread(env, camera_intrinsic):
    """Background planning: read observations, call NavDP, transform to world frame, update MPC."""
    global mpc

    while not stop_event.is_set():
        try:
            with input_lock:
                if planning_input.current_goal is None or \
                   planning_input.current_image is None or \
                   planning_input.current_depth is None or \
                   planning_input.camera_pos is None or \
                   planning_input.camera_rot is None:
                    time.sleep(0.01)
                    continue

                goal = planning_input.current_goal.copy()
                image = planning_input.current_image.copy()
                depth = planning_input.current_depth.copy()
                camera_pos = planning_input.camera_pos.copy()
                camera_rot = planning_input.camera_rot.copy()

            with output_lock:
                planning_output.is_planning = True

            trajectory_points_camera, all_trajectories_camera, all_values_camera = \
                pointgoal_step(goal, image, depth, port=args_cli.port)

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


scene_list = sorted(os.listdir(args_cli.scene_dir))
scene_list.sort()

scene_name = scene_list[args_cli.scene_index]
scene_path = os.path.join(args_cli.scene_dir, scene_name) + "/"
usd_path, init_path = find_usd_path(scene_path, task='pointgoal')

scene_config = PointNavSceneCfg()
scene_config.num_envs = args_cli.num_envs
scene_config.env_spacing = 0.0
scene_config.terrain = BENCH_TERRAIN_CFG
scene_config.terrain.usd_path = usd_path
scene_config.goal = GOAL_CFG
scene_config.robot = DINGO_CFG
scene_config.camera_sensor = DINGO_CameraCfg
scene_config.contact_sensor = DINGO_ContactCfg

env_config = DingoPointNavCfg()
env_config.scene = scene_config
env_config.sim.device = args_cli.device
env_config.events.reset_pose.params = {
    "init_point_path": init_path,
    'height_offset': 0.1,
    'robot_visible': False,
    'light_enabled': False
}

env = ManagerBasedRLEnv(env_config)
env = RslRlVecEnvWrapper(env)
adjust_usd_scale(scale=args_cli.scene_scale)
obs, infos = env.reset()

def step_env(action):
    """Normalize Gymnasium's 5-tuple and the older wrapper's 4-tuple."""
    result = env.step(action)
    if len(result) == 5:
        next_obs, rewards, terminated, truncated, extras = result
        dones = torch.logical_or(terminated, truncated)
        return next_obs, rewards, dones, extras
    return result


PREHEAT_STEPS = 10
for _ in range(PREHEAT_STEPS):
    action = torch.zeros((args_cli.num_envs, 2), device=env.unwrapped.device)
    obs, rewards, dones, infos = step_env(action)


if args_cli.builtin_pointgoal_steps > 0:
    controller = DifferentialController(
        name="builtin_pointgoal_control",
        wheel_radius=DINGO_WHEEL_RADIUS,
        wheel_base=DINGO_WHEEL_BASE,
    )
    writer = imageio.get_writer(args_cli.builtin_video, fps=10)
    print(f"[INFO] Running built-in PointGoal smoke test: {args_cli.builtin_video}")
    for step_index in range(args_cli.builtin_pointgoal_steps):
        goals = infos["observations"]["goal_pose"].cpu().numpy()[:, :2]
        images = infos["observations"]["rgb"].cpu().numpy()[:, :, :, :3]
        distance = np.linalg.norm(goals, axis=1)
        heading = np.arctan2(goals[:, 1], goals[:, 0])
        commands = []
        for env_index in range(args_cli.num_envs):
            linear = 0.0 if distance[env_index] < 1.0 else min(args_cli.speed, 0.35)
            angular = float(np.clip(1.5 * heading[env_index], -0.7, 0.7))
            wheel = controller.forward(np.array([linear, angular])).joint_velocities
            commands.append(wheel)

        frame = images[0].copy()
        frame = draw_box_with_text(
            frame, 0, 0, 520, 50,
            f"builtin PointGoal  distance={distance[0]:.2f}m heading={heading[0]:.2f}rad",
        )
        writer.append_data(frame)
        action = torch.as_tensor(np.stack(commands), device=env.unwrapped.device)
        obs, rewards, dones, infos = step_env(action)
        if bool(dones[0]):
            print(f"[INFO] Episode ended at step {step_index + 1}")
            break

    writer.close()
    env.close()
    simulation_app.close()
    print(f"BUILTIN_POINTGOAL_DONE video={args_cli.builtin_video}")
    raise SystemExit(0)

camera_intrinsic = env.unwrapped.scene.sensors['camera_sensor'].data.intrinsic_matrices[0]

planning_thread_obj = threading.Thread(
    target=planning_thread,
    args=(env, camera_intrinsic)
)
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

episode_num = args_cli.num_envs - 1
evaluation_metrics = []
save_dir = "./pointgoal_%s_%s/%s/" % (
    algo,
    args_cli.scene_dir.split("/")[-1],
    scene_path.split("/")[-2]
)
os.makedirs(save_dir, exist_ok=True)

euclidean = np.sqrt(
    np.square(infos['observations']['goal_pose'].cpu().numpy()[:, 0:2]).sum(axis=-1)
)

fps_writer = [
    imageio.get_writer(save_dir + "fps_%d.mp4" % i, fps=10)
    for i in range(scene_config.num_envs)
]

trajectory_length = np.zeros((scene_config.num_envs))
model_step_index = 0

while simulation_app.is_running():
    with torch.inference_mode():
        goals = infos['observations']['goal_pose'].cpu().numpy()[:, 0:2]
        images = infos['observations']['rgb'].cpu().numpy()[:, :, :, 0:3]
        depths = infos['observations']['depth'].cpu().numpy()[:, :, :]

        camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
        camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
        camera_rot_quat = camera_rot_quat[:, [1, 2, 3, 0]]
        camera_rot = R.from_quat(camera_rot_quat).as_matrix()

        with input_lock:
            planning_input.current_goal = goals.copy()
            planning_input.current_image = images.copy()
            planning_input.current_depth = depths.copy()
            planning_input.camera_pos = camera_pos.copy()
            planning_input.camera_rot = camera_rot.copy()

        robot_vel = env.unwrapped.scene.articulations['robot'].data.root_lin_vel_w[0, :2].norm().cpu().numpy()
        robot_ang_vel = env.unwrapped.scene.articulations['robot'].data.root_ang_vel_w[0, 2].cpu().numpy()

        x0 = np.stack([
            camera_pos[:, 0],
            camera_pos[:, 1],
            np.arctan2(camera_rot[:, 1, 0], camera_rot[:, 0, 0]),
            [robot_vel],
            [robot_ang_vel]
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
                vis_image = vis_manager[i].visualize_trajectory_global_with_people(
                    images[i],
                    depths[i][:, :, None],
                    camera_intrinsic.cpu().numpy(),
                    current_trajectory[i],
                    robot_pose=x0[i],
                    all_trajectories_points=current_all_trajectories[i],
                    all_trajectories_values=current_all_values[i]
                )

                if mpc is None:
                    continue

                t0 = time.time()
                opt_u_controls, opt_x_states = mpc.solve(x0[i, :3])
                print(f"MPC solve time: {time.time() - t0:.3f}s")

                v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]

                action = torch.tensor([v, w], device=env.unwrapped.device)
                action_cpu = action.cpu().numpy()
                joint_velocities = controller.forward(action_cpu).joint_velocities
                action_list.append(joint_velocities)

                vis_image = draw_box_with_text(
                    vis_image, 0, 0, 430, 50,
                    "desired lin.:%.2f ang.:%.2f" % (v, w)
                )
                vis_image = draw_box_with_text(
                    vis_image, 0, 50, 430, 50,
                    "actual lin.:%.2f ang.:%.2f" % (robot_vel, robot_ang_vel)
                )
                vis_image = draw_box_with_text(
                    vis_image, 0, 820, 430, 50,
                    "point goal:(%.2f, %.2f)" % (goals[i][0], goals[i][1])
                )
                cv2.imwrite(f"frame_{algo}_pointgoal_{scene_name}.png",
                            cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                fps_writer[i].append_data(vis_image)

            action = torch.as_tensor(np.stack(action_list, axis=0), device=env.unwrapped.device)
            obs, rewards, dones, infos = step_env(action)

            trajectory_length += (infos['observations']['policy'][:, 0] * env.unwrapped.step_dt).cpu().numpy()

        else:
            action = torch.zeros((args_cli.num_envs, 2), device=env.unwrapped.device)
            obs, rewards, dones, infos = step_env(action)
            print("Trajectory not ready; zero action")

        for i in range(args_cli.num_envs):
            if dones[i] == True:
                episode_num += 1

                navigator_reset(env_id=i, port=args_cli.port)

                success_flag = (np.sqrt(np.square(goals[i]).sum()) < 1.5).astype(np.float32)
                spl = np.clip(euclidean[i] / trajectory_length[i], 0, 1) * success_flag

                fps_writer[i].close()

                evaluation_metrics.append({
                    'success': success_flag,
                    'spl': spl,
                    'distance': euclidean[i]
                })

                write_metrics(evaluation_metrics, save_dir + "metric.csv")

                euclidean[i] = np.sqrt(
                    np.square(infos['observations']['goal_pose'].cpu().numpy()[:, 0:2]).sum(axis=-1)
                )[i]
                fps_writer[i] = imageio.get_writer(save_dir + "fps_%d.mp4" % episode_num, fps=10)
                trajectory_length[i] = 0.0

        if episode_num >= args_cli.num_episodes:
            break

        model_step_index += 1
        if args_cli.max_steps > 0 and model_step_index >= args_cli.max_steps:
            print(f"[INFO] Reached --max_steps={args_cli.max_steps}")
            break

stop_event.set()
for writer in fps_writer:
    writer.close()
env.close()
simulation_app.close()
print(f"MODEL_POINTGOAL_DONE steps={model_step_index} save_dir={save_dir}")
