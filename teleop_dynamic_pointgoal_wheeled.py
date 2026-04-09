import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("Teleoperation with dynamic environments")
parser.add_argument("-c", "--config_file", required=True, help="Path to a IRA config file", default="/isaac-sim/tools/agent_sdg/simple_nosave_config.yaml" )
parser.add_argument("--people_config", type=str, default="/workspace/NavDP/configs/people_config.yaml")
# parser.add_argument("--num_people", type=int, default=3)
parser.add_argument("--people_seed", type=int, default=3)
parser.add_argument(
    "--scene_dir", type=str, default="/workspace/NavDP/assets/dyn_scenes/cluttered_easy")
parser.add_argument(
    "--scene_index", type=int, default=0)
parser.add_argument(
    "--scene_scale", type=float, default=1.0)
parser.add_argument(
    "--stop_threshold", type=float, default=-2.0)
parser.add_argument(
    "--num_episodes", type=int, default=100)
parser.add_argument(
    "--port", type=int, default=9999)
args_cli = parser.parse_args()
CUSTOM_APP_PATH = "/workspace/IsaacLab/apps/isaacsim_4_5/isaaclab.python.dyn.kit" 
app_launcher = AppLauncher(headless=False, enable_cameras=True,
                            experience=CUSTOM_APP_PATH,
                            )
simulation_app = app_launcher.app

import omni
import cv2
import carb
import numpy as np
import imageio
import os
import csv
import torch
import open3d as o3d
import requests
import io
import json
import time
import threading
from queue import Queue
from typing import Optional, List, Tuple
from pynput import keyboard 
from scipy.spatial.transform import Rotation as R

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from wheeled_robots.controllers.differential_controller import DifferentialController

from utils_tasks.basic_utils import PlanningInput, PlanningOutput,draw_box_with_text, find_usd_path, adjust_usd_scale
from utils_tasks.client_utils import navigator_reset, pointgoal_step
from configs.robots import *
from configs.scenes import *
from configs.tasks import *
from utils_tasks.visualization_utils import VisualizationManager

# Global shared states
planning_input = PlanningInput()
planning_output = PlanningOutput()
input_lock = threading.Lock()
output_lock = threading.Lock()
stop_event = threading.Event()

def planning_thread(env, camera_intrinsic):
    """Thread function that continuously plans trajectories"""
    while not stop_event.is_set():
        try:
            # Get latest observations from shared state
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
            
            # Start timing planning
            planning_start = time.time()
            trajectory_points_camera, all_trajectories_camera, all_values_camera = pointgoal_step(goal, image, depth, port=args_cli.port)
            # print(trajectory_points_camera.shape,all_trajectories_camera.shape,all_values_camera.shape)
        
            # Transform trajectory from camera frame to world frame
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

            # Update shared state
            with output_lock:
                planning_output.trajectory_points_world = batch_optimal_points_world
                planning_output.all_trajectories_world = batch_all_points_world
                planning_output.all_values_camera = all_values_camera
                planning_output.is_planning = False
                planning_output.planning_error = None
            
            # Print planning timing
            planning_time = time.time() - planning_start
            # print(f"Planning time: {planning_time:.3f}s, Goal: [{goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}]")
                
        except Exception as e:
            print(f"Planning error: {e}")
            with output_lock:
                planning_output.is_planning = False
                planning_output.planning_error = str(e)
        # Small sleep to prevent CPU overload
        time.sleep(0.1)

def main():
    scene_path = os.path.join(args_cli.scene_dir,os.listdir(args_cli.scene_dir)[args_cli.scene_index]) + "/"
    usd_path,init_path = find_usd_path(scene_path,'pointgoal')

    episode_json_path = os.path.join(scene_path, "episode_0.json")
    if not os.path.exists(episode_json_path):
        print(f"Warning: Episode JSON not found at {episode_json_path}")
        episode_json_path = None
    else:
        print(f"Using episode JSON: {episode_json_path}")

    # Setup environment
    scene_config = PointNavSceneCfg()
    scene_config.num_envs = 1
    scene_config.env_spacing = 0.0
    scene_config.terrain = BENCH_TERRAIN_CFG
    scene_config.terrain.usd_path = usd_path
    scene_config.goal = GOAL_CFG
    scene_config.robot = DINGO_CFG
    scene_config.contact_sensor = DINGO_ContactCfg
    scene_config.camera_sensor = DINGO_CameraCfg

    scene_config.episode_json_path = episode_json_path

    env_config = DingoPointNavCfg()
    env_config.actions = DingoActionsCfg()
    env_config.scene = scene_config
    env_config.events.reset_pose.params = {"init_point_path": init_path,
                                           'height_offset': 0.1,
                                           'robot_visible': True, # False 
                                           'light_enabled': False} 
    # ===== 新增：配置 people simulation =====
    env_config.people_simulation = episode_json_path is not None

    env = ManagerBasedRLEnv(env_config)
    env = RslRlVecEnvWrapper(env)
    adjust_usd_scale(scale=args_cli.scene_scale)

    # ===== 新增：在 reset 之前排除机器人 =====
    if env_config.people_simulation:
        print("[INFO] Excluding robot from NavMesh...")
        import NavSchema
        import omni.usd
        
        stage = omni.usd.get_context().get_stage()
        
        # 获取机器人路径（正则表达式格式）
        robot_prim_path = env.unwrapped.scene.articulations['robot'].cfg.prim_path
        print(f"  Robot prim path pattern: {robot_prim_path}")
        
        # 方法 1: 使用 IsaacLab 的工具函数查找匹配的路径
        from isaaclab.sim.utils import find_matching_prim_paths
        
        try:
            actual_robot_paths = find_matching_prim_paths(robot_prim_path)
            print(f"  Found {len(actual_robot_paths)} robot(s): {actual_robot_paths}")
            
            for actual_path in actual_robot_paths:
                robot_prim = stage.GetPrimAtPath(actual_path)
                
                if robot_prim and robot_prim.IsValid():
                    if not robot_prim.HasAPI(NavSchema.NavMeshExcludeAPI):
                        try:
                            omni.kit.commands.execute(
                                "ApplyNavMeshAPICommand",
                                prim_path=actual_path,
                                api=NavSchema.NavMeshExcludeAPI
                            )
                            print(f"  ✓ Excluded robot at {actual_path}")
                        except Exception as e:
                            print(f"  ✗ Failed to exclude robot at {actual_path}: {e}")
                else:
                    print(f"  ✗ Robot prim not found at {actual_path}")
        
        except Exception as e:
            print(f"  ✗ Error finding robot paths: {e}")
            
            # 方法 2: 后备方案 - 手动构建路径
            print("  Trying fallback method...")
            for env_id in range(scene_config.num_envs):
                # 将 'env_.*' 替换为 'env_0', 'env_1' 等
                actual_path = robot_prim_path.replace('env_.*', f'env_{env_id}')
                print(f"  Trying path: {actual_path}")
                
                robot_prim = stage.GetPrimAtPath(actual_path)
                
                if robot_prim and robot_prim.IsValid():
                    if not robot_prim.HasAPI(NavSchema.NavMeshExcludeAPI):
                        try:
                            omni.kit.commands.execute(
                                "ApplyNavMeshAPICommand",
                                prim_path=actual_path,
                                api=NavSchema.NavMeshExcludeAPI
                            )
                            print(f"  ✓ Excluded robot at {actual_path}")
                        except Exception as e:
                            print(f"  ✗ Failed to exclude robot: {e}")
                else:
                    print(f"  ✗ Robot prim not valid at {actual_path}")
        
        # 等待几帧让 NavMesh 更新
        for _ in range(10):
            simulation_app.update()

    obs, infos = env.reset()
    camera_intrinsic = env.unwrapped.scene.sensors['camera_sensor'].data.intrinsic_matrices[0]

    # ========== 新增：获取初始位置并初始化vis_manager ==========
    camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
    camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
    camera_rot_quat = camera_rot_quat[:,[1, 2, 3, 0]]
    camera_rot = R.from_quat(camera_rot_quat).as_matrix()

    controller = DifferentialController(name="simple_control", wheel_radius=DINGO_WHEEL_RADIUS, wheel_base=DINGO_WHEEL_BASE)
    algo = navigator_reset(camera_intrinsic.cpu().numpy(),batch_size=scene_config.num_envs,stop_threshold=args_cli.stop_threshold,port=args_cli.port)
    vis_manager = [VisualizationManager(history_size=5) for i in range(scene_config.num_envs)]

    # 初始化vis_manager的全局地图原点
    for i in range(scene_config.num_envs):
        initial_pose = np.array([
            camera_pos[i, 0],
            camera_pos[i, 1],
            np.arctan2(camera_rot[i, 1, 0], camera_rot[i, 0, 0])
        ])
        vis_manager[i].reset(initial_robot_pose=initial_pose)

    # if scene_config.enable_navmesh:
    print("Waiting for NavMesh to be ready...")
    wait_count = 0
    while not env.unwrapped.scene.navmesh_ready and wait_count < 100:
        simulation_app.update()
        wait_count += 1
        if wait_count % 20 == 0:
            print(f"  Waiting... ({wait_count}/100)")
    
    if env.unwrapped.scene.navmesh_ready:
        print("✓ NavMesh ready!")
    else:
        print("⚠ NavMesh not ready, continuing without people")
            
    # Start planning thread
    planning_thread_obj = threading.Thread(target=planning_thread, args=(env, camera_intrinsic))
    planning_thread_obj.daemon = True
    planning_thread_obj.start()

    # Initialize tracking variables
    episode_num = 0
    trajectory_length = np.zeros((scene_config.num_envs))
    save_dir = "./teleop_pointgoal_%s_%s/%s/"%(algo,args_cli.scene_dir.split("/")[-1],scene_path.split("/")[-2])
    os.makedirs(save_dir, exist_ok=True)
    euclidean = np.sqrt(np.square(infos['observations']['goal_pose'].cpu().numpy()[:,0:2]).sum(axis=-1))
    fps_writer = [imageio.get_writer(save_dir + "fps_%d.mp4"%i, fps=10) for i in range(scene_config.num_envs)]
    
    # Initialize dones
    dones = torch.zeros(1, dtype=torch.bool, device="cuda:0")
    linear_vel = 0.0
    angular_vel = 0.0
    max_linear_vel = 0.5  # Maximum linear velocity
    max_angular_vel = 0.5  # Maximum angular velocity
    vel_step = 0.1  # Velocity increment/decrement step

    # Initialize keyboard state
    key_state = {
        'w': False,
        's': False,
        'a': False,
        'd': False,
        'enter': False
    }

    def on_press(key):
        try:
            if key.char in key_state:
                key_state[key.char] = True
        except AttributeError:
            if key == keyboard.Key.enter:
                key_state['enter'] = True

    def on_release(key):
        try:
            if key.char in key_state:
                key_state[key.char] = False
        except AttributeError:
            if key == keyboard.Key.enter:
                key_state['enter'] = False

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Main simulation loop
    while simulation_app.is_running():
        # Move CUDA operations to CPU before sharing
        goals = infos['observations']['goal_pose'].cpu().numpy()[:,0:2]
        images = infos['observations']['rgb'].cpu().numpy()[:,:,:,0:3]
        depths = infos['observations']['depth'].cpu().numpy()[:,:,:]
        # get all camera poses
        camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
        camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
        camera_rot_quat = camera_rot_quat[:,[1, 2, 3, 0]]
        camera_rot = R.from_quat(camera_rot_quat).as_matrix()
        
        # Update shared state with latest observations
        with input_lock:
            planning_input.current_goal = goals.copy()
            planning_input.current_image = images.copy()
            planning_input.current_depth = depths.copy()
            planning_input.camera_pos = camera_pos.copy()
            planning_input.camera_rot = camera_rot.copy()
        
        # based on the current world trajectory 
        robot_vel = env.unwrapped.scene.articulations['robot'].data.root_lin_vel_w[0, :2].norm().cpu().numpy()
        robot_ang_vel = env.unwrapped.scene.articulations['robot'].data.root_ang_vel_w[0, 2].cpu().numpy()
        x0 = np.stack([camera_pos[:,0], camera_pos[:,1], np.arctan2(camera_rot[:,1,0], camera_rot[:,0,0]), [robot_vel], [robot_ang_vel]],axis=-1)
        current_trajectory = None
        current_all_trajectories = None
        current_all_values = None
        with output_lock:
            if planning_output.trajectory_points_world is not None:
                current_trajectory = planning_output.trajectory_points_world.copy() if planning_output.trajectory_points_world is not None else None
                current_all_trajectories = planning_output.all_trajectories_world.copy() if planning_output.all_trajectories_world is not None else None
                current_all_values = planning_output.all_values_camera.copy() if planning_output.all_values_camera is not None else None
        
        # Handle keyboard input for control
        if key_state['w']:
            linear_vel = min(linear_vel + vel_step, max_linear_vel)
        elif key_state['s']:
            linear_vel = max(linear_vel - vel_step, -max_linear_vel)
        else:
            linear_vel = 0.0

        if key_state['a']:
            angular_vel = min(angular_vel + vel_step, max_angular_vel)
        elif key_state['d']:
            angular_vel = max(angular_vel - vel_step, -max_angular_vel)
        else:
            angular_vel = 0.0
        
        # Create visualization
        if current_trajectory is not None:
            for i in range(scene_config.num_envs):
                try:
                    goal_world = camera_pos[i] + camera_rot[i] @ np.array([goals[i][0], goals[i][1], 0.0])

                    # vis_image = vis_manager[i].visualize_trajectory(
                    #     images[i], depths[i][:,:,None], camera_intrinsic.cpu().numpy(),
                    #     current_trajectory[i],
                    #     robot_pose=x0[i],
                    #     all_trajectories_points=current_all_trajectories[i],
                    #     all_trajectories_values=current_all_values[i]
                    # )

                    vis_image = vis_manager[i].visualize_trajectory_global(
                        images[i], depths[i][:,:,None], camera_intrinsic.cpu().numpy(),
                        current_trajectory[i],
                        robot_pose=x0[i],
                        goal_position=goal_world[:2],  # 传入goal的世界坐标
                        all_trajectories_points=current_all_trajectories[i],
                        all_trajectories_values=current_all_values[i]
                    )
                    vis_image = draw_box_with_text(vis_image,0,820,430,50,"point goal:(%.2f, %.2f)"%(goals[i][0],goals[i][1]))
                    cv2.imwrite(f"frame_test_dyn.png", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                    fps_writer[i].append_data(vis_image)
                except:
                    pass
        # Control using keyboard input
        action = torch.tensor([linear_vel, angular_vel], device="cuda:0")
        action_cpu = action.cpu().numpy()
        joint_velocities = controller.forward(action_cpu).joint_velocities
        action = torch.as_tensor(joint_velocities, device="cuda:0").unsqueeze(0)
        
        obs, rewards, dones, infos = env.step(action)
        
        # Get robot velocities
        robot_vel = env.unwrapped.scene.articulations['robot'].data.root_lin_vel_w[0, :2].norm().cpu().numpy()
        robot_ang_vel = env.unwrapped.scene.articulations['robot'].data.root_ang_vel_w[0, 2].cpu().numpy()
        
        # Get actual joint velocities from Isaac Sim
        actual_joint_velocities = env.unwrapped.scene.articulations['robot'].data.joint_vel[0, :4].cpu().numpy()
        desired_joint_velocities = env.unwrapped.scene.articulations['robot'].data.joint_vel_target[0, :4].cpu().numpy()

        print(f"actual joint vel:{actual_joint_velocities}  desired joint vel:{desired_joint_velocities}")
        trajectory_length += (infos['observations']['policy'][0,0] * env.unwrapped.step_dt).cpu().numpy()
        
        # Print control status
        print(f"Linear vel: {linear_vel:.3f}, Angular vel: {angular_vel:.3f}, Actual vel: {robot_vel:.3f} {robot_ang_vel:.2f}")
        
        # Check for Enter key press to reset
        if key_state['enter']:
            print("Reset triggered by Enter key press")
            episode_num += 1
            navigator_reset(env_id=i,port=args_cli.port)
            fps_writer[i].close()
            euclidean[i] = np.sqrt(np.square(infos['observations']['goal_pose'].cpu().numpy()[:,0:2]).sum(axis=-1))[i]
            fps_writer[i] = imageio.get_writer(save_dir + "fps_%d.mp4"%episode_num, fps=10)
            trajectory_length[i] = 0.0
            # ========== 新增：重新初始化vis_manager的全局地图 ==========
            initial_pose = np.array([camera_pos[i, 0], camera_pos[i, 1], 
                                    np.arctan2(camera_rot[i, 1, 0], camera_rot[i, 0, 0])])
            vis_manager[i].reset(initial_robot_pose=initial_pose)

        for i in range(scene_config.num_envs):
            if dones[i] == True:
                episode_num += 1
                navigator_reset(env_id=i,port=args_cli.port)
                fps_writer[i].close()
                euclidean[i] = np.sqrt(np.square(infos['observations']['goal_pose'].cpu().numpy()[:,0:2]).sum(axis=-1))[i]
                fps_writer[i] = imageio.get_writer(save_dir + "fps_%d.mp4"%episode_num, fps=10)
                trajectory_length[i] = 0.0
                # ========== 新增：重新初始化vis_manager的全局地图 ==========
                initial_pose = np.array([camera_pos[i, 0], camera_pos[i, 1], 
                                        np.arctan2(camera_rot[i, 1, 0], camera_rot[i, 0, 0])])
                vis_manager[i].reset(initial_robot_pose=initial_pose)
    # Cleanup
    stop_event.set()
    planning_thread_obj.join()
    listener.stop()
    fps_writer[0].close()

if __name__ == "__main__":
    main()