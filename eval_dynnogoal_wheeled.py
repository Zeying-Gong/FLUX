import argparse

# ============ 只导入标准库 ============
parser = argparse.ArgumentParser(description="A script to run dynamic exploration with people")
parser.add_argument("--scene_dir", type=str, default="/workspace/NavDP/assets/dyn_scenes/isaacsim_scene")
parser.add_argument("--scene_index", type=int, default=2)
parser.add_argument("--scene_scale", type=float, default=1.0)
parser.add_argument("--stop_threshold", type=float, default=-3.0)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--speed", type=float, default=0.5)
parser.add_argument("--port", type=int, default=9999)
parser.add_argument("--gpu_id", type=int, default=0)
args_cli = parser.parse_args()

# ============ 设置独立的缓存目录 ============
import os
import sys

print(f"GPU {args_cli.gpu_id}, Scene {args_cli.scene_index}")
print(f"OMNI_USER_DATA_DIR: {os.environ.get('OMNI_USER_DATA_DIR', 'NOT SET')}")
print(f"CARB_APP_DATA_DIR: {os.environ.get('CARB_APP_DATA_DIR', 'NOT SET')}")

from isaaclab.app import AppLauncher
HEADLESS = True
MULTI_GPU = False # True # False
NUM_GPUS = 3

CUSTOM_APP_PATH = "/workspace/IsaacLab/apps/isaacsim_4_5/isaaclab.python.dyn.kit"

# 构建 AppLauncher 参数
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

# Isaac Sim 相关导入
import signal
import omni
import cv2
import carb
import numpy as np
import imageio
import os
import csv
import torch
import open3d as o3d
import asyncio
from scipy.spatial.transform import Rotation as R
from pxr import Usd, Sdf
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_rgbd
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from wheeled_robots.controllers.differential_controller import DifferentialController
import time
import threading

from utils_tasks.basic_utils import PlanningInput, PlanningOutput, find_usd_path, write_metrics, draw_box_with_text, adjust_usd_scale, cpu_pointcloud_from_array
from configs.robots import *
from configs.scenes import *
from configs.tasks import *
from utils_tasks.client_utils import navigator_reset, nogoal_step
from utils_tasks.visualization_utils import VisualizationManager
from utils_tasks.tracking_utils import MPC_Controller
from socialnav_metrics import SocialMetricsTracker, get_people_positions

planning_input = PlanningInput()
planning_output = PlanningOutput()
input_lock = threading.Lock()
output_lock = threading.Lock()
stop_event = threading.Event()
vis_manager = [VisualizationManager(history_size=5) for i in range(args_cli.num_envs)]
mpc = None


def update_occupancy(global_pcd, camera_int, current_pos, current_rot, robot_rgb, robot_depth):
    """更新占据栅格并计算探索面积"""
    filter_rgb = torch.tensor(robot_rgb, device=robot_rgb.device)
    filter_depth = torch.tensor(robot_depth, device=robot_depth.device)
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


def cleanup_simulation(env=None, simulation_app=None):
    """统一的清理函数"""
    print("[INFO] Starting cleanup...")
    
    try:
        # 1. 停止规划线程
        if 'stop_event' in globals():
            stop_event.set()
        if 'planning_thread_obj' in globals() and planning_thread_obj.is_alive():
            print("  Stopping planning thread...")
            planning_thread_obj.join(timeout=3)
        
        # 2. 关闭视频写入器
        if 'fps_writer' in globals():
            print("  Closing video writers...")
            for writer in fps_writer:
                try:
                    writer.close()
                except:
                    pass
        
        # 3. 提前清理 camera sensors
        if env is not None:
            print("  Detaching camera sensors...")
            try:
                unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
                if hasattr(unwrapped_env, 'scene') and hasattr(unwrapped_env.scene, 'sensors'):
                    camera_sensor = unwrapped_env.scene.sensors.get('camera_sensor')
                    if camera_sensor is not None:
                        if hasattr(camera_sensor, '_annotators'):
                            for annotator in camera_sensor._annotators:
                                try:
                                    annotator.detach()
                                except:
                                    pass
                        camera_sensor._annotators = []
                        camera_sensor._sensor_prims = []
            except Exception as e:
                print(f"  Warning: Camera cleanup failed: {e}")
        
        # 4. 清理环境
        if env is not None:
            print("  Closing environment...")
            try:
                unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
                if hasattr(unwrapped_env, 'scene'):
                    try:
                        unwrapped_env.scene.reset()
                    except:
                        pass
                env.close()
            except Exception as e:
                print(f"  Warning: Error closing env: {e}")
        
        # 5. 清理 simulation context
        print("  Clearing simulation context...")
        try:
            from isaaclab.sim import SimulationContext
            sim_context = SimulationContext.instance()
            if sim_context is not None:
                sim_context.clear_all_callbacks()
                SimulationContext.clear_instance()
        except Exception as e:
            print(f"  Warning: SimulationContext cleanup failed: {e}")
        
        # 6. 清理 Isaac Sim 组件
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
        
        # 7. 清理 GPU 缓存
        if torch.cuda.is_available():
            print("  Clearing CUDA cache...")
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        print("[INFO] Cleanup complete")
        
    except Exception as e:
        print(f"[ERROR] Cleanup failed: {e}")
        import traceback
        traceback.print_exc()


def signal_handler(sig, frame):
    """优雅退出处理"""
    print("\n" + "="*50)
    print("Received interrupt signal, shutting down...")
    print("="*50)
    
    global simulation_app, env
    
    cleanup_simulation(
        env=env if 'env' in globals() else None,
        simulation_app=simulation_app if 'simulation_app' in globals() else None
    )
    
    print("Exiting...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def setup_all_lighting(env):
    import omni.usd
    from pxr import UsdLux, Gf
    
    settings = carb.settings.get_settings()
    stage = omni.usd.get_context().get_stage()
    
    # 1. Viewport Camera Light
    settings.set("/rtx/useViewLightingMode", True)
    settings.set("/rtx/viewLightingMode", 1)
    
    # 2. 环境光和曝光
    settings.set("/rtx/sceneDb/ambientLightIntensity", 3.0)
    settings.set("/rtx/post/tonemap/enable", True)
    settings.set("/rtx/post/tonemap/exposure", 1.0)
    
    # 3. 为机器人相机添加光源
    camera_sensor = env.unwrapped.scene.sensors['camera_sensor']
    for i, camera_prim in enumerate(camera_sensor._sensor_prims):
        camera_path = str(camera_prim.GetPath())
        parent_path = '/'.join(camera_path.split('/')[:-1])
        light_path = f"{parent_path}/CameraLight"
        
        if not stage.GetPrimAtPath(light_path):
            sphere_light = UsdLux.SphereLight.Define(stage, light_path)
            sphere_light.CreateIntensityAttr(1000.0)
            sphere_light.CreateRadiusAttr(0.1)
            sphere_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
            print(f"[INFO] ✓ Added light to robot camera {i}")
    
    print("[INFO] ✓ All lighting setup complete")


def planning_thread(env, camera_intrinsic):
    global mpc
    """Thread function that continuously plans trajectories"""
    while not stop_event.is_set():
        try:
            # Get latest observations from shared state
            with input_lock:
                if planning_input.current_image is None or planning_input.current_depth is None or planning_input.camera_pos is None or planning_input.camera_rot is None:
                    time.sleep(0.01)
                    continue
                image = planning_input.current_image.copy()
                depth = planning_input.current_depth.copy()
                camera_pos = planning_input.camera_pos.copy()
                camera_rot = planning_input.camera_rot.copy()
            with output_lock:
                planning_output.is_planning = True
            
            # Start timing planning
            planning_start = time.time()
            trajectory_points_camera, all_trajectories_camera, all_values_camera = nogoal_step(image, depth, port=args_cli.port)
            
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
                mpc = MPC_Controller(trajectory_points_world,
                                     desired_v=args_cli.speed,
                                     v_max=args_cli.speed,
                                     w_max=args_cli.speed)
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
                
        except Exception as e:
            print(f"Planning error: {e}")
            with output_lock:
                planning_output.is_planning = False
                planning_output.planning_error = str(e)
        # Small sleep to prevent CPU overload
        time.sleep(0.1)


def main():
    # 获取场景目录列表
    scene_list = os.listdir(args_cli.scene_dir)
    scene_list.sort()

    scene_name = scene_list[args_cli.scene_index]
    scene_path = os.path.join(args_cli.scene_dir, scene_name) + "/"
    usd_path, init_path = find_usd_path(scene_path, 'pointgoal')

    # Validate episode JSON files
    print(f"[INFO] Validating episode files in {scene_path}")
    episode_json_files = []
    for episode_id in range(args_cli.num_episodes):
        episode_json_path = os.path.join(scene_path, f"episode_{episode_id}.json")
        if not os.path.exists(episode_json_path):
            raise RuntimeError(
                f"Missing episode file: {episode_json_path}\n"
                f"Expected {args_cli.num_episodes} episodes (0 to {args_cli.num_episodes-1})"
            )
        episode_json_files.append(episode_json_path)

    print(f"[INFO] Found all {args_cli.num_episodes} episode files ✓")

    first_episode_path = episode_json_files[0]

    # 使用 DynExploreSceneCfg - 这是新的场景配置
    scene_config = DynExploreSceneCfg()
    scene_config.num_envs = args_cli.num_envs
    scene_config.env_spacing = 0.0
    scene_config.terrain = BENCH_TERRAIN_CFG
    scene_config.terrain.usd_path = usd_path
    scene_config.robot = DINGO_CFG
    scene_config.camera_sensor = DINGO_CameraCfg
    scene_config.contact_sensor = DINGO_ContactCfg
    scene_config.metric_sensor = DINGO_MetricCameraCfg

    # Enable people simulation
    scene_config.people_simulation = True
    scene_config.episode_json_path = first_episode_path

    # 使用 DingoDynExploreCfg - 这是新的任务配置
    env_config = DingoDynExploreCfg()
    env_config.scene = scene_config
    env_config.events.reset_pose.params = {
        "episode_json_dir": scene_path,
        "num_episodes": args_cli.num_episodes,
        'height_offset': 0.1,
        'robot_visible': False,
        'light_enabled': False
    }

    env = ManagerBasedRLEnv(env_config)
    env = RslRlVecEnvWrapper(env)
    adjust_usd_scale(scale=args_cli.scene_scale)
    
    episode_steps = np.zeros((scene_config.num_envs,), dtype=np.int64)
    
    # Setup lighting for specific scenes
    if scene_name in ['Hospital', 'Jetracer', 'Office']:
        setup_all_lighting(env)
        for _ in range(5):
            simulation_app.update()

    # warm-up
    PREHEAT_STEPS = 10
    for _ in range(PREHEAT_STEPS):
        action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
        obs, rewards, dones, infos = env.step(action)
        
    camera_intrinsic = env.unwrapped.scene.sensors['camera_sensor'].data.intrinsic_matrices[0]

    planning_thread_obj = threading.Thread(target=planning_thread, args=(env, camera_intrinsic))
    planning_thread_obj.daemon = True
    planning_thread_obj.start()

    controller = DifferentialController(name="simple_control",
                                        wheel_radius=DINGO_WHEEL_RADIUS,
                                        wheel_base=DINGO_WHEEL_BASE)
    algo = navigator_reset(camera_intrinsic.cpu().numpy(), batch_size=scene_config.num_envs, stop_threshold=args_cli.stop_threshold, port=args_cli.port)

    if algo == "fallback_algo":
        print("[ERROR] Navigator server connection failed!")
        print(f"[INFO] Please start the server: python server.py --port {args_cli.port}")
        cleanup_simulation(env, simulation_app)
        sys.exit(1)

    print(f"[INFO] Connected to navigator server, algorithm: {algo}")
    
    episode_num = 0
    evaluation_metrics = []
    current_episode_idx = 0
    save_dir = f"./metrics/dynnogoal_{algo}_{args_cli.scene_dir.split('/')[-1]}/{scene_path.split('/')[-2]}/"
    os.makedirs(save_dir, exist_ok=True)

    fps_writer = [imageio.get_writer(save_dir + f"fps_{i}.mp4", fps=10) for i in range(scene_config.num_envs)]
    
    # Exploration tracking
    global_pcds = [o3d.geometry.PointCloud() for i in range(scene_config.num_envs)]
    navigable_pcds = [o3d.geometry.PointCloud() for i in range(scene_config.num_envs)]
    explore_areas = np.zeros((scene_config.num_envs))
    trajectory_length = np.zeros((scene_config.num_envs))

    # Social metrics tracking
    social_metrics_trackers = [SocialMetricsTracker() for _ in range(scene_config.num_envs)]
    env.env.social_metrics_trackers = social_metrics_trackers

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

    # Wait for NavMesh
    if scene_config.people_simulation:
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
                images = infos['observations']['rgb'].cpu().numpy()[:, :, :, 0:3]
                depths = infos['observations']['depth'].cpu().numpy()[:, :, :]
                
                # Update occupancy map with metric sensor
                metric_camera_pos = env.unwrapped.scene.sensors["metric_sensor"].data.pos_w
                metric_camera_rot = env.unwrapped.scene.sensors["metric_sensor"].data.quat_w_ros
                metric_camera_int = env.unwrapped.scene.sensors["metric_sensor"].data.intrinsic_matrices
                metric_rgb = infos['observations']["metric_rgb"]
                metric_depth = infos['observations']["metric_depth"]
                
                for i in range(scene_config.num_envs):
                    global_pcds[i], navigable_pcds[i], explore_areas[i] = update_occupancy(
                        global_pcds[i],
                        metric_camera_int[i],
                        metric_camera_pos[i],
                        metric_camera_rot[i],
                        metric_rgb[i] / 255.0,
                        metric_depth[i],
                    )
                
                # Get camera poses
                camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
                camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
                camera_rot_quat = camera_rot_quat[:, [1, 2, 3, 0]]
                camera_rot = R.from_quat(camera_rot_quat).as_matrix()
                
                with input_lock:
                    planning_input.current_image = images.copy()
                    planning_input.current_depth = depths.copy()
                    planning_input.camera_pos = camera_pos.copy()
                    planning_input.camera_rot = camera_rot.copy()

                # Get robot velocity
                robot_vel = env.unwrapped.scene.articulations['robot'].data.root_lin_vel_w[0, :2].norm().cpu().numpy()
                robot_ang_vel = env.unwrapped.scene.articulations['robot'].data.root_ang_vel_w[0, 2].cpu().numpy()

                x0 = np.stack([camera_pos[:, 0], camera_pos[:, 1], np.arctan2(camera_rot[:, 1, 0], camera_rot[:, 0, 0]), [robot_vel], [robot_ang_vel]], axis=-1)
                
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
                        # Get people positions for social metrics
                        people_positions, people_char_paths, pos_get_flag = get_people_positions(env)
                        
                        if pos_get_flag:
                            # Update social metrics
                            social_metrics_trackers[i].update(camera_pos[i], people_positions, env.unwrapped.step_dt)
                            
                            # Visualize with people
                            vis_image = vis_manager[i].visualize_trajectory_global_with_people(
                                images[i], depths[i][:, :, None], camera_intrinsic.cpu().numpy(),
                                current_trajectory[i],
                                robot_pose=x0[i],
                                goal_position=None,  # No goal in exploration
                                all_trajectories_points=current_all_trajectories[i],
                                all_trajectories_values=current_all_values[i],
                                people_positions=people_positions,
                                people_positions_dict=people_char_paths,
                            )
                        else:
                            # Fallback to visualization without people
                            vis_image = vis_manager[i].visualize_trajectory(
                                images[i], depths[i][:, :, None], camera_intrinsic.cpu().numpy(),
                                current_trajectory[i],
                                robot_pose=x0[i],
                                all_trajectories_points=current_all_trajectories[i],
                                all_trajectories_values=current_all_values[i]
                            )
                        
                        if mpc is None:
                            continue
                        
                        t0 = time.time()
                        opt_u_controls, opt_x_states = mpc.solve(x0[i, :3])
                        v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]
                        action = torch.tensor([v, w], device="cuda:0")
                        action_cpu = action.cpu().numpy()
                        joint_velocities = controller.forward(action_cpu).joint_velocities
                        action_list.append(joint_velocities)
                        
                        try:
                            vis_image = draw_box_with_text(vis_image, 0, 0, 430, 50, f"desired lin.:{v:.2f} ang.:{w:.2f}")
                            vis_image = draw_box_with_text(vis_image, 0, 50, 430, 50, f"actual lin.:{robot_vel:.2f} ang.:{robot_ang_vel:.2f}")
                            if current_all_values is not None:
                                vis_image = draw_box_with_text(vis_image, 0, 770, 430, 50, f"critic max:{np.max(current_all_values[i]):.2f} min:{np.min(current_all_values[i]):.2f}")
                            vis_image = draw_box_with_text(vis_image, 0, 820, 430, 50, f"explore area:{explore_areas[i]:.2f} m²")
                            if frame_count > 0:
                                cv2.imwrite(f"frame_{algo}_dynnogoal_{scene_name}.png", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                                fps_writer[i].append_data(vis_image)
                            frame_count += 1
                        except:
                            pass
                        
                    action = torch.as_tensor(np.stack(action_list, axis=0), device="cuda:0")
                    obs, rewards, dones, infos = env.step(action)
                    episode_steps += 1
                    trajectory_length += (infos['observations']['policy'][:, 0] * env.unwrapped.step_dt).cpu().numpy()
                else:
                    action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
                    obs, rewards, dones, infos = env.step(action)
                    episode_steps += 1
                    print("No trajectory available, using zero action")
                
                for i in range(args_cli.num_envs):
                    if dones[i] == True:
                        episode_num += 1
                        navigator_reset(env_id=i, port=args_cli.port)
                        
                        # Get social metrics
                        social_metrics = social_metrics_trackers[i].get_metrics()
                        
                        evaluation_metrics.append({
                            'episode': current_episode_idx,
                            'time': episode_steps[i] * env.env.step_dt,
                            'area': explore_areas[i],
                            'trajectory_length': trajectory_length[i],
                            # Social metrics
                            'collision': social_metrics['collision'],
                            'collision_count': social_metrics['collision_count'],
                            'min_distance': social_metrics['min_distance'],
                            'avg_distance': social_metrics['avg_distance'],
                            'psi_count': social_metrics['psi_count'],
                            'psi_time': social_metrics['psi_time'],
                            'total_time': social_metrics['total_time'],
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
                        
                        # Check if all episodes completed
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
                                        except:
                                            pass
                                    camera_sensor._annotators = []
                            except Exception as e:
                                print(f"Warning: Camera pre-cleanup failed: {e}")
                            
                            cleanup_simulation(env, simulation_app)
                            return
                        
                        # Reset for next episode
                        if hasattr(env.env, '_recent_positions'):
                            env.env._recent_positions.clear()
                        with output_lock:
                            planning_output.trajectory_points_world = None
                            planning_output.all_trajectories_world = None
                            planning_output.all_values_camera = None
                            planning_output.is_planning = False
                            planning_output.planning_error = None
                        
                        # Update scene episode path and reset people
                        new_episode_path = os.path.join(scene_path, f"episode_{current_episode_idx}.json")
                        env.unwrapped.scene.cfg.episode_json_path = new_episode_path
                        
                        # Reset people for new episode
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
                        
                        fps_writer[i] = imageio.get_writer(save_dir + f"fps_{current_episode_idx}.mp4", fps=10)
                        trajectory_length[i] = 0.0
                        episode_steps[i] = 0
                        global_pcds[i] = o3d.geometry.PointCloud()
                        navigable_pcds[i] = o3d.geometry.PointCloud()
                        explore_areas[i] = 0
                        
                        # Reset vis_manager
                        camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
                        camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
                        camera_rot_quat = camera_rot_quat[:, [1, 2, 3, 0]]
                        camera_rot = R.from_quat(camera_rot_quat).as_matrix()
                        initial_pose = np.array([camera_pos[i, 0], camera_pos[i, 1],
                                                np.arctan2(camera_rot[i, 1, 0], camera_rot[i, 0, 0])])
                        vis_manager[i].reset(initial_robot_pose=initial_pose)
                        frame_count = 0
                        
                        # Reset social metrics tracker
                        social_metrics_trackers[i].reset()
                        
                        break  # Process one reset at a time
                
                if episode_num > args_cli.num_episodes:
                    break
                    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected!")
        signal_handler(signal.SIGINT, None)
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