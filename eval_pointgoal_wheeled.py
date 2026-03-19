"""
点目标导航评估脚本 (客户端)
功能：在 Isaac Sim 中评估 NavDP 模型的点目标导航性能

工作流程：
1. 启动 Isaac Sim 仿真环境
2. 创建异步规划线程（调用 NavDP Server）
3. 主循环：获取观测 → 规划轨迹 → MPC控制 → 执行动作
4. 记录评估指标（成功率、SPL、距离）

命令示例：
python eval_pointgoal_wheeled.py \
    --port 8888 \
    --scene_dir /path/to/scenes \
    --scene_index 0 \
    --scene_scale 0.01
"""
import argparse
from omni.isaac.lab.app import AppLauncher

# ============ 解析命令行参数 ============
parser = argparse.ArgumentParser(description="点目标导航评估脚本")
parser.add_argument(
    "--scene_dir", type=str, default="./asset_scenes/cluttered_easy",
    help="场景文件夹路径"
)
parser.add_argument(
    "--scene_index", type=int, default=8,
    help="场景索引（文件夹内的第几个场景）"
)
parser.add_argument(
    "--scene_scale", type=float, default=1.0,
    help="场景缩放比例（如 0.01 表示缩小100倍）"
)
parser.add_argument(
    "--stop_threshold", type=float, default=-3.0,
    help="探索转向的停止阈值（Critic价值）"
)
parser.add_argument(
    "--num_envs", type=int, default=1,
    help="并行环境数量（通常为1）"
)
parser.add_argument(
    "--num_episodes", type=int, default=100,
    help="评估回合数"
)
parser.add_argument(
    "--speed", type=float, default=0.5,
    help="期望速度 (m/s)"
)
parser.add_argument(
    "--port", type=int, default=8888,
    help="NavDP Server 的端口号"
)
args_cli = parser.parse_args()

# ============ 启动 Isaac Sim ============
app_launcher = AppLauncher(
    headless=True,         # 无头模式（不显示GUI）
    enable_cameras=True    # 启用相机渲染
)
simulation_app = app_launcher.app

# ============ 导入依赖（必须在 AppLauncher 之后导入） ============
import omni
import cv2
import carb
import numpy as np
import imageio
import os
import csv
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from pxr import Usd, Sdf
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
from wheeled_robots.controllers.differential_controller import DifferentialController
import torchvision.transforms as F
import time
import threading

# 项目工具
from utils_tasks.basic_utils import PlanningInput, PlanningOutput, find_usd_path, write_metrics, draw_box_with_text,adjust_usd_scale
from configs.robots import *    # 机器人配置（Dingo）
from configs.scenes import *    # 场景配置
from configs.tasks import *     # 任务配置（点目标导航）
from utils_tasks.client_utils import navigator_reset, pointgoal_step  # HTTP 客户端
from utils_tasks.visualization_utils import VisualizationManager  # 轨迹可视化
from utils_tasks.tracking_utils import MPC_Controller  # MPC 轨迹跟踪

# ============ 全局共享变量（主线程与规划线程通信） ============
planning_input = PlanningInput()   # 输入缓存：目标、图像、深度、相机位姿
planning_output = PlanningOutput() # 输出缓存：轨迹、价值、规划状态
input_lock = threading.Lock()      # 输入锁：保护 planning_input
output_lock = threading.Lock()     # 输出锁：保护 planning_output
stop_event = threading.Event()     # 停止事件：用于优雅关闭规划线程
vis_manager = [VisualizationManager(history_size=5) for i in range(args_cli.num_envs)]  # 可视化管理器
mpc = None  # MPC 控制器（在规划线程中初始化）

def planning_thread(env, camera_intrinsic):
    """
    规划线程函数（异步执行，不阻塞主仿真循环）
    
    功能：
    1. 从共享输入缓存读取观测（RGB、Depth、Goal）
    2. 调用 NavDP Server 进行轨迹规划
    3. 将轨迹从相机坐标系转换到世界坐标系
    4. 将结果写入共享输出缓存
    5. 初始化 MPC 控制器
    
    Args:
        env: Isaac Sim 环境
        camera_intrinsic: 相机内参矩阵 (3x3)
    
    线程安全：
    - 使用 input_lock 保护读取操作
    - 使用 output_lock 保护写入操作
    - 10Hz 规划频率（sleep 0.1s）
    """
    global mpc
    
    while not stop_event.is_set():
        try:
            # ===== 步骤1：从共享缓存读取最新观测 =====
            with input_lock:
                # 检查数据完整性
                if planning_input.current_goal is None or \
                   planning_input.current_image is None or \
                   planning_input.current_depth is None or \
                   planning_input.camera_pos is None or \
                   planning_input.camera_rot is None:
                    time.sleep(0.01)  # 数据未就绪，等待
                    continue
                
                # 拷贝数据（避免主线程修改导致数据不一致）
                goal = planning_input.current_goal.copy()           # (num_envs, 2) 相对目标坐标
                image = planning_input.current_image.copy()         # (num_envs, H, W, 3) RGB
                depth = planning_input.current_depth.copy()         # (num_envs, H, W, 1) Depth
                camera_pos = planning_input.camera_pos.copy()       # (num_envs, 3) 相机世界坐标
                camera_rot = planning_input.camera_rot.copy()       # (num_envs, 3, 3) 相机旋转矩阵
            
            # 设置规划状态标志
            with output_lock:
                planning_output.is_planning = True
            
            # ===== 步骤2：调用 NavDP Server 进行规划 =====
            planning_start = time.time()
            trajectory_points_camera, all_trajectories_camera, all_values_camera = \
                pointgoal_step(goal, image, depth, port=args_cli.port)
            # 返回值：
            #   trajectory_points_camera: (num_envs, 24, 3) 最优轨迹（相机坐标系）
            #   all_trajectories_camera: (num_envs, 16, 24, 3) 所有候选轨迹
            #   all_values_camera: (num_envs, 16) Critic 价值分数
            # ===== 步骤3：坐标转换（相机坐标系 → 世界坐标系）=====
            # 最优轨迹转换
            batch_optimal_points_world = []
            for idx in range(trajectory_points_camera.shape[0]):
                trajectory_points_world = []
                for i, point in enumerate(trajectory_points_camera[idx]):
                    if i < 0:
                        continue
                    # point: [Δx, Δy, Δθ] in camera frame
                    # 只取前两维（x, y），忽略角度
                    point_local = np.array([point[0], point[1], 0.0])
                    
                    # 坐标变换: P_world = P_camera + R_camera @ P_local
                    point_world = camera_pos[idx] + camera_rot[idx] @ point_local
                    trajectory_points_world.append(point_world[:2])  # 只保留 (x, y)
                
                trajectory_points_world = np.array(trajectory_points_world)
                batch_optimal_points_world.append(trajectory_points_world)
                
                # ===== 步骤4：初始化 MPC 控制器 =====
                mpc = MPC_Controller(
                    trajectory_points_world,    # 世界坐标系下的轨迹点
                    desired_v=args_cli.speed,   # 期望线速度 (m/s)
                    v_max=args_cli.speed,       # 最大线速度
                    w_max=args_cli.speed        # 最大角速度 (rad/s)
                )
            batch_optimal_points_world = np.array(batch_optimal_points_world)
           
            # ===== 步骤5：转换所有候选轨迹（用于可视化）=====
            batch_all_points_world = []
            for idx in range(all_trajectories_camera.shape[0]):
                all_trajectories_world = []
                for traj_camera in all_trajectories_camera[idx]:  # 16条候选轨迹
                    traj_world = []
                    for point in traj_camera:  # 24个点
                        point_local = np.array([point[0], point[1], 0.0])
                        point_world = camera_pos[idx] + camera_rot[idx] @ point_local
                        traj_world.append(point_world[:2])
                    all_trajectories_world.append(np.array(traj_world))
                batch_all_points_world.append(all_trajectories_world)
            batch_all_points_world = np.array(batch_all_points_world)

            # ===== 步骤6：更新共享输出缓存 =====
            with output_lock:
                planning_output.trajectory_points_world = batch_optimal_points_world  # 最优轨迹（世界坐标）
                planning_output.all_trajectories_world = batch_all_points_world       # 所有轨迹（世界坐标）
                planning_output.all_values_camera = all_values_camera                  # Critic 价值
                planning_output.is_planning = False  # 规划完成
                planning_output.planning_error = None
            
            # 打印规划耗时
            planning_time = time.time() - planning_start
            # print(f"Planning time: {planning_time:.3f}s, Goal: [{goal[0]:.2f}, {goal[1]:.2f}]")
                
        except Exception as e:
            # 规划失败处理
            print(f"Planning error: {e}")
            with output_lock:
                planning_output.is_planning = False
                planning_output.planning_error = str(e)
        
        # 控制规划频率为 10Hz（每 0.1 秒规划一次）
        time.sleep(0.1)

# ============ 环境初始化 ============

# ===== 步骤1：加载场景 =====
scene_path = os.path.join(
    args_cli.scene_dir,
    os.listdir(args_cli.scene_dir)[args_cli.scene_index]
) + "/"
usd_path, init_path = find_usd_path(scene_path, task='pointgoal')
# usd_path: 场景的 USD 文件路径
# init_path: 初始化位置的 CSV 文件路径

# ===== 步骤2：配置场景 =====
scene_config = PointNavSceneCfg()
scene_config.num_envs = args_cli.num_envs    # 并行环境数
scene_config.env_spacing = 0.0               # 环境间距（单环境为0）
scene_config.terrain = BENCH_TERRAIN_CFG     # 地形配置
scene_config.terrain.usd_path = usd_path     # USD场景文件
scene_config.goal = GOAL_CFG                 # 目标配置
scene_config.robot = DINGO_CFG               # Dingo 机器人配置
scene_config.camera_sensor = DINGO_CameraCfg # 相机传感器配置
scene_config.contact_sensor = DINGO_ContactCfg # 接触传感器配置

# ===== 步骤3：配置任务 =====
env_config = DingoPointNavCfg()
env_config.scene = scene_config
env_config.events.reset_pose.params = {
    "init_point_path": init_path,   # 初始位置文件
    'height_offset': 0.1,           # 高度偏移（避免穿模）
    'robot_visible': False,         # 机器人不可见（避免自身影响观测）
    'light_enabled': False          # 禁用额外光源
}

# ===== 步骤4：创建环境 =====
env = ManagerBasedRLEnv(env_config)
env = RslRlVecEnvWrapper(env)  # RL 包装器
adjust_usd_scale(scale=args_cli.scene_scale)  # 调整场景缩放
_, infos = env.reset()

# ===== 步骤5：预热（让物理引擎稳定）=====
PREHEAT_STEPS = 10
for _ in range(PREHEAT_STEPS):
    action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
    obs, rewards, dones, infos = env.step(action)
    
# ===== 步骤6：获取相机内参 =====
camera_intrinsic = env.unwrapped.scene.sensors['camera_sensor'].data.intrinsic_matrices[0]
# shape: (3, 3)

# ===== 步骤7：启动异步规划线程 =====
planning_thread_obj = threading.Thread(
    target=planning_thread, 
    args=(env, camera_intrinsic)
)
planning_thread_obj.daemon = True  # 守护线程（主线程退出时自动关闭）
planning_thread_obj.start()

# ===== 步骤8：初始化控制器 =====
controller = DifferentialController(
    name="simple_control", 
    wheel_radius=DINGO_WHEEL_RADIUS,  # 轮子半径 (m)
    wheel_base=DINGO_WHEEL_BASE       # 轮距 (m)
)

# ===== 步骤9：重置 NavDP Agent =====
algo = navigator_reset(
    camera_intrinsic.cpu().numpy(),
    batch_size=scene_config.num_envs,
    stop_threshold=args_cli.stop_threshold,
    port=args_cli.port
)

# ===== 步骤10：初始化评估变量 =====
episode_num = args_cli.num_envs - 1  # 当前回合编号
evaluation_metrics = []               # 评估指标列表
save_dir = "./pointgoal_%s_%s/%s/" % (
    algo,
    args_cli.scene_dir.split("/")[-1],
    scene_path.split("/")[-2]
)
os.makedirs(save_dir, exist_ok=True)

# 初始欧几里得距离（用于计算 SPL）
euclidean = np.sqrt(
    np.square(infos['observations']['goal_pose'].cpu().numpy()[:, 0:2]).sum(axis=-1)
)

# 视频写入器
fps_writer = [
    imageio.get_writer(save_dir + "fps_%d.mp4" % i, fps=10) 
    for i in range(scene_config.num_envs)
]

# 轨迹长度累计
trajectory_length = np.zeros((scene_config.num_envs))

# ============ 主仿真循环 ============
while simulation_app.is_running():
    with torch.inference_mode():
        # ===== 步骤1：获取最新观测 =====
        goals = infos['observations']['goal_pose'].cpu().numpy()[:, 0:2]  # (num_envs, 2) 相对目标
        images = infos['observations']['rgb'].cpu().numpy()[:, :, :, 0:3]  # (num_envs, H, W, 3) RGB
        depths = infos['observations']['depth'].cpu().numpy()[:, :, :]     # (num_envs, H, W) Depth
        
        # 获取相机位姿（世界坐标系）
        camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()  # (num_envs, 3)
        camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()  # (num_envs, 4)
        camera_rot_quat = camera_rot_quat[:, [1, 2, 3, 0]]  # 调整四元数顺序 (w,x,y,z) → (x,y,z,w)
        camera_rot = R.from_quat(camera_rot_quat).as_matrix()  # 转换为旋转矩阵 (num_envs, 3, 3)
        
        # ===== 步骤2：更新规划线程的输入缓存 =====
        with input_lock:
            planning_input.current_goal = goals.copy()
            planning_input.current_image = images.copy()
            planning_input.current_depth = depths.copy()
            planning_input.camera_pos = camera_pos.copy()
            planning_input.camera_rot = camera_rot.copy()

        # ===== 步骤3：获取机器人当前状态 =====
        robot_vel = env.unwrapped.scene.articulations['robot'].data.root_lin_vel_w[0, :2].norm().cpu().numpy()  # 线速度 (m/s)
        robot_ang_vel = env.unwrapped.scene.articulations['robot'].data.root_ang_vel_w[0, 2].cpu().numpy()       # 角速度 (rad/s)

        # 构建 MPC 初始状态: [x, y, θ, v, ω]
        x0 = np.stack([
            camera_pos[:, 0],                                    # x 坐标
            camera_pos[:, 1],                                    # y 坐标
            np.arctan2(camera_rot[:, 1, 0], camera_rot[:, 0, 0]),  # 朝向角 θ
            [robot_vel],                                         # 线速度
            [robot_ang_vel]                                      # 角速度
        ], axis=-1)
        
        # ===== 步骤4：从规划线程读取最新轨迹 =====
        current_trajectory = None
        current_all_trajectories = None
        current_all_values = None
        with output_lock:
            if planning_output.trajectory_points_world is not None:
                current_trajectory = planning_output.trajectory_points_world.copy()
                current_all_trajectories = planning_output.all_trajectories_world.copy()
                current_all_values = planning_output.all_values_camera.copy()
        
        # ===== 步骤5：控制执行（如果轨迹可用）=====
        if current_trajectory is not None:
            control_start = time.time()
            action_list = []
            
            for i in range(args_cli.num_envs):
                # ===== 5.1 可视化轨迹 =====
                vis_image = vis_manager[i].visualize_trajectory(
                    images[i],                           # RGB 图像
                    depths[i][:, :, None],               # Depth 图像
                    camera_intrinsic.cpu().numpy(),      # 相机内参
                    current_trajectory[i],               # 最优轨迹
                    robot_pose=x0[i],                    # 当前机器人位姿
                    all_trajectories_points=current_all_trajectories[i],  # 所有候选轨迹
                    all_trajectories_values=current_all_values[i]         # Critic 价值
                )
                
                # ===== 5.2 MPC 求解最优控制 =====
                if mpc is None:
                    continue
                
                t0 = time.time()
                opt_u_controls, opt_x_states = mpc.solve(x0[i, :3])  # 输入: [x, y, θ]
                print(f"MPC 求解耗时: {time.time() - t0:.3f}s")
                
                # 取第2步的控制量（第1步是当前状态，第2步是下一时刻）
                v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]
                # v: 线速度 (m/s)
                # w: 角速度 (rad/s)
                
                # ===== 5.3 差速运动学：(v, ω) → (v_left, v_right) =====
                action = torch.tensor([v, w], device="cuda:0")
                action_cpu = action.cpu().numpy()
                joint_velocities = controller.forward(action_cpu).joint_velocities
                # joint_velocities: [v_left, v_right] 左右轮速度 (rad/s)
                action_list.append(joint_velocities)
                
                # ===== 5.4 可视化文本信息 =====
                try:
                    # 期望速度
                    vis_image = draw_box_with_text(
                        vis_image, 0, 0, 430, 50,
                        "期望 lin.:%.2f ang.:%.2f" % (v, w)
                    )
                    # 实际速度
                    vis_image = draw_box_with_text(
                        vis_image, 0, 50, 430, 50,
                        "实际 lin.:%.2f ang.:%.2f" % (robot_vel, robot_ang_vel)
                    )
                    # Critic 价值范围
                    if current_all_values is not None:
                        vis_image = draw_box_with_text(
                            vis_image, 0, 770, 430, 50,
                            "Critic max:%.2f min:%.2f" % (
                                np.max(current_all_values[i]),
                                np.min(current_all_values[i])
                            )
                        )
                    # 目标坐标
                    vis_image = draw_box_with_text(
                        vis_image, 0, 820, 430, 50,
                        "点目标:(%.2f, %.2f)" % (goals[i][0], goals[i][1])
                    )
                    # 保存帧
                    cv2.imwrite(f"frame_test.png", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                    fps_writer[i].append_data(vis_image)
                except:
                    pass
            
            # ===== 5.5 执行动作 =====
            action = torch.as_tensor(np.stack(action_list, axis=0), device="cuda:0")
            obs, rewards, dones, infos = env.step(action)
            
            # 获取实际关节速度（用于调试）
            actual_joint_velocities = env.unwrapped.scene.articulations['robot'].data.joint_vel[0, :2].cpu().numpy()
            desired_joint_velocities = env.unwrapped.scene.articulations['robot'].data.joint_vel_target[0, :2].cpu().numpy()
            
            # 累计轨迹长度
            trajectory_length += (infos['observations']['policy'][:, 0] * env.unwrapped.step_dt).cpu().numpy()
        
        else:
            # ===== 轨迹未就绪，使用零动作 =====
            action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
            obs, rewards, dones, infos = env.step(action)
            print("轨迹未就绪，使用零动作")
        
        # ===== 步骤6：检查回合结束 =====
        for i in range(args_cli.num_envs):
            if dones[i] == True:
                episode_num += 1
                
                # 重置 NavDP Agent（清空历史队列）
                navigator_reset(env_id=i, port=args_cli.port)
                
                # ===== 计算评估指标 =====
                # 成功标准：距离目标 < 1.5米
                success_flag = (np.sqrt(np.square(goals[i]).sum()) < 1.5).astype(np.float32)
                
                # SPL (Success weighted by Path Length)
                # SPL = (最短路径 / 实际路径) * 成功标志
                # 用于评估路径效率：1.0 表示最优路径，越小越绕远
                spl = np.clip(euclidean[i] / trajectory_length[i], 0, 1) * success_flag
                
                # 关闭当前视频
                fps_writer[i].close()
                
                # 记录指标
                evaluation_metrics.append({
                    'success': success_flag,        # 是否成功 (0/1)
                    'spl': spl,                     # SPL 指标
                    'distance': euclidean[i]        # 初始距离
                })
                
                # 写入 CSV 文件
                write_metrics(evaluation_metrics, save_dir + "metric.csv")
                
                # 重置变量（准备下一回合）
                euclidean[i] = np.sqrt(
                    np.square(infos['observations']['goal_pose'].cpu().numpy()[:, 0:2]).sum(axis=-1)
                )[i]
                fps_writer[i] = imageio.get_writer(save_dir + "fps_%d.mp4" % episode_num, fps=10)
                trajectory_length[i] = 0.0
        
        # ===== 步骤7：检查是否完成所有回合 =====
        if episode_num > args_cli.num_episodes:
            break
       
                
   

        
