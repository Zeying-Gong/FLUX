"""
flux/train_grpo.py  —  多场景 + 三任务联合 GRPO 训练

支持的任务：
  dynpointgoal  单人动态点目标导航   (goal = 行人相对坐标, pointgoal 接口)
  dynnogoal     多人环境自由探索      (goal = 全零,         nogoal   接口)
  socialnav     多人环境静态点导航    (goal = 目标点相对坐标, pointgoal 接口)

多场景：--scene_dirs 接受多个目录，每个目录下有 N 个子场景（easy_0…easy_9）。
       每个 episode 结束后，按轮换策略切换 (scene_dir × sub_scene × task)。

用法示例：
    isaacsim-python baselines/flux/train_grpo.py \
        --checkpoint baselines/flux_wo_rl/checkpoints/checkpoint-11710navdp.ckpt \
        --scene_dirs assets/dyn_scenes/cluttered_easy assets/dyn_scenes/isaacsim_scene \
        --tasks dynpointgoal dynnogoal socialnav \
        --num_episodes 3000 \
        --save_dir baselines/flux/checkpoints_rl \
        --gpu_id 1 --train_gpu_id 1 \
        --lr 3e-5 --update_interval 32 --save_interval 100

单场景单任务（兼容旧用法）：
    isaacsim-python baselines/flux/train_grpo.py \
        --checkpoint ... \
        --scene_dirs assets/dyn_scenes/cluttered_easy \
        --scene_index 0 \
        --tasks dynpointgoal \
        --num_episodes 1000 ...
"""

import argparse
import os
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ============ CLI 参数（AppLauncher 之前）============
parser = argparse.ArgumentParser()
# 场景
parser.add_argument("--scene_dirs", type=str, nargs="+",
                    default=["/workspace/NavDP/assets/dyn_scenes/isaacsim_scene"],
                    help="一个或多个场景根目录，每个目录下含多个子场景文件夹")
parser.add_argument("--scene_index", type=int, default=None,
                    help="固定子场景索引（不设则轮换所有子场景）")
parser.add_argument("--scene_scale", type=float, default=1.0)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=2000)
parser.add_argument("--speed", type=float, default=0.5)
parser.add_argument("--stop_threshold", type=float, default=-3.0)
parser.add_argument("--gpu_id", type=int, default=0)
# 任务
parser.add_argument("--tasks", type=str, nargs="+",
                    default=["socialnav"],
                    choices=["dynpointgoal", "dynnogoal", "socialnav"],
                    help="联合训练的任务列表，多个任务按 episode 轮换")
# RL 训练
parser.add_argument("--checkpoint", type=str,
                    default="baselines/flux_wo_rl/checkpoints/checkpoint-11710navdp.ckpt")
parser.add_argument("--save_dir", type=str, default="baselines/flux/checkpoints_rl")
parser.add_argument("--save_interval", type=int, default=300)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--update_interval", type=int, default=16)
parser.add_argument("--sample_num", type=int, default=16)
parser.add_argument("--unfreeze_last_n", type=int, default=0)
parser.add_argument("--train_gpu_id", type=int, default=0)
parser.add_argument("--normalization_config", type=str, default=None)
parser.add_argument("--episodes_per_scene", type=int, default=20,
                    help="每个场景训练多少个 episode 后切换到下一个场景")
args_cli = parser.parse_args()

# ============ 启动 Isaac Sim ============
from isaaclab.app import AppLauncher

CUSTOM_APP_PATH = "/workspace/IsaacLab/apps/isaacsim_4_5/isaaclab.python.dyn.kit"
app_launcher = AppLauncher(
    headless=True,
    enable_cameras=True,
    experience=CUSTOM_APP_PATH,
    device=f"cuda:{args_cli.gpu_id}",
)
simulation_app = app_launcher.app

# ============ 其余 import ============
import signal
import asyncio
import threading
import time
import numpy as np
import torch
import cv2
import imageio
from scipy.spatial.transform import Rotation as R

import omni
import carb
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from wheeled_robots.controllers.differential_controller import DifferentialController

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "baselines", "flux"))

from utils_tasks.basic_utils import (
    PlanningInput, PlanningOutput, find_usd_path, write_metrics,
    draw_box_with_text, adjust_usd_scale,
)
from configs.robots import *
from configs.scenes import *
from configs.tasks import *
from utils_tasks.visualization_utils import VisualizationManager
from utils_tasks.tracking_utils import MPC_Controller
from socialnav_metrics import SocialMetricsTracker, get_people_positions
from policy_agent import GRPO_Agent
import json

RESUME_STATE_FILE = os.path.join(args_cli.save_dir, "resume_state.json")
# ============ 全局状态 ============
stop_event      = threading.Event()
input_lock      = threading.Lock()
output_lock     = threading.Lock()
planning_input  = PlanningInput()
planning_output = PlanningOutput()
vis_manager     = [VisualizationManager(history_size=5) for _ in range(args_cli.num_envs)]
mpc             = None
TRAIN_DEVICE    = f"cuda:{args_cli.train_gpu_id}"
SIM_DEVICE      = f"cuda:{args_cli.gpu_id}"


# ============================================================
# 任务调度器
# ============================================================

# class TaskScheduler:
#     """
#     维护 (scene_dir, sub_scene_name, task) 的轮换列表。
#     next() 返回下一个组合并自动循环。
#     """
#     def __init__(self, scene_dirs, tasks, fixed_scene_index=None):
#         self.tasks  = tasks
#         self.combos = []
#         for sd in scene_dirs:
#             sub_scenes = sorted(os.listdir(sd))
#             if fixed_scene_index is not None:
#                 sub_scenes = [sub_scenes[fixed_scene_index]]
#             for ss in sub_scenes:
#                 for t in tasks:
#                     self.combos.append((sd, ss, t))
#         if not self.combos:
#             raise RuntimeError("No valid (scene, task) combinations found.")
#         self._idx = 0
#         print(f"[TaskScheduler] {len(self.combos)} training combinations:")
#         for c in self.combos:
#             print(f"  {c[2]:16s}  {os.path.join(c[0], c[1])}")

#     def current(self):
#         return self.combos[self._idx]

#     def next(self):
#         self._idx = (self._idx + 1) % len(self.combos)
#         return self.combos[self._idx]

#     def __len__(self):
#         return len(self.combos)

class TaskScheduler:
    def __init__(self, scene_dirs, tasks, fixed_scene_index=None,
                 episodes_per_scene=20):          # ← 新增参数
        self.tasks  = tasks
        self.episodes_per_scene = episodes_per_scene
        self._episode_in_current_scene = 0        # 当前场景已跑的 episode 数
        self.combos = []

        for sd in scene_dirs:
            sub_scenes = sorted(os.listdir(sd))
            if fixed_scene_index is not None:
                sub_scenes = [sub_scenes[fixed_scene_index]]
            for ss in sub_scenes:
                for t in tasks:
                    self.combos.append((sd, ss, t))

        if not self.combos:
            raise RuntimeError("No valid (scene, task) combinations found.")
        self._idx = 0
        print(f"[TaskScheduler] {len(self.combos)} training combinations "
              f"(episodes_per_scene={episodes_per_scene}):")
        for c in self.combos:
            print(f"  {c[2]:16s}  {os.path.join(c[0], c[1])}")

    def current(self):
        return self.combos[self._idx]

    def next(self):
        self._idx = (self._idx + 1) % len(self.combos)
        self._episode_in_current_scene = 0        # ← 重置计数
        return self.combos[self._idx]

    def on_episode_done(self):
        """
        每个 episode 结束时调用。
        返回 True 表示需要切换场景（重建 env），False 表示只切换 task/episode_file。
        """
        self._episode_in_current_scene += 1
        if self._episode_in_current_scene >= self.episodes_per_scene:
            return True   # 需要切换场景
        return False

    def __len__(self):
        return len(self.combos)


# ============================================================
# 清理 & 信号
# ============================================================

def cleanup_simulation(env=None, sim_app=None):
    print("[INFO] Cleanup...")
    stop_event.set()
    try:
        if env is not None:
            try:
                sensor = env.unwrapped.scene.sensors.get("camera_sensor")
                if sensor and hasattr(sensor, "_annotators"):
                    for ann in sensor._annotators:
                        try: ann.detach()
                        except: pass
                    sensor._annotators = []
            except: pass
            try: 
                if env is not None:
                    safe_close_env(env, sim_app)  # ← 替换原来的 env.close()
            except: pass
        try:
            from isaaclab.sim import SimulationContext
            ctx = SimulationContext.instance()
            if ctx:
                ctx.clear_all_callbacks()
                SimulationContext.clear_instance()
        except: pass
        if sim_app is not None:
            try:
                import omni.usd
                omni.usd.get_context().close_stage()
                for _ in range(3): sim_app.update()
            except: pass
            try: sim_app.close()
            except: pass
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
    except Exception as e:
        print(f"[ERROR] Cleanup: {e}")


def safe_close_env(env, sim_app):
    """
    安全关闭 IsaacLab 环境。
    必须先手动 detach 相机 annotators，再调用 env.close()，
    否则 Isaac Sim 的弱引用在 clear_all_callbacks 时已失效，
    导致 camera.__del__ 里 annotator.detach() 触发 ReferenceError + abort。
    """
    try:
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        if hasattr(unwrapped, 'scene') and hasattr(unwrapped.scene, 'sensors'):
            for sensor_name in ['camera_sensor', 'metric_sensor']:
                sensor = unwrapped.scene.sensors.get(sensor_name)
                if sensor is None:
                    continue
                if hasattr(sensor, '_annotators'):
                    for ann in sensor._annotators:
                        try:
                            ann.detach()
                        except Exception:
                            pass
                    sensor._annotators = []
                if hasattr(sensor, '_sensor_prims'):
                    sensor._sensor_prims = []
    except Exception as e:
        print(f"  [safe_close_env] annotator detach warning: {e}")

    # 让 Isaac Sim 处理完待完成的帧，再关闭
    for _ in range(3):
        try:
            sim_app.update()
        except Exception:
            break

    try:
        env.close()
    except Exception as e:
        print(f"  [safe_close_env] env.close() warning: {e}")


def signal_handler(sig, frame):
    print("\nInterrupt received, shutting down...")
    cleanup_simulation(
        env=env if "env" in globals() else None,
        sim_app=simulation_app,
    )
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================
# 照明
# ============================================================

def setup_all_lighting(env):
    from pxr import UsdLux, Gf
    stage = omni.usd.get_context().get_stage()
    carb.settings.get_settings().set("/rtx/sceneDb/ambientLightIntensity", 3.0)
    sensor = env.unwrapped.scene.sensors["camera_sensor"]
    for prim in sensor._sensor_prims:
        parent = "/".join(str(prim.GetPath()).split("/")[:-1])
        lp = f"{parent}/CameraLight"
        if not stage.GetPrimAtPath(lp):
            sl = UsdLux.SphereLight.Define(stage, lp)
            sl.CreateIntensityAttr(1000.0)
            sl.CreateRadiusAttr(0.1)
            sl.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))


# ============================================================
# 规划线程（pointgoal / nogoal 两种接口）
# ============================================================

def planning_thread_func(agent: GRPO_Agent, camera_intrinsic, current_task_ref: list):
    """
    current_task_ref 是长度为 1 的 list，主线程可随时修改 [0] 来切换任务，
    无需重启线程。
    """
    global mpc
    while not stop_event.is_set():
        try:
            with input_lock:
                task = current_task_ref[0]
                need_goal = (task != "dynnogoal")
                if planning_input.current_image is None or planning_input.current_depth is None:
                    time.sleep(0.01)
                    continue
                if need_goal and planning_input.current_goal is None:
                    time.sleep(0.01)
                    continue
                goal       = planning_input.current_goal.copy() if need_goal else None
                image      = planning_input.current_image.copy()
                depth      = planning_input.current_depth.copy()
                camera_pos = planning_input.camera_pos.copy()
                camera_rot = planning_input.camera_rot.copy()

            with output_lock:
                planning_output.is_planning = True

            if task == "dynnogoal":
                execute_traj, all_traj, all_values = agent.step_nogoal(image, depth)
            else:
                execute_traj, all_traj, all_values = agent.step_pointgoal(goal, image, depth)

            # camera → world
            batch_opt = []
            for idx in range(execute_traj.shape[0]):
                pts = []
                for pt in execute_traj[idx]:
                    pw = camera_pos[idx] + camera_rot[idx] @ np.array([pt[0], pt[1], 0.0])
                    pts.append(pw[:2])
                pts = np.array(pts)
                batch_opt.append(pts)
                mpc = MPC_Controller(pts,
                                     desired_v=args_cli.speed,
                                     v_max=args_cli.speed,
                                     w_max=args_cli.speed)
            batch_opt = np.array(batch_opt)

            batch_all = []
            for idx in range(all_traj.shape[0]):
                aworld = []
                for traj in all_traj[idx]:
                    tw = []
                    for pt in traj:
                        pw = camera_pos[idx] + camera_rot[idx] @ np.array([pt[0], pt[1], 0.0])
                        tw.append(pw[:2])
                    aworld.append(np.array(tw))
                batch_all.append(aworld)
            batch_all = np.array(batch_all)

            with output_lock:
                planning_output.trajectory_points_world = batch_opt
                planning_output.all_trajectories_world  = batch_all
                planning_output.all_values_camera       = all_values
                planning_output.is_planning             = False
                planning_output.planning_error          = None

        except Exception as e:
            print(f"[PlanningThread] Error: {e}")
            with output_lock:
                planning_output.is_planning = False
                planning_output.planning_error = str(e)
        time.sleep(0.05)

def save_resume_state(total_episode_idx, scene_ep_idx, task_idx):
    state = {
        "total_episode_idx": total_episode_idx,
        "scene_ep_idx": scene_ep_idx,
        "task_combo_idx": task_idx,
        "timestamp": time.time(),
    }
    with open(RESUME_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"[Resume] Saved state: ep={total_episode_idx}")

def load_resume_state():
    if os.path.exists(RESUME_STATE_FILE):
        with open(RESUME_STATE_FILE) as f:
            state = json.load(f)
        print(f"[Resume] Loaded: ep={state['total_episode_idx']}")
        return state
    return None
# ============================================================
# 奖励函数
# ============================================================

# ---- 奖励尺度设计原则 ----
# 机器人速度约 0.5m/s，控制频率约 10Hz，每步移动约 0.05m
# r_approach 每步最大约 0.05 × 5.0 = 0.25
# r_social 应远小于 r_approach，否则导航信号被淹没
# 目标：正常行驶时 mean_reward ≈ 0，成功时 >> 0，碰撞时 << 0

INTIMATE = 0.45   # 亲密距离：严重惩罚
COMFORT  = 1.2    # 舒适距离：轻微惩罚

# 惩罚系数：设计为每步最坏情况（贴近1人）约 -0.3，不超过 r_approach 量级
_SOCIAL_INTIMATE_PEN = -0.5   # 原来 -5.0，降低 10x
_SOCIAL_COMFORT_PEN  = -0.1   # 原来线性 max -1.0，降低 10x


def _social_penalty(camera_pos, people_positions):
    """
    社交距离惩罚。系数经过缩放，使其不掩盖导航方向信号。
    - INTIMATE zone (< 0.45m): 固定 -0.5 / 人
    - COMFORT zone (0.45~1.2m): 线性 0 ~ -0.1 / 人
    """
    r = 0.0
    for ppos in people_positions:
        d = np.linalg.norm(camera_pos[:2] - np.array(ppos)[:2])
        if d < INTIMATE:
            r += _SOCIAL_INTIMATE_PEN
        elif d < COMFORT:
            r += _SOCIAL_COMFORT_PEN * (COMFORT - d) / (COMFORT - INTIMATE)
    return r


def reward_dynpointgoal(camera_pos, target_pos, prev_dist, success, stuck,
                         people_positions, timeout=False):
    cur_dist   = np.linalg.norm(camera_pos[:2] - target_pos[:2])
    r_arrive   = 20.0 if success else 0.0
    # 双向progress：靠近奖励，远离惩罚
    r_progress = (prev_dist - cur_dist) * 5.0
    r_stuck    = -1.0 if stuck else 0.0
    r_timeout  = -3.0 if timeout else 0.0
    return r_arrive + r_progress + r_stuck + r_timeout, cur_dist


# def reward_socialnav(camera_pos, goal_pos_world, prev_dist, success, stuck, people_positions):
#     """
#     多人场景导航到静态目标点，与 dynpointgoal 结构相同。
#     返回 (reward, new_dist)
#     """
#     cur_dist   = np.linalg.norm(camera_pos[:2] - goal_pos_world[:2])
#     r_arrive   = 10.0 if success else 0.0
#     # r_approach = (prev_dist - cur_dist) * 5.0
#     r_progress = max(0, (prev_dist - cur_dist)) * 2.0   # 只奖励靠近，不惩罚远离
#     r_social   = _social_penalty(camera_pos, people_positions)
#     r_stuck    = -1.0 if stuck else 0.0
#     return r_arrive + r_progress + r_social + r_stuck, cur_dist

def reward_socialnav(camera_pos, goal_pos_world, prev_dist, success, stuck,
                      people_positions, timeout=False):
    cur_dist   = np.linalg.norm(camera_pos[:2] - goal_pos_world[:2])
    r_arrive   = 20.0 if success else 0.0
    # 双向progress：靠近奖励，远离惩罚
    r_progress = (prev_dist - cur_dist) * 5.0
    r_social   = _social_penalty(camera_pos, people_positions)
    r_stuck    = -1.0 if stuck else 0.0
    r_timeout  = -3.0 if timeout else 0.0
    return r_arrive + r_progress + r_social + r_stuck + r_timeout, cur_dist


# def reward_dynnogoal(camera_pos, prev_pos, people_positions, stuck):
#     """
#     无目标自由探索。
#     - r_explore:   移动距离 × 2.0（鼓励运动）
#     - r_social:    社交距离惩罚
#     - r_collision: 碰撞 -5.0（原 -15，降低避免方差过大）
#     - r_survive:   每步存活 +0.05
#     """
#     move_dist   = np.linalg.norm(camera_pos[:2] - prev_pos[:2])
#     r_explore   = move_dist * 2.0
#     # r_social    = _social_penalty(camera_pos, people_positions)
#     r_stuck    = -1.0 if stuck else 0.0     # 原 -15，降低方差
#     r_survive   = 0.05
#     return r_explore + r_stuck + r_survive

def reward_dynnogoal(camera_pos, prev_pos, people_positions):
    move_dist = np.linalg.norm(camera_pos[:2] - prev_pos[:2])

    # 鼓励移动，系数适中
    r_explore = move_dist * 8.0

    # 放宽阈值：0.05->0.03，避免转弯时被惩罚
    r_forward_bonus = 0.15 if move_dist > 0.03 else -0.05

    # 生存奖励
    r_survive = 0.1

    return r_explore + r_forward_bonus + r_survive

# ============================================================
# 辅助函数
# ============================================================

def get_target_person_pos(env):
    try:
        from omni.anim.people.scripts.global_character_position_manager import (
            GlobalCharacterPositionManager,
        )
        mgr   = GlobalCharacterPositionManager.get_instance()
        chars = mgr.get_all_managed_characters()
        if not chars:
            return None
        pos = mgr.get_character_current_pos(list(chars)[0])
        return np.array([float(pos[0]), float(pos[1]), float(pos[2])])
    except Exception:
        return None


def count_scene_episodes(scene_path):
    """
    统计场景目录中连续的 episode_N.json 文件数量。
    从 episode_0.json 开始，遇到第一个缺失则停止。
    至少需要 1 个 episode，否则报错。
    """
    count = 0
    while os.path.exists(os.path.join(scene_path, f"episode_{count}.json")):
        count += 1
    if count == 0:
        raise RuntimeError(f"No episode_0.json found in {scene_path}")
    return count


def build_env(scene_dir, sub_scene_name, num_envs, scene_scale):
    """
    创建 IsaacLab 环境。

    三个任务（dynpointgoal / socialnav / dynnogoal）共用同一套
    SocialNavSceneCfg + DingoSocialNavCfg 配置（多人场景）。
    任务切换时只改 goal 计算方式和奖励函数，不重建环境，
    避免 Isaac Sim 在同一进程内 env.close() 导致的 weakref abort。
    """
    scene_path     = os.path.join(scene_dir, sub_scene_name) + "/"
    usd_path, _    = find_usd_path(scene_path, "pointgoal")
    scene_ep_count = count_scene_episodes(scene_path)
    first_ep       = os.path.join(scene_path, "episode_0.json")

    print(f"[build_env] {sub_scene_name} | {scene_ep_count} episode files | "
          f"using SocialNav scene (shared by all tasks)")

    sc = SocialNavSceneCfg()
    sc.num_envs          = num_envs
    sc.env_spacing       = 0.0
    sc.terrain           = BENCH_TERRAIN_CFG
    sc.terrain.usd_path  = usd_path
    sc.goal              = GOAL_CFG
    sc.robot             = DINGO_CFG
    sc.camera_sensor     = DINGO_CameraCfg
    sc.contact_sensor    = DINGO_ContactCfg
    sc.people_simulation = True
    sc.episode_json_path = first_ep

    ec = DingoSocialNavCfg()
    ec.scene = sc
    ec.events.reset_pose.params = {
        "episode_json_dir": scene_path,
        "num_episodes":     scene_ep_count,
        "height_offset":    0.1,
        "robot_visible":    False,
        "light_enabled":    False,
    }

    # 等待渲染管线就绪
    for _ in range(10):
        simulation_app.update()

    env = ManagerBasedRLEnv(ec)
    env = RslRlVecEnvWrapper(env)
    adjust_usd_scale(scale=scene_scale)
    return env, sc, scene_path, scene_ep_count


def wait_for_people(env, sim_app, max_wait=500):
    wait = 0
    while not env.unwrapped.scene.navmesh_ready and wait < 100:
        sim_app.update(); wait += 1
    print("✓ NavMesh ready" if env.unwrapped.scene.navmesh_ready else "⚠ NavMesh not ready")
    wait = 0
    while env.unwrapped.scene._people_setup_in_progress and wait < max_wait:
        sim_app.update(); wait += 1
    print("[INFO] People ready.")


def reset_people_for_episode(env, sim_app, episode_path, max_wait=500):
    env.unwrapped.scene.cfg.episode_json_path = episode_path
    if env.unwrapped.scene.people is not None or env.unwrapped.scene._people_setup_in_progress:
        while env.unwrapped.scene._people_setup_in_progress:
            sim_app.update()
        asyncio.ensure_future(
            env.unwrapped.scene._reset_people_for_episode(episode_path))
        wait = 0
        while not env.unwrapped.scene._people_setup_in_progress and wait < 10:
            sim_app.update(); wait += 1
        wait = 0
        while env.unwrapped.scene._people_setup_in_progress and wait < max_wait:
            sim_app.update(); wait += 1

# def find_latest_rl_checkpoint(save_dir):
#     import glob
#     import re
#     ckpts = glob.glob(os.path.join(save_dir, "rl_checkpoint_ep*.ckpt"))
#     if not ckpts:
#         return None
    
#     def extract_ep(path):
#         m = re.search(r"rl_checkpoint_ep(\d+)\.ckpt", os.path.basename(path))
#         return int(m.group(1)) if m else -1
    
#     ckpts = sorted(ckpts, key=extract_ep)
#     latest = ckpts[-1]
#     print(f"[Resume] Found {len(ckpts)} checkpoints, "
#           f"ep range: {extract_ep(ckpts[0])} ~ {extract_ep(latest)}")
#     return latest

def find_latest_rl_checkpoint(save_dir):
    import glob, re
    # 同时兼容旧的 ep 命名和新的 update 命名
    ckpts = glob.glob(os.path.join(save_dir, "rl_checkpoint_update*.ckpt"))
    if not ckpts:
        ckpts = glob.glob(os.path.join(save_dir, "rl_checkpoint_ep*.ckpt"))
    if not ckpts:
        return None

    def extract_num(path):
        m = re.search(r"rl_checkpoint_(?:update|ep)(\d+)\.ckpt", os.path.basename(path))
        return int(m.group(1)) if m else -1

    ckpts = sorted(ckpts, key=extract_num)
    latest = ckpts[-1]
    print(f"[Resume] Found {len(ckpts)} checkpoints, latest: {os.path.basename(latest)}")
    return latest

# ============================================================
# 主训练循环
# ============================================================

def main():
    global mpc, env

    # ---- 任务调度器 ----
    scheduler = TaskScheduler(
        scene_dirs=args_cli.scene_dirs,
        tasks=args_cli.tasks,
        fixed_scene_index=args_cli.scene_index,
        episodes_per_scene=args_cli.episodes_per_scene,   # ← 新增
    )

    # ---- 提前处理resume，确定实际使用的checkpoint ----
    resume_state = load_resume_state()
    actual_checkpoint = args_cli.checkpoint  # 默认用命令行传入的
    initial_total_episode_idx = 0
    initial_scene_ep_idx = 0

    if resume_state:
        initial_total_episode_idx = resume_state["total_episode_idx"]
        initial_scene_ep_idx = resume_state["scene_ep_idx"]
        scheduler._idx = resume_state["task_combo_idx"]
        _, _, task = scheduler.current()
        latest_ckpt = find_latest_rl_checkpoint(args_cli.save_dir)
        if latest_ckpt:
            actual_checkpoint = latest_ckpt
            print(f"[Resume] Using checkpoint: {latest_ckpt}")
        print(f"[Resume] Resuming from ep={initial_total_episode_idx}, "
              f"scene_ep={initial_scene_ep_idx}")

    scene_dir, sub_scene_name, task = scheduler.current()

    print(f"\n[INFO] First combo: task={task}  "
          f"scene={os.path.join(scene_dir, sub_scene_name)}")

    env, scene_config, scene_path, scene_ep_count = build_env(
        scene_dir, sub_scene_name,
        args_cli.num_envs, args_cli.scene_scale,
    )
    episode_steps = np.zeros((args_cli.num_envs,), dtype=np.int64)

    if sub_scene_name in ["Hospital", "Jetracer", "Office"]:
        setup_all_lighting(env)
        for _ in range(5): simulation_app.update()

    # 热身
    for _ in range(10):
        obs, rewards, dones, infos = env.step(
            torch.zeros((args_cli.num_envs, 2), device=SIM_DEVICE))

    camera_intrinsic = (env.unwrapped.scene.sensors["camera_sensor"]
                        .data.intrinsic_matrices[0])

    # ---- GRPO Agent（使用resume确定的actual_checkpoint）----
    agent = GRPO_Agent(
        checkpoint_path=actual_checkpoint,   # ← 修复：不再用args_cli.checkpoint
        image_intrinsic=camera_intrinsic.cpu().numpy(),
        temporal_depth=16,
        cfm_num_steps=5,
        device=TRAIN_DEVICE,
        normalization_config=args_cli.normalization_config,
        lr=args_cli.lr,
        update_interval=args_cli.update_interval,
        sample_num=args_cli.sample_num,
        unfreeze_decoder_last_n=args_cli.unfreeze_last_n,
        save_dir=args_cli.save_dir,
        save_interval=args_cli.save_interval,
    )
    agent.reset(batch_size=args_cli.num_envs, stop_threshold=args_cli.stop_threshold)

    # ---- 规划线程 ----
    current_task_ref = [task]

    planning_th = threading.Thread(
        target=planning_thread_func,
        args=(agent, camera_intrinsic, current_task_ref),
        daemon=True,
    )
    planning_th.start()

    controller = DifferentialController(
        name="simple_control",
        wheel_radius=DINGO_WHEEL_RADIUS,
        wheel_base=DINGO_WHEEL_BASE,
    )

    wait_for_people(env, simulation_app)
    if hasattr(env.env, "_recent_positions"):
        env.env._recent_positions.clear()

    # resume时恢复到对应的episode文件
    if resume_state and initial_scene_ep_idx > 0:
        restore_ep_path = os.path.join(scene_path,
                                       f"episode_{initial_scene_ep_idx}.json")
        reset_people_for_episode(env, simulation_app, restore_ep_path)
        print(f"[Resume] Restored episode file: episode_{initial_scene_ep_idx}.json")

    # 初始化 vis_manager
    camera_pos = (env.unwrapped.scene.sensors["camera_sensor"]
                  .data.pos_w.cpu().numpy())
    camera_rot_quat = (env.unwrapped.scene.sensors["camera_sensor"]
                       .data.quat_w_world.cpu().numpy())
    camera_rot_quat = camera_rot_quat[:, [1, 2, 3, 0]]
    camera_rot = R.from_quat(camera_rot_quat).as_matrix()
    for i in range(args_cli.num_envs):
        vis_manager[i].reset(initial_robot_pose=np.array([
            camera_pos[i, 0], camera_pos[i, 1],
            np.arctan2(camera_rot[i, 1, 0], camera_rot[i, 0, 0]),
        ]))

    # ---- 训练状态（使用resume的初始值，不再硬编码0）----
    MIN_EPISODE_TIME = 5.0  # 低于此秒数认为是spawn碰撞，丢弃episode

    evaluation_metrics = []
    total_episode_idx  = initial_total_episode_idx   # ← 修复：从resume点开始
    scene_ep_idx       = initial_scene_ep_idx        # ← 修复：从resume点开始
    trajectory_length  = np.zeros((args_cli.num_envs,))
    social_metrics_trackers = [SocialMetricsTracker() for _ in range(args_cli.num_envs)]

    # per-episode 状态（done 时清零）
    prev_dist_to_target = [None] * args_cli.num_envs
    initial_dist        = [None] * args_cli.num_envs
    prev_camera_pos     = [None] * args_cli.num_envs

    def _make_save_dir(t, sn):
        d = os.path.join(args_cli.save_dir, "metrics", f"{t}_{sn}")
        os.makedirs(d, exist_ok=True)
        return d

    save_dir   = _make_save_dir(task, sub_scene_name)
    fps_writer = [
        imageio.get_writer(
            os.path.join(save_dir, f"fps_ep{total_episode_idx}_env{i}.mp4"), fps=10)
        for i in range(args_cli.num_envs)
    ]
    frame_count = 0

    print(f"\n{'='*60}")
    print(f"  GRPO Joint Training")
    print(f"  Tasks         : {args_cli.tasks}")
    print(f"  Scene dirs    : {args_cli.scene_dirs}")
    print(f"  Total episodes: {args_cli.num_episodes}")
    print(f"  Start episode : {total_episode_idx}")
    print(f"  Checkpoint    : {actual_checkpoint}")
    print(f"  Update interval: {args_cli.update_interval} steps")
    print(f"  LR: {args_cli.lr}  |  Save: {args_cli.save_dir}")
    print(f"{'='*60}\n")

    try:
        while simulation_app.is_running():
            with torch.inference_mode():

                # ======== 获取观测 ========
                images = infos["observations"]["rgb"].cpu().numpy()[:, :, :, 0:3]
                depths = infos["observations"]["depth"].cpu().numpy()[:, :, :]

                camera_pos = (env.unwrapped.scene.sensors["camera_sensor"]
                              .data.pos_w.cpu().numpy())
                camera_rot_quat = (env.unwrapped.scene.sensors["camera_sensor"]
                                   .data.quat_w_world.cpu().numpy())
                camera_rot_quat = camera_rot_quat[:, [1, 2, 3, 0]]
                camera_rot = R.from_quat(camera_rot_quat).as_matrix()

                robot_vel = (env.unwrapped.scene.articulations["robot"]
                             .data.root_lin_vel_w[0, :2].norm().cpu().numpy())
                robot_ang_vel = (env.unwrapped.scene.articulations["robot"]
                                 .data.root_ang_vel_w[0, 2].cpu().numpy())
                x0 = np.stack([
                    camera_pos[:, 0], camera_pos[:, 1],
                    np.arctan2(camera_rot[:, 1, 0], camera_rot[:, 0, 0]),
                    [robot_vel], [robot_ang_vel],
                ], axis=-1)

                # ======== 任务特定目标 ========
                goals            = None
                target_pos_world = None

                if task == "dynpointgoal":
                    target_pos_world = get_target_person_pos(env)
                    if target_pos_world is None:
                        obs, rewards, dones, infos = env.step(
                            torch.zeros((args_cli.num_envs, 2), device=SIM_DEVICE))
                        continue
                    goals = np.zeros((args_cli.num_envs, 3))
                    for i in range(args_cli.num_envs):
                        rel = camera_rot[i].T @ (target_pos_world[:3] - camera_pos[i])
                        goals[i] = rel[:3]

                elif task == "socialnav":
                    goals_xy = infos["observations"]["goal_pose"].cpu().numpy()[:, 0:2]
                    goals = np.concatenate(
                        [goals_xy, np.zeros((args_cli.num_envs, 1))], axis=1)
                    target_pos_world = (camera_pos[0]
                                        + camera_rot[0] @ np.array([goals[0, 0], goals[0, 1], 0.0]))

                with input_lock:
                    planning_input.current_goal  = goals.copy() if goals is not None else None
                    planning_input.current_image  = images.copy()
                    planning_input.current_depth  = depths.copy()
                    planning_input.camera_pos     = camera_pos.copy()
                    planning_input.camera_rot     = camera_rot.copy()

                for i in range(args_cli.num_envs):
                    if task in ("dynpointgoal", "socialnav") and prev_dist_to_target[i] is None:
                        d0 = np.linalg.norm(camera_pos[i, :2] - target_pos_world[:2])
                        prev_dist_to_target[i] = d0
                        initial_dist[i]        = d0
                        print(f"[INFO] Init dist to target: {d0:.2f}m (task={task})")
                    if task == "dynnogoal" and prev_camera_pos[i] is None:
                        prev_camera_pos[i] = camera_pos[i].copy()

                current_trajectory = None
                current_all_traj   = None
                current_all_values = None
                with output_lock:
                    if planning_output.trajectory_points_world is not None:
                        current_trajectory = planning_output.trajectory_points_world.copy()
                        current_all_traj   = planning_output.all_trajectories_world.copy()
                        current_all_values = planning_output.all_values_camera.copy()

            # ======== MPC 控制 ========
            if current_trajectory is not None:
                action_list = []
                for i in range(args_cli.num_envs):
                    people_positions, people_char_paths, pos_get_flag = get_people_positions(env)

                    if pos_get_flag:
                        social_metrics_trackers[i].update(
                            camera_pos[i], people_positions, env.unwrapped.step_dt)
                        goal_vis = target_pos_world[:2] if target_pos_world is not None else None
                        vis_image = vis_manager[i].visualize_trajectory_global_with_people(
                            images[i], depths[i][:, :, None],
                            camera_intrinsic.cpu().numpy(),
                            current_trajectory[i],
                            robot_pose=x0[i],
                            goal_position=goal_vis,
                            all_trajectories_points=current_all_traj[i],
                            all_trajectories_values=current_all_values[i],
                            people_positions=people_positions,
                            people_positions_dict=people_char_paths,
                        )
                    else:
                        vis_image = vis_manager[i].visualize_trajectory_global(
                            images[i], depths[i][:, :, None],
                            camera_intrinsic.cpu().numpy(),
                            current_trajectory[i],
                            robot_pose=x0[i],
                            goal_position=target_pos_world[:2] if target_pos_world is not None else None,
                            all_trajectories_points=current_all_traj[i],
                            all_trajectories_values=current_all_values[i],
                        )

                    if mpc is None:
                        continue
                    opt_u, _ = mpc.solve(x0[i, :3])
                    v, w = opt_u[1, 0], opt_u[1, 1]
                    jv = controller.forward(np.array([v, w])).joint_velocities
                    action_list.append(jv)

                    try:
                        vis_image = draw_box_with_text(vis_image, 0, 0, 430, 50,
                            f"[{task}] lin:{v:.2f} ang:{w:.2f}")
                        vis_image = draw_box_with_text(vis_image, 0, 50, 430, 50,
                            f"actual lin:{robot_vel:.2f} ang:{robot_ang_vel:.2f}")
                        if target_pos_world is not None:
                            dt = np.linalg.norm(camera_pos[i, :2] - target_pos_world[:2])
                            vis_image = draw_box_with_text(vis_image, 0, 820, 430, 50,
                                f"target dist:{dt:.2f}m  ep:{total_episode_idx}")
                        if frame_count > 0:
                            fps_writer[i].append_data(vis_image)
                        frame_count += 1
                    except Exception:
                        pass

                action = (torch.as_tensor(np.stack(action_list, axis=0), device=SIM_DEVICE)
                          if action_list else
                          torch.zeros((args_cli.num_envs, 2), device=SIM_DEVICE))

                obs, rewards, dones, infos = env.step(action)
                episode_steps += 1
                trajectory_length += (
                    infos["observations"]["policy"][:, 0] * env.unwrapped.step_dt
                ).cpu().numpy()

                # ======== 奖励 → GRPO buffer ========
                for i in range(args_cli.num_envs):
                    people_pos_list, _, _ = get_people_positions(env)
                    ep_time_so_far = episode_steps[i] * env.unwrapped.step_dt

                    if task == "dynpointgoal":
                        success = bool(infos["log"].get(
                            "Episode_Termination/arrive_goal", False))
                        stuck   = bool(infos["log"].get(
                            "Episode_Termination/stuck", False))
                        timeout = bool(infos["log"].get(
                            "Episode_Termination/max_episode_length", False))
                        r, nd = reward_dynpointgoal(
                            camera_pos[i], target_pos_world,
                            prev_dist_to_target[i], success, stuck, people_pos_list,
                            timeout=timeout)
                        prev_dist_to_target[i] = nd

                    elif task == "socialnav":
                        success = bool(infos["log"].get(
                            "Episode_Termination/arrive_goal", False))
                        stuck   = bool(infos["log"].get(
                            "Episode_Termination/stuck", False))
                        timeout = bool(infos["log"].get(
                            "Episode_Termination/max_episode_length", False))
                        r, nd = reward_socialnav(
                            camera_pos[i], target_pos_world,
                            prev_dist_to_target[i], success, stuck, people_pos_list,
                            timeout=timeout)
                        prev_dist_to_target[i] = nd

                    else:  # dynnogoal
                        prev_p = prev_camera_pos[i] if prev_camera_pos[i] is not None \
                                else camera_pos[i]
                        r = reward_dynnogoal(camera_pos[i], prev_p, people_pos_list)
                        prev_camera_pos[i] = camera_pos[i].copy()

                    # 3秒后才开始记录，跳过spawn不稳定阶段
                    with torch.set_grad_enabled(True):
                        if ep_time_so_far > 3.0:
                            agent.record_reward(r, dones[i].item())

                with torch.set_grad_enabled(True):
                    agent.maybe_update()

            else:
                # 等待规划线程生成第一条轨迹
                obs, rewards, dones, infos = env.step(
                    torch.zeros((args_cli.num_envs, 2), device=SIM_DEVICE))
                episode_steps += 1

            # ======== Episode 结束 ========
            for i in range(args_cli.num_envs):
                if not dones[i]:
                    continue

                ep_time = episode_steps[i] * env.unwrapped.step_dt
                print(f"[Episode] steps={episode_steps[i]} | time={ep_time:.2f}s | "
                      f"traj_len={trajectory_length[i]:.2f}m | task={task}")

                # ---- spawn碰撞过滤：时间过短的episode直接丢弃 ----
                if ep_time < MIN_EPISODE_TIME:
                    print(f"[FILTER] Ep {total_episode_idx} discarded "
                          f"(time={ep_time:.1f}s < {MIN_EPISODE_TIME}s, likely spawn collision)")
                    # 清空这个episode污染的buffer数据
                    agent.buffer.clear()
                    # 重置episode级状态，但不推进total_episode_idx
                    trajectory_length[i]   = 0.0
                    episode_steps[i]       = 0
                    prev_dist_to_target[i] = None
                    initial_dist[i]        = None
                    prev_camera_pos[i]     = None
                    social_metrics_trackers[i].reset()
                    agent.reset_env(i)
                    if hasattr(env.env, "_recent_positions"):
                        env.env._recent_positions.clear()
                    break

                # ---- 正常episode处理 ----
                social_metrics = social_metrics_trackers[i].get_metrics()
                ep_metric = {
                    "total_episode": total_episode_idx,
                    "scene": sub_scene_name,
                    "task": task,
                    "time": ep_time,
                    "steps": int(episode_steps[i]),
                    "trajectory_length": trajectory_length[i],
                    **{k: social_metrics[k] for k in [
                        "collision", "collision_count", "min_distance",
                        "avg_distance", "psi_count", "psi_time", "PSC", "SC"
                    ]},
                }
                if task in ("dynpointgoal", "socialnav"):
                    success = float(infos["log"].get(
                        "Episode_Termination/arrive_goal", False))
                    final_d = (np.linalg.norm(camera_pos[i, :2] - target_pos_world[:2])
                               if target_pos_world is not None else float("inf"))
                    spl = (np.clip(initial_dist[i] / max(trajectory_length[i], 0.01), 0, 1)
                           * success) if initial_dist[i] else 0.0
                    ep_metric.update({"success": success, "final_dist": final_d, "spl": spl})

                evaluation_metrics.append(ep_metric)
                print(f"\n=== Ep {total_episode_idx} | task={task} | scene={sub_scene_name} ===")
                for k, v in ep_metric.items():
                    print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

                try:
                    fps_writer[i].close()
                except Exception as e:
                    print(f"[WARN] fps_writer close: {e}")
                write_metrics(evaluation_metrics, os.path.join(save_dir, "metric.csv"))

                save_resume_state(total_episode_idx, scene_ep_idx, scheduler._idx)

                total_episode_idx += 1
                if total_episode_idx >= args_cli.num_episodes:
                    print(f"\n[INFO] Training complete!")
                    agent.save_final("rl_final.ckpt")
                    cleanup_simulation(env, simulation_app)
                    return

                # ---- 判断是否需要切换场景 ----
                need_scene_switch = scheduler.on_episode_done()

                if need_scene_switch:
                    # ======== 切换场景：需要重建 env ========
                    _, _, task = scheduler.next()
                    current_task_ref[0] = task
                    new_scene_dir, new_sub_scene, _ = scheduler.current()

                    print(f"\n[SceneSwitch] After {args_cli.episodes_per_scene} episodes, "
                        f"switching to: {new_sub_scene}  task={task}")

                    # 关闭旧 env
                    stop_event.set()
                    try:
                        fps_writer[i].close()
                    except Exception:
                        pass

                    safe_close_env(env, simulation_app)

                    for _ in range(10):
                        simulation_app.update()

                    # 重建新 env
                    stop_event.clear()
                    env, scene_config, scene_path, scene_ep_count = build_env(
                        new_scene_dir, new_sub_scene,
                        args_cli.num_envs, args_cli.scene_scale,
                    )
                    scene_ep_idx = 0

                    if new_sub_scene in ["Hospital", "Jetracer", "Office"]:
                        setup_all_lighting(env)
                        for _ in range(5):
                            simulation_app.update()

                    # 热身
                    for _ in range(10):
                        obs, rewards, dones, infos = env.step(
                            torch.zeros((args_cli.num_envs, 2), device=SIM_DEVICE))

                    camera_intrinsic = (env.unwrapped.scene.sensors["camera_sensor"]
                                        .data.intrinsic_matrices[0])

                    wait_for_people(env, simulation_app)

                else:
                    # ======== 只切换 task / episode 文件，不重建 env ========
                    _, _, task = scheduler.next()
                    current_task_ref[0] = task

                    scene_ep_idx = (scene_ep_idx + 1) % scene_ep_count
                    new_ep_path  = os.path.join(scene_path, f"episode_{scene_ep_idx}.json")
                    reset_people_for_episode(env, simulation_app, new_ep_path)

                    print(f"[INFO] Next: task={task}  ep_file={scene_ep_idx}")

                # ---- episode-end 强制更新 ----
                if len(agent.buffer) > 0:
                    print(f"[GRPO] Episode-end forced update, buffer_len={len(agent.buffer)}")
                    with torch.set_grad_enabled(True):
                        agent._grpo_update()
                    agent.buffer.clear()

                # ---- 重置 episode 级状态 ----
                save_dir      = _make_save_dir(task, scheduler.current()[1])
                fps_writer[i] = imageio.get_writer(
                    os.path.join(save_dir, f"fps_ep{total_episode_idx}_env{i}.mp4"), fps=10)
                trajectory_length[i]    = 0.0
                episode_steps[i]        = 0
                prev_dist_to_target[i]  = None
                initial_dist[i]         = None
                prev_camera_pos[i]      = None
                frame_count             = 0
                social_metrics_trackers[i].reset()
                agent.reset_env(i)

                if hasattr(env.env, "_recent_positions"):
                    env.env._recent_positions.clear()

                camera_pos = (env.unwrapped.scene.sensors["camera_sensor"]
                            .data.pos_w.cpu().numpy())
                camera_rot_quat = (env.unwrapped.scene.sensors["camera_sensor"]
                                .data.quat_w_world.cpu().numpy())
                camera_rot_quat = camera_rot_quat[:, [1, 2, 3, 0]]
                camera_rot = R.from_quat(camera_rot_quat).as_matrix()
                vis_manager[i].reset(initial_robot_pose=np.array([
                    camera_pos[i, 0], camera_pos[i, 1],
                    np.arctan2(camera_rot[i, 1, 0], camera_rot[i, 0, 0]),
                ]))
                break

    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback; traceback.print_exc()
    finally:
        agent.save_final("rl_interrupted.ckpt")
        cleanup_simulation(
            env=env if "env" in locals() else None,
            sim_app=simulation_app,
        )


if __name__ == "__main__":
    main()