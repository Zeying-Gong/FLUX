#!/usr/bin/env python3
"""
tracking_episode_collection.py
──────────────────────────────
跟踪任务数据采集 (v3)

关键修复
─────────
这一版的核心修复是 **NavMesh bake 必须在相机 / articulation 之前**。

原因：相机一旦初始化就会注册 SyntheticData OmniGraph，挂上每帧
post-process tick。而 `bake_navmesh` 的 cache 恢复路径会通过 sublayer
注入 NavMeshVolume，触发 stage re-compose，使相机的 graph 引用失效
(`Graph(INVALID)`)。失效的 graph 在每次 `simulation_app.update()` 时抛
异常，打断 NavMesh bake 系统的 update tick，最终 `get_navmesh()` 返回
None。

正确顺序（对齐 test_dataset_pipeline.py）：
    1. open_stage
    2. bake_navmesh   ← 干净 stage，没有相机
    3. World + initialize_physics
    4. add_reference robot + ArticulationView
    5. spawn camera   ← 此时 stage 已稳定
    6. 进入每个 episode：清角色 → spawn 角色 → reset robot → 控制循环

因此本版不再调用 `IsaacSimEvaluator.setup()`（它把 camera/articulation 都
放在 open_stage 之后立刻做了，没法在中间插 bake），而是手动按正确顺序
组装。控制时直接用 ArticulationView 和 Camera 对象，不通过 evaluator
封装 —— 简单直接，也不引入新的状态。

用法
────
    /isaac-sim/python.sh tracking_episode_collection.py \\
        --usda /workspace/SAGE-3D_Official/SAGE-3D_data/usda/839873.usda \\
        --episode_dir /workspace/SAGE-3D_Official/SAGE-3D_data/v1_tracking_episodes \\
        --scene_id 839873 --start_idx 0 --end_idx 1 \\
        --save_images --image_save_dir /tmp/tracking_images \\
        --output_metrics /tmp/tracking_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = "/workspace/FLUX"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Tracking episode data collection")
    p.add_argument("--usda", required=True,
                   help="Path to scene USD file (e.g. .../usda/839873.usda)")
    p.add_argument("--episode_dir", required=True,
                   help="Root dir containing per-scene subfolders with episode_*.json")
    p.add_argument("--scene_id", required=True,
                   help="Scene subfolder name (e.g. 839873)")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx",   type=int, default=-1,
                   help="Exclusive end index; -1 means all")
    p.add_argument("--max_steps", type=int, default=500,
                   help="Maximum control ticks per episode")

    # NavMesh
    p.add_argument("--collision_root",  default="/World/scene_collision")
    p.add_argument("--volume_padding",  type=float, default=1.2)
    p.add_argument("--fallback_size",   type=float, default=100.0)
    p.add_argument("--warmup_frames",   type=int,   default=60)
    p.add_argument("--cache_dir",       default=None)
    p.add_argument("--force_rebake",    action="store_true")
    p.add_argument("--semantic_map_json", default=None)

    # Robot
    p.add_argument("--robot_usd",
                   default="/workspace/FLUX/assets/robots/dingo_fixed.usd")
    p.add_argument("--robot_z_height", type=float, default=0.1)

    # Differential drive (Dingo)
    p.add_argument("--wheel_radius", type=float, default=0.0762)
    p.add_argument("--wheel_base",   type=float, default=0.32)

    # Controller
    p.add_argument("--control_gain_linear",  type=float, default=1.0)
    p.add_argument("--control_gain_angular", type=float, default=2.0)
    p.add_argument("--max_linear_vel",       type=float, default=0.8)
    p.add_argument("--max_angular_vel",      type=float, default=1.5)
    p.add_argument("--physics_dt",  type=float, default=1.0 / 60.0)
    p.add_argument("--decimation",  type=int,   default=9,
                   help="Physics substeps per control tick (9 * 1/60 = 0.15s)")

    # Tracking metric thresholds
    p.add_argument("--tracking_dist_min",     type=float, default=1.0)
    p.add_argument("--tracking_dist_max",     type=float, default=3.0)
    p.add_argument("--tracking_angle_thresh", type=float, default=60.0)
    p.add_argument("--loss_dist_thresh",      type=float, default=5.0)

    # IO
    p.add_argument("--save_images", action="store_true")
    p.add_argument("--save_image_every", type=int, default=5)
    p.add_argument("--image_save_dir",   type=str, default="./tracking_images")
    p.add_argument("--output_metrics",   type=str, default="./tracking_metrics.csv")
    p.add_argument("--headless", action="store_true", default=False)
    return p.parse_args()


ARGS = parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# SimulationApp — must come BEFORE any omni / isaacsim imports
# ═══════════════════════════════════════════════════════════════════════════
from isaacsim import SimulationApp

EXP_PATH = os.environ.get("EXP_PATH")
if EXP_PATH is None:
    print("[FATAL] EXP_PATH env var not set.")
    sys.exit(1)

CUSTOM_APP_PATH = os.path.join(
    EXP_PATH, "isaacsim.exp.action_and_event_data_generation.base.kit"
)

simulation_app = SimulationApp(
    launch_config={
        "renderer": "RayTracedLighting",
        "headless": ARGS.headless,
        "enable_cameras": True,
        "crash_reporter/enabled": False,
        "crash_reporter/skip_old_dump_upload": True,
    },
    experience=CUSTOM_APP_PATH,
)


# ═══════════════════════════════════════════════════════════════════════════
# Post-launch imports
# ═══════════════════════════════════════════════════════════════════════════
import carb
import omni.usd
import omni.timeline
from pxr import Usd, Sdf, UsdGeom

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import SingleArticulation as ArticulationView
from isaacsim.sensors.camera import Camera as IsaacCamera

from sage_utils.navmesh_utils import cache_paths, bake_navmesh, safe_query_random_point
from sage_utils.people_utils import (
    load_character_assets,
    spawn_character,
    load_default_skeleton_and_animations,
    bind_animation_graph_to_characters,
    attach_behavior_scripts_to_characters,
)
from isaacsim.replicator.agent.core.settings import PrimPaths
from omni.kit.viewport.utility import get_active_viewport
from sage_utils.people_utils import frame_viewport_on

# ═══════════════════════════════════════════════════════════════════════════
# Constants (mirror evaluator.py)
# ═══════════════════════════════════════════════════════════════════════════
ROBOT_PRIM_PATH    = "/World/Robot"
CAMERA_PRIM_PATH   = "/World/Robot/base_link/front_cam"
CHARACTERS_ROOT    = "/World/Characters"
ROBOT_JOINT_NAMES  = ["left_wheel_joint", "right_wheel_joint"]
CAM_W, CAM_H       = 640, 360


# ═══════════════════════════════════════════════════════════════════════════
# Sim helpers
# ═══════════════════════════════════════════════════════════════════════════
def update_sim(steps: int = 1):
    for _ in range(steps):
        simulation_app.update()


def open_stage(usda_path: str) -> bool:
    if not os.path.exists(usda_path):
        print(f"[FATAL] USDA not found: {usda_path}")
        return False
    print(f"[STAGE] Opening: {usda_path}")
    omni.usd.get_context().open_stage(usda_path)
    update_sim(30)
    waited = 0
    while omni.usd.get_context().get_stage_loading_status()[1] > 0:
        simulation_app.update()
        waited += 1
        if waited > 3000:
            print("[WARN] Stage still loading after 30s.")
            break
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[FATAL] Stage failed to open.")
        return False
    print(f"[STAGE] Loaded. Default prim: {stage.GetDefaultPrim().GetPath()}")
    return True


def hide_navmesh_volumes():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    hidden = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() == "NavMeshVolume":
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden += 1
    print(f"[NavMesh] Hidden {hidden} NavMeshVolume prim(s).")


def clear_all_characters():
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(CHARACTERS_ROOT)
    if root and root.IsValid():
        for prim in list(root.GetChildren()):
            stage.RemovePrim(prim.GetPath())
        print("[Clear] All characters removed.")

# ═══════════════════════════════════════════════════════════════════════════
# Robot & camera spawning (manually, NOT via IsaacSimEvaluator.setup)
# ═══════════════════════════════════════════════════════════════════════════
def add_robot_and_articulation(world: World) -> ArticulationView:
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(ROBOT_PRIM_PATH).IsValid():
        if not os.path.exists(ARGS.robot_usd):
            raise FileNotFoundError(f"Robot USD not found: {ARGS.robot_usd}")
        print(f"[Robot] Adding reference: {ARGS.robot_usd}")
        add_reference_to_stage(ARGS.robot_usd, ROBOT_PRIM_PATH)
        update_sim(60)  # give the reference time to fully compose

    robot = ArticulationView(prim_path=ROBOT_PRIM_PATH, name="robot_view")
    world.scene.add(robot)
    world.reset()
    update_sim(10)

    if robot.num_dof is None or robot.num_dof == 0:
        raise RuntimeError("Robot articulation has 0 dofs — bad USD or wrong root?")
    print(f"[Robot] num_dof = {robot.num_dof}, dof_names = {list(robot.dof_names)}")
    return robot


def spawn_policy_camera() -> IsaacCamera:
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/World/Robot/base_link").IsValid():
        print("[WARN] /World/Robot/base_link not found; camera will be created "
              "but may not follow the robot. Check the robot USD hierarchy.")

    cam_trans_ros = np.array([0.0, 0.0, 0.3], dtype=np.float64)
    cam_rot_ros   = np.array([-0.5, 0.5, -0.5, 0.5], dtype=np.float64)

    cam = IsaacCamera(
        prim_path=CAMERA_PRIM_PATH,
        resolution=(CAM_W, CAM_H),
        translation=cam_trans_ros,
    )
    for _ in range(5):
        simulation_app.update()
    cam.initialize()
    cam.set_local_pose(
        translation=cam_trans_ros,
        orientation=cam_rot_ros,
        camera_axes="ros",
    )
    cam.set_focal_length(1.4)
    cam.set_focus_distance(0.205)
    cam.set_horizontal_aperture(1.88)
    cam.set_clipping_range(0.01, 100.0)
    cam.add_distance_to_image_plane_to_frame()
    for _ in range(5):
        simulation_app.update()
    print(f"[Camera] Policy camera initialized at {CAMERA_PRIM_PATH}")
    return cam

def position_chase_camera(
    robot,
    distance: float = 4.0,
    height: float = 2.5,
    behind_offset_along_yaw: bool = True,
):
    """
    一次性把 active viewport camera 放到机器人后上方，朝向机器人。
    调用之后用户可以自由用鼠标在 viewport 里旋转/缩放/平移。

    Args:
        robot: ArticulationView (用来取当前位姿)
        distance: 相机到机器人的水平距离 (m)
        height:   相机在机器人之上的高度 (m)
        behind_offset_along_yaw: True = 沿机器人朝向的反方向后退；
                                 False = 沿世界 -X 方向后退 (与机器人朝向无关)
    """
    from pxr import Gf, UsdGeom
    from omni.kit.viewport.utility import get_active_viewport

    robot_pos, robot_yaw = get_robot_pose(robot)

    # 1) 算相机世界坐标：机器人后方 (沿 yaw 反向) + 上方
    if behind_offset_along_yaw:
        back_x = -math.cos(robot_yaw)
        back_y = -math.sin(robot_yaw)
    else:
        back_x, back_y = -1.0, 0.0
    cam_pos = Gf.Vec3d(
        float(robot_pos[0] + back_x * distance),
        float(robot_pos[1] + back_y * distance),
        float(robot_pos[2] + height),
    )
    target_pos = Gf.Vec3d(float(robot_pos[0]),
                          float(robot_pos[1]),
                          float(robot_pos[2]) + 0.3)  # 看机器人略偏上一点

    # 2) 拿到 active viewport 的 camera prim
    viewport = get_active_viewport()
    if viewport is None:
        print("[ChaseCam] No active viewport, skip.")
        return
    cam_path = viewport.get_active_camera()
    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath(cam_path)
    if not cam_prim or not cam_prim.IsValid():
        print(f"[ChaseCam] Camera prim not found: {cam_path}")
        return

    # 3) 构造 lookAt 矩阵 (USD 相机看向 -Z, up 用 +Z)
    #    Gf.Matrix4d.SetLookAt 给出的是 view matrix (world->cam)，
    #    我们要的是 cam->world，所以取逆。
    up = Gf.Vec3d(0.0, 0.0, 1.0)
    view = Gf.Matrix4d().SetLookAt(cam_pos, target_pos, up)
    cam_world_xform = view.GetInverse()

    # 4) 算 parent-space 的 local transform 并写回 (覆盖现有 xformOps)
    xformable = UsdGeom.Xformable(cam_prim)
    imageable = UsdGeom.Imageable(cam_prim)
    parent_xform = imageable.ComputeParentToWorldTransform(Usd.TimeCode.Default())
    local_xform  = cam_world_xform * parent_xform.GetInverse()

    # 清掉旧的 xformOps，写入一个 transform op，这样后续 viewport 操作正常
    xformable.ClearXformOpOrder()
    op = xformable.AddTransformOp()
    op.Set(local_xform)

    # 维护 center-of-interest，让后续鼠标轨道旋转围绕机器人
    coi_attr_name = "omni:kit:centerOfInterest"
    coi_attr = cam_prim.GetAttribute(coi_attr_name)
    if not coi_attr:
        coi_attr = cam_prim.CreateAttribute(
            coi_attr_name, Sdf.ValueTypeNames.Vector3d, custom=True,
            variability=Sdf.VariabilityUniform,
        )
    # COI 在 camera local space，方向是 -Z，长度 = 相机到目标的距离
    coi_len = (cam_pos - target_pos).GetLength()
    coi_attr.Set(Gf.Vec3d(0.0, 0.0, -coi_len))

    print(f"[ChaseCam] {cam_path} -> pos={tuple(cam_pos)}, "
          f"looking at robot at {tuple(target_pos)}")

# ═══════════════════════════════════════════════════════════════════════════
# Episode → characters
# ═══════════════════════════════════════════════════════════════════════════
def spawn_characters_from_episode(
    episode: dict,
    char_pool: List[str],
) -> Dict[str, str]:
    spawn_positions = episode["characters"]["spawn_positions"]
    commands_dict   = episode["characters"]["commands"]

    char_paths: Dict[str, str] = {}
    for idx, (char_name, spawn_data) in enumerate(spawn_positions.items()):
        pos = spawn_data["pos"]
        usd = char_pool[idx % len(char_pool)]
        prim_path = spawn_character(simulation_app, idx, usd, pos)
        char_paths[char_name] = prim_path
        print(f"[Episode] Spawned {char_name} at {pos}  ->  {prim_path}")

    load_default_skeleton_and_animations(simulation_app)
    update_sim(30)
    bind_animation_graph_to_characters(simulation_app)
    update_sim(30)

    # ✅ 先写入命令数据
    _write_episode_commands(commands_dict)
    update_sim(30)

    # ✅ 再挂载行为脚本（此时脚本能读到数据）
    attach_behavior_scripts_to_characters(simulation_app)
    update_sim(60)

    return char_paths


def _write_episode_commands(commands_dict: dict):
    stage = omni.usd.get_context().get_stage()
    parent_path = str(PrimPaths.characters_parent_path())

    for char_name, cmds in commands_dict.items():
        prim_path = f"{parent_path}/{char_name}"
        char_prim = stage.GetPrimAtPath(prim_path)
        if not char_prim or not char_prim.IsValid():
            print(f"[WARN] Character prim {prim_path} not found, skip")
            continue

        skelroot = None
        for desc in Usd.PrimRange(char_prim):
            if desc.GetTypeName() == "SkelRoot":
                skelroot = desc
                break
        if skelroot is None:
            print(f"[WARN] No SkelRoot under {prim_path}, skip")
            continue

        command_strings: List[str] = []
        path_data: Dict[int, list] = {}
        goto_idx = 0
        for cmd in cmds:
            cmd_name = cmd.get("cmd", "")
            params   = cmd.get("params", [])
            command_strings.append(f"{cmd_name} " + " ".join(str(p) for p in params))
            if cmd_name == "GoTo":
                if "path" in cmd and len(cmd["path"]) > 0:
                    path_data[goto_idx] = cmd["path"]
                goto_idx += 1

        sd_attr = skelroot.GetAttribute("omni:scripting:scriptData")
        if not sd_attr:
            sd_attr = skelroot.CreateAttribute(
                "omni:scripting:scriptData", Sdf.ValueTypeNames.StringArray)
        sd_attr.Set(command_strings)

        if path_data:
            pd_attr = skelroot.GetAttribute("omni:scripting:pathData")
            if not pd_attr:
                pd_attr = skelroot.CreateAttribute(
                    "omni:scripting:pathData", Sdf.ValueTypeNames.String)
            pd_attr.Set(json.dumps(path_data))

        print(f"[Episode] {char_name}: {len(command_strings)} commands, "
              f"{len(path_data)} paths written.")


# ═══════════════════════════════════════════════════════════════════════════
# Pose helpers
# ═══════════════════════════════════════════════════════════════════════════
def _find_skelroot(prim) -> Optional[Usd.Prim]:
    for desc in Usd.PrimRange(prim):
        if desc.GetTypeName() == "SkelRoot":
            return desc
    return None


def get_character_pose(char_prim_path: str) -> Tuple[np.ndarray, float]:
    stage = omni.usd.get_context().get_stage()
    prim  = stage.GetPrimAtPath(char_prim_path)
    if not prim or not prim.IsValid():
        return np.zeros(3), 0.0
    skel = _find_skelroot(prim) or prim
    tl = omni.timeline.get_timeline_interface()
    try:
        tc = Usd.TimeCode(tl.get_current_time() * tl.get_time_codes_per_seconds())
    except Exception:
        tc = Usd.TimeCode.Default()
    xform = UsdGeom.Xformable(skel)
    world_transform = xform.ComputeLocalToWorldTransform(tc)
    trans = world_transform.ExtractTranslation()
    q     = world_transform.ExtractRotation().GetQuaternion()
    w     = q.GetReal()
    ix, iy, iz = q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]
    yaw = math.atan2(2.0 * (w * iz + ix * iy),
                     1.0 - 2.0 * (iy * iy + iz * iz))
    return np.array([trans[0], trans[1], trans[2]]), yaw


def get_robot_pose(robot) -> Tuple[np.ndarray, float]:
    pos, quat = robot.get_world_pose()
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    yaw = math.atan2(2.0 * (w * z + x * y),
                     1.0 - 2.0 * (y * y + z * z))
    return np.array([pos[0], pos[1], pos[2]]), yaw

def reset_robot(robot, start_pos_xy, start_yaw, world: World):
    pos = np.array([start_pos_xy[0], start_pos_xy[1], ARGS.robot_z_height],
                   dtype=np.float32)
    quat = np.array([math.cos(start_yaw / 2.0),
                     0.0, 0.0,
                     math.sin(start_yaw / 2.0)], dtype=np.float32)
    robot.set_world_pose(position=pos, orientation=quat)
    nd = int(robot.num_dof)
    try:
        robot.set_joint_velocities(np.zeros(nd, dtype=np.float32))
    except Exception:
        pass
    try:
        robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
        robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
    except Exception:
        pass
    for _ in range(15):
        world.step(render=False)
    check_pos, check_yaw = get_robot_pose(robot)
    print(f"[Reset] commanded pos={pos.tolist()}, "
          f"yaw={math.degrees(start_yaw):.1f} deg "
          f"-> actual pos={check_pos.tolist()}, "
          f"yaw={math.degrees(check_yaw):.1f} deg")


# ═══════════════════════════════════════════════════════════════════════════
# Controller
# ═══════════════════════════════════════════════════════════════════════════
def compute_control(robot_pos, robot_yaw, target_pos,
                    gain_linear, gain_angular,
                    max_linear, max_angular) -> Tuple[float, float]:
    delta = target_pos[:2] - robot_pos[:2]
    dist  = float(np.linalg.norm(delta))
    if dist < 0.01:
        return 0.0, 0.0
    desired_yaw = math.atan2(float(delta[1]), float(delta[0]))
    angle_diff  = math.atan2(math.sin(desired_yaw - robot_yaw),
                             math.cos(desired_yaw - robot_yaw))
    forward_gate = max(0.0, math.cos(angle_diff))
    linear  = float(np.clip(gain_linear * dist * forward_gate, 0.0, max_linear))
    angular = float(np.clip(gain_angular * angle_diff, -max_angular, max_angular))
    return linear, angular


def diff_drive_joint_vels(linear, angular, wheel_radius, wheel_base):
    left  = (linear - angular * wheel_base / 2.0) / wheel_radius
    right = (linear + angular * wheel_base / 2.0) / wheel_radius
    return left, right


def resolve_wheel_dof_order(robot) -> Tuple[int, int]:
    names = list(robot.dof_names)
    try:
        left_idx  = names.index(ROBOT_JOINT_NAMES[0])
        right_idx = names.index(ROBOT_JOINT_NAMES[1])
    except ValueError:
        print(f"[WARN] {ROBOT_JOINT_NAMES} not in dof_names; falling back to 0,1")
        return 0, 1
    print(f"[Robot] left  wheel dof = {left_idx} ({names[left_idx]})")
    print(f"[Robot] right wheel dof = {right_idx} ({names[right_idx]})")
    return left_idx, right_idx


# ═══════════════════════════════════════════════════════════════════════════
# Image saving
# ═══════════════════════════════════════════════════════════════════════════
def save_rgb_depth(cam: IsaacCamera, step_idx: int, save_dir: str, episode_id: int):
    try:
        from PIL import Image
        rgb_raw   = cam.get_rgb()
        depth_raw = cam.get_depth()
        if rgb_raw is None or depth_raw is None:
            return
        rgb   = np.asarray(rgb_raw)
        depth = np.nan_to_num(np.asarray(depth_raw).astype(np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0)
        episode_dir = os.path.join(save_dir, f"episode_{episode_id:04d}")
        rgb_dir     = os.path.join(episode_dir, "rgb")
        depth_dir   = os.path.join(episode_dir, "depth")
        os.makedirs(rgb_dir,   exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        Image.fromarray(rgb).save(os.path.join(rgb_dir, f"{step_idx:05d}.png"))
        depth_mm = (depth * 1000.0).astype(np.uint16)
        Image.fromarray(depth_mm, mode="I;16").save(
            os.path.join(depth_dir, f"{step_idx:05d}.png"))
    except Exception as e:
        print(f"[WARN] Image save failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Episode helpers
# ═══════════════════════════════════════════════════════════════════════════
def resolve_target_character(episode: dict,
                             char_paths: Dict[str, str]) -> Optional[str]:
    target_name = None
    if "tracking" in episode and isinstance(episode["tracking"], dict):
        target_name = episode["tracking"].get("target_character")
    if target_name and target_name in char_paths:
        return char_paths[target_name]
    if target_name is not None:
        print(f"[WARN] tracking.target_character='{target_name}' not in spawn "
              f"list {list(char_paths)}, falling back to first")
    if char_paths:
        first_name = next(iter(char_paths))
        print(f"[Episode] target = '{first_name}' (auto)")
        return char_paths[first_name]
    return None


def extract_ep_id(p: str) -> int:
    try:    return int(os.path.basename(p).split("_")[1].split(".")[0])
    except: return 999999


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> int:
    # ── 1. open_stage (clean, no robot, no camera) ────────────────────────
    if not open_stage(ARGS.usda):
        return 1

    # ── 2. NavMesh bake — MUST come before any camera ─────────────────────
    navvols_usda = cache_paths(ARGS.usda, ARGS.cache_dir)
    print(f"[Cache] NavVols USDA : {navvols_usda}")
    nav_ok, inav = bake_navmesh(
        app=simulation_app,
        navvols_usda=navvols_usda,
        force_rebake=ARGS.force_rebake,
        collision_root=ARGS.collision_root,
        volume_padding=ARGS.volume_padding,
        fallback_size=ARGS.fallback_size,
        warmup_frames=ARGS.warmup_frames,
        semantic_map_json=ARGS.semantic_map_json,
    )
    if not nav_ok:
        print("[FATAL] NavMesh bake failed")
        simulation_app.close()
        return 1
    nm = inav.get_navmesh()
    if nm is None:
        print("[FATAL] get_navmesh() returned None after successful bake")
        simulation_app.close()
        return 1
    rp = safe_query_random_point(nm)
    print(f"[NavMesh] OK. Sample point: {tuple(rp) if rp is not None else None}")

    hide_navmesh_volumes()
    update_sim(10)

    # ── 3. World + physics ────────────────────────────────────────────────
    print("[World] Creating World + physicsScene...")
    world = World(physics_dt=ARGS.physics_dt, rendering_dt=ARGS.physics_dt * 3)
    world.initialize_physics()
    world.reset()
    update_sim(10)
    print("[World] Physics initialized.")

    # ── 4. Robot (articulation) ───────────────────────────────────────────
    robot = add_robot_and_articulation(world)
    left_idx, right_idx = resolve_wheel_dof_order(robot)

    # ── 5. Policy camera (LAST — only after stage is stable) ──────────────
    cam = spawn_policy_camera()

    # ── 6. Character asset pool ───────────────────────────────────────────
    char_pool = load_character_assets()
    print(f"[People] {len(char_pool)} character assets loaded.")

    # ── 7. Locate episode JSONs ───────────────────────────────────────────
    ep_dir = os.path.join(ARGS.episode_dir, ARGS.scene_id)
    if not os.path.isdir(ep_dir):
        print(f"[FATAL] Episode dir not found: {ep_dir}")
        simulation_app.close()
        return 1
    candidates = sorted(glob.glob(os.path.join(ep_dir, "episode_*.json")),
                        key=extract_ep_id)
    if not candidates:
        print(f"[FATAL] No episode_*.json found in {ep_dir}")
        simulation_app.close()
        return 1
    end_idx  = len(candidates) if ARGS.end_idx == -1 else ARGS.end_idx
    selected = candidates[ARGS.start_idx:end_idx]
    print(f"[Episode] Selected {len(selected)}/{len(candidates)} episodes")

    # ── 8. Start timeline ─────────────────────────────────────────────────
    tl = omni.timeline.get_timeline_interface()
    tl.set_current_time(0.0)
    tl.play()
    update_sim(5)

    all_metrics: List[dict] = []

    # ── 9. Per-episode loop ───────────────────────────────────────────────
    for ep_path in selected:
        ep_id = extract_ep_id(ep_path)
        print(f"\n{'='*60}\nEpisode {ep_id}\n{'='*60}")

        with open(ep_path, "r") as f:
            episode = json.load(f)["episode"]

        # 9a. Clear & spawn pedestrians
        clear_all_characters()
        update_sim(10)
        char_paths = spawn_characters_from_episode(episode, char_pool)

        # 9b. Resolve target
        target_prim_path = resolve_target_character(episode, char_paths)
        if target_prim_path is None:
            print("[ERROR] No target character resolvable, skip episode")
            continue

        # 9c. Reset robot
        robot_block         = episode["robot"]
        robot_start_pos_xyz = robot_block["start_pos"]
        robot_start_yaw     = float(robot_block.get("start_orientation", 0.0))
        print(f"[Episode] start_pos={robot_start_pos_xyz}, "
              f"start_yaw={math.degrees(robot_start_yaw):.1f} deg, "
              f"target_prim={target_prim_path}")
        reset_robot(robot, robot_start_pos_xyz[:2], robot_start_yaw, world)

        # Frame the viewport above the robot for debug visibility (once per episode)
        # frame_viewport_on(ROBOT_PRIM_PATH)
        position_chase_camera(robot, distance=4.0, height=2.5)

        # Warm up pedestrian behavior scripts
        for _ in range(30):
            world.step(render=False)
            simulation_app.update()

        # 9d. Control loop
        step           = 0
        tracking_steps = 0
        distances:     List[float] = []
        heading_errors: List[float] = []
        done_reason = "max_steps"

        while step < ARGS.max_steps:
            robot_pos, robot_yaw = get_robot_pose(robot)
            target_pos, _        = get_character_pose(target_prim_path)

            delta = target_pos[:2] - robot_pos[:2]
            dist_to_target = float(np.linalg.norm(delta))
            if dist_to_target > 1e-6:
                desired_yaw = math.atan2(float(delta[1]), float(delta[0]))
            else:
                desired_yaw = robot_yaw
            angle_err_deg = abs(math.degrees(
                math.atan2(math.sin(desired_yaw - robot_yaw),
                           math.cos(desired_yaw - robot_yaw))))

            distances.append(dist_to_target)
            heading_errors.append(angle_err_deg)

            in_window = (ARGS.tracking_dist_min <= dist_to_target
                         <= ARGS.tracking_dist_max)
            in_view   = (angle_err_deg <= ARGS.tracking_angle_thresh)
            if in_window and in_view:
                tracking_steps += 1

            if dist_to_target > ARGS.loss_dist_thresh:
                print(f"[EP{ep_id}] Target lost at step {step}, "
                      f"dist={dist_to_target:.2f}")
                done_reason = "target_lost"
                break

            linear, angular = compute_control(
                robot_pos, robot_yaw, target_pos,
                ARGS.control_gain_linear, ARGS.control_gain_angular,
                ARGS.max_linear_vel, ARGS.max_angular_vel,
            )
            left_v, right_v = diff_drive_joint_vels(
                linear, angular, ARGS.wheel_radius, ARGS.wheel_base)

            nd = int(robot.num_dof)
            vel_cmd = np.zeros(nd, dtype=np.float32)
            vel_cmd[left_idx]  = left_v
            vel_cmd[right_idx] = right_v
            try:
                robot.apply_action(ArticulationAction(joint_velocities=vel_cmd))
            except Exception as e:
                print(f"[ERROR] apply_action failed: {e}")
                done_reason = "control_error"
                break

            # Advance one control tick:
            #   decimation × physics step + one omni update (anim.people tick)
            for _ in range(ARGS.decimation):
                world.step(render=False)
            simulation_app.update()

            if ARGS.save_images and step % ARGS.save_image_every == 0:
                save_rgb_depth(cam, step, ARGS.image_save_dir, ep_id)

            step += 1

        # 9e. Episode summary
        episode_length = step
        tracking_rate  = tracking_steps / max(episode_length, 1)
        success = (bool(distances)
                   and tracking_rate >= 0.8
                   and distances[-1] <= ARGS.tracking_dist_max)

        metrics = {
            "episode_id":            ep_id,
            "success":               int(success),
            "episode_length":        episode_length,
            "tracking_rate":         tracking_rate,
            "collision":             0,
            "initial_dist":          distances[0]  if distances else 0.0,
            "final_dist":            distances[-1] if distances else 0.0,
            "avg_dist":              float(np.mean(distances))      if distances      else 0.0,
            "avg_heading_error_deg": float(np.mean(heading_errors)) if heading_errors else 0.0,
            "done_reason":           done_reason,
        }
        all_metrics.append(metrics)
        print(f"[EP{ep_id}] done | success={success} len={episode_length} "
              f"tr={tracking_rate:.3f} reason={done_reason}")

        # Stop the robot before the next episode
        try:
            nd = int(robot.num_dof)
            robot.apply_action(ArticulationAction(
                joint_velocities=np.zeros(nd, dtype=np.float32)))
        except Exception:
            pass
        for _ in range(5):
            world.step(render=False)
            simulation_app.update()

    # ── 10. Save metrics ──────────────────────────────────────────────────
    if all_metrics:
        out_dir = os.path.dirname(os.path.abspath(ARGS.output_metrics)) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(ARGS.output_metrics, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"[Metrics] Saved {len(all_metrics)} rows to {ARGS.output_metrics}")
    else:
        print("[Metrics] No episodes were collected.")

    print("\n[Finished] Cleaning up...")
    try:    tl.stop()
    except: pass
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())