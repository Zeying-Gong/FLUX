#!/usr/bin/env python3
"""
tracking_episode_collection.py
──────────────────────────────
跟踪任务数据采集 (v4)

v4 变化
─────────
- 预读 episode 0，把 robot/行人/相机的初始化全部对齐到 episode 0 的最终位姿，
  这样程序刚启动时画面就是 episode 0 的样子，方便观察。
- 抽出 setup_episode() helper，episode 切换流程更清晰。
- 修正 control loop 被误缩进进 warmup for 循环的 bug。
- timeline 一次性 play，整个 run 期间不停 (避免 stop 销毁 PhysX view)。

正确顺序：
    1. open_stage
    2. bake_navmesh                ← 干净 stage，没有相机
    3. World + initialize_physics
    4. 预读 episode 0
    5. add_reference robot @ ep0 start_pos
    6. init_chase_camera 贴上去
    7. spawn policy camera
    8. tl.play() —— 永不停
    9. 预 spawn episode 0 行人 (不挂行为脚本)
    10. 进入 episode 循环：
        - ep0: 跳过 spawn，直接 attach behavior + 控制
        - ep1+: clear → spawn → reset → attach behavior + 控制

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
    p.add_argument("--usda", default=None,
                help="Path to scene USD file. If omitted, auto-derived from "
                        "--usda_root and --scene_id.")
    p.add_argument("--usda_root",
                default="/workspace/SAGE-3D_Official/SAGE-3D_data/usda",
                help="Root dir of <scene_id>.usda files.")
    p.add_argument("--episode_dir", required=True,
                help="Root dir containing per-scene subfolders. Subfolder may "
                        "be named '<scene_id>' or '<NNNN>_<scene_id>'.")
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
    p.add_argument("--follow_distance", type=float, default=1.5,
               help="Desired following distance behind pedestrian (m)")

    # Chase camera
    p.add_argument("--chase_distance", type=float, default=1.5,
                   help="Third-person camera horizontal distance behind robot (m)")
    p.add_argument("--chase_height", type=float, default=0.8,
                   help="Third-person camera vertical height above robot (m)")
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
from pxr import Usd, Sdf, UsdGeom, Gf

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
# Constants
# ═══════════════════════════════════════════════════════════════════════════
ROBOT_PRIM_PATH    = "/World/Robot"
CAMERA_PRIM_PATH   = "/World/Robot/base_link/front_cam"
CHARACTERS_ROOT    = "/World/Characters"
ROBOT_JOINT_NAMES  = ["left_wheel_joint", "right_wheel_joint"]
CAM_W, CAM_H       = 640, 360
# ═══════════════════════════════════════════════════════════════════════════
# NavMesh-based pursuit constants
# ═══════════════════════════════════════════════════════════════════════════
REPLAN_PERIOD     = 10     # control ticks
REPLAN_TARGET_THR = 0.5    # m, follow-point displacement triggers re-plan
LOOKAHEAD_DIST    = 1.0    # m, carrot distance ahead of robot along path
WAYPOINT_REACH    = 0.3    # m, advance path index when robot within this
MAX_PLAN_FAILS    = 5      # consecutive failures → end episode
NM_AGENT_RADIUS   = 0.25   # m, agent radius for NavMesh query
# ─── Stuck detection / recovery ─────────────────────────────────────────────
STUCK_WINDOW          = 8       # control ticks to look back
STUCK_POS_THRESH      = 0.05    # m — moved less than this over the window
STUCK_YAW_THRESH_DEG  = 5.0     # deg — turned less than this over the window
STUCK_CMD_THRESH      = 0.5     # rad/s — only count as "commanded to move" above this
RECOVERY_DURATION     = 8       # control ticks to execute backup
RECOVERY_LINEAR       = -0.4    # m/s during recovery (negative = reverse)
RECOVERY_ANGULAR_MAG  = 1.2     # rad/s, sign chosen at recovery start
MAX_RECOVERIES        = 3       # consecutive recoveries before giving up
DIRECT_FOLLOW_RANGE = 2.5    # m, within this distance use direct follow, not navmesh
DIRECT_DEADZONE_POS = 0.25   # m
DIRECT_DEADZONE_ANG = 10.0   # deg

def compute_follow_point(target_pos: np.ndarray,
                         target_yaw: float,
                         follow_distance: float) -> np.ndarray:
    """Point that is `follow_distance` m directly behind the target
    along its current yaw (XY plane). Z taken from target."""
    bx = -math.cos(target_yaw)
    by = -math.sin(target_yaw)
    return np.array([
        float(target_pos[0]) + bx * follow_distance,
        float(target_pos[1]) + by * follow_distance,
        float(target_pos[2]),
    ], dtype=np.float64)


def snap_to_navmesh(nm, xyz: np.ndarray, search_radius: float = 1.0):
    """Project an arbitrary point onto the nearest navmesh location.
    Returns numpy array [x,y,z] or None."""
    try:
        pt = carb.Float3(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        result = nm.query_closest_point(pt, search_radius)
        if result is None or result[0] is None:
            return None
        snapped = result[0]
        return np.array([float(snapped[0]),
                         float(snapped[1]),
                         float(snapped[2])], dtype=np.float64)
    except Exception:
        return None


def plan_navmesh_path(nm, from_xyz: np.ndarray, to_xyz: np.ndarray,
                      agent_radius: float = NM_AGENT_RADIUS):
    """Wrap nm.query_shortest_path; returns list of [x,y,z] or None."""
    try:
        a = carb.Float3(float(from_xyz[0]), float(from_xyz[1]), float(from_xyz[2]))
        b = carb.Float3(float(to_xyz[0]),   float(to_xyz[1]),   float(to_xyz[2]))
        path = nm.query_shortest_path(a, b, agent_radius=agent_radius)
    except Exception:
        return None
    if path is None:
        return None
    if hasattr(path, "get_points"):
        pts = path.get_points()
    elif hasattr(path, "points"):
        pts = path.points
    else:
        pts = path
    if pts is None or len(pts) == 0:
        return None
    return [[float(p[0]), float(p[1]), float(p[2])] for p in pts]


def pick_carrot(path: List[List[float]],
                path_idx: int,
                robot_pos: np.ndarray,
                lookahead: float = LOOKAHEAD_DIST) -> Tuple[np.ndarray, int]:
    """Walk forward along `path` from `path_idx` and return the first point
    that is at least `lookahead` m away from robot_pos. Also advance path_idx
    past waypoints we are already on top of."""
    # 1) Advance past waypoints we've already reached
    while path_idx < len(path) - 1:
        wp = np.asarray(path[path_idx], dtype=np.float64)
        if float(np.linalg.norm(wp[:2] - robot_pos[:2])) < WAYPOINT_REACH:
            path_idx += 1
        else:
            break
    # 2) Find a carrot point at least `lookahead` away (or path end)
    carrot_idx = path_idx
    while carrot_idx < len(path) - 1:
        wp = np.asarray(path[carrot_idx], dtype=np.float64)
        if float(np.linalg.norm(wp[:2] - robot_pos[:2])) >= lookahead:
            break
        carrot_idx += 1
    carrot = np.asarray(path[carrot_idx], dtype=np.float64)
    return carrot, path_idx

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
# Robot & camera spawning
# ═══════════════════════════════════════════════════════════════════════════
def add_robot_and_articulation(
    world: World,
    initial_xy: Optional[Tuple[float, float]] = None,
) -> ArticulationView:
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(ROBOT_PRIM_PATH).IsValid():
        if not os.path.exists(ARGS.robot_usd):
            raise FileNotFoundError(f"Robot USD not found: {ARGS.robot_usd}")
        print(f"[Robot] Adding reference: {ARGS.robot_usd}")
        add_reference_to_stage(ARGS.robot_usd, ROBOT_PRIM_PATH)

        # 先在 USD 层把根 prim translate 到目标点。
        # 此时 PhysX 还没接管这个 articulation，写 xformOp 是合法的，
        # 而且不会被物理覆盖。等 world.reset() 注册 articulation 时，
        # PhysX 会以这个 transform 作为初始位姿。
        if initial_xy is not None:
            robot_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
            xformable = UsdGeom.Xformable(robot_prim)
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(
                float(initial_xy[0]),
                float(initial_xy[1]),
                float(ARGS.robot_z_height),
            ))
            print(f"[Robot] USD-translated to "
                  f"({initial_xy[0]:.2f}, {initial_xy[1]:.2f}, "
                  f"{ARGS.robot_z_height:.2f}) before physics takes over.")

        update_sim(60)  # 让 reference 完全 compose

    robot = ArticulationView(prim_path=ROBOT_PRIM_PATH, name="robot_view")
    world.scene.add(robot)

    if robot.num_dof is None or robot.num_dof == 0:
        raise RuntimeError("Robot articulation has 0 dofs")
    print(f"[Robot] num_dof = {robot.num_dof}, dof_names = {list(robot.dof_names)}")
    return robot


def spawn_policy_camera() -> IsaacCamera:
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/World/Robot/base_link").IsValid():
        print("[WARN] /World/Robot/base_link not found; camera will be created "
              "but may not follow the robot.")

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


# ═══════════════════════════════════════════════════════════════════════════
# Chase camera (third-person)
# ═══════════════════════════════════════════════════════════════════════════
_CHASE_CAM_STATE = {
    "cam_prim": None,
    "xform_op": None,
    "coi_attr": None,
    "distance": 1.5,
    "height":   0.8,
    "behind_offset_along_yaw": True,
    "look_at_z_offset": 0.3,
}


def init_chase_camera(
    distance: float = 1.5,
    height: float = 0.8,
    behind_offset_along_yaw: bool = True,
    look_at_z_offset: float = 0.3,
):
    """
    一次性初始化第三人称相机：拿到 active viewport 的相机 prim，
    清掉它的 xformOps 并装上一个 transform op，方便每步直接写矩阵。
    """
    viewport = get_active_viewport()
    if viewport is None:
        print("[ChaseCam] No active viewport.")
        return False
    cam_path = viewport.get_active_camera()
    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath(cam_path)
    if not cam_prim or not cam_prim.IsValid():
        print(f"[ChaseCam] Camera prim not found: {cam_path}")
        return False

    xformable = UsdGeom.Xformable(cam_prim)
    xformable.ClearXformOpOrder()
    op = xformable.AddTransformOp()

    coi_attr_name = "omni:kit:centerOfInterest"
    coi_attr = cam_prim.GetAttribute(coi_attr_name)
    if not coi_attr:
        coi_attr = cam_prim.CreateAttribute(
            coi_attr_name, Sdf.ValueTypeNames.Vector3d, custom=True,
            variability=Sdf.VariabilityUniform,
        )

    _CHASE_CAM_STATE.update({
        "cam_prim": cam_prim,
        "xform_op": op,
        "coi_attr": coi_attr,
        "distance": float(distance),
        "height":   float(height),
        "behind_offset_along_yaw": behind_offset_along_yaw,
        "look_at_z_offset": float(look_at_z_offset),
    })
    print(f"[ChaseCam] Initialized on {cam_path}, "
          f"distance={distance}m, height={height}m")
    return True


def update_chase_camera(robot):
    """每个控制 tick 调一次：把相机贴到机器人后上方。"""
    state = _CHASE_CAM_STATE
    cam_prim = state["cam_prim"]
    op       = state["xform_op"]
    if cam_prim is None or op is None:
        return

    robot_pos, robot_yaw = get_robot_pose(robot)

    if state["behind_offset_along_yaw"]:
        back_x = -math.cos(robot_yaw)
        back_y = -math.sin(robot_yaw)
    else:
        back_x, back_y = -1.0, 0.0

    d = state["distance"]
    h = state["height"]
    cam_pos = Gf.Vec3d(
        float(robot_pos[0] + back_x * d),
        float(robot_pos[1] + back_y * d),
        float(robot_pos[2] + h),
    )
    target_pos = Gf.Vec3d(
        float(robot_pos[0]),
        float(robot_pos[1]),
        float(robot_pos[2]) + state["look_at_z_offset"],
    )

    up = Gf.Vec3d(0.0, 0.0, 1.0)
    view = Gf.Matrix4d().SetLookAt(cam_pos, target_pos, up)
    cam_world_xform = view.GetInverse()

    imageable = UsdGeom.Imageable(cam_prim)
    parent_xform = imageable.ComputeParentToWorldTransform(Usd.TimeCode.Default())
    local_xform = cam_world_xform * parent_xform.GetInverse()
    op.Set(local_xform)

    coi_len = (cam_pos - target_pos).GetLength()
    state["coi_attr"].Set(Gf.Vec3d(0.0, 0.0, -coi_len))


# ═══════════════════════════════════════════════════════════════════════════
# Episode → characters
# ═══════════════════════════════════════════════════════════════════════════
def spawn_characters_from_episode(
    episode: dict,
    char_pool: List[str],
) -> Dict[str, str]:
    """只 spawn 角色 + 写 scriptData，不挂行为脚本。
    
    行为脚本必须由调用方在"一切就绪后"挂载，否则脚本一挂上就开始消费
    scriptData，导致行人在 setup 阶段就跑出去了。
    """
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

    _write_episode_commands(commands_dict)
    update_sim(30)

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
def compute_control(
    robot_pos, robot_yaw, target_pos,
    gain_linear, gain_angular,
    max_linear, max_angular,
    desired_distance: float = 1.5,
    pos_deadzone: float = 0.2,
    angle_deadzone_deg: float = 5.0
) -> Tuple[float, float]:
    """
    跟随控制器：
    - 保持与目标点 (行人) 的平面距离在 desired_distance 附近
    - 允许前进和后退，最大速度 ±max_linear
    - 当距离和朝向误差都在死区内时，完全停止，避免原地转圈
    """
    delta = target_pos[:2] - robot_pos[:2]
    dist = float(np.linalg.norm(delta))

    if dist > 1e-6:
        desired_yaw = math.atan2(float(delta[1]), float(delta[0]))
    else:
        desired_yaw = robot_yaw
    angle_diff = math.atan2(math.sin(desired_yaw - robot_yaw),
                            math.cos(desired_yaw - robot_yaw))

    dist_error = dist - desired_distance
    linear = gain_linear * dist_error
    linear = np.clip(linear, -max_linear, max_linear)

    angular = gain_angular * angle_diff
    angular = np.clip(angular, -max_angular, max_angular)

    if (abs(dist_error) < pos_deadzone and
        abs(math.degrees(angle_diff)) < angle_deadzone_deg):
        linear = 0.0
        angular = 0.0

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

def resolve_scene_paths(usda_root: str,
                        usda_override: Optional[str],
                        episode_root: str,
                        scene_id: str) -> Tuple[str, str]:
    """
    Resolve (usda_path, episode_dir) from scene_id.

    - usda_path: usda_override if given, else f"{usda_root}/{scene_id}.usda"
    - episode_dir: look under episode_root for a subdir named exactly
      <scene_id>, or <NNNN>_<scene_id> (e.g. '0033_839873').
      If multiple matches exist, pick the lexicographically first one and warn.
    """
    # 1. USDA
    if usda_override:
        usda_path = usda_override
    else:
        usda_path = os.path.join(usda_root, f"{scene_id}.usda")
    if not os.path.exists(usda_path):
        raise FileNotFoundError(f"USDA not found: {usda_path}")

    # 2. Episode dir
    direct = os.path.join(episode_root, scene_id)
    if os.path.isdir(direct):
        return usda_path, direct

    # 匹配 <prefix>_<scene_id>，比如 0033_839873
    suffix = f"_{scene_id}"
    matches = []
    if os.path.isdir(episode_root):
        for name in os.listdir(episode_root):
            full = os.path.join(episode_root, name)
            if not os.path.isdir(full):
                continue
            if name == scene_id or name.endswith(suffix):
                matches.append(full)

    if not matches:
        raise FileNotFoundError(
            f"No episode subdir found under {episode_root} for scene_id "
            f"'{scene_id}' (looked for '{scene_id}' or '*_<{scene_id}>')."
        )
    matches.sort()
    if len(matches) > 1:
        print(f"[WARN] Multiple episode dirs match scene_id={scene_id}: "
              f"{matches}. Using first: {matches[0]}")
    return usda_path, matches[0]

def extract_ep_id(p: str) -> int:
    try:    return int(os.path.basename(p).split("_")[1].split(".")[0])
    except: return 999999


def setup_episode_full(
    episode: dict,
    char_pool: List[str],
    robot,
    world: World,
) -> Tuple[Dict[str, str], Optional[str]]:
    """
    完整的 episode setup (用于第 2 个及之后的 episode)：
    清掉旧角色 → spawn 新角色 → reset robot → 相机贴上 → 挂行为脚本 → 等脚本初始化。
    返回 (char_paths, target_prim_path)。
    """
    # 清掉旧角色
    clear_all_characters()
    update_sim(10)

    # spawn 新角色 (不挂行为脚本)
    char_paths = spawn_characters_from_episode(episode, char_pool)

    # resolve target
    target_prim_path = resolve_target_character(episode, char_paths)
    if target_prim_path is None:
        return char_paths, None

    # reset robot
    robot_block         = episode["robot"]
    robot_start_pos_xyz = robot_block["start_pos"]
    robot_start_yaw     = float(robot_block.get("start_orientation", 0.0))
    print(f"[Episode setup] start_pos={robot_start_pos_xyz}, "
          f"start_yaw={math.degrees(robot_start_yaw):.1f} deg, "
          f"target_prim={target_prim_path}")
    reset_robot(robot, robot_start_pos_xyz[:2], robot_start_yaw, world)

    # 相机贴上
    update_chase_camera(robot)
    for _ in range(3):
        simulation_app.update()
        update_chase_camera(robot)

    # 挂行为脚本 —— 行人开始执行 GoTo
    attach_behavior_scripts_to_characters(simulation_app)

    # 等脚本 on_play 真正触发，机器人保持静止
    for _ in range(60):
        world.step(render=False)
        simulation_app.update()
        update_chase_camera(robot)

    return char_paths, target_prim_path


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> int:
    # ── 0. Resolve paths from scene_id ────────────────────────────────────
    try:
        usda_path, ep_dir = resolve_scene_paths(
            usda_root=ARGS.usda_root,
            usda_override=ARGS.usda,
            episode_root=ARGS.episode_dir,
            scene_id=ARGS.scene_id,
        )
    except FileNotFoundError as e:
        print(f"[FATAL] {e}")
        return 1
    print(f"[Paths] scene_id   = {ARGS.scene_id}")
    print(f"[Paths] usda_path  = {usda_path}")
    print(f"[Paths] episode_dir = {ep_dir}")

    # ── 1. open_stage ─────────────────────────────────────────────────────
    if not open_stage(usda_path):
        return 1

    # ── 2. NavMesh bake ───────────────────────────────────────────────────
    navvols_usda = cache_paths(usda_path, ARGS.cache_dir)
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

    # ── 3. Locate episode JSONs ───────────────────────────────────────────
    candidates = sorted(glob.glob(os.path.join(ep_dir, "episode_*.json")),
                        key=extract_ep_id)
    if not candidates:
        print(f"[FATAL] No episode_*.json found in {ep_dir}")
        simulation_app.close()
        return 1

    end_idx  = len(candidates) if ARGS.end_idx == -1 else ARGS.end_idx
    selected = candidates[ARGS.start_idx:end_idx]
    print(f"[Episode] Selected {len(selected)}/{len(candidates)} episodes")

    # ── 3.5  预读第一个 episode —— 用它的 robot start_pos 决定 robot 初始位置 ──
    with open(selected[0], "r") as f:
        first_episode = json.load(f)["episode"]
    first_robot_block   = first_episode["robot"]
    first_start_pos_xyz = first_robot_block["start_pos"]
    first_start_yaw     = float(first_robot_block.get("start_orientation", 0.0))
    init_xy = (float(first_start_pos_xyz[0]), float(first_start_pos_xyz[1]))
    print(f"[Init] First episode robot start: xy={init_xy}, "
          f"yaw={math.degrees(first_start_yaw):.1f} deg")

    # ── 4. World + physics ────────────────────────────────────────────────
    print("[World] Creating World + physicsScene...")
    world = World(physics_dt=ARGS.physics_dt, rendering_dt=ARGS.physics_dt * 3)
    world.initialize_physics()
    update_sim(5)
    print("[World] Physics initialized.")

    # ── 5. Robot — 直接生成在 episode 0 的 start_pos ─────────────────────
    robot = add_robot_and_articulation(world, initial_xy=init_xy)
    left_idx, right_idx = resolve_wheel_dof_order(robot)

    world.reset()
    update_sim(10)

    # 用 set_world_pose 兜一次底，让 PhysX 也对齐到 episode 0 start_pos + yaw
    init_quat = np.array(
        [math.cos(first_start_yaw / 2.0), 0.0, 0.0,
         math.sin(first_start_yaw / 2.0)], dtype=np.float32)
    robot.set_world_pose(
        position=np.array([init_xy[0], init_xy[1], ARGS.robot_z_height],
                          dtype=np.float32),
        orientation=init_quat,
    )
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
    for _ in range(10):
        world.step(render=False)
    init_pos, init_yaw = get_robot_pose(robot)
    print(f"[Robot] settled at {init_pos.tolist()}, "
          f"yaw={math.degrees(init_yaw):.1f} deg")

    # ── 6. 第三人称相机 (机器人已经在 episode 0 起点了，相机直接贴上) ────
    init_chase_camera(distance=ARGS.chase_distance, height=ARGS.chase_height)
    update_chase_camera(robot)
    update_sim(3)

    # ── 7. Policy camera ──────────────────────────────────────────────────
    cam = spawn_policy_camera()

    # ── 8. Character asset pool ───────────────────────────────────────────
    char_pool = load_character_assets()
    print(f"[People] {len(char_pool)} character assets loaded.")

    # ── 9. Start timeline (一次性 play，整个 run 期间不停) ──────────────
    tl = omni.timeline.get_timeline_interface()
    tl.set_current_time(0.0)
    tl.play()
    update_sim(5)

    # ── 10. 预 spawn 第一个 episode 的角色 (不挂行为脚本) ────────────────
    #     这一步让程序刚启动时画面就是 episode 0 的样子：
    #     - 机器人在 episode 0 的 start_pos ✅
    #     - 行人在 episode 0 的 spawn_positions ✅
    #     - 相机贴在机器人后方 ✅
    #     - 行为脚本尚未挂载，行人在 spawn 点做 idle 动画 (不会乱跑)
    print("\n[Pre-spawn] Setting up first episode characters...")
    pre_char_paths = spawn_characters_from_episode(first_episode, char_pool)
    pre_target_prim_path = resolve_target_character(first_episode, pre_char_paths)

    # 给相机几帧时间渲染初始画面
    for _ in range(30):
        simulation_app.update()
        update_chase_camera(robot)

    print("[Pre-spawn] Initial scene ready. Entering episode loop...")

    all_metrics: List[dict] = []

    # ── 11. Per-episode loop ──────────────────────────────────────────────
    for ep_idx, ep_path in enumerate(selected):
        ep_id = extract_ep_id(ep_path)
        print(f"\n{'='*60}\nEpisode {ep_id}\n{'='*60}")

        is_first = (ep_idx == 0)

        if is_first:
            episode = first_episode
            char_paths = pre_char_paths
            target_prim_path = pre_target_prim_path
            if target_prim_path is None:
                print("[ERROR] No target character resolvable in first episode, skip")
                continue
            # ↓ 加这几行
            target_name = next((k for k, v in char_paths.items()
                                if v == target_prim_path), "?")
            target_init_pos, _ = get_character_pose(target_prim_path)
            robot_pos, _ = get_robot_pose(robot)
            init_dist = float(np.linalg.norm(target_init_pos[:2] - robot_pos[:2]))
            print(f"[Episode 0] target='{target_name}' prim={target_prim_path} "
                f"target_pos={target_init_pos[:2]} robot_pos={robot_pos[:2]} "
                f"init_dist={init_dist:.2f}m")
            print("[Episode 0] Attaching behavior scripts (first time)...")

            attach_behavior_scripts_to_characters(simulation_app)

            # 等脚本 on_play 真正触发
            for _ in range(60):
                world.step(render=False)
                # simulation_app.update()
                update_chase_camera(robot)
        else:
            # 后续 episode：完整跑一遍 setup
            with open(ep_path, "r") as f:
                episode = json.load(f)["episode"]
            char_paths, target_prim_path = setup_episode_full(
                episode, char_pool, robot, world
            )
            if target_prim_path is None:
                print("[ERROR] No target character resolvable, skip episode")
                continue

        # ── Control loop ──────────────────────────────────────────────────
        step           = 0
        tracking_steps = 0
        distances:     List[float] = []
        heading_errors: List[float] = []
        done_reason = "max_steps"

        # ── NavMesh pursuit state (per episode) ──────────────────────────
        pursuit_state = {
            "path": None,
            "path_idx": 0,
            "last_plan_step": -10**9,
            "last_goal": None,
            "plan_fail_count": 0,
            # ── stuck recovery ──
            "pose_history": [],            # list of (pos_xy, yaw)
            "recovery_steps_left": 0,
            "recovery_linear": 0.0,
            "recovery_angular": 0.0,
            "consecutive_recoveries": 0,
        }

        while step < ARGS.max_steps:
            robot_pos, robot_yaw   = get_robot_pose(robot)
            target_pos, target_yaw = get_character_pose(target_prim_path)

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
            # ── 0. Stuck detection & recovery ─────────────────────────────
            pursuit_state["pose_history"].append(
                (robot_pos[:2].copy(), float(robot_yaw)))
            if len(pursuit_state["pose_history"]) > STUCK_WINDOW:
                pursuit_state["pose_history"].pop(0)

            # If currently in recovery, just keep executing backup motion.
            if pursuit_state["recovery_steps_left"] > 0:
                lin = pursuit_state["recovery_linear"]
                ang = pursuit_state["recovery_angular"]
                left_v, right_v = diff_drive_joint_vels(
                    lin, ang, ARGS.wheel_radius, ARGS.wheel_base)
                vel_cmd = np.zeros(int(robot.num_dof), dtype=np.float32)
                vel_cmd[left_idx]  = left_v
                vel_cmd[right_idx] = right_v
                robot.apply_action(ArticulationAction(joint_velocities=vel_cmd))

                if step % 5 == 0:
                    print(f"[EP{ep_id}] step={step} RECOVERY "
                          f"lin={lin:.2f} ang={ang:.2f} "
                          f"left={pursuit_state['recovery_steps_left']}")

                for _ in range(ARGS.decimation):
                    world.step(render=False)
                    simulation_app.update()
                simulation_app.update()
                update_chase_camera(robot)
                if ARGS.save_images and step % ARGS.save_image_every == 0:
                    save_rgb_depth(cam, step, ARGS.image_save_dir, ep_id)

                pursuit_state["recovery_steps_left"] -= 1
                # On the last recovery tick, force a replan next iteration
                # by clearing the current path.
                if pursuit_state["recovery_steps_left"] == 0:
                    pursuit_state["path"] = None
                    pursuit_state["pose_history"].clear()
                    print(f"[EP{ep_id}] Recovery finished, will replan.")
                step += 1
                continue

            # Otherwise, check if we look stuck (and have been commanding motion).
            if len(pursuit_state["pose_history"]) >= STUCK_WINDOW:
                first_pos, first_yaw = pursuit_state["pose_history"][0]
                last_pos,  last_yaw  = pursuit_state["pose_history"][-1]
                pos_delta = float(np.linalg.norm(last_pos - first_pos))
                yaw_delta = abs(math.degrees(math.atan2(
                    math.sin(last_yaw - first_yaw),
                    math.cos(last_yaw - first_yaw))))

                # Were we actually *trying* to move? Look at the wheels we just
                # commanded. `vel_cmd` from the previous iteration isn't in
                # scope here, so use the joint state.
                try:
                    cur_vels = robot.get_joint_velocities()
                    cmd_magnitude = float(np.max(np.abs(cur_vels)))
                except Exception:
                    cmd_magnitude = 0.0

                commanded_motion = cmd_magnitude > STUCK_CMD_THRESH
                truly_stuck = (pos_delta < STUCK_POS_THRESH and
                               yaw_delta < STUCK_YAW_THRESH_DEG and
                               commanded_motion)

                if truly_stuck:
                    pursuit_state["consecutive_recoveries"] += 1
                    if pursuit_state["consecutive_recoveries"] > MAX_RECOVERIES:
                        print(f"[EP{ep_id}] Stuck {MAX_RECOVERIES}+ times in a "
                              f"row, ending episode.")
                        done_reason = "stuck"
                        robot.apply_action(ArticulationAction(
                            joint_velocities=np.zeros(int(robot.num_dof),
                                                      dtype=np.float32)))
                        for _ in range(3):
                            world.step(render=False)
                            simulation_app.update()
                            update_chase_camera(robot)
                        break

                    # Choose recovery direction: turn *away* from where we
                    # were trying to go (so backing up swings the nose off
                    # the obstacle). Use the current carrot if we have one,
                    # else the target.
                    if pursuit_state["path"] is not None:
                        try:
                            ref_pt, _ = pick_carrot(
                                pursuit_state["path"],
                                pursuit_state["path_idx"],
                                robot_pos, lookahead=LOOKAHEAD_DIST)
                            ref_xy = ref_pt[:2]
                        except Exception:
                            ref_xy = target_pos[:2]
                    else:
                        ref_xy = target_pos[:2]
                    dx = float(ref_xy[0] - robot_pos[0])
                    dy = float(ref_xy[1] - robot_pos[1])
                    desired_yaw = math.atan2(dy, dx)
                    yaw_err = math.atan2(math.sin(desired_yaw - robot_yaw),
                                         math.cos(desired_yaw - robot_yaw))
                    # If goal is on the left (yaw_err > 0), swing right while
                    # reversing, and vice versa.
                    ang_sign = -1.0 if yaw_err > 0 else 1.0

                    pursuit_state["recovery_steps_left"] = RECOVERY_DURATION
                    pursuit_state["recovery_linear"]    = RECOVERY_LINEAR
                    pursuit_state["recovery_angular"]   = ang_sign * RECOVERY_ANGULAR_MAG
                    pursuit_state["pose_history"].clear()
                    print(f"[EP{ep_id}] step={step} STUCK detected "
                          f"(pos_delta={pos_delta:.3f}m, "
                          f"yaw_delta={yaw_delta:.1f}deg, "
                          f"cmd_mag={cmd_magnitude:.2f}). "
                          f"Starting recovery: lin={RECOVERY_LINEAR} "
                          f"ang={ang_sign * RECOVERY_ANGULAR_MAG:.2f} "
                          f"(consec={pursuit_state['consecutive_recoveries']})")
                    # Fall through to next iteration's recovery branch
                    step += 1
                    continue
                else:
                    # If we *are* moving, decay the consecutive counter.
                    if pos_delta > STUCK_POS_THRESH * 3:
                        pursuit_state["consecutive_recoveries"] = 0
            # ── Short-range direct-follow mode ────────────────────────────
            # When the target is close, navmesh pursuit becomes degenerate
            # (follow_point lands behind the robot and we spin trying to
            # chase it). Switch to direct face-and-hold behaviour.
            if dist_to_target <= DIRECT_FOLLOW_RANGE:
                # Vector to the target (not the follow point).
                dx = float(target_pos[0] - robot_pos[0])
                dy = float(target_pos[1] - robot_pos[1])
                heading_to_target = math.atan2(dy, dx)
                yaw_err = math.atan2(math.sin(heading_to_target - robot_yaw),
                                     math.cos(heading_to_target - robot_yaw))

                dist_err = dist_to_target - ARGS.follow_distance

                # Full deadzone: in the sweet spot AND facing the target → stop.
                if (abs(dist_err) < DIRECT_DEADZONE_POS and
                    abs(math.degrees(yaw_err)) < DIRECT_DEADZONE_ANG):
                    linear, angular = 0.0, 0.0
                else:
                    # If target is mostly behind us (|yaw_err| > 90°), just
                    # rotate in place — don't try to translate, otherwise
                    # we'd drive *away* from them.
                    if abs(yaw_err) > math.radians(90.0):
                        linear  = 0.0
                        angular = float(np.clip(
                            ARGS.control_gain_angular * yaw_err,
                            -ARGS.max_angular_vel, ARGS.max_angular_vel))
                    else:
                        # Standard follow:
                        # - move forward/backward to close distance error
                        # - rotate to face target
                        # - scale linear by cos(yaw_err) so we don't drive
                        #   sideways while still turning
                        linear = ARGS.control_gain_linear * dist_err * math.cos(yaw_err)
                        linear = float(np.clip(linear,
                                               -ARGS.max_linear_vel,
                                               ARGS.max_linear_vel))
                        angular = ARGS.control_gain_angular * yaw_err
                        angular = float(np.clip(angular,
                                                -ARGS.max_angular_vel,
                                                ARGS.max_angular_vel))

                left_v, right_v = diff_drive_joint_vels(
                    linear, angular, ARGS.wheel_radius, ARGS.wheel_base)
                vel_cmd = np.zeros(int(robot.num_dof), dtype=np.float32)
                vel_cmd[left_idx]  = left_v
                vel_cmd[right_idx] = right_v
                robot.apply_action(ArticulationAction(joint_velocities=vel_cmd))

                # Invalidate any current navmesh path — we'll replan when
                # the target gets far again.
                pursuit_state["path"] = None

                if step % 20 == 0:
                    print(f"[EP{ep_id}] step={step} DIRECT-FOLLOW "
                          f"dist={dist_to_target:.2f} "
                          f"dist_err={dist_err:+.2f} "
                          f"yaw_err={math.degrees(yaw_err):+.1f}deg "
                          f"cmd=(L={left_v:.2f},R={right_v:.2f})")

                for _ in range(ARGS.decimation):
                    world.step(render=False)
                    simulation_app.update()
                simulation_app.update()
                update_chase_camera(robot)

                if ARGS.save_images and step % ARGS.save_image_every == 0:
                    save_rgb_depth(cam, step, ARGS.image_save_dir, ep_id)

                step += 1
                continue
            # ── 1. Compute follow point (behind target along target yaw) ──
            follow_point_raw = compute_follow_point(
                target_pos, target_yaw, ARGS.follow_distance)
            follow_point = snap_to_navmesh(nm, follow_point_raw, search_radius=1.5)
            if follow_point is None:
                # Fallback: snap target itself
                follow_point = snap_to_navmesh(nm, target_pos, search_radius=1.5)
            if follow_point is None:
                # Last-ditch fallback: target position as-is
                follow_point = np.asarray(target_pos, dtype=np.float64)

            # ── 2. Decide whether to replan ──────────────────────────────
            need_replan = False
            if pursuit_state["path"] is None:
                need_replan = True
            elif (step - pursuit_state["last_plan_step"]) >= REPLAN_PERIOD:
                need_replan = True
            elif pursuit_state["last_goal"] is not None:
                goal_shift = float(np.linalg.norm(
                    follow_point[:2] - pursuit_state["last_goal"][:2]))
                if goal_shift > REPLAN_TARGET_THR:
                    need_replan = True
            elif pursuit_state["path_idx"] >= len(pursuit_state["path"]) - 1:
                # Walked the whole path → force a replan next tick
                need_replan = True

            if need_replan:
                # Snap robot pos onto navmesh too — query_shortest_path is
                # picky about off-mesh start points.
                robot_on_nm = snap_to_navmesh(nm, robot_pos, search_radius=1.0)
                start_xyz = robot_on_nm if robot_on_nm is not None else robot_pos
                new_path = plan_navmesh_path(nm, start_xyz, follow_point)

                if new_path is not None and len(new_path) >= 1:
                    pursuit_state["path"]            = new_path
                    pursuit_state["path_idx"]        = 0
                    pursuit_state["last_plan_step"]  = step
                    pursuit_state["last_goal"]       = follow_point.copy()
                    pursuit_state["plan_fail_count"] = 0
                else:
                    pursuit_state["plan_fail_count"] += 1
                    if pursuit_state["plan_fail_count"] >= MAX_PLAN_FAILS:
                        print(f"[EP{ep_id}] Planning failed "
                              f"{pursuit_state['plan_fail_count']} times in a "
                              f"row, ending episode.")
                        done_reason = "planning_failed"
                        # Stop the wheels before breaking out
                        robot.apply_action(ArticulationAction(
                            joint_velocities=np.zeros(int(robot.num_dof),
                                                      dtype=np.float32)))
                        for _ in range(3):
                            world.step(render=False)
                            simulation_app.update()
                            update_chase_camera(robot)
                        break
                    # Hold position for this tick and retry next time.
                    robot.apply_action(ArticulationAction(
                        joint_velocities=np.zeros(int(robot.num_dof),
                                                  dtype=np.float32)))
                    for _ in range(ARGS.decimation):
                        world.step(render=False)
                        simulation_app.update()
                    simulation_app.update()
                    update_chase_camera(robot)
                    if ARGS.save_images and step % ARGS.save_image_every == 0:
                        save_rgb_depth(cam, step, ARGS.image_save_dir, ep_id)
                    step += 1
                    continue

            # ── 3. Pick carrot on current path & drive towards it ────────
            carrot, new_idx = pick_carrot(
                pursuit_state["path"], pursuit_state["path_idx"], robot_pos,
                lookahead=LOOKAHEAD_DIST)
            pursuit_state["path_idx"] = new_idx

            # End-of-path: if the last waypoint is also within follow_distance
            # of the target, fall back to direct pursuit so we don't oscillate.
            at_path_end = (new_idx >= len(pursuit_state["path"]) - 1)
            if at_path_end:
                # Use direct pursuit toward the actual follow point so the
                # closing-distance term in compute_control engages.
                carrot = follow_point

            # Use existing controller, but drive toward carrot rather than
            # the raw target pos. Important: when chasing a carrot on a path,
            # we want zero standoff (drive to the point), not follow_distance.
            # The follow-distance behavior is already baked into where the
            # follow_point sits behind the target.
            if at_path_end:
                # Standoff is already encoded in follow_point's location, so
                # ask the controller to drive *onto* it.
                desired_standoff = 0.0
            else:
                desired_standoff = 0.0

            linear, angular = compute_control(
                robot_pos, robot_yaw, carrot,
                ARGS.control_gain_linear, ARGS.control_gain_angular,
                ARGS.max_linear_vel, ARGS.max_angular_vel,
                desired_distance=desired_standoff,
                pos_deadzone=0.15,
                angle_deadzone_deg=4.0,
            )
            left_v, right_v = diff_drive_joint_vels(
                linear, angular, ARGS.wheel_radius, ARGS.wheel_base)

            vel_cmd = np.zeros(int(robot.num_dof), dtype=np.float32)
            vel_cmd[left_idx]  = left_v
            vel_cmd[right_idx] = right_v
            robot.apply_action(ArticulationAction(joint_velocities=vel_cmd))

            if step % 20 == 0:
                actual_vels = robot.get_joint_velocities()
                path_len = len(pursuit_state["path"]) if pursuit_state["path"] else 0
                print(f"[EP{ep_id}] step={step} "
                      f"pos={robot_pos[:2]} yaw={math.degrees(robot_yaw):.1f} "
                      f"target={target_pos[:2]} dist={dist_to_target:.2f} "
                      f"carrot={carrot[:2]} path_idx={pursuit_state['path_idx']}/{path_len} "
                      f"cmd=(L={left_v:.2f},R={right_v:.2f}) "
                      f"plan_fails={pursuit_state['plan_fail_count']}")

            # 1 control tick = decimation × physics step + 1 omni update
            for _ in range(ARGS.decimation):
                world.step(render=False)
                simulation_app.update()
            simulation_app.update()
            update_chase_camera(robot)

            if ARGS.save_images and step % ARGS.save_image_every == 0:
                save_rgb_depth(cam, step, ARGS.image_save_dir, ep_id)

            step += 1

        # ── Episode summary ───────────────────────────────────────────────
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
        robot.apply_action(ArticulationAction(
            joint_velocities=np.zeros(int(robot.num_dof), dtype=np.float32)))
        for _ in range(5):
            world.step(render=False)
            simulation_app.update()

    # ── 12. Save metrics ──────────────────────────────────────────────────
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