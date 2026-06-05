#!/usr/bin/env python3
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

PROJECT_ROOT = "/workspace/FLUX"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_CLOTHING_PROFILES = None

def parse_args():
    p = argparse.ArgumentParser(description="Tracking episode data collection")
    p.add_argument("--usda", default=None)
    p.add_argument("--usda_root",
                default="/workspace/SAGE-3D_Official/SAGE-3D_data/usda")
    p.add_argument("--episode_dir", required=True)
    p.add_argument("--scene_id", required=True)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx",   type=int, default=-1)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--collision_root",  default="/World/scene_collision")
    p.add_argument("--volume_padding",  type=float, default=1.2)
    p.add_argument("--fallback_size",   type=float, default=100.0)
    p.add_argument("--warmup_frames",   type=int,   default=60)
    p.add_argument("--cache_dir",       default=None)
    p.add_argument("--force_rebake",    action="store_true")
    p.add_argument("--semantic_map_json", default=None)
    p.add_argument("--robot_usd",
                   default="/workspace/FLUX/assets/robots/dingo_fixed.usd")
    p.add_argument("--robot_z_height", type=float, default=0.1)
    p.add_argument("--wheel_radius", type=float, default=0.0762)
    p.add_argument("--wheel_base",   type=float, default=0.32)
    p.add_argument("--control_gain_linear",  type=float, default=1.0)
    p.add_argument("--control_gain_angular", type=float, default=2.0)
    p.add_argument("--max_linear_vel",       type=float, default=0.8)
    p.add_argument("--max_angular_vel",      type=float, default=1.5)
    p.add_argument("--physics_dt",  type=float, default=1.0 / 60.0)
    p.add_argument("--decimation",  type=int,   default=9)
    p.add_argument("--tracking_dist_min",     type=float, default=1.0)
    p.add_argument("--tracking_dist_max",     type=float, default=3.0)
    p.add_argument("--tracking_angle_thresh", type=float, default=60.0)
    p.add_argument("--loss_dist_thresh",      type=float, default=5.0)
    p.add_argument("--save_images", action="store_true")
    p.add_argument("--save_image_every", type=int, default=5)
    p.add_argument("--image_save_dir",   type=str, default="./tracking_images")
    p.add_argument("--output_metrics",   type=str, default="./tracking_metrics.csv")
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--follow_distance", type=float, default=2.0)
    p.add_argument("--chase_distance", type=float, default=1.5)
    p.add_argument("--chase_height", type=float, default=0.8)
    p.add_argument("--profiles_json",
                   default="/workspace/FLUX/sage_utils/character_clothing_profiles.json",
                   help="Profiles JSON for asset-name -> usd_path lookup.")
    p.add_argument("--robot_type", choices=["dingo", "go2", "g1"], default="dingo",
                   help="Robot type: dingo (differential drive), "
                        "go2 (Unitree Go2 quadruped, kinematic), "
                        "g1 (Unitree G1 humanoid, kinematic)")
    return p.parse_args()


ARGS = parse_args()

# ── Robot-type presets ──────────────────────────────────────────────────────
_ROBOT_PRESETS = {
    "dingo": {
        "usd":         "/workspace/FLUX/assets/robots/dingo_fixed.usd",
        "z_height":    0.1,
        "camera_link": "base_link",
        "drive_mode":  "diff_drive",
        "cam_trans":   [0.0, 0.0, 0.3],
    },
    "go2": {
        "usd":         "/workspace/FLUX/assets/isaacsim_assets/Assets/Isaac/4.5/"
                       "Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd",
        "z_height":    0.40,
        "camera_link": "base",
        "drive_mode":  "kinematic",
        "cam_trans":   [0.0, 0.0, 0.3],
    },
    "g1": {
        "usd":         "/workspace/FLUX/assets/isaacsim_assets/Assets/Isaac/4.5/"
                       "Isaac/IsaacLab/Robots/Unitree/G1/g1.usd",
        "z_height":    0.74,
        "camera_link": "pelvis",
        "drive_mode":  "kinematic",
        # pelvis is the ArticulationRoot (waist level). [0,0,0.3] would land inside
        # the torso mesh → black images. Push forward (X) and above torso top (Z).
        "cam_trans":   [0.2, 0.0, 0.45],
    },
}
_preset = _ROBOT_PRESETS[ARGS.robot_type]
# Only apply preset defaults when user hasn't overridden them
if ARGS.robot_usd == "/workspace/FLUX/assets/robots/dingo_fixed.usd":
    ARGS.robot_usd = _preset["usd"]
if ARGS.robot_z_height == 0.1:
    ARGS.robot_z_height = _preset["z_height"]
ROBOT_DRIVE_MODE   = _preset["drive_mode"]   # "diff_drive" or "kinematic"
_ROBOT_CAMERA_LINK = _preset["camera_link"]  # link name under /World/Robot/

# ── 跟踪距离统一配置 ──────────────────────────────────────────
FOLLOW_DIST        = ARGS.follow_distance          # 机器人目标跟随距离
DIRECT_FOLLOW_RANGE = max(FOLLOW_DIST + 1.0, 3.0)  # 切换直接跟随的阈值，始终比follow_dist大1m
# tracking window 自动跟随 follow_distance 浮动
TRACKING_DIST_MIN  = max(FOLLOW_DIST - 1.0, 1.0)
TRACKING_DIST_MAX  = FOLLOW_DIST + 1.5

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


import carb
import omni.usd
import omni.timeline
from pxr import Usd, Sdf, UsdGeom, UsdPhysics, Gf
try:
    from pxr import PhysxSchema as _PhysxSchema
except ImportError:
    _PhysxSchema = None

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
from clothing_recolor import recolor_character
from clothing_appearance import ClothingProfiles
from isaacsim.replicator.agent.core.settings import PrimPaths
from omni.kit.viewport.utility import get_active_viewport
from sage_utils.people_utils import frame_viewport_on


ROBOT_PRIM_PATH    = "/World/Robot"
CAMERA_PRIM_PATH   = f"/World/Robot/{_ROBOT_CAMERA_LINK}/front_cam"
CHARACTERS_ROOT    = "/World/Characters"
ROBOT_JOINT_NAMES  = (["left_wheel_joint", "right_wheel_joint"]
                      if ROBOT_DRIVE_MODE == "diff_drive" else [])
CAM_W, CAM_H       = 640, 360

REPLAN_PERIOD     = 10
REPLAN_TARGET_THR = 0.5
LOOKAHEAD_DIST    = 1.0
WAYPOINT_REACH    = 0.3
MAX_PLAN_FAILS    = 5
NM_AGENT_RADIUS   = 0.25

STUCK_WINDOW          = 8
STUCK_POS_THRESH      = 0.05
STUCK_YAW_THRESH_DEG  = 5.0
STUCK_CMD_THRESH      = 0.5
RECOVERY_DURATION     = 8
RECOVERY_LINEAR       = -0.4
RECOVERY_ANGULAR_MAG  = 1.2
MAX_RECOVERIES        = 3

# DIRECT_FOLLOW_RANGE = 2.5
DIRECT_DEADZONE_POS = 0.25
DIRECT_DEADZONE_ANG = 10.0

LOS_RAY_Z_OFFSET    = 0.9
LOS_CAM_Z_OFFSET    = 0.3
LOS_HIT_TOLERANCE   = 0.3

VANTAGE_RADII          = [1.5, 2.0, 2.5]
VANTAGE_NUM_SAMPLES    = 12
VANTAGE_MIN_ROBOT_DIST = 0.5

SMOOTH_ALPHA_NORMAL = 0.4


_PHYSX_SQ = None

def _get_physx_sq():
    global _PHYSX_SQ
    if _PHYSX_SQ is None:
        from omni.physx import get_physx_scene_query_interface
        _PHYSX_SQ = get_physx_scene_query_interface()
    return _PHYSX_SQ


def check_visibility(robot_pos: np.ndarray,
                     target_pos: np.ndarray,
                     target_prim_path: str) -> Tuple[bool, str, float]:
    origin_np = np.array([
        float(robot_pos[0]),
        float(robot_pos[1]),
        float(robot_pos[2]) + LOS_CAM_Z_OFFSET,
    ], dtype=np.float64)
    target_np = np.array([
        float(target_pos[0]),
        float(target_pos[1]),
        float(target_pos[2]) + LOS_RAY_Z_OFFSET,
    ], dtype=np.float64)

    direction_np = target_np - origin_np
    full_dist = float(np.linalg.norm(direction_np))
    if full_dist < 1e-3:
        return True, "", 0.0
    direction_np /= full_dist

    origin    = carb.Float3(float(origin_np[0]),    float(origin_np[1]),    float(origin_np[2]))
    direction = carb.Float3(float(direction_np[0]), float(direction_np[1]), float(direction_np[2]))

    sq = _get_physx_sq()
    try:
        hit = sq.raycast_closest(origin, direction, full_dist + 0.5)
    except Exception as e:
        print(f"[LoS] raycast_closest exception: {e}")
        return True, "", full_dist

    if not hit or not hit.get("hit", False):
        return True, "", full_dist

    hit_prim = hit.get("rigidBody", "") or ""
    hit_dist = float(hit.get("distance", full_dist))

    if hit_prim.startswith(target_prim_path):
        return True, hit_prim, hit_dist

    if hit_dist >= (full_dist - LOS_HIT_TOLERANCE):
        return True, hit_prim, hit_dist

    return False, hit_prim, hit_dist


def compute_follow_point(target_pos: np.ndarray,
                         target_yaw: float,
                         follow_distance: float) -> np.ndarray:
    bx = -math.cos(target_yaw)
    by = -math.sin(target_yaw)
    return np.array([
        float(target_pos[0]) + bx * follow_distance,
        float(target_pos[1]) + by * follow_distance,
        float(target_pos[2]),
    ], dtype=np.float64)


def snap_to_navmesh(nm, xyz: np.ndarray, search_radius: float = 1.0):
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
    while path_idx < len(path) - 1:
        wp = np.asarray(path[path_idx], dtype=np.float64)
        if float(np.linalg.norm(wp[:2] - robot_pos[:2])) < WAYPOINT_REACH:
            path_idx += 1
        else:
            break
    carrot_idx = path_idx
    while carrot_idx < len(path) - 1:
        wp = np.asarray(path[carrot_idx], dtype=np.float64)
        if float(np.linalg.norm(wp[:2] - robot_pos[:2])) >= lookahead:
            break
        carrot_idx += 1
    carrot = np.asarray(path[carrot_idx], dtype=np.float64)
    return carrot, path_idx


def find_visible_vantage_point(
    nm,
    target_pos: np.ndarray,
    target_prim_path: str,
    robot_pos: np.ndarray,
) -> Optional[np.ndarray]:
    best_pt = None
    best_score = float("inf")

    for radius in VANTAGE_RADII:
        for i in range(VANTAGE_NUM_SAMPLES):
            angle = 2.0 * math.pi * i / VANTAGE_NUM_SAMPLES
            cand = np.array([
                float(target_pos[0]) + radius * math.cos(angle),
                float(target_pos[1]) + radius * math.sin(angle),
                float(target_pos[2]),
            ], dtype=np.float64)

            cand_nm = snap_to_navmesh(nm, cand, search_radius=0.8)
            if cand_nm is None:
                continue
            if float(np.linalg.norm(cand_nm[:2] - robot_pos[:2])) \
                    < VANTAGE_MIN_ROBOT_DIST:
                continue

            vis, _, _ = check_visibility(cand_nm, target_pos, target_prim_path)
            if not vis:
                continue

            path = plan_navmesh_path(nm, robot_pos, cand_nm)
            if path is None or len(path) < 1:
                continue

            plen = 0.0
            for j in range(len(path) - 1):
                p0 = np.asarray(path[j],     dtype=np.float64)
                p1 = np.asarray(path[j + 1], dtype=np.float64)
                plen += float(np.linalg.norm(p1[:2] - p0[:2]))

            if plen < best_score:
                best_score = plen
                best_pt    = cand_nm

        if best_pt is not None:
            break

    return best_pt


def smooth_cmd(state, lin, ang, alpha):
    lin = alpha * lin + (1.0 - alpha) * state["prev_linear"]
    ang = alpha * ang + (1.0 - alpha) * state["prev_angular"]
    state["prev_linear"]  = lin
    state["prev_angular"] = ang
    return lin, ang


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


def add_robot_and_articulation(
    world: World,
    initial_xy: Optional[Tuple[float, float]] = None,
) -> ArticulationView:
    global _ROBOT_ARTICULATION_PATH

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(ROBOT_PRIM_PATH).IsValid():
        if not os.path.exists(ARGS.robot_usd):
            raise FileNotFoundError(f"Robot USD not found: {ARGS.robot_usd}")
        print(f"[Robot] Adding reference: {ARGS.robot_usd}")
        add_reference_to_stage(ARGS.robot_usd, ROBOT_PRIM_PATH)

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

        # Disable gravity on all robot rigid bodies before first physics update.
        _setup_kinematic_root(stage)

        update_sim(60)

    # Discover the ArticulationRoot prim — may be nested inside a sub-prim
    # depending on the USD file's default prim structure.
    art_path = _find_articulation_root_path(stage, ROBOT_PRIM_PATH)
    if art_path is None:
        print(f"[Robot] No ArticulationRootAPI found under {ROBOT_PRIM_PATH}. "
              f"Printing prim tree:")
        _print_prim_tree(stage, ROBOT_PRIM_PATH)
        raise RuntimeError(f"Could not find ArticulationRoot under {ROBOT_PRIM_PATH}")
    if art_path != ROBOT_PRIM_PATH:
        print(f"[Robot] ArticulationRoot discovered at {art_path} "
              f"(USD nesting: not directly at {ROBOT_PRIM_PATH})")
    _ROBOT_ARTICULATION_PATH = art_path

    robot = ArticulationView(prim_path=art_path, name="robot_view")
    world.scene.add(robot)

    if robot.num_dof is None or robot.num_dof == 0:
        raise RuntimeError("Robot articulation has 0 dofs")
    print(f"[Robot] num_dof = {robot.num_dof}, dof_names = {list(robot.dof_names)}")
    return robot

def make_policy_cam_orientation(pitch_deg: float = 0.0) -> np.ndarray:
    """
    返回 policy camera 的局部朝向四元数（xyzw，camera_axes='ros'）。
    pitch_deg > 0 = 上仰（看向更高处）
    pitch_deg < 0 = 下俯
    """
    # 基础旋转：水平朝前（ROS convention，0°俯仰）
    # w, x, y, z = 0.5, -0.5, 0.5, -0.5  →  xyzw = [-0.5, 0.5, -0.5, 0.5]
    q_base = np.array([0.5, -0.5, 0.5, -0.5])   # wxyz

    # 上仰旋转：绕局部 Y 轴（在Isaac base_link坐标里）
    half = math.radians(pitch_deg) / 2.0
    q_tilt = np.array([math.cos(half), 0.0, math.sin(half), 0.0])  # wxyz

    # 复合：q_final = q_base * q_tilt
    def qmul(a, b):  # wxyz × wxyz
        w = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
        x = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]
        y = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1]
        z = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
        return np.array([w, x, y, z])

    q = qmul(q_base, q_tilt)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)  # → xyzw
    
def spawn_policy_camera() -> IsaacCamera:
    stage = omni.usd.get_context().get_stage()

    # For legged robots the ArticulationRoot may be nested (e.g. /World/Robot/go2_description).
    # Attach the camera to the articulation root prim so it follows the robot body.
    if _ROBOT_ARTICULATION_PATH and _ROBOT_ARTICULATION_PATH != ROBOT_PRIM_PATH:
        cam_prim_path = f"{_ROBOT_ARTICULATION_PATH}/front_cam"
    else:
        cam_prim_path = CAMERA_PRIM_PATH

    cam_parent = "/".join(cam_prim_path.split("/")[:-1])
    if not stage.GetPrimAtPath(cam_parent).IsValid():
        print(f"[WARN] Camera parent {cam_parent} not found; "
              "camera will be created but may not follow the robot.")

    cam_trans_ros = np.array(_preset.get("cam_trans", [0.0, 0.0, 0.3]), dtype=np.float64)
    # cam_rot_ros   = np.array([-0.5, 0.5, -0.5, 0.5], dtype=np.float64)
    CAMERA_PITCH_DEG = 0.0
    #     pitch_deg > 0 = 上仰（看向更高处）
    #     pitch_deg < 0 = 下俯
    cam_rot_ros   = make_policy_cam_orientation(CAMERA_PITCH_DEG)

    cam = IsaacCamera(
        prim_path=cam_prim_path,
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
    print(f"[Camera] Policy camera initialized at {cam_prim_path}")
    return cam


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


def spawn_characters_from_episode(
    episode: dict,
    char_pool: List[str],
    profiles: Optional[ClothingProfiles] = None,
) -> Dict[str, str]:
    spawn_positions = episode["characters"]["spawn_positions"]
    commands_dict   = episode["characters"]["commands"]
    appearance      = episode.get("appearance") or {}
    by_char         = appearance.get("by_character", {}) if appearance else {}

    stage = omni.usd.get_context().get_stage()

    char_paths: Dict[str, str] = {}
    char_appearance_used: Dict[str, dict] = {}  # 记录每个角色用的 appearance

    for idx, (char_name, spawn_data) in enumerate(spawn_positions.items()):
        pos = spawn_data["pos"]

        # 决定资产 USD:优先用 appearance 指定的 asset → profiles 查 usd_path
        usd = None
        app_entry = by_char.get(char_name)
        if app_entry and profiles is not None:
            asset_name = app_entry.get("asset")
            asset_idx = profiles.index_of(asset_name) if asset_name else None
            if asset_idx is not None and 0 <= asset_idx < len(char_pool):
                # 用 char_pool(5.1, animation-graph 兼容)里对应 index 的 USD,
                # 而不是 profiles 的本地 4.5 路径 —— 后者与当前 animation graph
                # 不匹配,会导致 skel:animationGraph 绑定失败 → T-pose。
                usd = char_pool[asset_idx]
            else:
                print(f"[Episode] WARN asset '{asset_name}' index lookup failed; "
                      f"falling back to char_pool[idx].")
        if usd is None:
            usd = char_pool[idx % len(char_pool)]

        prim_path = spawn_character(simulation_app, idx, usd, pos)
        char_paths[char_name] = prim_path
        if app_entry:
            char_appearance_used[char_name] = app_entry
        print(f"[Episode] Spawned {char_name} at {pos}  ->  {prim_path}  "
              f"(asset={app_entry.get('asset') if app_entry else 'pool'})")

    load_default_skeleton_and_animations(simulation_app)
    update_sim(30)
    bind_animation_graph_to_characters(simulation_app)
    update_sim(30)

    _write_episode_commands(commands_dict)
    update_sim(30)

    return char_paths

def recolor_episode_characters(episode: dict,
                               char_paths: Dict[str, str]) -> None:
    """在角色动画完全就绪后,按 appearance 给所有角色染色。"""
    appearance = episode.get("appearance") or {}
    by_char = appearance.get("by_character", {}) if appearance else {}
    if not by_char:
        return
    stage = omni.usd.get_context().get_stage()
    for char_name, prim_path in char_paths.items():
        entry = by_char.get(char_name)
        if not entry:
            continue
        parts = entry.get("parts", {})
        if not parts:
            continue
        status = recolor_character(stage, prim_path, parts)
        n_ok = sum(1 for s in status.values() if s.startswith("recolored"))
        n_def = sum(1 for s in status.values() if s == "skip_default")
        print(f"[Recolor] {char_name}: {n_ok} recolored, {n_def} default-kept "
              f"| {status}")
    # return

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
    global _KIN_POS, _KIN_YAW
    pos = np.array([start_pos_xy[0], start_pos_xy[1], ARGS.robot_z_height],
                   dtype=np.float32)
    quat = np.array([math.cos(start_yaw / 2.0),
                     0.0, 0.0,
                     math.sin(start_yaw / 2.0)], dtype=np.float32)
    robot.set_world_pose(position=pos, orientation=quat)

    # Sync desired-state globals so kinematic_move continues from correct origin.
    if ROBOT_DRIVE_MODE == "kinematic":
        _KIN_POS[:] = [float(start_pos_xy[0]), float(start_pos_xy[1]),
                       float(ARGS.robot_z_height)]
        _KIN_YAW    = float(start_yaw)

    nd = int(robot.num_dof)
    try:
        robot.set_joint_velocities(np.zeros(nd, dtype=np.float32))
    except Exception:
        pass
    if ROBOT_DRIVE_MODE == "kinematic" and _REST_JOINT_POSITIONS is not None:
        try:
            robot.set_joint_positions(_REST_JOINT_POSITIONS)
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


def compute_control(
    robot_pos, robot_yaw, target_pos,
    gain_linear, gain_angular,
    max_linear, max_angular,
    desired_distance: float = 1.5,
    pos_deadzone: float = 0.2,
    angle_deadzone_deg: float = 5.0
) -> Tuple[float, float]:
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


# Module-level wheel DOF indices, set in main() for diff_drive robots.
_WHEEL_DOF_LEFT  = 0
_WHEEL_DOF_RIGHT = 1


def kinematic_move(robot, linear: float, angular: float) -> None:
    """Advance the legged robot one physics step along the commanded velocity.

    State is integrated in _KIN_POS/_KIN_YAW (NOT read from physics) so that
    joint reaction torques and residual dynamics can't accumulate into drift or
    spin.  After setting the pose, all linear/angular/joint velocities are zeroed
    to drain any physics-induced motion before the next step.

    Call this AFTER world.step() so it overrides whatever PhysX computed.
    """
    global _KIN_POS, _KIN_YAW
    dt = ARGS.physics_dt

    # Integrate desired state — independent of physics.
    _KIN_YAW += angular * dt
    _KIN_POS[0] += linear * math.cos(_KIN_YAW) * dt
    _KIN_POS[1] += linear * math.sin(_KIN_YAW) * dt
    _KIN_POS[2]  = float(ARGS.robot_z_height)

    new_pos  = _KIN_POS.astype(np.float32)
    new_quat = np.array([
        math.cos(_KIN_YAW / 2.0), 0.0, 0.0, math.sin(_KIN_YAW / 2.0),
    ], dtype=np.float32)

    robot.set_world_pose(position=new_pos, orientation=new_quat)

    # Hold joints at standing rest pose to prevent leg flailing.
    if _REST_JOINT_POSITIONS is not None:
        robot.set_joint_positions(_REST_JOINT_POSITIONS)

    # Zero all velocities — joint stiffness creates reaction torques that spin the
    # root body if allowed to accumulate across steps.
    try:
        robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
        robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
    except Exception:
        pass
    try:
        robot.set_joint_velocities(np.zeros(int(robot.num_dof), dtype=np.float32))
    except Exception:
        pass


def _apply_drive(robot, linear: float, angular: float) -> None:
    """Apply a velocity command for diff-drive robots (no-op for kinematic robots)."""
    if ROBOT_DRIVE_MODE != "diff_drive":
        return
    lv, rv = diff_drive_joint_vels(linear, angular, ARGS.wheel_radius, ARGS.wheel_base)
    vels = np.zeros(int(robot.num_dof), dtype=np.float32)
    vels[_WHEEL_DOF_LEFT]  = lv
    vels[_WHEEL_DOF_RIGHT] = rv
    robot.apply_action(ArticulationAction(joint_velocities=vels))


def _stop_drive(robot) -> None:
    """Zero wheel velocities for diff-drive robots (no-op for kinematic robots)."""
    if ROBOT_DRIVE_MODE != "diff_drive":
        return
    robot.apply_action(ArticulationAction(
        joint_velocities=np.zeros(int(robot.num_dof), dtype=np.float32)))


# Cached rest-pose joint positions for kinematic robots (built once after world.reset).
_REST_JOINT_POSITIONS: Optional[np.ndarray] = None

# Desired kinematic state — integrated independently of physics to avoid drift.
_KIN_POS = np.zeros(3, dtype=np.float64)
_KIN_YAW = 0.0

# Per-robot standing joint positions (from Isaac Lab configs).
_ROBOT_REST_POSES = {
    "go2": {
        "FL_hip_joint": 0.1,  "FR_hip_joint": -0.1,
        "RL_hip_joint": 0.1,  "RR_hip_joint": -0.1,
        "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5, "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
    },
    "g1": {
        "left_hip_pitch_joint":  -0.20, "right_hip_pitch_joint": -0.20,
        "left_knee_joint":        0.42, "right_knee_joint":       0.42,
        "left_ankle_pitch_joint": -0.23, "right_ankle_pitch_joint": -0.23,
        "left_elbow_pitch_joint":  0.87, "right_elbow_pitch_joint":  0.87,
        "left_shoulder_roll_joint":  0.16, "right_shoulder_roll_joint": -0.16,
        "left_shoulder_pitch_joint": 0.35, "right_shoulder_pitch_joint": 0.35,
    },
}


def _build_rest_joint_positions(robot) -> np.ndarray:
    """Build (once) and cache the standing rest-pose array ordered by dof_names."""
    global _REST_JOINT_POSITIONS
    if _REST_JOINT_POSITIONS is not None:
        return _REST_JOINT_POSITIONS
    nd = int(robot.num_dof)
    positions = np.zeros(nd, dtype=np.float32)
    rest_by_name = _ROBOT_REST_POSES.get(ARGS.robot_type, {})
    for i, name in enumerate(robot.dof_names):
        if name in rest_by_name:
            positions[i] = float(rest_by_name[name])
    _REST_JOINT_POSITIONS = positions
    print(f"[KinRobot] Rest joint positions ({ARGS.robot_type}): "
          f"{ {n: round(float(positions[i]),3) for i,n in enumerate(robot.dof_names)} }")
    return _REST_JOINT_POSITIONS


def _find_articulation_root_path(stage, search_root: str) -> Optional[str]:
    """Return the prim path that carries PhysicsArticulationRootAPI under search_root.

    USD references can nest the robot inside a sub-prim named after the file's
    default prim (e.g. go2.usd → /World/Robot/go2_description/...), so we must
    traverse rather than assume a fixed depth.
    """
    base = stage.GetPrimAtPath(search_root)
    if not base or not base.IsValid():
        return None
    for prim in Usd.PrimRange(base):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(prim.GetPath())
    return None


def _find_first_rigid_body_path(stage, search_root: str) -> Optional[str]:
    """Return the path of the first prim carrying PhysicsRigidBodyAPI under search_root."""
    base = stage.GetPrimAtPath(search_root)
    if not base or not base.IsValid():
        return None
    for prim in Usd.PrimRange(base):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(prim.GetPath())
    return None


def _print_prim_tree(stage, root_path: str, max_depth: int = 4) -> None:
    """Debug helper: print prim types under root_path."""
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        print(f"[PrimTree] {root_path} does not exist")
        return
    for prim in Usd.PrimRange(root):
        depth = len(str(prim.GetPath()).split("/")) - len(root_path.split("/"))
        if depth > max_depth:
            continue
        indent = "  " * depth
        apis = [s for s in prim.GetAppliedSchemas()]
        print(f"[PrimTree] {indent}{prim.GetPath()}  [{prim.GetTypeName()}]  {apis}")


# Discovered at load time; used for ArticulationView and camera parent.
_ROBOT_ARTICULATION_PATH: Optional[str] = None


def _setup_kinematic_root(stage) -> None:
    """Disable gravity on every rigid body of the robot so it won't free-fall.

    IMPORTANT: We do NOT set kinematic_enabled=True here.  Setting that flag on
    an articulation root removes the body from the PhysX articulation solver and
    breaks ArticulationView entirely.  Using PhysxRigidBodyAPI.disableGravity
    achieves the same 'no falling' behaviour while keeping the articulation intact.
    """
    if ROBOT_DRIVE_MODE == "diff_drive":
        return
    if _PhysxSchema is None:
        print("[KinRobot] WARN: PhysxSchema not available — gravity not disabled; "
              "robot may fall.")
        return

    base = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    if not base or not base.IsValid():
        print(f"[KinRobot] WARN: {ROBOT_PRIM_PATH} not valid.")
        return

    count = 0
    for prim in Usd.PrimRange(base):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        physx_rb = _PhysxSchema.PhysxRigidBodyAPI.Get(stage, prim.GetPath())
        if not physx_rb:
            physx_rb = _PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        attr = physx_rb.GetDisableGravityAttr()
        if attr:
            attr.Set(True)
        else:
            physx_rb.CreateDisableGravityAttr(True)
        count += 1

    print(f"[KinRobot] Disabled gravity on {count} rigid bodies under {ROBOT_PRIM_PATH}.")


def resolve_wheel_dof_order(robot) -> Tuple[int, int]:
    if ROBOT_DRIVE_MODE != "diff_drive":
        print(f"[Robot] {ARGS.robot_type} uses kinematic control — no wheel DOFs.")
        return -1, -1
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
    if usda_override:
        usda_path = usda_override
    else:
        usda_path = os.path.join(usda_root, f"{scene_id}.usda")
    if not os.path.exists(usda_path):
        raise FileNotFoundError(f"USDA not found: {usda_path}")

    direct = os.path.join(episode_root, scene_id)
    if os.path.isdir(direct):
        return usda_path, direct

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
    clear_all_characters()
    update_sim(10)

    char_paths = spawn_characters_from_episode(episode, char_pool, _CLOTHING_PROFILES)
    target_prim_path = resolve_target_character(episode, char_paths)
    if target_prim_path is None:
        return char_paths, None

    robot_block         = episode["robot"]
    robot_start_pos_xyz = robot_block["start_pos"]
    robot_start_yaw     = float(robot_block.get("start_orientation", 0.0))
    print(f"[Episode setup] start_pos={robot_start_pos_xyz}, "
          f"start_yaw={math.degrees(robot_start_yaw):.1f} deg, "
          f"target_prim={target_prim_path}")
    reset_robot(robot, robot_start_pos_xyz[:2], robot_start_yaw, world)

    update_chase_camera(robot)
    for _ in range(3):
        simulation_app.update()
        update_chase_camera(robot)

    attach_behavior_scripts_to_characters(simulation_app)

    for _ in range(60):
        world.step(render=False)
        update_chase_camera(robot)
        simulation_app.update()

    # 动画/脚本就绪后再染色,避免打断 skeleton 初始化
    recolor_episode_characters(episode, char_paths)
    for _ in range(10):
        world.step(render=False)
        update_chase_camera(robot)
        simulation_app.update()

    return char_paths, target_prim_path


def main() -> int:
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

    if not open_stage(usda_path):
        return 1

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

    candidates = sorted(glob.glob(os.path.join(ep_dir, "episode_*.json")),
                        key=extract_ep_id)
    if not candidates:
        print(f"[FATAL] No episode_*.json found in {ep_dir}")
        simulation_app.close()
        return 1

    end_idx  = len(candidates) if ARGS.end_idx == -1 else ARGS.end_idx
    selected = candidates[ARGS.start_idx:end_idx]
    print(f"[Episode] Selected {len(selected)}/{len(candidates)} episodes")

    with open(selected[0], "r") as f:
        first_episode = json.load(f)["episode"]
    first_robot_block   = first_episode["robot"]
    first_start_pos_xyz = first_robot_block["start_pos"]
    first_start_yaw     = float(first_robot_block.get("start_orientation", 0.0))
    init_xy = (float(first_start_pos_xyz[0]), float(first_start_pos_xyz[1]))
    print(f"[Init] First episode robot start: xy={init_xy}, "
          f"yaw={math.degrees(first_start_yaw):.1f} deg")

    print("[World] Creating World + physicsScene...")
    world = World(physics_dt=ARGS.physics_dt, rendering_dt=ARGS.physics_dt * 3)
    world.initialize_physics()
    update_sim(5)
    print("[World] Physics initialized.")

    robot = add_robot_and_articulation(world, initial_xy=init_xy)
    left_idx, right_idx = resolve_wheel_dof_order(robot)
    global _WHEEL_DOF_LEFT, _WHEEL_DOF_RIGHT
    _WHEEL_DOF_LEFT, _WHEEL_DOF_RIGHT = left_idx, right_idx

    world.reset()
    update_sim(10)

    # For kinematic robots: build rest pose (needs dof_names, available after reset),
    # apply it immediately, and sync desired-state globals to the first episode start.
    if ROBOT_DRIVE_MODE == "kinematic":
        global _KIN_POS, _KIN_YAW
        _build_rest_joint_positions(robot)
        try:
            robot.set_joint_positions(_REST_JOINT_POSITIONS)
        except Exception as e:
            print(f"[KinRobot] Could not set initial joint positions: {e}")
        _KIN_POS[:] = [float(init_xy[0]), float(init_xy[1]), float(ARGS.robot_z_height)]
        _KIN_YAW    = float(first_start_yaw)

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

    init_chase_camera(distance=ARGS.chase_distance, height=ARGS.chase_height)
    update_chase_camera(robot)
    update_sim(3)

    cam = spawn_policy_camera()

    char_pool = load_character_assets()
    # print(char_pool)
    print(f"[People] {len(char_pool)} character assets loaded.")

    # 载入 profiles(资产名 -> usd_path)
    clothing_profiles = None
    clothing_profiles = ClothingProfiles(ARGS.profiles_json)
    global _CLOTHING_PROFILES
    _CLOTHING_PROFILES = clothing_profiles
    print(f"[People] Loaded clothing profiles: {clothing_profiles.num_assets} assets")

    tl = omni.timeline.get_timeline_interface()
    tl.set_current_time(0.0)
    tl.play()
    update_sim(5)

    print("\n[Pre-spawn] Setting up first episode characters...")
    pre_char_paths = spawn_characters_from_episode(first_episode, char_pool, clothing_profiles)
    pre_target_prim_path = resolve_target_character(first_episode, pre_char_paths)

    for _ in range(30):
        update_chase_camera(robot)
        simulation_app.update()

    print("[Pre-spawn] Initial scene ready. Entering episode loop...")

    all_metrics: List[dict] = []

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

            for _ in range(60):
                world.step(render=False)
                update_chase_camera(robot)
                simulation_app.update()

            # 动画就绪后染色
            recolor_episode_characters(episode, pre_char_paths)
            for _ in range(10):
                world.step(render=False)
                simulation_app.update()
        else:
            with open(ep_path, "r") as f:
                episode = json.load(f)["episode"]
            char_paths, target_prim_path = setup_episode_full(
                episode, char_pool, robot, world
            )
            if target_prim_path is None:
                print("[ERROR] No target character resolvable, skip episode")
                continue

        step           = 0
        tracking_steps = 0
        distances:     List[float] = []
        heading_errors: List[float] = []
        done_reason = "max_steps"

        pursuit_state = {
            "path": None,
            "path_idx": 0,
            "last_plan_step": -10**9,
            "last_goal": None,
            "plan_fail_count": 0,
            "pose_history": [],
            "recovery_steps_left": 0,
            "recovery_linear": 0.0,
            "recovery_angular": 0.0,
            "consecutive_recoveries": 0,
            "any_recovery": False,
            "prev_linear": 0.0,
            "prev_angular": 0.0,
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

            pursuit_state["pose_history"].append(
                (robot_pos[:2].copy(), float(robot_yaw)))
            if len(pursuit_state["pose_history"]) > STUCK_WINDOW:
                pursuit_state["pose_history"].pop(0)

            if pursuit_state["recovery_steps_left"] > 0:
                lin = pursuit_state["recovery_linear"]
                ang = pursuit_state["recovery_angular"]
                _apply_drive(robot, lin, ang)

                pursuit_state["prev_linear"]  = lin
                pursuit_state["prev_angular"] = ang

                if step % 5 == 0:
                    print(f"[EP{ep_id}] step={step} RECOVERY "
                          f"lin={lin:.2f} ang={ang:.2f} "
                          f"left={pursuit_state['recovery_steps_left']}")

                for _ in range(ARGS.decimation):
                    world.step(render=False)
                    if ROBOT_DRIVE_MODE == "kinematic":
                        kinematic_move(robot, lin, ang)
                    update_chase_camera(robot)
                    simulation_app.update()
                if ARGS.save_images and step % ARGS.save_image_every == 0:
                    save_rgb_depth(cam, step, ARGS.image_save_dir, ep_id)

                pursuit_state["recovery_steps_left"] -= 1
                if pursuit_state["recovery_steps_left"] == 0:
                    pursuit_state["path"] = None
                    pursuit_state["pose_history"].clear()
                    pursuit_state["prev_linear"]  = 0.0
                    pursuit_state["prev_angular"] = 0.0
                    print(f"[EP{ep_id}] Recovery finished, will replan.")
                step += 1
                continue

            if len(pursuit_state["pose_history"]) >= STUCK_WINDOW:
                first_pos, first_yaw = pursuit_state["pose_history"][0]
                last_pos,  last_yaw  = pursuit_state["pose_history"][-1]
                pos_delta = float(np.linalg.norm(last_pos - first_pos))
                yaw_delta = abs(math.degrees(math.atan2(
                    math.sin(last_yaw - first_yaw),
                    math.cos(last_yaw - first_yaw))))

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
                    pursuit_state["any_recovery"] = True
                    if pursuit_state["consecutive_recoveries"] > MAX_RECOVERIES:
                        print(f"[EP{ep_id}] Stuck {MAX_RECOVERIES}+ times in a "
                              f"row, ending episode.")
                        done_reason = "stuck"
                        _stop_drive(robot)
                        for _ in range(3):
                            world.step(render=False)
                            update_chase_camera(robot)
                            simulation_app.update()
                        break

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
                    step += 1
                    continue
                else:
                    if pos_delta > STUCK_POS_THRESH * 3:
                        pursuit_state["consecutive_recoveries"] = 0

            visible, blocker_prim, hit_dist = check_visibility(
                robot_pos, target_pos, target_prim_path)
            if not visible and step % 10 == 0:
                print(f"[EP{ep_id}] step={step} OCCLUDED "
                      f"by {blocker_prim} at {hit_dist:.2f}m "
                      f"(full_dist={float(np.linalg.norm(target_pos[:2]-robot_pos[:2])):.2f}m)")

            in_window = (TRACKING_DIST_MIN <= dist_to_target
                         <= ARGS.tracking_dist_max)
            in_view   = (angle_err_deg <= ARGS.tracking_angle_thresh)
            if in_window and in_view and visible:
                tracking_steps += 1

            handled_by_detour = False
            if not visible:
                vantage = find_visible_vantage_point(
                    nm, target_pos, target_prim_path, robot_pos)

                if vantage is not None:
                    robot_on_nm = snap_to_navmesh(nm, robot_pos,
                                                  search_radius=1.0)
                    start_xyz = (robot_on_nm if robot_on_nm is not None
                                 else robot_pos)
                    new_path = plan_navmesh_path(nm, start_xyz, vantage)

                    if new_path is not None and len(new_path) >= 1:
                        pursuit_state["path"]            = new_path
                        pursuit_state["path_idx"]        = 0
                        pursuit_state["last_plan_step"]  = step
                        pursuit_state["last_goal"]       = vantage.copy()
                        pursuit_state["plan_fail_count"] = 0

                        carrot, new_idx = pick_carrot(
                            pursuit_state["path"], 0, robot_pos,
                            lookahead=LOOKAHEAD_DIST)
                        pursuit_state["path_idx"] = new_idx

                        linear, angular = compute_control(
                            robot_pos, robot_yaw, carrot,
                            ARGS.control_gain_linear, ARGS.control_gain_angular,
                            ARGS.max_linear_vel, ARGS.max_angular_vel,
                            desired_distance=0.0,
                            pos_deadzone=0.15,
                            angle_deadzone_deg=4.0,
                        )
                        linear, angular = smooth_cmd(
                            pursuit_state, linear, angular,
                            SMOOTH_ALPHA_NORMAL)

                        _apply_drive(robot, linear, angular)

                        if step % 10 == 0:
                            print(f"[EP{ep_id}] step={step} DETOUR "
                                  f"vantage={vantage[:2]} "
                                  f"path_len={len(new_path)} "
                                  f"carrot={carrot[:2]} "
                                  f"lin={linear:.2f} ang={angular:.2f}")

                        for _ in range(ARGS.decimation):
                            world.step(render=False)
                            if ROBOT_DRIVE_MODE == "kinematic":
                                kinematic_move(robot, linear, angular)
                            update_chase_camera(robot)
                            simulation_app.update()

                        if ARGS.save_images and step % ARGS.save_image_every == 0:
                            save_rgb_depth(cam, step, ARGS.image_save_dir, ep_id)
                        step += 1
                        handled_by_detour = True
                else:
                    if step % 20 == 0:
                        print(f"[EP{ep_id}] step={step} OCCLUDED, "
                              f"no vantage found — falling through to direct.")

            if handled_by_detour:
                continue

            if dist_to_target <= DIRECT_FOLLOW_RANGE:
                dx = float(target_pos[0] - robot_pos[0])
                dy = float(target_pos[1] - robot_pos[1])
                heading_to_target = math.atan2(dy, dx)
                yaw_err = math.atan2(math.sin(heading_to_target - robot_yaw),
                                     math.cos(heading_to_target - robot_yaw))

                dist_err = dist_to_target - FOLLOW_DIST

                if (visible and
                    abs(dist_err) < DIRECT_DEADZONE_POS and
                    abs(math.degrees(yaw_err)) < DIRECT_DEADZONE_ANG):
                    linear, angular = 0.0, 0.0
                elif not visible:
                    angular = ARGS.control_gain_angular * yaw_err
                    angular = float(np.clip(angular,
                                            -ARGS.max_angular_vel,
                                            ARGS.max_angular_vel))
                    if abs(yaw_err) < math.radians(60.0):
                        linear = 0.4 * math.cos(yaw_err)
                    else:
                        linear = 0.0
                else:
                    if abs(yaw_err) > math.radians(90.0):
                        linear  = 0.0
                        angular = float(np.clip(
                            ARGS.control_gain_angular * yaw_err,
                            -ARGS.max_angular_vel, ARGS.max_angular_vel))
                    else:
                        linear = ARGS.control_gain_linear * dist_err * math.cos(yaw_err)
                        linear = float(np.clip(linear,
                                               -ARGS.max_linear_vel,
                                               ARGS.max_linear_vel))
                        angular = ARGS.control_gain_angular * yaw_err
                        angular = float(np.clip(angular,
                                                -ARGS.max_angular_vel,
                                                ARGS.max_angular_vel))

                linear, angular = smooth_cmd(
                    pursuit_state, linear, angular, SMOOTH_ALPHA_NORMAL)

                _apply_drive(robot, linear, angular)

                pursuit_state["path"] = None

                if step % 20 == 0:
                    print(f"[EP{ep_id}] step={step} DIRECT-FOLLOW "
                          f"dist={dist_to_target:.2f} "
                          f"dist_err={dist_err:+.2f} "
                          f"yaw_err={math.degrees(yaw_err):+.1f}deg "
                          f"visible={visible} "
                          f"lin={linear:.2f} ang={angular:.2f}")

                for _ in range(ARGS.decimation):
                    world.step(render=False)
                    if ROBOT_DRIVE_MODE == "kinematic":
                        kinematic_move(robot, linear, angular)
                    update_chase_camera(robot)
                    simulation_app.update()

                if ARGS.save_images and step % ARGS.save_image_every == 0:
                    save_rgb_depth(cam, step, ARGS.image_save_dir, ep_id)

                step += 1
                continue

            follow_point_raw = compute_follow_point(
                target_pos, target_yaw, FOLLOW_DIST)
            follow_point = snap_to_navmesh(nm, follow_point_raw, search_radius=1.5)
            if follow_point is None:
                follow_point = snap_to_navmesh(nm, target_pos, search_radius=1.5)
            if follow_point is None:
                follow_point = np.asarray(target_pos, dtype=np.float64)

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
                need_replan = True

            if need_replan:
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
                        _stop_drive(robot)
                        for _ in range(3):
                            world.step(render=False)
                            update_chase_camera(robot)
                            simulation_app.update()
                        break
                    _stop_drive(robot)
                    for _ in range(ARGS.decimation):
                        world.step(render=False)
                        update_chase_camera(robot)
                        simulation_app.update()
                    if ARGS.save_images and step % ARGS.save_image_every == 0:
                        save_rgb_depth(cam, step, ARGS.image_save_dir, ep_id)
                    step += 1
                    continue

            carrot, new_idx = pick_carrot(
                pursuit_state["path"], pursuit_state["path_idx"], robot_pos,
                lookahead=LOOKAHEAD_DIST)
            pursuit_state["path_idx"] = new_idx

            at_path_end = (new_idx >= len(pursuit_state["path"]) - 1)
            if at_path_end:
                carrot = follow_point

            desired_standoff = 0.0

            linear, angular = compute_control(
                robot_pos, robot_yaw, carrot,
                ARGS.control_gain_linear, ARGS.control_gain_angular,
                ARGS.max_linear_vel, ARGS.max_angular_vel,
                desired_distance=desired_standoff,
                pos_deadzone=0.15,
                angle_deadzone_deg=4.0,
            )
            linear, angular = smooth_cmd(
                pursuit_state, linear, angular, SMOOTH_ALPHA_NORMAL)

            _apply_drive(robot, linear, angular)

            if step % 20 == 0:
                path_len = len(pursuit_state["path"]) if pursuit_state["path"] else 0
                print(f"[EP{ep_id}] step={step} "
                      f"pos={robot_pos[:2]} yaw={math.degrees(robot_yaw):.1f} "
                      f"target={target_pos[:2]} dist={dist_to_target:.2f} "
                      f"carrot={carrot[:2]} path_idx={pursuit_state['path_idx']}/{path_len} "
                      f"lin={linear:.2f} ang={angular:.2f} "
                      f"plan_fails={pursuit_state['plan_fail_count']}")

            for _ in range(ARGS.decimation):
                world.step(render=False)
                if ROBOT_DRIVE_MODE == "kinematic":
                    kinematic_move(robot, linear, angular)
                update_chase_camera(robot)
                simulation_app.update()

            if ARGS.save_images and step % ARGS.save_image_every == 0:
                save_rgb_depth(cam, step, ARGS.image_save_dir, ep_id)

            step += 1

        episode_length = step
        tracking_rate  = tracking_steps / max(episode_length, 1)
        had_recovery   = pursuit_state["any_recovery"]
        success = (bool(distances)
                   and tracking_rate >= 0.8
                   and distances[-1] <= ARGS.tracking_dist_max
                   and not had_recovery)

        if had_recovery and done_reason == "max_steps":
            done_reason = "had_recovery"

        metrics = {
            "episode_id":            ep_id,
            "success":               int(success),
            "had_recovery":          int(had_recovery),
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
              f"tr={tracking_rate:.3f} had_recovery={had_recovery} reason={done_reason}")

        _stop_drive(robot)
        for _ in range(5):
            world.step(render=False)
            update_chase_camera(robot)
            simulation_app.update()

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