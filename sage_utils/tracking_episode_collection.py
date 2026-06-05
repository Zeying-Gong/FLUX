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
    p.add_argument("--episode_dir", required=True,
                   help="Root dir of tracking episodes (contains scene subdirs), "
                        "OR a direct path to a specific scene subdir "
                        "(in which case --scene_id is inferred from the basename).")
    p.add_argument("--scene_id", default=None,
                   help="Scene ID, e.g. '0001_839920'.  Inferred from the basename "
                        "of --episode_dir when omitted.")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx",   type=int, default=-1)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--collision_root",  default="/World/scene_collision")
    p.add_argument("--volume_padding",  type=float, default=1.2)
    p.add_argument("--fallback_size",   type=float, default=100.0)
    p.add_argument("--warmup_frames",   type=int,   default=60)
    p.add_argument("--cache_dir",       default=None)
    p.add_argument("--force_rebake",    action="store_true")
    p.add_argument("--semantic_maps_root",
                   default="/workspace/SAGE-3D_Official/SAGE-3D_data/semantic_maps_v2",
                   help="Directory containing 2D_Semantic_Map_<scene_id>_Complete.json files.")
    p.add_argument("--semantic_map_json", default=None,
                   help="Explicit path to semantic map JSON.  "
                        "Auto-derived from --semantic_maps_root and scene_id when omitted.")
    p.add_argument("--occ_scale",        type=float, default=0.1)
    p.add_argument("--robot_radius_2d",  type=float, default=0.3)
    p.add_argument("--robot_usd",
                   default="/workspace/FLUX/assets/robots/dingo_fixed.usd")
    p.add_argument("--robot_z_height", type=float, default=0.1)
    p.add_argument("--wheel_radius", type=float, default=0.0762)
    p.add_argument("--wheel_base",   type=float, default=0.32)
    p.add_argument("--control_gain_linear",  type=float, default=1.0)
    p.add_argument("--control_gain_angular", type=float, default=3.0)
    p.add_argument("--max_linear_vel",       type=float, default=1.2)
    p.add_argument("--max_angular_vel",      type=float, default=2.5)
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

FOLLOW_DIST       = ARGS.follow_distance
TRACKING_DIST_MIN = max(FOLLOW_DIST - 1.0, 1.0)

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
from isaacsim.core.prims import RigidPrim as RigidPrimView
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
# Camera horizontal half-FOV: 2*atan(aperture/2 / focal_length) = 2*atan(0.94/1.4)
_CAM_FOV_HALF_DEG = math.degrees(math.atan(1.88 / 2.0 / 1.4))   # ≈ 33.9°

REPLAN_PERIOD     = 10
REPLAN_TARGET_THR = 0.5
LOOKAHEAD_DIST    = 1.0
WAYPOINT_REACH    = 0.3
MAX_PLAN_FAILS    = 5
NM_AGENT_RADIUS   = 0.25

STUCK_WINDOW          = 8
STUCK_POS_THRESH      = 0.05
STUCK_YAW_THRESH_DEG  = 5.0

# ── Character-done detection ────────────────────────────────────────
# Primary: read omni:scripting:commandsDone written by CharacterBehavior when its
# command queue empties.  Grace period gives the robot time to reach final position.
DONE_GRACE_STEPS = 30   # extra steps after done flag set (~3 s at 10 Hz control)

# ── Sidestep parameters ─────────────────────────────────────────────
# When STANDOFF-FAIL fires (no valid standoff position found), the robot tries
# to escape sideways instead of standing frozen in front of the target.
SIDESTEP_LINEAR       = 0.35   # forward speed used while turning to dodge (m/s)
SIDESTEP_CHECK_DIST   = 0.55   # how far ahead to probe occ before choosing side (m)
STUCK_CMD_THRESH      = 0.5
RECOVERY_DURATION     = 8
RECOVERY_LINEAR       = -0.4
RECOVERY_ANGULAR_MAG  = 1.2
MAX_RECOVERIES        = 3

DIRECT_DEADZONE_ANG  = 15.0   # yaw error below which angular=0 (was 10°)
DIRECT_DEADZONE_ANG_TIGHT = 8.0   # secondary tight zone for prev_angular reset

LOS_RAY_Z_OFFSET    = 0.9
LOS_CAM_Z_OFFSET    = 0.3
LOS_HIT_TOLERANCE   = 0.3

VANTAGE_RADII          = [1.5, 2.0, 2.5]
VANTAGE_NUM_SAMPLES    = 12
VANTAGE_MIN_ROBOT_DIST = 0.5

SMOOTH_ALPHA_NORMAL = 0.55   # higher = more responsive (was 0.4)
SMOOTH_ALPHA_TRACK  = 0.30   # more damped for TRACK mode to prevent yaw oscillation

# ── Oracle target evasion & LoS parameters ─────────────────────────
TARGET_EVADE_LOOKAHEAD_S  = 2.0   # seconds ahead to predict target position
LOS_ANG_TARGET_WEIGHT     = 0.40  # fraction of angular blended toward target vs waypoint

ROBOT_PRE_TRACK_FRAMES = 20   # frames robot moves toward target before characters start

# ── Collision / contact force detection ────────────────────────────
COLLISION_FORCE_THRESHOLD = 400.0   # N·s impulse threshold (per contact pair per step)

_robot_rigid_view: Optional[RigidPrimView] = None


def setup_robot_contact_report(stage) -> bool:
    """Enable PhysX contact reporting on every rigid body of the robot (for RigidPrimView)."""
    if _PhysxSchema is None:
        return False
    base = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    if not base or not base.IsValid():
        return False
    count = 0
    for prim in Usd.PrimRange(base):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        api = _PhysxSchema.PhysxContactReportAPI.Get(stage, prim.GetPath())
        if not api:
            api = _PhysxSchema.PhysxContactReportAPI.Apply(prim)
        thr = api.GetThresholdAttr()
        if thr:
            thr.Set(0.0)   # report all contacts; we apply threshold in Python
        else:
            api.CreateThresholdAttr(0.0)
        count += 1
    print(f"[Collision] Contact report enabled on {count} robot rigid bodies.")
    return True


def setup_robot_contact_sensor(world: "World") -> None:
    """Create a RigidPrimView for the robot base to track net contact forces."""
    global _robot_rigid_view
    if _ROBOT_ARTICULATION_PATH is None:
        return
    try:
        _robot_rigid_view = RigidPrimView(
            prim_paths_expr=_ROBOT_ARTICULATION_PATH,
            name="robot_base_contact_view",
            track_contact_forces=True,
        )
        world.scene.add(_robot_rigid_view)
        print(f"[Collision] RigidPrimView contact sensor on {_ROBOT_ARTICULATION_PATH}.")
    except Exception as e:
        print(f"[Collision] WARN: RigidPrimView setup failed: {e}")
        _robot_rigid_view = None


def read_robot_contact_force() -> float:
    """Return max-norm net contact force on the robot base (N). 0.0 on error."""
    if _robot_rigid_view is None:
        return 0.0
    try:
        forces = _robot_rigid_view.get_net_contact_forces(dt=ARGS.physics_dt)
        if forces is None:
            return 0.0
        arr = np.asarray(forces)
        return float(np.max(np.linalg.norm(arr.reshape(-1, 3), axis=1)))
    except Exception:
        return 0.0


def reset_collision_state() -> None:
    pass  # stateless: contact force read directly each step


# ── Backward-safety & abnormal-state helpers ───────────────────────
_GLOBAL_OCC: Optional[tuple] = None   # set in main() after occ is loaded

# Count of kinematic steps where linear motion was blocked by occ grid.
_KIN_BLOCKED_TOTAL = 0


def _backward_clear(robot_pos: np.ndarray, robot_yaw: float, occ,
                    probe_dist: float = 0.60) -> bool:
    """Return True if the cell probe_dist behind the robot is obstacle-free."""
    if occ is None:
        return True
    bx = float(robot_pos[0]) - probe_dist * math.cos(robot_yaw)
    by = float(robot_pos[1]) - probe_dist * math.sin(robot_yaw)
    return _occ_is_free(occ, bx, by)


def _detect_in_obstacle(pos: np.ndarray, occ) -> bool:
    """Return True if pos is inside an obstacle cell (abnormal state)."""
    if occ is None:
        return False
    return not _occ_is_free(occ, float(pos[0]), float(pos[1]))


def _find_nearest_free(pos: np.ndarray, occ,
                       max_steps: int = 30) -> Optional[np.ndarray]:
    """Spiral outward from pos to find the nearest obstacle-free cell."""
    if occ is None:
        return None
    _, _, _, scale = occ
    for r in range(1, max_steps + 1):
        for angle_deg in range(0, 360, 15):
            a = math.radians(angle_deg)
            cx = float(pos[0]) + r * scale * math.cos(a)
            cy = float(pos[1]) + r * scale * math.sin(a)
            if _occ_is_free(occ, cx, cy):
                return np.array([cx, cy, float(pos[2])], dtype=np.float64)
    return None


# ── Pedestrian prediction & avoidance ──────────────────────────────
PED_WALK_SPEED_M_S     = 1.2   # assumed character walking speed
PED_FORECAST_HORIZON_S = 3.0   # look-ahead window for path prediction
PED_FORECAST_SAMPLES   = 9     # number of sample points over horizon
PED_DYN_RADIUS_M       = 0.55  # virtual obstacle radius per predicted position
PED_REACT_STOP_M       = 0.6   # stop if non-target character within this distance ahead
PED_REACT_SLOW_M       = 1.4   # slow-down zone starts here
PED_REACT_FRONT_DEG    = 100   # react only when character is within ±100° ahead

# Module-level caches, reset per episode.
_CHAR_TRAJ_CACHE: Dict[str, List[List[float]]] = {}  # char_name → flat path pts
_PED_POS_HIST:   Dict[str, List[np.ndarray]]   = {}  # char_name → recent positions
_PED_HIST_LEN = 4

# Oracle target trajectory (populated per episode from episode GoTo commands).
_TARGET_TRAJ_CACHE: List[List[float]] = []
# Recent target positions used for velocity-based approach detection (fallback).
_TARGET_POS_HIST:   List[np.ndarray]  = []
_TARGET_POS_HIST_LEN = 6


def build_char_traj_cache(episode: dict,
                           char_paths: Dict[str, str],
                           target_prim_path: str) -> None:
    """Extract ordered GoTo path points for every non-target character."""
    global _CHAR_TRAJ_CACHE, _PED_POS_HIST
    _CHAR_TRAJ_CACHE.clear()
    _PED_POS_HIST.clear()
    target_name = next((k for k, v in char_paths.items() if v == target_prim_path), None)
    commands = episode.get("characters", {}).get("commands", {})
    for char_name, cmds in commands.items():
        if char_name == target_name or char_name not in char_paths:
            continue
        pts: List[List[float]] = []
        for cmd in cmds:
            if cmd.get("cmd") == "GoTo":
                pts.extend(cmd.get("path", []))
        if pts:
            _CHAR_TRAJ_CACHE[char_name] = pts


def build_target_traj_cache(episode: dict, char_paths: Dict[str, str],
                             target_prim_path: str) -> None:
    """Extract oracle GoTo path for the target character (used for proactive evasion)."""
    global _TARGET_TRAJ_CACHE, _TARGET_POS_HIST
    _TARGET_TRAJ_CACHE.clear()
    _TARGET_POS_HIST.clear()
    target_name = next((k for k, v in char_paths.items() if v == target_prim_path), None)
    if target_name is None:
        return
    commands = episode.get("characters", {}).get("commands", {}).get(target_name, [])
    pts: List[List[float]] = []
    for cmd in commands:
        if cmd.get("cmd") == "GoTo":
            pts.extend(cmd.get("path", []))
    _TARGET_TRAJ_CACHE = pts
    print(f"[Oracle] Target '{target_name}' path: {len(pts)} oracle points loaded")


def predict_target_future_pos(target_pos: np.ndarray,
                               horizon_s: float = TARGET_EVADE_LOOKAHEAD_S
                               ) -> Optional[np.ndarray]:
    """Walk oracle GoTo path forward from current target pos by horizon_s * walk_speed."""
    if not _TARGET_TRAJ_CACHE:
        return None
    horizon_m = PED_WALK_SPEED_M_S * horizon_s
    pos2d = target_pos[:2]
    start_i = _nearest_path_idx(pos2d, _TARGET_TRAJ_CACHE)
    accumulated = 0.0
    for i in range(start_i, len(_TARGET_TRAJ_CACHE) - 1):
        seg_dx = float(_TARGET_TRAJ_CACHE[i + 1][0]) - float(_TARGET_TRAJ_CACHE[i][0])
        seg_dy = float(_TARGET_TRAJ_CACHE[i + 1][1]) - float(_TARGET_TRAJ_CACHE[i][1])
        seg_len = math.hypot(seg_dx, seg_dy)
        if accumulated + seg_len >= horizon_m:
            rem = horizon_m - accumulated
            frac = rem / max(seg_len, 1e-6)
            return np.array([
                float(_TARGET_TRAJ_CACHE[i][0]) + frac * seg_dx,
                float(_TARGET_TRAJ_CACHE[i][1]) + frac * seg_dy,
                float(target_pos[2]),
            ], dtype=np.float64)
        accumulated += seg_len
    return np.array([float(_TARGET_TRAJ_CACHE[-1][0]),
                     float(_TARGET_TRAJ_CACHE[-1][1]),
                     float(target_pos[2])], dtype=np.float64)


def _find_evade_position(nm, target_pos: np.ndarray,
                          target_future_pos: np.ndarray,
                          robot_pos: np.ndarray, occ) -> Optional[np.ndarray]:
    """Find a lateral standoff position when target walks toward robot.

    The evasion point is perpendicular to the target's future motion direction,
    at FOLLOW_DIST or TRACKING_DIST_MIN from the target's predicted future position.
    We prefer the side requiring less robot travel.
    """
    motion = (target_future_pos[:2] - target_pos[:2]).astype(np.float64)
    mot_len = float(np.linalg.norm(motion))
    if mot_len < 0.25:
        return None
    mot_dir = motion / mot_len
    perp = np.array([-mot_dir[1], mot_dir[0]], dtype=np.float64)   # 90° left of motion

    best_pt, best_cost = None, float("inf")
    for sign in [1.0, -1.0]:
        for radius in [FOLLOW_DIST, TRACKING_DIST_MIN]:
            cand_xy = target_future_pos[:2] + sign * perp * radius
            cand = np.array([cand_xy[0], cand_xy[1], float(target_pos[2])],
                            dtype=np.float64)
            if occ is not None and not _occ_is_free(occ, cand[0], cand[1]):
                continue
            cand_nm = snap_to_navmesh(nm, cand, search_radius=1.0)
            if cand_nm is None:
                continue
            cost = float(np.linalg.norm(cand_nm[:2] - robot_pos[:2]))
            if cost < best_cost:
                best_cost, best_pt = cost, cand_nm
    return best_pt


def _los_blend_angular(robot_pos: np.ndarray, robot_yaw: float,
                        target_pos: np.ndarray, ang_nav: float,
                        weight: float = LOS_ANG_TARGET_WEIGHT) -> float:
    """Blend waypoint-facing angular with target-facing angular for LoS maintenance.

    weight=1.0 → always face target; weight=0.0 → pure navigation.
    Near-distance guard prevents jitter when target and robot coincide.
    """
    dx = float(target_pos[0] - robot_pos[0])
    dy = float(target_pos[1] - robot_pos[1])
    if math.hypot(dx, dy) < 0.4:
        return ang_nav
    yaw_to_tgt = math.atan2(dy, dx)
    ang_err_tgt = math.atan2(math.sin(yaw_to_tgt - robot_yaw),
                              math.cos(yaw_to_tgt - robot_yaw))
    ang_to_tgt = float(np.clip(ARGS.control_gain_angular * ang_err_tgt,
                                -ARGS.max_angular_vel, ARGS.max_angular_vel))
    return weight * ang_to_tgt + (1.0 - weight) * ang_nav


def _nearest_path_idx(pos2d: np.ndarray, path: List[List[float]]) -> int:
    """Index of the path point closest to pos2d."""
    best_i, best_d = 0, float("inf")
    for i, pt in enumerate(path):
        d = math.hypot(float(pt[0]) - pos2d[0], float(pt[1]) - pos2d[1])
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def predict_char_trajectory(char_name: str,
                             char_prim_path: str) -> List[np.ndarray]:
    """
    Return PED_FORECAST_SAMPLES positions predicted along the character's
    known GoTo path, starting from their current map position.
    Falls back to linear extrapolation from velocity history if no path is available.
    """
    horizon_m = PED_WALK_SPEED_M_S * PED_FORECAST_HORIZON_S
    traj = _CHAR_TRAJ_CACHE.get(char_name)
    curr_pos, _ = get_character_pose(char_prim_path)
    pos2d = curr_pos[:2]

    if not traj:
        # Fallback: linear extrapolation from velocity history
        hist = _PED_POS_HIST.get(char_name, [])
        if len(hist) >= 2:
            step_dt = ARGS.physics_dt * ARGS.decimation
            vel = (hist[-1] - hist[-2]) / max(step_dt, 1e-6)
            return [
                np.array([pos2d[0] + vel[0] * t,
                          pos2d[1] + vel[1] * t, 0.0], dtype=np.float64)
                for t in np.linspace(0, PED_FORECAST_HORIZON_S,
                                     PED_FORECAST_SAMPLES + 1)[1:]
            ]
        return []

    start_i = _nearest_path_idx(pos2d, traj)
    samples: List[np.ndarray] = []
    dist_targets = [horizon_m * (k + 1) / PED_FORECAST_SAMPLES
                    for k in range(PED_FORECAST_SAMPLES)]
    accumulated = 0.0
    i = start_i

    for dist_target in dist_targets:
        while i < len(traj) - 1:
            seg_dx = float(traj[i + 1][0]) - float(traj[i][0])
            seg_dy = float(traj[i + 1][1]) - float(traj[i][1])
            seg_len = math.hypot(seg_dx, seg_dy)
            if accumulated + seg_len >= dist_target:
                rem = dist_target - accumulated
                frac = rem / max(seg_len, 1e-6)
                samples.append(np.array([
                    float(traj[i][0]) + frac * seg_dx,
                    float(traj[i][1]) + frac * seg_dy, 0.0
                ], dtype=np.float64))
                break
            accumulated += seg_len
            i += 1
        else:
            # Past end of path: hold at last point
            samples.append(np.array(
                [float(traj[-1][0]), float(traj[-1][1]), 0.0], dtype=np.float64))

    return samples


def update_ped_vel_hist(char_paths: Dict[str, str],
                         target_prim_path: str) -> None:
    """Record current position of each non-target character (for velocity fallback)."""
    for char_name, prim_path in char_paths.items():
        if prim_path == target_prim_path:
            continue
        pos, _ = get_character_pose(prim_path)
        hist = _PED_POS_HIST.setdefault(char_name, [])
        hist.append(pos[:2].copy())
        if len(hist) > _PED_HIST_LEN:
            hist.pop(0)


def get_all_ped_forecasts(char_paths: Dict[str, str],
                           target_prim_path: str) -> List[np.ndarray]:
    """Aggregate predicted positions for every non-target character."""
    out: List[np.ndarray] = []
    for char_name, prim_path in char_paths.items():
        if prim_path == target_prim_path:
            continue
        out.extend(predict_char_trajectory(char_name, prim_path))
    return out


def occ_with_ped_forecasts(occ, ped_forecasts: List[np.ndarray]):
    """Return a temporary OCC tuple with predicted ped positions added as obstacles.
    The original grid is not modified."""
    if occ is None or not ped_forecasts:
        return occ
    grid, max_x, max_y, scale = occ
    dyn = grid.copy()
    H, W = dyn.shape
    r_px = max(1, int(math.ceil(PED_DYN_RADIUS_M / scale)))
    for pos in ped_forecasts:
        cpx = int(round((max_x + float(pos[0])) / scale))
        cpy = int(round((max_y + float(pos[1])) / scale))
        for dy in range(-r_px, r_px + 1):
            for dx in range(-r_px, r_px + 1):
                if dx * dx + dy * dy <= r_px * r_px:
                    nx, ny = cpx + dx, cpy + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        dyn[ny, nx] = 1
    return (dyn, max_x, max_y, scale)


def apply_ped_reactive_avoidance(
    robot_pos: np.ndarray,
    robot_yaw: float,
    linear: float,
    angular: float,
    char_paths: Dict[str, str],
    target_prim_path: str,
) -> Tuple[float, float]:
    """Scale linear velocity down (or to 0) when a non-target pedestrian is
    close and ahead of the robot.  Angular is left unchanged."""
    min_front_dist = float("inf")
    for char_name, prim_path in char_paths.items():
        if prim_path == target_prim_path:
            continue
        ped_pos, _ = get_character_pose(prim_path)
        dx = float(ped_pos[0]) - float(robot_pos[0])
        dy = float(ped_pos[1]) - float(robot_pos[1])
        dist = math.hypot(dx, dy)
        if dist >= PED_REACT_SLOW_M:
            continue
        rel_deg = abs(math.degrees(
            math.atan2(math.sin(math.atan2(dy, dx) - robot_yaw),
                       math.cos(math.atan2(dy, dx) - robot_yaw))))
        if rel_deg <= PED_REACT_FRONT_DEG:
            min_front_dist = min(min_front_dist, dist)

    if min_front_dist <= PED_REACT_STOP_M:
        return 0.0, angular
    if min_front_dist < PED_REACT_SLOW_M:
        scale = ((min_front_dist - PED_REACT_STOP_M) /
                 (PED_REACT_SLOW_M - PED_REACT_STOP_M))
        return linear * scale, angular
    return linear, angular


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

    if hit_prim.startswith(ROBOT_PRIM_PATH):
        return True, hit_prim, hit_dist

    if hit_prim.startswith(target_prim_path):
        return True, hit_prim, hit_dist

    if hit_dist >= (full_dist - LOS_HIT_TOLERANCE):
        return True, hit_prim, hit_dist

    return False, hit_prim, hit_dist



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


def _load_occ_grid(sem_json: str, scale_m: float, robot_r: float):
    """Load occupancy grid from semantic map JSON with interior masking.

    Mirrors load_2d_map() in generate_episode_fast_tracking.py:
      - Strategy A: uses floor labels to define interior.
      - Strategy B: infers interior from building footprint (walls+furniture).
    Exterior pixels are forced to obstacle so the robot cannot plan outside.
    Robot-radius inflation uses EDT for accuracy (vs. iterative dilation).

    Returns (grid, max_x, max_y, scale) where grid[py,px]==0 means free,
    or None on failure.
    """
    from scipy.ndimage import (binary_dilation, binary_fill_holes,
                                binary_closing, distance_transform_edt,
                                label as _scilabel)
    try:
        with open(sem_json, encoding="utf-8") as f:
            raw = json.load(f)
        items = raw["instances"] if isinstance(raw, dict) and "instances" in raw else raw
        meta  = raw.get("meta", {}) if isinstance(raw, dict) else {}
        if all(k in meta for k in ("x_min", "x_max", "y_min", "y_max")):
            min_x, max_x = float(meta["x_min"]), float(meta["x_max"])
            min_y, max_y = float(meta["y_min"]), float(meta["y_max"])
        else:
            coords = [(float(y), float(x))
                      for inst in items
                      for y, x in inst.get("mask_coords_m", [])]
            if not coords:
                return None
            min_y = min(c[0] for c in coords); max_y = max(c[0] for c in coords)
            min_x = min(c[1] for c in coords); max_x = max(c[1] for c in coords)

        h = int(np.ceil((max_y - min_y) / scale_m)) + 1
        w = int(np.ceil((max_x - min_x) / scale_m)) + 1

        # ── walls + unable areas → raw_grid ──────────────────────────────
        raw_grid = np.zeros((h, w), dtype=np.uint8)
        for inst in items:
            label = str(inst.get("category_label", "")).lower()
            if label not in ("wall", "unable area", "unable_area"):
                continue
            for y_m, x_m in inst.get("mask_coords_m", []):
                try:
                    py = int(round((float(y_m) - min_y) / scale_m))
                    px = int(round((float(x_m) - min_x) / scale_m))
                    if 0 <= py < h and 0 <= px < w:
                        raw_grid[py, px] = 1
                except (ValueError, TypeError):
                    pass

        # ── furniture mask (used only by Strategy B footprint) ───────────
        furniture_labels = ("table", "chair", "sofa", "bed", "wardrobe",
                            "desk", "counter", "cabinet")
        furniture_mask = np.zeros((h, w), dtype=np.uint8)
        for inst in items:
            label = str(inst.get("category_label", "")).lower()
            if not any(k in label for k in furniture_labels):
                continue
            for y_m, x_m in inst.get("mask_coords_m", []):
                try:
                    py = int(round((float(y_m) - min_y) / scale_m))
                    px = int(round((float(x_m) - min_x) / scale_m))
                    if 0 <= py < h and 0 <= px < w:
                        furniture_mask[py, px] = 1
                except (ValueError, TypeError):
                    pass

        # ── interior mask ─────────────────────────────────────────────────
        floor_px = 0
        interior_mask = np.zeros((h, w), dtype=np.uint8)
        for inst in items:
            label = str(inst.get("category_label", "")).lower()
            if label != "floor":
                continue
            for y_m, x_m in inst.get("mask_coords_m", []):
                try:
                    py = int(round((float(y_m) - min_y) / scale_m))
                    px = int(round((float(x_m) - min_x) / scale_m))
                    if 0 <= py < h and 0 <= px < w:
                        interior_mask[py, px] = 1; floor_px += 1
                except (ValueError, TypeError):
                    pass

        if floor_px > 50:
            # Strategy A: floor labels are available
            print(f"[OccMap] Strategy A: floor labels ({floor_px} px).")
            interior_mask = binary_dilation(interior_mask.astype(bool), iterations=2)
            interior_mask = binary_fill_holes(interior_mask).astype(np.uint8)
        else:
            # Strategy B: infer interior from building footprint
            free_now = (raw_grid == 0).astype(np.uint8)
            border_mask = np.zeros_like(free_now, dtype=bool)
            border_mask[0, :] = border_mask[-1, :] = True
            border_mask[:, 0] = border_mask[:, -1] = True
            labeled_free, _ = _scilabel(free_now, structure=np.ones((3, 3), dtype=np.int32))
            border_lbls = set(np.unique(labeled_free[border_mask]).tolist()); border_lbls.discard(0)
            border_free_px = sum(int((labeled_free == lbl).sum()) for lbl in border_lbls)
            total_free_px = int(free_now.sum())
            border_ratio = border_free_px / max(total_free_px, 1)
            print(f"[OccMap] Strategy B: border-free ratio = {100.0*border_ratio:.1f}%.")

            if border_ratio < 0.15:
                print("[OccMap] Walls already enclose the scene; skip morph.")
                interior_mask = free_now
            else:
                building_seed = ((raw_grid == 1) | (furniture_mask == 1)).astype(bool)
                bridge_px = max(1, int(round(0.4 / scale_m)))
                building_blob = binary_dilation(
                    building_seed,
                    structure=np.ones((2*bridge_px+1, 2*bridge_px+1), dtype=bool))
                close_px = max(1, int(round(0.5 / scale_m)))
                building_solid = binary_closing(
                    building_blob,
                    structure=np.ones((2*close_px+1, 2*close_px+1), dtype=bool))
                building_solid = binary_fill_holes(building_solid)

                outside_cands = (~building_solid).astype(np.uint8)
                lbl_out, n_out = _scilabel(outside_cands,
                                           structure=np.ones((3, 3), dtype=np.int32))
                exterior_mask = np.zeros_like(outside_cands, dtype=bool)
                if n_out > 0:
                    border_out_lbls = set(np.unique(lbl_out[border_mask]).tolist())
                    border_out_lbls.discard(0)
                    for lbl in border_out_lbls:
                        exterior_mask |= (lbl_out == lbl)
                interior_mask = ((~exterior_mask) & (raw_grid == 0)).astype(np.uint8)

                labeled, n_lbl = _scilabel(interior_mask,
                                           structure=np.ones((3, 3), dtype=np.int32))
                if n_lbl > 1:
                    sizes = [(lbl, int((labeled == lbl).sum()))
                             for lbl in range(1, n_lbl + 1)]
                    sizes.sort(key=lambda x: -x[1])
                    threshold = max(50, int(0.1 * sizes[0][1]))
                    keep = {lbl for lbl, sz in sizes if sz >= threshold}
                    interior_mask = np.isin(labeled, list(keep)).astype(np.uint8)
                    print(f"[OccMap] Kept {len(keep)} interior components.")

        interior_px = int(interior_mask.sum())
        total_free  = int((raw_grid == 0).sum())
        if total_free == 0 or interior_px / max(total_free, 1) < 0.05:
            print(f"[OccMap] WARN: interior too small "
                  f"({100.0*interior_px/max(total_free,1):.1f}%); falling back.")
            interior_mask = (raw_grid == 0).astype(np.uint8)

        print(f"[OccMap] Interior: {interior_px}/{total_free} free px "
              f"({100.0*interior_px/max(total_free,1):.1f}%).")
        raw_grid[interior_mask == 0] = 1

        # ── robot-radius inflation via EDT (matches load_2d_map) ─────────
        if robot_r > 0:
            dist_m = distance_transform_edt(raw_grid == 0, sampling=scale_m)
            grid = (dist_m <= robot_r).astype(np.uint8)
        else:
            grid = raw_grid.copy()

        print(f"[OccMap] Loaded {sem_json} → grid {w}×{h} px, "
              f"scale={scale_m}m, obstacles={int(grid.sum())}")
        return (grid, max_x, max_y, scale_m)
    except Exception as e:
        print(f"[OccMap] Failed to load: {e}")
        return None


def _occ_is_free(occ, ix: float, iy: float) -> bool:
    """Return True if Isaac-world point (ix,iy) is obstacle-free in the occupancy grid."""
    grid, max_x, max_y, scale = occ
    px = int(round((max_x + ix) / scale))
    py = int(round((max_y + iy) / scale))
    H, W = grid.shape
    return 0 <= px < W and 0 <= py < H and grid[py, px] == 0


def find_standoff_position(
    nm,
    target_pos: np.ndarray,
    robot_pos: np.ndarray,
    occ=None,
) -> Optional[np.ndarray]:
    """Find a valid standoff position at TRACKING_DIST_MIN–FOLLOW_DIST from target.
    occ (occupancy map) is the primary obstacle filter; navmesh snap is secondary.
    Angular sweep starts at the robot's current bearing to minimise lateral movement."""
    robot_angle = math.atan2(
        float(robot_pos[1]) - float(target_pos[1]),
        float(robot_pos[0]) - float(target_pos[0]),
    )
    offsets = [0,
               math.pi / 6, -math.pi / 6,
               math.pi / 3, -math.pi / 3,
               math.pi / 2, -math.pi / 2,
               2 * math.pi / 3, -2 * math.pi / 3,
               5 * math.pi / 6, -5 * math.pi / 6,
               math.pi]
    for radius in [TRACKING_DIST_MIN, FOLLOW_DIST]:
        for offset in offsets:
            angle = robot_angle + offset
            cand = np.array([
                float(target_pos[0]) + radius * math.cos(angle),
                float(target_pos[1]) + radius * math.sin(angle),
                float(target_pos[2]),
            ], dtype=np.float64)
            if occ is not None and not _occ_is_free(occ, cand[0], cand[1]):
                continue
            cand_nm = snap_to_navmesh(nm, cand, search_radius=0.8)
            if cand_nm is None:
                continue
            if float(np.linalg.norm(cand_nm[:2] - robot_pos[:2])) < 0.15:
                continue
            return cand_nm
    return None


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


def _robot_pretract_step(robot, target_pos_xyz: np.ndarray) -> Tuple[float, float]:
    """Simple proportional controller used during pre-tracking warmup.
    Orients the robot toward target spawn and approaches to FOLLOW_DIST.
    Returns (linear, angular) applied this step."""
    robot_pos, robot_yaw = get_robot_pose(robot)
    dx = float(target_pos_xyz[0]) - float(robot_pos[0])
    dy = float(target_pos_xyz[1]) - float(robot_pos[1])
    dist = math.hypot(dx, dy)
    desired_yaw = math.atan2(dy, dx) if dist > 1e-3 else robot_yaw
    yaw_err = math.atan2(math.sin(desired_yaw - robot_yaw),
                          math.cos(desired_yaw - robot_yaw))
    angular = float(np.clip(ARGS.control_gain_angular * yaw_err,
                             -ARGS.max_angular_vel, ARGS.max_angular_vel))
    if dist > FOLLOW_DIST:
        linear = float(np.clip(ARGS.control_gain_linear * (dist - FOLLOW_DIST),
                                0.0, ARGS.max_linear_vel * 0.6))
    else:
        linear = 0.0
    _apply_drive(robot, linear, angular)
    return linear, angular


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


_CMDS_DONE_LAST_VAL: Optional[bool] = None  # tracks last-seen value to suppress repeated logs

def _read_commands_done(char_prim_path: str, step: int = -1) -> bool:
    """Return True when CharacterBehavior has set commandsDone=True on the SkelRoot."""
    global _CMDS_DONE_LAST_VAL
    try:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(char_prim_path)
        if not prim or not prim.IsValid():
            return False
        skelroot = _find_skelroot(prim) or prim
        attr = skelroot.GetAttribute("omni:scripting:commandsDone")
        if not attr or not attr.IsValid():
            val = False
        else:
            val = bool(attr.Get())
        if val != _CMDS_DONE_LAST_VAL:
            print(f"[cmdsDone] step={step} prim={char_prim_path} "
                  f"skelroot={skelroot.GetPath()} attr_exists={attr.IsValid() if attr else False} "
                  f"val={val} (was {_CMDS_DONE_LAST_VAL})")
            _CMDS_DONE_LAST_VAL = val
        return val
    except Exception as e:
        print(f"[cmdsDone] step={step} ERROR: {e}")
        return False


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
    global _KIN_POS, _KIN_YAW, _KIN_BLOCKED_TOTAL
    dt = ARGS.physics_dt

    # Integrate desired state — independent of physics.
    _KIN_YAW += angular * dt
    new_x = _KIN_POS[0] + linear * math.cos(_KIN_YAW) * dt
    new_y = _KIN_POS[1] + linear * math.sin(_KIN_YAW) * dt

    # ── Obstacle guard: refuse to enter an occupied occ cell ──────────
    # This prevents backward recovery from driving the kinematic body
    # into walls and generating high contact forces → collision abort.
    if _GLOBAL_OCC is not None and linear != 0.0:
        if not _occ_is_free(_GLOBAL_OCC, new_x, new_y):
            new_x = _KIN_POS[0]
            new_y = _KIN_POS[1]
            _KIN_BLOCKED_TOTAL += 1

    _KIN_POS[0] = new_x
    _KIN_POS[1] = new_y
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


def save_occ_debug(
    occ,
    robot_pos: np.ndarray,
    robot_yaw: float,
    target_pos: np.ndarray,
    nav_goal,
    path,
    step_idx: int,
    save_dir: str,
    episode_id: int,
    mode: str = "",
):
    """Save occupancy-map debug frame: obstacles + robot/target/nav_goal/path overlay."""
    if occ is None:
        return
    try:
        from PIL import Image, ImageDraw

        grid, max_x, max_y, scale = occ
        H, W = grid.shape

        def to_px(ix, iy):
            px = int(round((max_x + float(ix)) / scale))
            py = int(round((max_y + float(iy)) / scale))
            # Flip y so the image matches matplotlib origin="lower" (y increases upward)
            return px, (H - 1 - py)

        # Full-map view (matches generate_episode_fast_tracking.py save_vis)
        vis = np.where(grid[::-1, :][..., None] == 1,
                       np.array([50, 50, 50], dtype=np.uint8),
                       np.array([240, 240, 240], dtype=np.uint8)).astype(np.uint8)
        img  = Image.fromarray(vis)
        draw = ImageDraw.Draw(img)

        def local(ix, iy):
            return to_px(ix, iy)

        DOT = max(4, int(0.25 / scale))

        # Tracking range dashed circles around target
        for r_m, col in [(TRACKING_DIST_MIN, (220, 60, 60)),
                         (float(ARGS.tracking_dist_max), (60, 180, 60))]:
            r_p = max(1, int(r_m / scale))
            tx, ty = local(target_pos[0], target_pos[1])
            for deg in range(0, 360, 12):
                a0, a1 = math.radians(deg), math.radians(deg + 6)
                p0 = (int(tx + r_p * math.cos(a0)), int(ty - r_p * math.sin(a0)))
                p1 = (int(tx + r_p * math.cos(a1)), int(ty - r_p * math.sin(a1)))
                draw.line([p0, p1], fill=col, width=1)

        # Navmesh path
        if path is not None and len(path) > 1:
            pts = [local(p[0], p[1]) for p in path]
            for j in range(len(pts) - 1):
                draw.line([pts[j], pts[j + 1]], fill=(0, 200, 255), width=2)
            for pt in pts:
                draw.ellipse([pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2],
                             fill=(0, 160, 220))

        # Nav goal: orange circle + crosshair
        if nav_goal is not None:
            gx, gy = local(nav_goal[0], nav_goal[1])
            d = DOT + 2
            draw.ellipse([gx-d, gy-d, gx+d, gy+d],
                         fill=(255, 165, 0), outline=(180, 90, 0), width=2)
            draw.line([gx - d - 2, gy, gx + d + 2, gy], fill=(180, 90, 0), width=2)
            draw.line([gx, gy - d - 2, gx, gy + d + 2], fill=(180, 90, 0), width=2)

        # Target: green circle
        tx, ty = local(target_pos[0], target_pos[1])
        draw.ellipse([tx - DOT, ty - DOT, tx + DOT, ty + DOT],
                     fill=(0, 210, 0), outline=(0, 130, 0), width=2)

        # Robot: blue circle + heading arrow
        rx, ry = local(robot_pos[0], robot_pos[1])
        draw.ellipse([rx - DOT, ry - DOT, rx + DOT, ry + DOT],
                     fill=(30, 120, 255), outline=(0, 60, 190), width=2)
        arr_len = DOT + 6
        draw.line([rx, ry,
                   int(rx + arr_len * math.cos(robot_yaw)),
                   int(ry - arr_len * math.sin(robot_yaw))],
                  fill=(255, 255, 0), width=2)

        # Mode + step text overlay
        label = f"{mode} #{step_idx:05d}" if mode else f"#{step_idx:05d}"
        draw.rectangle([0, 0, len(label) * 6 + 4, 13], fill=(0, 0, 0))
        draw.text((2, 1), label, fill=(255, 255, 80))

        episode_dir = os.path.join(save_dir, f"episode_{episode_id:04d}")
        occ_dir = os.path.join(episode_dir, "debug_occ")
        os.makedirs(occ_dir, exist_ok=True)
        img.save(os.path.join(occ_dir, f"{step_idx:05d}.png"))
    except Exception as e:
        print(f"[WARN] OCC debug save failed: {e}")


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
        # scene_id may be "0001_839920"; the USDA is named "839920.usda" (numeric suffix only)
        usda_stem = scene_id.split("_")[-1] if "_" in scene_id else scene_id
        usda_path = os.path.join(usda_root, f"{usda_stem}.usda")
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

    # Phase 1: Static physics warmup — physics settles, nothing moves.
    for _ in range(40):
        world.step(render=False)
        update_chase_camera(robot)
        simulation_app.update()

    # Phase 2: Recolor while all characters are still at spawn positions.
    recolor_episode_characters(episode, char_paths)

    # Phase 3: Robot pre-tracks toward target spawn; characters still idle.
    target_spawn = episode["characters"]["spawn_positions"]["Character"]["pos"]
    target_spawn_np = np.array(target_spawn, dtype=np.float64)
    print("[Setup] Robot pre-tracking toward target spawn (characters idle)...")
    for _ in range(ROBOT_PRE_TRACK_FRAMES):
        lin, ang = _robot_pretract_step(robot, target_spawn_np)
        world.step(render=False)
        if ROBOT_DRIVE_MODE == "kinematic":
            kinematic_move(robot, lin, ang)
        update_chase_camera(robot)
        simulation_app.update()

    # Phase 4: Attach behavior scripts with barrier CLOSED so characters don't
    # start executing commands during the 30-frame internal warmup inside
    # attach_behavior_scripts_to_characters.  The main loop opens the barrier.
    carb.settings.get_settings().set("/exts/people_sim/character_barrier_open", False)
    attach_behavior_scripts_to_characters(simulation_app)

    return char_paths, target_prim_path


def main() -> int:
    # ── Infer scene_id from episode_dir basename when not supplied ──────────
    episode_root = ARGS.episode_dir
    scene_id = ARGS.scene_id
    if scene_id is None:
        scene_id = os.path.basename(os.path.normpath(episode_root))
        episode_root = os.path.dirname(os.path.normpath(episode_root))
        print(f"[Paths] scene_id inferred from episode_dir: '{scene_id}'")

    # ── Auto-derive semantic_map_json from scene_id when not supplied ───────
    sem_map_json = ARGS.semantic_map_json
    if sem_map_json is None:
        sem_map_json = os.path.join(
            ARGS.semantic_maps_root,
            f"2D_Semantic_Map_{scene_id}_Complete.json",
        )
        print(f"[Paths] semantic_map_json auto-derived: '{sem_map_json}'")

    try:
        usda_path, ep_dir = resolve_scene_paths(
            usda_root=ARGS.usda_root,
            usda_override=ARGS.usda,
            episode_root=episode_root,
            scene_id=scene_id,
        )
    except FileNotFoundError as e:
        print(f"[FATAL] {e}")
        return 1
    print(f"[Paths] scene_id   = {scene_id}")
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
        semantic_map_json=sem_map_json,
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

    occ = None
    if sem_map_json and os.path.exists(sem_map_json):
        occ = _load_occ_grid(sem_map_json, ARGS.occ_scale, ARGS.robot_radius_2d)
    elif sem_map_json:
        print(f"[OccMap] WARN: semantic_map_json not found: '{sem_map_json}', skipping.")
    global _GLOBAL_OCC
    _GLOBAL_OCC = occ   # make occ grid accessible to kinematic_move & helpers

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

    setup_robot_contact_report(omni.usd.get_context().get_stage())

    world.reset()
    update_sim(10)

    setup_robot_contact_sensor(world)

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
            # Phase 1: Static physics warmup.
            print("[Episode 0] Phase 1: static warmup (40 frames)...")
            for _ in range(40):
                world.step(render=False)
                update_chase_camera(robot)
                simulation_app.update()

            # Phase 2: Recolor while everything is static.
            recolor_episode_characters(episode, pre_char_paths)

            # Phase 3: Robot pre-tracks toward target spawn; characters still idle.
            target_spawn = episode["characters"]["spawn_positions"]["Character"]["pos"]
            target_spawn_np = np.array(target_spawn, dtype=np.float64)
            print("[Episode 0] Phase 3: robot pre-tracking toward target spawn...")
            for _ in range(ROBOT_PRE_TRACK_FRAMES):
                lin, ang = _robot_pretract_step(robot, target_spawn_np)
                world.step(render=False)
                if ROBOT_DRIVE_MODE == "kinematic":
                    kinematic_move(robot, lin, ang)
                update_chase_camera(robot)
                simulation_app.update()

            # Phase 4: Attach behavior scripts with barrier CLOSED (same as setup_episode_full).
            print("[Episode 0] Phase 4: attaching behavior scripts (barrier closed; "
                  "main loop will open it)...")
            carb.settings.get_settings().set(
                "/exts/people_sim/character_barrier_open", False)
            attach_behavior_scripts_to_characters(simulation_app)
        else:
            with open(ep_path, "r") as f:
                episode = json.load(f)["episode"]
            char_paths, target_prim_path = setup_episode_full(
                episode, char_pool, robot, world
            )
            if target_prim_path is None:
                print("[ERROR] No target character resolvable, skip episode")
                continue

        # Build pedestrian trajectory cache for proactive avoidance.
        build_char_traj_cache(episode, char_paths, target_prim_path)
        build_target_traj_cache(episode, char_paths, target_prim_path)
        reset_collision_state()

        # Open the character barrier — characters start executing commands now,
        # guaranteed AFTER setup warmup frames so commandsDone is never stale.
        global _CMDS_DONE_LAST_VAL
        _CMDS_DONE_LAST_VAL = None  # reset log-suppression tracker for new episode
        carb.settings.get_settings().set("/exts/people_sim/character_barrier_open", True)
        print(f"[EP{ep_id}] Character barrier opened — commands start now.")

        step           = 0
        tracking_steps = 0
        collision_count = 0
        distances:     List[float] = []
        heading_errors: List[float] = []
        done_reason = "max_steps"
        _viz_mode  = "INIT"
        _prev_viz_mode = "INIT"   # for mode-transition logging
        ep_min_ped_dist = float("inf")   # track closest non-target ped distance
        # Character-done detection state
        _tgt_done      = False
        _tgt_done_step = -1

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
            "_last_mode": "",   # for mode-switch smoother reset
        }

        while step < ARGS.max_steps:
            robot_pos, robot_yaw   = get_robot_pose(robot)
            target_pos, _ = get_character_pose(target_prim_path)

            # ── Abnormal state detection ──────────────────────────────────
            # Kinematic drift or a runaway backward command can land the robot
            # inside an obstacle cell.  Detect and recover to nearest free cell
            # so subsequent navmesh plans don't start from an invalid position.
            if ROBOT_DRIVE_MODE == "kinematic" and _detect_in_obstacle(robot_pos, occ):
                nearest_free = _find_nearest_free(robot_pos, occ)
                print(f"[EP{ep_id}] step={step} ABNORMAL STATE: robot "
                      f"({robot_pos[0]:.2f},{robot_pos[1]:.2f}) inside obstacle! "
                      f"_KIN_BLOCKED_TOTAL={_KIN_BLOCKED_TOTAL} "
                      f"nearest_free={nearest_free[:2].tolist() if nearest_free is not None else None}")
                if nearest_free is not None:
                    _KIN_POS[:] = [float(nearest_free[0]), float(nearest_free[1]),
                                   float(ARGS.robot_z_height)]
                    snap_quat = np.array([math.cos(robot_yaw / 2.0), 0.0, 0.0,
                                          math.sin(robot_yaw / 2.0)], dtype=np.float32)
                    robot.set_world_pose(
                        position=_KIN_POS.astype(np.float32), orientation=snap_quat)
                    print(f"[EP{ep_id}] step={step} Snapped kinematic pos → "
                          f"({nearest_free[0]:.2f},{nearest_free[1]:.2f})")
                pursuit_state["path"] = None   # force replan from clean position

            # Track recent target positions (for velocity-based EVADE fallback).
            _TARGET_POS_HIST.append(target_pos[:2].copy())
            if len(_TARGET_POS_HIST) > _TARGET_POS_HIST_LEN:
                _TARGET_POS_HIST.pop(0)

            # Update pedestrian velocity history and build dynamic OCC.
            update_ped_vel_hist(char_paths, target_prim_path)
            ped_fc = get_all_ped_forecasts(char_paths, target_prim_path)
            dyn_occ = occ_with_ped_forecasts(occ, ped_fc)

            # ── Collision check ──────────────────────────────────────────
            contact_force = read_robot_contact_force()
            if contact_force >= COLLISION_FORCE_THRESHOLD:
                collision_count += 1
                if step % 5 == 0 or collision_count == 1:
                    print(f"[EP{ep_id}] step={step} COLLISION "
                          f"force={contact_force:.1f} N count={collision_count}")
                if collision_count >= 3:
                    done_reason = "collision"
                    _stop_drive(robot)
                    for _ in range(3):
                        world.step(render=False)
                        update_chase_camera(robot)
                        simulation_app.update()
                    break

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

            # ── Per-step state log (always printed so logs are parseable) ──
            _tgt_approach_vel = 0.0   # positive = target approaching robot
            _tgt_vel_dbg = ""
            if len(_TARGET_POS_HIST) >= 3:
                _sdt = ARGS.physics_dt * ARGS.decimation
                _tv = (_TARGET_POS_HIST[-1] - _TARGET_POS_HIST[-3]) / (2 * _sdt)
                _tv_spd = float(np.linalg.norm(_tv))
                _to_r = robot_pos[:2] - target_pos[:2]
                _tgt_approach_vel = float(np.dot(_tv, _to_r / max(float(np.linalg.norm(_to_r)), 0.1)))
                _tgt_vel_dbg = f" tgt_v={_tv_spd:.2f}m/s appr={_tgt_approach_vel:+.2f}m/s"

            # ── Non-target pedestrian: distance + angle + camera FOV ────
            _ped_min_step = float("inf")
            _ped_detail_parts: List[str] = []
            for _pn, _pp_path in char_paths.items():
                if _pp_path == target_prim_path:
                    continue
                _pp_pos, _ = get_character_pose(_pp_path)
                _pd = float(np.linalg.norm(_pp_pos[:2] - robot_pos[:2]))
                # Angle relative to robot heading (negative = right, positive = left)
                _pdx = float(_pp_pos[0] - robot_pos[0])
                _pdy = float(_pp_pos[1] - robot_pos[1])
                _pang = math.degrees(math.atan2(
                    math.sin(math.atan2(_pdy, _pdx) - robot_yaw),
                    math.cos(math.atan2(_pdy, _pdx) - robot_yaw)))
                _in_cam = abs(_pang) <= _CAM_FOV_HALF_DEG
                _in_trk = abs(_pang) <= ARGS.tracking_angle_thresh
                _fov_tag = "CAM" if _in_cam else ("TRK" if _in_trk else "OUT")
                _ped_detail_parts.append(
                    f"{_pn}:d={_pd:.2f}m ang={_pang:+.1f}° fov={_fov_tag}")
                if _pd < _ped_min_step:
                    _ped_min_step = _pd
            if _ped_min_step < float("inf"):
                ep_min_ped_dist = min(ep_min_ped_dist, _ped_min_step)

            # Target angle relative to robot heading (separate from dist which is already tracked)
            _tgt_ang = math.degrees(math.atan2(
                math.sin(math.atan2(float(target_pos[1]-robot_pos[1]),
                                    float(target_pos[0]-robot_pos[0])) - robot_yaw),
                math.cos(math.atan2(float(target_pos[1]-robot_pos[1]),
                                    float(target_pos[0]-robot_pos[0])) - robot_yaw)))
            _tgt_in_cam = abs(_tgt_ang) <= _CAM_FOV_HALF_DEG

            _ped_dbg = (f" | ped_min={_ped_min_step:.2f}m ep_min={ep_min_ped_dist:.2f}m"
                        f" [{' | '.join(_ped_detail_parts)}]"
                        if _ped_detail_parts else "")

            print(f"[ST] ep={ep_id} s={step:4d} {_viz_mode:<12s} "
                  f"R({robot_pos[0]:+7.3f},{robot_pos[1]:+7.3f}) "
                  f"yaw{math.degrees(robot_yaw):+6.1f}° "
                  f"T({target_pos[0]:+7.3f},{target_pos[1]:+7.3f}) "
                  f"dist={dist_to_target:5.3f}m ang={_tgt_ang:+5.1f}° "
                  f"tgt_cam={'Y' if _tgt_in_cam else 'N'}"
                  f"{_tgt_vel_dbg}{_ped_dbg}")

            # ── Mode-transition log (uses mode from end of previous step) ──
            if _viz_mode != _prev_viz_mode:
                print(f"[MODE] ep={ep_id} s={step:4d} "
                      f"{_prev_viz_mode} → {_viz_mode}  "
                      f"dist={dist_to_target:.3f}m ang_err={angle_err_deg:.1f}°")
                _prev_viz_mode = _viz_mode

            pursuit_state["pose_history"].append(
                (robot_pos[:2].copy(), float(robot_yaw)))
            if len(pursuit_state["pose_history"]) > STUCK_WINDOW:
                pursuit_state["pose_history"].pop(0)

            # ── Character-done detection ──────────────────────────────────
            # Read the commandsDone flag written by CharacterBehavior when its
            # command queue empties — no velocity heuristic needed.
            if not _tgt_done and _read_commands_done(target_prim_path, step):
                _tgt_done      = True
                _tgt_done_step = step
                print(f"[EP{ep_id}] step={step} TARGET DONE "
                      f"(commandsDone flag set, dist={dist_to_target:.2f}m)")
            if _tgt_done and (step - _tgt_done_step) >= DONE_GRACE_STEPS:
                done_reason = "char_done"
                print(f"[EP{ep_id}] step={step} EPISODE END: char_done "
                      f"(grace {DONE_GRACE_STEPS} steps elapsed, "
                      f"final dist={dist_to_target:.2f}m)")
                break

            if pursuit_state["recovery_steps_left"] > 0:
                _viz_mode = "RECOVERY"
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
                    save_occ_debug(occ, robot_pos, robot_yaw, target_pos,
                                   pursuit_state.get("last_goal"), pursuit_state.get("path"),
                                   step, ARGS.image_save_dir, ep_id, _viz_mode)

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

                # For kinematic robots joint velocities are zeroed every step, so
                # get_joint_velocities() always returns ~0 and can never indicate
                # commanded motion.  Use the smoothed command history instead.
                if ROBOT_DRIVE_MODE == "kinematic":
                    cmd_magnitude = abs(pursuit_state["prev_linear"]) + abs(
                        pursuit_state["prev_angular"])
                else:
                    try:
                        cur_vels = robot.get_joint_velocities()
                        cmd_magnitude = float(np.max(np.abs(cur_vels)))
                    except Exception:
                        cmd_magnitude = 0.0

                # For kinematic mode a lower threshold is appropriate because
                # prev_linear/angular are smoothed (max ~1.2 + 2.5 = 3.7 total).
                _stuck_thresh = 0.1 if ROBOT_DRIVE_MODE == "kinematic" else STUCK_CMD_THRESH
                commanded_motion = cmd_magnitude > _stuck_thresh
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

                    # Check backward clearance before triggering reverse recovery.
                    # If a wall is behind the robot, use spin-only to avoid collision.
                    back_ok = _backward_clear(robot_pos, robot_yaw, occ,
                                              probe_dist=0.60)
                    recovery_lin = RECOVERY_LINEAR if back_ok else 0.0
                    if not back_ok:
                        print(f"[EP{ep_id}] step={step} STUCK: backward BLOCKED "
                              f"(occ grid), switching to spin-only recovery "
                              f"(ang={ang_sign * RECOVERY_ANGULAR_MAG:.2f})")

                    # Adaptive duration: spin just enough to face the waypoint/target,
                    # plus a small buffer.  Fixed RECOVERY_DURATION (8 steps) caused
                    # severe overshoot when yaw_err was small (e.g. 30°) — robot ended
                    # up 177° away from target after recovery.
                    _deg_per_step = math.degrees(
                        RECOVERY_ANGULAR_MAG * ARGS.physics_dt * ARGS.decimation)
                    _needed = int(abs(math.degrees(yaw_err)) / max(_deg_per_step, 1.0))
                    _adaptive_dur = max(3, min(RECOVERY_DURATION, _needed + 2))

                    pursuit_state["recovery_steps_left"] = _adaptive_dur
                    pursuit_state["recovery_linear"]    = recovery_lin
                    pursuit_state["recovery_angular"]   = ang_sign * RECOVERY_ANGULAR_MAG
                    pursuit_state["pose_history"].clear()
                    print(f"[EP{ep_id}] step={step} STUCK detected "
                          f"(pos_delta={pos_delta:.3f}m, "
                          f"yaw_delta={yaw_delta:.1f}deg, "
                          f"cmd_mag={cmd_magnitude:.2f}). "
                          f"Recovery: lin={recovery_lin:.2f} "
                          f"ang={ang_sign * RECOVERY_ANGULAR_MAG:.2f} "
                          f"steps={_adaptive_dur} (yaw_err={math.degrees(yaw_err):+.1f}°) "
                          f"back_ok={back_ok} "
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

            # ── Oracle evasion: step aside when target walks toward robot ──
            # Multi-horizon scan: when target is already close, a single 2-second
            # prediction overshoots (target has passed the robot by then).
            # Scan 4 horizons; evade if ANY puts the target within TRACKING_DIST_MIN.
            # t_reach caps the adaptive horizon so we detect close fast-approaching targets.
            if (not pursuit_state["recovery_steps_left"]
                    and _TARGET_TRAJ_CACHE
                    and dist_to_target < FOLLOW_DIST + 1.5):
                t_reach = dist_to_target / max(PED_WALK_SPEED_M_S, 0.3)
                _best_tf, _min_fd = None, float("inf")
                for _h in sorted({0.5, 1.0,
                                  float(min(t_reach, TARGET_EVADE_LOOKAHEAD_S)),
                                  TARGET_EVADE_LOOKAHEAD_S}):
                    _h = float(max(0.3, min(_h, TARGET_EVADE_LOOKAHEAD_S)))
                    _tf = predict_target_future_pos(target_pos, horizon_s=_h)
                    if _tf is not None:
                        _d = float(np.linalg.norm(_tf[:2] - robot_pos[:2]))
                        if _d < _min_fd:
                            _min_fd, _best_tf = _d, _tf
                        if _d < TRACKING_DIST_MIN:
                            break
                evade_future_dist = _min_fd
                evade_tgt_future = (_best_tf
                                    if _min_fd < TRACKING_DIST_MIN else None)

                # ── Velocity-history fallback ─────────────────────────────
                # When oracle path is unavailable or multi-horizon still misses
                # (target already very close), use smoothed velocity from history.
                if evade_tgt_future is None and len(_TARGET_POS_HIST) >= 4:
                    _sdt = ARGS.physics_dt * ARGS.decimation
                    _tv2 = ((_TARGET_POS_HIST[-1] - _TARGET_POS_HIST[-4])
                            / (3 * _sdt))
                    _tv_spd2 = float(np.linalg.norm(_tv2))
                    if _tv_spd2 > 0.15:
                        _to_r2 = robot_pos[:2] - target_pos[:2]
                        _dist_r2 = float(np.linalg.norm(_to_r2))
                        _app2 = float(np.dot(
                            _tv2 / _tv_spd2,
                            _to_r2 / max(_dist_r2, 0.1)))
                        # Only trigger if actively approaching (> 0.25 m/s toward robot)
                        if _app2 > 0.25 and dist_to_target < TRACKING_DIST_MIN + 1.0:
                            _t_pred = dist_to_target / max(_app2, 0.3)
                            _vf = np.array([
                                float(target_pos[0]) + float(_tv2[0]) * _t_pred,
                                float(target_pos[1]) + float(_tv2[1]) * _t_pred,
                                float(target_pos[2]),
                            ], dtype=np.float64)
                            _vf_dist = float(np.linalg.norm(_vf[:2] - robot_pos[:2]))
                            if _vf_dist < TRACKING_DIST_MIN:
                                evade_tgt_future = _vf
                                evade_future_dist = _vf_dist
                                print(f"[EP{ep_id}] step={step} EVADE-VEL fallback: "
                                      f"approach={_app2:.2f}m/s dist={dist_to_target:.2f}m "
                                      f"pred_dist={_vf_dist:.2f}m")

                if evade_tgt_future is not None:
                    evade_goal = _find_evade_position(
                        nm, target_pos, evade_tgt_future, robot_pos, occ)
                    if (evade_goal is not None and
                            float(np.linalg.norm(
                                evade_goal[:2] - robot_pos[:2])) > 0.45):
                        _viz_mode = "EVADE"
                        robot_on_nm = snap_to_navmesh(nm, robot_pos, search_radius=1.0)
                        if robot_on_nm is None:
                            robot_on_nm = snap_to_navmesh(nm, robot_pos, search_radius=3.0)
                        start_xyz = (robot_on_nm if robot_on_nm is not None
                                     else robot_pos)
                        evade_path = plan_navmesh_path(nm, start_xyz, evade_goal)
                        if evade_path and len(evade_path) >= 1:
                            pursuit_state["path"]           = evade_path
                            pursuit_state["path_idx"]       = 0
                            pursuit_state["last_plan_step"] = step
                            pursuit_state["last_goal"]      = evade_goal.copy()
                            carrot, new_idx = pick_carrot(
                                evade_path, 0, robot_pos, lookahead=LOOKAHEAD_DIST)
                            pursuit_state["path_idx"] = new_idx
                            linear, angular = compute_control(
                                robot_pos, robot_yaw, carrot,
                                ARGS.control_gain_linear, ARGS.control_gain_angular,
                                ARGS.max_linear_vel, ARGS.max_angular_vel,
                                desired_distance=0.0, pos_deadzone=0.15,
                                angle_deadzone_deg=4.0)
                            angular = _los_blend_angular(
                                robot_pos, robot_yaw, target_pos, angular)
                            linear, angular = smooth_cmd(
                                pursuit_state, linear, angular, SMOOTH_ALPHA_NORMAL)
                            linear, angular = apply_ped_reactive_avoidance(
                                robot_pos, robot_yaw, linear, angular,
                                char_paths, target_prim_path)
                            if linear < 0 and not _backward_clear(
                                    robot_pos, robot_yaw, occ):
                                linear = 0.0
                            _apply_drive(robot, linear, angular)
                            pursuit_state["_last_mode"] = "EVADE"
                            if step % 10 == 0:
                                print(f"[EP{ep_id}] step={step} EVADE "
                                      f"future_dist={evade_future_dist:.2f}m "
                                      f"t_reach={t_reach:.1f}s "
                                      f"goal={evade_goal[:2].tolist()} "
                                      f"lin={linear:.2f} ang={angular:.2f}")
                            for _ in range(ARGS.decimation):
                                world.step(render=False)
                                if ROBOT_DRIVE_MODE == "kinematic":
                                    kinematic_move(robot, linear, angular)
                                update_chase_camera(robot)
                                simulation_app.update()
                            if ARGS.save_images and step % ARGS.save_image_every == 0:
                                save_rgb_depth(cam, step, ARGS.image_save_dir, ep_id)
                                save_occ_debug(occ, robot_pos, robot_yaw, target_pos,
                                               evade_goal, evade_path,
                                               step, ARGS.image_save_dir, ep_id, "EVADE")
                            step += 1
                            continue

            handled_by_detour = False
            if not visible:
                vantage = find_visible_vantage_point(
                    nm, target_pos, target_prim_path, robot_pos)

                if vantage is not None:
                    _viz_mode = "DETOUR"
                    robot_on_nm = snap_to_navmesh(nm, robot_pos, search_radius=1.0)
                    if robot_on_nm is None:
                        robot_on_nm = snap_to_navmesh(nm, robot_pos, search_radius=3.0)
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
                        # LoS: partial blend toward target while heading to vantage
                        angular = _los_blend_angular(robot_pos, robot_yaw,
                                                      target_pos, angular,
                                                      weight=LOS_ANG_TARGET_WEIGHT * 0.6)
                        linear, angular = smooth_cmd(
                            pursuit_state, linear, angular,
                            SMOOTH_ALPHA_NORMAL)
                        linear, angular = apply_ped_reactive_avoidance(
                            robot_pos, robot_yaw, linear, angular,
                            char_paths, target_prim_path)
                        # Backward safety: clamp negative linear if path blocked
                        if linear < 0 and not _backward_clear(robot_pos, robot_yaw, occ):
                            if step % 10 == 0:
                                print(f"[EP{ep_id}] step={step} DETOUR: "
                                      f"backward blocked, clamp lin {linear:.2f}→0")
                            linear = 0.0
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
                            save_occ_debug(occ, robot_pos, robot_yaw, target_pos,
                                           pursuit_state.get("last_goal"), pursuit_state.get("path"),
                                           step, ARGS.image_save_dir, ep_id, _viz_mode)
                        step += 1
                        handled_by_detour = True
                else:
                    if step % 20 == 0:
                        print(f"[EP{ep_id}] step={step} OCCLUDED, "
                              f"no vantage found — falling through to direct.")

            if handled_by_detour:
                continue

            dx = float(target_pos[0] - robot_pos[0])
            dy = float(target_pos[1] - robot_pos[1])
            yaw_err = math.atan2(math.sin(math.atan2(dy, dx) - robot_yaw),
                                 math.cos(math.atan2(dy, dx) - robot_yaw))
            angular_base = float(np.clip(ARGS.control_gain_angular * yaw_err,
                                         -ARGS.max_angular_vel, ARGS.max_angular_vel))
            if abs(math.degrees(yaw_err)) < DIRECT_DEADZONE_ANG:
                angular_base = 0.0

            if in_window:
                _viz_mode = "TRACK"
                pursuit_state["path"] = None

                # ── Mode-switch smoother reset ────────────────────────────
                # Entering TRACK from another mode: prev_angular carries momentum
                # from navigation that's inappropriate for stand-and-rotate control.
                if pursuit_state["_last_mode"] != "TRACK":
                    pursuit_state["prev_angular"] = 0.0
                    pursuit_state["prev_linear"]  = 0.0

                # ── Distance regulation: small linear when off-FOLLOW_DIST ─
                # Only back up when target is stationary or approaching.
                # When target is receding (appr < -0.3 m/s), suppress negative
                # linear so the robot follows instead of widening the gap.
                dist_err = dist_to_target - FOLLOW_DIST
                _tgt_receding = _tgt_approach_vel < -0.3
                if abs(dist_err) > 0.30:
                    _lin_lo = 0.0 if _tgt_receding else -ARGS.max_linear_vel * 0.25
                    track_linear = float(np.clip(
                        ARGS.control_gain_linear * dist_err * 0.40,
                        _lin_lo,
                        ARGS.max_linear_vel * 0.25))
                else:
                    track_linear = 0.0

                linear, angular = track_linear, angular_base
                _ang_raw_track = angular
                # Use more-damped alpha in TRACK to suppress yaw overshoot/oscillation.
                linear, angular = smooth_cmd(pursuit_state, linear, angular, SMOOTH_ALPHA_TRACK)
                _ang_after_smooth_track = angular
                linear, angular = apply_ped_reactive_avoidance(
                    robot_pos, robot_yaw, linear, angular, char_paths, target_prim_path)
                # Backward safety guard on distance-regulation linear.
                if linear < 0 and not _backward_clear(robot_pos, robot_yaw, occ):
                    linear = 0.0
                # Dead-stop: once aligned, zero angular AND flush the smoother so that
                # exponential decay of prev_angular can't cause left-right oscillation.
                if abs(math.degrees(yaw_err)) < DIRECT_DEADZONE_ANG:
                    angular = 0.0
                    pursuit_state["prev_angular"] = 0.0
                if abs(dist_err) <= 0.30:
                    linear = 0.0
                    pursuit_state["prev_linear"] = 0.0
                _apply_drive(robot, linear, angular)
                pursuit_state["_last_mode"] = "TRACK"
                # Always print TRACK; include angular breakdown every step for turn analysis.
                print(f"[EP{ep_id}] step={step} TRACK "
                      f"dist={dist_to_target:.3f} dist_err={dist_err:+.3f} "
                      f"yaw_err={math.degrees(yaw_err):+.1f}° "
                      f"ang_raw={_ang_raw_track:+.3f} ang_smooth={_ang_after_smooth_track:+.3f} "
                      f"ang_final={angular:+.3f} lin={linear:.3f}")
                for _ in range(ARGS.decimation):
                    world.step(render=False)
                    if ROBOT_DRIVE_MODE == "kinematic":
                        kinematic_move(robot, linear, angular)
                    update_chase_camera(robot)
                    simulation_app.update()
                if ARGS.save_images and step % ARGS.save_image_every == 0:
                    save_rgb_depth(cam, step, ARGS.image_save_dir, ep_id)
                    save_occ_debug(occ, robot_pos, robot_yaw, target_pos,
                                   pursuit_state.get("last_goal"), pursuit_state.get("path"),
                                   step, ARGS.image_save_dir, ep_id, _viz_mode)
                step += 1
                continue

            if dist_to_target < TRACKING_DIST_MIN:
                _viz_mode = "STANDOFF"
                nav_goal = find_standoff_position(nm, target_pos, robot_pos, dyn_occ)
                if nav_goal is None:
                    # ── Sidestep escape ───────────────────────────────────
                    # No valid standoff position found (typically the robot is
                    # backed against a wall).  Instead of freezing in place, try
                    # to dodge sideways: choose the lateral direction (left/right
                    # of the robot-to-target vector) that has a free occ cell,
                    # then turn toward it while moving forward slightly.
                    # This keeps the target in view while escaping the dead zone.
                    _dx_rt = float(target_pos[0] - robot_pos[0])
                    _dy_rt = float(target_pos[1] - robot_pos[1])
                    _dist_rt = math.hypot(_dx_rt, _dy_rt)
                    _rt_norm = (_dx_rt / max(_dist_rt, 1e-6),
                                _dy_rt / max(_dist_rt, 1e-6))
                    # perpendicular: left = CCW 90°, right = CW 90°
                    _perp_L = (-_rt_norm[1],  _rt_norm[0])
                    _perp_R = ( _rt_norm[1], -_rt_norm[0])
                    _probe  = SIDESTEP_CHECK_DIST
                    _free_L = _occ_is_free(occ,
                        robot_pos[0] + _probe * _perp_L[0],
                        robot_pos[1] + _probe * _perp_L[1]) if occ else True
                    _free_R = _occ_is_free(occ,
                        robot_pos[0] + _probe * _perp_R[0],
                        robot_pos[1] + _probe * _perp_R[1]) if occ else True

                    if _free_L or _free_R:
                        # Pick the freer side; break ties by preferring the side
                        # that requires less rotation from current heading.
                        _side = _perp_L if _free_L else _perp_R
                        if _free_L and _free_R:
                            _ang_L = abs(math.degrees(math.atan2(
                                math.sin(math.atan2(_perp_L[1], _perp_L[0]) - robot_yaw),
                                math.cos(math.atan2(_perp_L[1], _perp_L[0]) - robot_yaw))))
                            _ang_R = abs(math.degrees(math.atan2(
                                math.sin(math.atan2(_perp_R[1], _perp_R[0]) - robot_yaw),
                                math.cos(math.atan2(_perp_R[1], _perp_R[0]) - robot_yaw))))
                            _side = _perp_L if _ang_L <= _ang_R else _perp_R
                        _side_yaw = math.atan2(_side[1], _side[0])
                        _side_err = math.atan2(math.sin(_side_yaw - robot_yaw),
                                               math.cos(_side_yaw - robot_yaw))
                        angular = float(np.clip(
                            ARGS.control_gain_angular * _side_err,
                            -ARGS.max_angular_vel, ARGS.max_angular_vel))
                        linear  = SIDESTEP_LINEAR
                        _viz_mode = "SIDESTEP"
                        if step % 10 == 0:
                            _side_name = "L" if _side is _perp_L else "R"
                            print(f"[EP{ep_id}] step={step} SIDESTEP-{_side_name} "
                                  f"dist={dist_to_target:.2f}m "
                                  f"side_err={math.degrees(_side_err):+.1f}° "
                                  f"lin={linear:.2f} ang={angular:.2f}")
                    else:
                        # Both sides blocked — fall back to original: face target, wait
                        _viz_mode = "STANDOFF-FAIL"
                        linear, angular = 0.0, angular_base
                        if step % 20 == 0:
                            print(f"[EP{ep_id}] step={step} STANDOFF-FAIL(both blocked) "
                                  f"dist={dist_to_target:.2f}")

                    pursuit_state["path"] = None
                    linear, angular = smooth_cmd(pursuit_state, linear, angular, SMOOTH_ALPHA_NORMAL)
                    _apply_drive(robot, linear, angular)
                    pursuit_state["_last_mode"] = _viz_mode
                    for _ in range(ARGS.decimation):
                        world.step(render=False)
                        if ROBOT_DRIVE_MODE == "kinematic":
                            kinematic_move(robot, linear, angular)
                        update_chase_camera(robot)
                        simulation_app.update()
                    if ARGS.save_images and step % ARGS.save_image_every == 0:
                        save_rgb_depth(cam, step, ARGS.image_save_dir, ep_id)
                        save_occ_debug(occ, robot_pos, robot_yaw, target_pos,
                                       None, None,
                                       step, ARGS.image_save_dir, ep_id, _viz_mode)
                    step += 1
                    continue
                path_end_dist = 0.0
            else:
                _viz_mode = "APPROACH"
                nav_goal = snap_to_navmesh(nm, target_pos, search_radius=2.0)
                if nav_goal is None:
                    nav_goal = target_pos.copy()
                path_end_dist = TRACKING_DIST_MIN

            need_replan = (
                pursuit_state["path"] is None
                or (step - pursuit_state["last_plan_step"]) >= REPLAN_PERIOD
                or (pursuit_state["last_goal"] is not None
                    and float(np.linalg.norm(
                        nav_goal[:2] - pursuit_state["last_goal"][:2])) > REPLAN_TARGET_THR)
                or pursuit_state["path_idx"] >= len(pursuit_state["path"]) - 1
            )

            if need_replan:
                robot_on_nm = snap_to_navmesh(nm, robot_pos, search_radius=1.0)
                if robot_on_nm is None:
                    robot_on_nm = snap_to_navmesh(nm, robot_pos, search_radius=3.0)
                start_xyz = robot_on_nm if robot_on_nm is not None else robot_pos
                new_path = plan_navmesh_path(nm, start_xyz, nav_goal)
                if new_path is not None and len(new_path) >= 1:
                    pursuit_state["path"]            = new_path
                    pursuit_state["path_idx"]        = 0
                    pursuit_state["last_plan_step"]  = step
                    pursuit_state["last_goal"]       = nav_goal.copy()
                    pursuit_state["plan_fail_count"] = 0
                else:
                    if _viz_mode == "STANDOFF":
                        # STANDOFF planning failure is transient (target too close);
                        # fall back to angular-only control without counting as a failure.
                        pursuit_state["path"] = None
                        linear = 0.0
                        angular = angular_base
                        linear, angular = smooth_cmd(
                            pursuit_state, linear, angular, SMOOTH_ALPHA_NORMAL)
                        _apply_drive(robot, linear, angular)
                        pursuit_state["_last_mode"] = "STANDOFF-PLANFAIL"
                        if step % 20 == 0:
                            print(f"[EP{ep_id}] step={step} STANDOFF-PLANFAIL "
                                  f"dist={dist_to_target:.2f} ang={angular:.2f}")
                        for _ in range(ARGS.decimation):
                            world.step(render=False)
                            if ROBOT_DRIVE_MODE == "kinematic":
                                kinematic_move(robot, linear, angular)
                            update_chase_camera(robot)
                            simulation_app.update()
                        if ARGS.save_images and step % ARGS.save_image_every == 0:
                            save_rgb_depth(cam, step, ARGS.image_save_dir, ep_id)
                            save_occ_debug(occ, robot_pos, robot_yaw, target_pos,
                                           None, None,
                                           step, ARGS.image_save_dir, ep_id, "STANDOFF-PF")
                        step += 1
                        continue
                    pursuit_state["plan_fail_count"] += 1
                    if pursuit_state["plan_fail_count"] >= MAX_PLAN_FAILS:
                        print(f"[EP{ep_id}] Planning failed {pursuit_state['plan_fail_count']}x, ending.")
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
                        save_occ_debug(occ, robot_pos, robot_yaw, target_pos,
                                       pursuit_state.get("last_goal"), None,
                                       step, ARGS.image_save_dir, ep_id, "PLAN-FAIL")
                    step += 1
                    continue

            carrot, new_idx = pick_carrot(
                pursuit_state["path"], pursuit_state["path_idx"], robot_pos,
                lookahead=LOOKAHEAD_DIST)
            pursuit_state["path_idx"] = new_idx
            at_path_end = new_idx >= len(pursuit_state["path"]) - 1
            if at_path_end:
                carrot = nav_goal
            linear, angular = compute_control(
                robot_pos, robot_yaw, carrot,
                ARGS.control_gain_linear, ARGS.control_gain_angular,
                ARGS.max_linear_vel, ARGS.max_angular_vel,
                desired_distance=path_end_dist if at_path_end else 0.0,
                pos_deadzone=0.15,
                angle_deadzone_deg=4.0,
            )
            # LoS: blend angular toward target to keep target in camera frame
            _ang_nav = angular
            angular = _los_blend_angular(robot_pos, robot_yaw, target_pos, angular)
            _ang_los = angular
            linear, angular = smooth_cmd(pursuit_state, linear, angular, SMOOTH_ALPHA_NORMAL)
            _ang_smooth = angular
            linear, angular = apply_ped_reactive_avoidance(
                robot_pos, robot_yaw, linear, angular, char_paths, target_prim_path)
            # Backward safety: clamp negative linear when wall/obstacle is behind robot
            if linear < 0 and not _backward_clear(robot_pos, robot_yaw, occ):
                print(f"[EP{ep_id}] step={step} {_viz_mode}: "
                      f"backward blocked, clamp lin {linear:.2f}→0")
                linear = 0.0
            _apply_drive(robot, linear, angular)
            pursuit_state["_last_mode"] = _viz_mode   # APPROACH or STANDOFF
            path_len = len(pursuit_state["path"]) if pursuit_state["path"] else 0
            print(f"[EP{ep_id}] step={step} {_viz_mode} "
                  f"dist={dist_to_target:.3f}m carrot=({carrot[0]:+.2f},{carrot[1]:+.2f}) "
                  f"idx={pursuit_state['path_idx']}/{path_len} "
                  f"ang_nav={_ang_nav:+.3f} ang_los={_ang_los:+.3f} "
                  f"ang_smooth={_ang_smooth:+.3f} ang_final={angular:+.3f} lin={linear:.3f}")
            for _ in range(ARGS.decimation):
                world.step(render=False)
                if ROBOT_DRIVE_MODE == "kinematic":
                    kinematic_move(robot, linear, angular)
                update_chase_camera(robot)
                simulation_app.update()
            if ARGS.save_images and step % ARGS.save_image_every == 0:
                save_rgb_depth(cam, step, ARGS.image_save_dir, ep_id)
                save_occ_debug(occ, robot_pos, robot_yaw, target_pos,
                               pursuit_state.get("last_goal"), pursuit_state.get("path"),
                               step, ARGS.image_save_dir, ep_id, _viz_mode)
            step += 1

        episode_length = step
        tracking_rate  = tracking_steps / max(episode_length, 1)
        had_recovery   = pursuit_state["any_recovery"]
        success = (bool(distances)
                   and tracking_rate >= 0.8
                   and distances[-1] <= ARGS.tracking_dist_max
                   and not had_recovery
                   and done_reason in ("max_steps", "char_done"))

        if had_recovery and done_reason == "max_steps":
            done_reason = "had_recovery"

        metrics = {
            "episode_id":            ep_id,
            "success":               int(success),
            "had_recovery":          int(had_recovery),
            "episode_length":        episode_length,
            "tracking_rate":         tracking_rate,
            "collision":             collision_count,
            "initial_dist":          distances[0]  if distances else 0.0,
            "final_dist":            distances[-1] if distances else 0.0,
            "avg_dist":              float(np.mean(distances))      if distances      else 0.0,
            "avg_heading_error_deg": float(np.mean(heading_errors)) if heading_errors else 0.0,
            "done_reason":           done_reason,
        }
        all_metrics.append(metrics)
        print(f"[EP{ep_id}] done | success={success} len={episode_length} "
              f"tr={tracking_rate:.3f} had_recovery={had_recovery} reason={done_reason} "
              f"ep_min_ped_dist={ep_min_ped_dist:.3f}m")

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