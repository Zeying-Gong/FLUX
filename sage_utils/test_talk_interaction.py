"""
test_talk_interaction.py
─────────────────────────
验证 SAGE-3D 场景中两个 character 之间的 Talk 交互。

spawn 点采样完全复用 generate_episode_json 的 2D semantic map + ESDF 方案：
  1. 从 semantic map 构建 occupancy grid（wall/unable area 为障碍）
  2. ESDF 膨胀（人物半径）得到可行走区域
  3. 随机采样 ESDF 值足够大的自由点（离墙够远）
  4. A* 验证两点连通、路径长度合适
  5. 坐标经 _sm_to_isaac() 变换到 Isaac 坐标系

Talk 命令格式：  Character Talk Character_01 10
  - Character 主动走向 Character_01（1.5~2m 内自动触发对话动画）
  - 不需要任何 prim 或 offset

Usage
-----
    python test_talk_interaction.py \
        --usda         /workspace/SAGE-3D_Official/SAGE-3D_data/usda/839873.usda \
        --semantic_map_json /workspace/SAGE-3D_Official/SAGE-3D_data/semantic_maps/2D_Semantic_Map_0033_839873_Complete.json
"""
from __future__ import annotations
import argparse
import heapq
import json
import math
import os
import random
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--usda",          required=True)
    p.add_argument("--semantic_map_json",  required=True)
    p.add_argument("--output_dir",    default="/tmp/talk_test")
    p.add_argument("--episode_id",    type=int,   default=0)
    p.add_argument("--talk_duration", type=float, default=10.0)
    p.add_argument("--num_waypoints", type=int,   default=2)

    # 2D map params — same defaults as generate_episode_json
    p.add_argument("--scale_m_per_px",  type=float, default=0.05)
    p.add_argument("--robot_radius_2d", type=float, default=0.4,
                   help="ESDF inflation radius (metres)")
    p.add_argument("--min_path_m",      type=float, default=2.5)
    p.add_argument("--max_path_m",      type=float, default=6.0)

    # NavMesh / sim params
    p.add_argument("--collision_root",  default="/World/scene_collision")
    p.add_argument("--volume_padding",  type=float, default=1.2)
    p.add_argument("--fallback_size",   type=float, default=100.0)
    p.add_argument("--warmup_frames",   type=int,   default=60)
    p.add_argument("--cache_dir",       default=None)
    p.add_argument("--force_rebake",    action="store_true")
    p.add_argument("--seed",            type=int,   default=0)
    return p.parse_args()


ARGS = parse_args()

# ─── SimulationApp — identical to test_dataset_pipeline.py ───────────────────
from isaacsim import SimulationApp

_exp = os.environ.get("EXP_PATH")
if _exp is None:
    print("[FATAL] EXP_PATH env var not set.")
    sys.exit(1)

CUSTOM_APP_PATH = os.path.join(
    _exp, "isaacsim.exp.action_and_event_data_generation.base.kit")

simulation_app = SimulationApp(
    launch_config={
        "renderer":   "RayTracedLighting",
        "headless":   False,
        "enable_cameras": True,
        "crash_reporter/enabled":              False,
        "crash_reporter/skip_old_dump_upload": True,
    },
    experience=CUSTOM_APP_PATH,
)

# ─── Post-launch imports ──────────────────────────────────────────────────────
import carb
import omni.usd
import omni.timeline
from pxr import Sdf, Usd, UsdGeom

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sage_utils.navmesh_utils import cache_paths, bake_navmesh
from sage_utils.people_utils import (
    load_character_assets, spawn_character,
    load_default_skeleton_and_animations,
    bind_animation_graph_to_characters,
    attach_behavior_scripts_to_characters,
    frame_viewport_on,
)
import omni.kit.commands
from isaacsim.replicator.agent.core.settings import PrimPaths
try:
    from pxr import NavSchema
except ImportError:
    NavSchema = None
from isaacsim.replicator.agent.core.stage_util import CharacterUtil


def _update(n: int = 1):
    for _ in range(n):
        simulation_app.update()


def _hide_navmesh_volumes():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    hidden = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() == "NavMeshVolume":
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden += 1
    print(f"[NavMesh] Hidden {hidden} NavMeshVolume prim(s).")


def open_stage(usda_path: str) -> bool:
    if not os.path.exists(usda_path):
        print(f"[FATAL] USDA not found: {usda_path}")
        return False
    print(f"[STAGE] Opening: {usda_path}")
    omni.usd.get_context().open_stage(usda_path)
    _update(30)
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


# ═══════════════════════════════════════════════════════════════════════════════
# 2D map helpers — copied verbatim from generate_episode_json
# ═══════════════════════════════════════════════════════════════════════════════

def _load_2d_map(semantic_map_json: str, robot_radius_m: float, scale: float):
    with open(semantic_map_json, encoding="utf-8") as f:
        sem_data = json.load(f)

    all_y, all_x = [], []
    for inst in sem_data:
        for y, x in inst.get("mask_coords_m", []):
            try:
                all_y.append(float(y))
                all_x.append(float(x))
            except (ValueError, TypeError):
                continue

    if not all_y:
        return None, None, None, None, None, None

    min_y, max_y = min(all_y), max(all_y)
    min_x, max_x = min(all_x), max(all_x)
    h = int(np.ceil((max_y - min_y) / scale)) + 1
    w = int(np.ceil((max_x - min_x) / scale)) + 1

    grid = np.zeros((h, w), dtype=np.uint8)
    for inst in sem_data:
        label = str(inst.get("category_label", "")).lower()
        if label in ("wall", "unable area"):
            for y_m, x_m in inst.get("mask_coords_m", []):
                try:
                    py = int(round((float(y_m) - min_y) / scale))
                    px = int(round((float(x_m) - min_x) / scale))
                    if 0 <= py < h and 0 <= px < w:
                        grid[py, px] = 1
                except (ValueError, TypeError):
                    continue

    if robot_radius_m > 0:
        dist_m = distance_transform_edt(grid == 0, sampling=scale)
        grid = (dist_m <= robot_radius_m).astype(np.uint8)

    esdf = distance_transform_edt(grid == 0, sampling=scale)
    return grid, esdf, min_x, min_y, scale, sem_data


def _world_to_px(x_m, y_m, min_x, min_y, scale):
    px = int(round((float(x_m) - min_x) / scale))
    py = int(round((float(y_m) - min_y) / scale))
    return px, py


def _px_to_world(px, py, min_x, min_y, scale):
    x_m = min_x + (px + 0.5) * scale
    y_m = min_y + (py + 0.5) * scale
    return x_m, y_m


def _sm_to_isaac(x_m, y_m, min_x, max_x, min_y, max_y,
                  flip_x=True, flip_y=True, negate=True):
    """Official trajectory_2d_to_3d.py coordinate transform."""
    px, py = float(x_m), float(y_m)
    if flip_x:
        px = (min_x + max_x) - px
    if flip_y:
        py = (min_y + max_y) - py
    if negate:
        px, py = -px, -py
    return px, py


def _astar(grid, start_px, goal_px):
    H, W = grid.shape
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    open_set = [(0.0, start_px)]
    came_from = {}
    g = {start_px: 0.0}
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal_px:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return path[::-1]
        for d in dirs:
            nx, ny = cur[0]+d[0], cur[1]+d[1]
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if grid[ny, nx] == 1:
                continue
            nb = (nx, ny)
            tg = g[cur] + math.hypot(d[0], d[1])
            if nb not in g or tg < g[nb]:
                came_from[nb] = cur
                g[nb] = tg
                f = tg + math.hypot(nx-goal_px[0], ny-goal_px[1])
                heapq.heappush(open_set, (f, nb))
    return None


def _sample_free_point_2d(grid, esdf, rng, min_clearance_m=0.4):
    H, W = grid.shape
    for _ in range(500):
        py = rng.randint(0, H-1)
        px = rng.randint(0, W-1)
        if grid[py, px] == 0 and esdf[py, px] >= min_clearance_m:
            return (px, py)
    for _ in range(500):
        py = rng.randint(0, H-1)
        px = rng.randint(0, W-1)
        if grid[py, px] == 0:
            return (px, py)
    return None


def _path_length_px(path_px: list, scale: float) -> float:
    if not path_px or len(path_px) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(path_px)):
        total += math.hypot(path_px[i][0]-path_px[i-1][0],
                             path_px[i][1]-path_px[i-1][1]) * scale
    return total


def sample_point_isaac(grid, esdf, rng,
                        min_x, min_y, scale,
                        map_min_x, map_max_x, map_min_y, map_max_y,
                        min_clearance_m=0.5):
    """Sample a free 2D point and convert to Isaac coordinates."""
    px_py = _sample_free_point_2d(grid, esdf, rng, min_clearance_m)
    if px_py is None:
        return None
    x_2d, y_2d = _px_to_world(px_py[0], px_py[1], min_x, min_y, scale)
    ix, iy = _sm_to_isaac(x_2d, y_2d, map_min_x, map_max_x, map_min_y, map_max_y)
    return (ix, iy, 0.0), px_py


def plan_path_isaac(start_px, goal_px, grid, scale,
                     min_x, min_y,
                     map_min_x, map_max_x, map_min_y, map_max_y):
    """A* in pixel space → Isaac coordinate path points."""
    def snap(px_py):
        H, W = grid.shape
        px, py = px_py
        if 0 <= px < W and 0 <= py < H and grid[py, px] == 0:
            return px_py
        from collections import deque
        visited = set()
        q = deque([(px, py, 0)])
        while q:
            cx, cy, d = q.popleft()
            if d > 30:
                break
            if 0 <= cx < W and 0 <= cy < H and grid[cy, cx] == 0:
                return (cx, cy)
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nb = (cx+dx, cy+dy)
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb[0], nb[1], d+1))
        return None

    sp = snap(start_px)
    gp = snap(goal_px)
    if sp is None or gp is None:
        return None
    path_px = _astar(grid, sp, gp)
    if path_px is None:
        return None
    result = []
    for (px, py) in path_px:
        x_2d, y_2d = _px_to_world(px, py, min_x, min_y, scale)
        ix, iy = _sm_to_isaac(x_2d, y_2d, map_min_x, map_max_x, map_min_y, map_max_y)
        result.append([ix, iy, 0.0])
    return result


# ─── Write commands into SkelRoot scriptData / pathData ──────────────────────

def _write_episode_commands(stage, commands_dict: dict):
    parent_path = str(PrimPaths.characters_parent_path())

    for char_name, cmds in commands_dict.items():
        char_prim_path = f"{parent_path}/{char_name}"
        char_prim = stage.GetPrimAtPath(char_prim_path)
        if not char_prim.IsValid():
            print(f"[Episode] WARNING: prim not found for {char_name}")
            continue

        skelroot = None
        for desc in Usd.PrimRange(char_prim):
            if desc.GetTypeName() == "SkelRoot":
                skelroot = desc
                break
        if skelroot is None:
            print(f"[Episode] WARNING: no SkelRoot under {char_prim_path}")
            continue

        command_strings: list[str] = []
        path_data: dict[int, list] = {}
        goto_index = 0

        for cmd in cmds:
            cmd_name = cmd.get("cmd", "")
            params   = cmd.get("params", [])
            command_strings.append(f"{cmd_name} " + " ".join(str(p) for p in params))
            if cmd_name == "GoTo":
                if "path" in cmd and cmd["path"]:
                    path_data[goto_index] = cmd["path"]
                    print(f"[Episode]   {char_name} GoTo#{goto_index}: "
                          f"{len(cmd['path'])} pts")
                goto_index += 1
            elif cmd_name == "Talk":
                print(f"[Episode]   {char_name} Talk → {params[0]}  {params[1]}s")
            elif cmd_name == "Idle":
                print(f"[Episode]   {char_name} Idle {params[0]}s")

        sd_attr = skelroot.GetAttribute("omni:scripting:scriptData")
        if not sd_attr:
            sd_attr = skelroot.CreateAttribute("omni:scripting:scriptData",
                                               Sdf.ValueTypeNames.StringArray)
        sd_attr.Set(command_strings)

        if path_data:
            pd_attr = skelroot.GetAttribute("omni:scripting:pathData")
            if not pd_attr:
                pd_attr = skelroot.CreateAttribute("omni:scripting:pathData",
                                                   Sdf.ValueTypeNames.String)
            pd_attr.Set(json.dumps(path_data))

        print(f"[Episode] {char_name}: {command_strings}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    rng = random.Random(ARGS.seed)
    np.random.seed(ARGS.seed)

    os.makedirs(ARGS.output_dir, exist_ok=True)
    scene_id = os.path.splitext(os.path.basename(ARGS.usda))[0]
    out_json  = os.path.join(ARGS.output_dir,
                             f"episode_{ARGS.episode_id}_talk_{scene_id}.json")

    print("=" * 65)
    print("   SAGE-3D  Talk Interaction Test")
    print("=" * 65)
    print(f"   Scene        : {ARGS.usda}")
    print(f"   Semantic map : {ARGS.semantic_map_json}")
    print(f"   Talk duration: {ARGS.talk_duration}s")
    print("=" * 65)

    # ── 1. Load 2D map (same as generate_episode_json) ────────────────────
    print(f"\n[2D] Loading semantic map: {ARGS.semantic_map_json}")
    grid, esdf, min_x, min_y, scale, sem_data = _load_2d_map(
        ARGS.semantic_map_json,
        robot_radius_m = ARGS.robot_radius_2d,
        scale          = ARGS.scale_m_per_px,
    )
    if grid is None:
        print("[FATAL] 2D map load failed")
        simulation_app.close()
        return 1

    # Compute map bounds for coordinate transform
    all_y, all_x = [], []
    for inst in sem_data:
        for y, x in inst.get("mask_coords_m", []):
            try:
                all_y.append(float(y)); all_x.append(float(x))
            except: pass
    map_min_x, map_max_x = min(all_x), max(all_x)
    map_min_y, map_max_y = min(all_y), max(all_y)
    print(f"[2D] grid={grid.shape}  free={int((grid==0).sum())} px  "
          f"scale={scale}m/px  ESDF max={esdf.max():.2f}m")

    # ── 2. Sample two connected spawn points via A* ───────────────────────
    # Spawn characters close enough that Talk triggers without best_point_around.
    # talk.py line 269: skips navigation when target_pos_distance < TalkDistance (~1.5m)
    SPAWN_MIN_M = 1.2
    SPAWN_MAX_M = 2.0
    print(f"\n[2D] Sampling spawn points (close-proximity {SPAWN_MIN_M}~{SPAWN_MAX_M}m for Talk)...")

    pos_a_isaac = pos_b_isaac = None
    pos_a_px    = pos_b_px    = None
    path_ab = None

    for attempt in range(500):
        ra = _sample_free_point_2d(grid, esdf, rng,
                                    min_clearance_m=ARGS.robot_radius_2d * 1.5)
        rb = _sample_free_point_2d(grid, esdf, rng,
                                    min_clearance_m=ARGS.robot_radius_2d * 1.5)
        if ra is None or rb is None:
            continue

        # Euclidean pre-filter: keep them close
        d_px = math.hypot(ra[0]-rb[0], ra[1]-rb[1]) * scale
        if d_px < SPAWN_MIN_M or d_px > SPAWN_MAX_M:
            continue

        path_px = _astar(grid, ra, rb)
        if path_px is None:
            continue
        pl = _path_length_px(path_px, scale)
        if not (SPAWN_MIN_M <= pl <= SPAWN_MAX_M):
            continue

        # Convert to Isaac coordinates
        xa_2d, ya_2d = _px_to_world(ra[0], ra[1], min_x, min_y, scale)
        xb_2d, yb_2d = _px_to_world(rb[0], rb[1], min_x, min_y, scale)
        ixa, iya = _sm_to_isaac(xa_2d, ya_2d, map_min_x, map_max_x, map_min_y, map_max_y)
        ixb, iyb = _sm_to_isaac(xb_2d, yb_2d, map_min_x, map_max_x, map_min_y, map_max_y)

        pos_a_isaac = (ixa, iya, 0.0)
        pos_b_isaac = (ixb, iyb, 0.0)
        pos_a_px    = ra
        pos_b_px    = rb

        # Convert A* path to Isaac coords for Character_00's pre-walk
        path_ab = []
        for (px, py) in path_px:
            x_2d, y_2d = _px_to_world(px, py, min_x, min_y, scale)
            ix, iy = _sm_to_isaac(x_2d, y_2d, map_min_x, map_max_x, map_min_y, map_max_y)
            path_ab.append([ix, iy, 0.0])

        print(f"  Found after {attempt+1} attempts  path={pl:.1f}m")
        break

    if pos_a_isaac is None:
        print("[FATAL] Could not find two connected points after 300 attempts")
        simulation_app.close()
        return 1

    char0_name = CharacterUtil.get_character_name_by_index(0)
    char1_name = CharacterUtil.get_character_name_by_index(1)

    print(f"  {char0_name}: isaac=({pos_a_isaac[0]:.3f}, {pos_a_isaac[1]:.3f})")
    print(f"  {char1_name}: isaac=({pos_b_isaac[0]:.3f}, {pos_b_isaac[1]:.3f})")

    # ── 3. Build base commands (waypoints added after NavMesh bake) ─────────
    char0_cmds = [
        {"cmd": "Idle", "params": ["8"]},
        {"cmd": "Talk", "params": [char1_name, str(ARGS.talk_duration)]},
    ]
    char1_cmds = [
        {"cmd": "Idle", "params": [str(int(ARGS.talk_duration * 2))]},
    ]

    spawn_positions = {
        char0_name: {"pos": list(pos_a_isaac), "rot": 0.0},
        char1_name: {"pos": list(pos_b_isaac), "rot": 0.0},
    }
    commands_dict = {
        char0_name: char0_cmds,
        char1_name: char1_cmds,
    }

    # ── 5. Save JSON ──────────────────────────────────────────────────────
    episode = {
        "episode": {
            "episode_id": ARGS.episode_id,
            "robot": {
                "start_pos": [pos_a_isaac[0]+5, pos_a_isaac[1]+5, 0.0],
                "goal_pos":  [pos_b_isaac[0]-5, pos_b_isaac[1]-5, 0.0],
                "start_orientation": 0.0,
            },
            "characters": {
                "num_characters": 2,
                "spawn_positions": spawn_positions,
                "commands": commands_dict,
            },
        }
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(episode, f, indent=4)
    print(f"\n[SAVED] {out_json}")
    print("\n[CMD SEQUENCE]")
    for cname, cmds in commands_dict.items():
        for cmd in cmds:
            print(f"  {cname}: {cmd['cmd']:<6}  " +
                  "  ".join(cmd.get("params", [])))

    # ── 6. Open stage ─────────────────────────────────────────────────────
    if not open_stage(ARGS.usda):
        simulation_app.close()
        return 1

    # ── 7. Bake NavMesh ───────────────────────────────────────────────────
    navvols_usda = cache_paths(ARGS.usda, ARGS.cache_dir)
    nav_ok, inav = bake_navmesh(
        app               = simulation_app,
        navvols_usda      = navvols_usda,
        force_rebake      = ARGS.force_rebake,
        collision_root    = ARGS.collision_root,
        volume_padding    = ARGS.volume_padding,
        fallback_size     = ARGS.fallback_size,
        warmup_frames     = ARGS.warmup_frames,
        semantic_map_json = ARGS.semantic_map_json,
    )
    if not nav_ok:
        print("[FATAL] NavMesh bake failed.")
        simulation_app.close()
        return 2

    _hide_navmesh_volumes()
    _update(10)

    # ── 8. Physics world ──────────────────────────────────────────────────
    from isaacsim.core.api import World
    world = World(physics_dt=1.0/60.0, rendering_dt=1.0/30.0)
    world.initialize_physics()
    world.reset()
    _update(10)

    nm = inav.get_navmesh()

    # ── 9. Sample waypoints validated by BOTH 2D A* and NavMesh ──────────
    def nm_connected(pa_isaac, pb_isaac) -> bool:
        """Check NavMesh path exists with agent_radius=0.3."""
        result = nm.query_shortest_path(
            carb.Float3(float(pa_isaac[0]), float(pa_isaac[1]), 0.0),
            carb.Float3(float(pb_isaac[0]), float(pb_isaac[1]), 0.0),
            agent_radius=0.3)
        if result is None:
            return False
        pts = result.get_points() if hasattr(result, "get_points") else result
        return pts is not None and len(pts) >= 2

    def sample_wp_validated(from_px, from_isaac):
        """Sample waypoint valid in both 2D A* and NavMesh."""
        for _ in range(300):
            wp_px = _sample_free_point_2d(grid, esdf, rng,
                                           min_clearance_m=ARGS.robot_radius_2d)
            if wp_px is None:
                continue
            d_px = _path_length_px([from_px, wp_px], scale)
            if not (2.0 <= d_px <= 7.0):
                continue
            path_px = _astar(grid, from_px, wp_px)
            if path_px is None:
                continue
            x_2d, y_2d = _px_to_world(wp_px[0], wp_px[1], min_x, min_y, scale)
            ix, iy = _sm_to_isaac(x_2d, y_2d, map_min_x, map_max_x, map_min_y, map_max_y)
            wp_isaac = [ix, iy, 0.0]
            # NavMesh double-check
            if not nm_connected(from_isaac, wp_isaac):
                continue
            path_isaac = []
            for (px, py) in path_px:
                x2, y2 = _px_to_world(px, py, min_x, min_y, scale)
                ix2, iy2 = _sm_to_isaac(x2, y2, map_min_x, map_max_x,
                                          map_min_y, map_max_y)
                path_isaac.append([ix2, iy2, 0.0])
            return wp_px, wp_isaac, path_isaac
        return None, None, None

    print("\n[2D+NM] Sampling NavMesh-validated waypoints...")
    prev_px, prev_isaac = pos_a_px, list(pos_a_isaac)
    for _ in range(ARGS.num_waypoints):
        wp_px, wp_isaac, wp_path = sample_wp_validated(prev_px, prev_isaac)
        if wp_isaac:
            char0_cmds.append({
                "cmd":    "GoTo",
                "params": [f"{wp_isaac[0]:.4f}", f"{wp_isaac[1]:.4f}", "0.0000", "_"],
                "path":   wp_path,
            })
            prev_px, prev_isaac = wp_px, wp_isaac
            print(f"  Character  wp: ({wp_isaac[0]:.3f}, {wp_isaac[1]:.3f})")
        else:
            print(f"  Character  wp: not found")

    prev_px, prev_isaac = pos_b_px, list(pos_b_isaac)
    for _ in range(ARGS.num_waypoints):
        wp_px, wp_isaac, wp_path = sample_wp_validated(prev_px, prev_isaac)
        if wp_isaac:
            char1_cmds.append({
                "cmd":    "GoTo",
                "params": [f"{wp_isaac[0]:.4f}", f"{wp_isaac[1]:.4f}", "0.0000", "_"],
                "path":   wp_path,
            })
            prev_px, prev_isaac = wp_px, wp_isaac
            print(f"  Character_01 wp: ({wp_isaac[0]:.3f}, {wp_isaac[1]:.3f})")
        else:
            print(f"  Character_01 wp: not found")

    # ── 10. Spawn characters — identical to test_dataset_pipeline.py ──────
    char_pool = load_character_assets()
    print(f"\n[PEOPLE] {len(char_pool)} asset(s) available.")

    stage = omni.usd.get_context().get_stage()
    char_prims: list[str] = []

    for i, (cname, sp_data) in enumerate(spawn_positions.items()):
        pos = sp_data["pos"]
        usd = char_pool[i % len(char_pool)]
        prim_p = spawn_character(simulation_app, i, usd, pos)
        char_prims.append(prim_p)
        print(f"[Episode] Spawned {cname} at ({pos[0]:.3f}, {pos[1]:.3f})")

        # Exclude this character from NavMesh so other characters can
        # path-find through their position (fixes query_shortest_path → None)
        if NavSchema is not None:
            try:
                omni.kit.commands.execute(
                    "ApplyNavMeshAPICommand",
                    prim_path=prim_p,
                    api=NavSchema.NavMeshExcludeAPI,
                )
                print(f"  [NavMesh] Excluded {prim_p} from NavMesh")
            except Exception as e:
                print(f"  [NavMesh] Could not exclude {prim_p}: {e}")
        else:
            print(f"  [NavMesh] NavSchema not available, skipping exclusion")

    load_default_skeleton_and_animations(simulation_app)
    _update(30)
    bind_animation_graph_to_characters(simulation_app)
    _update(60)

    # NavMeshExcludeAPI takes effect at runtime without re-baking.
    # Give the engine a few frames to process the API change.
    _update(30)

    # Rebuild commands_dict with final waypoints
    commands_dict = {char0_name: char0_cmds, char1_name: char1_cmds}
    # Also update episode JSON with validated waypoints
    episode["episode"]["characters"]["commands"] = commands_dict
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(episode, f, indent=4)
    print(f"[SAVED] Updated JSON with NavMesh-validated waypoints → {out_json}")

    _write_episode_commands(stage, commands_dict)
    _update(10)

    attach_behavior_scripts_to_characters(simulation_app)
    _update(60)

    # ── 11. Play ──────────────────────────────────────────────────────────
    if char_prims:
        frame_viewport_on(char_prims[0])
        _update(5)

    tl = omni.timeline.get_timeline_interface()
    tl.set_current_time(0.0)
    tl.play()

    print(f"\n[HOLD] Watching Talk animation — press Ctrl+C to quit.\n")
    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        print("\n[HOLD] Ctrl+C received, shutting down.")

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())