"""
generate_episode_json.py
────────────────────────
Offline episode-data generator.  Opens a USDA scene, bakes its NavMesh,
then samples robot start/goal + N pedestrian spawn positions and GoTo
waypoint chains, and writes the result to an episode_<id>.json file.

Output directory is automatically derived from the USDA scene name:
    <usda_dir>/../../episodes/<scene_id>/episode_<id>.json
e.g. for .../usda/839873.usda → .../episodes/839873/episode_0.json

Pass --overwrite to replace existing episode files (default: skip them).

The JSON schema is compatible with people_simulation.py:

    {
      "episode": {
        "episode_id": <int>,
        "robot": {
          "start_pos": [x, y, z],
          "goal_pos":  [x, y, z],
          "start_orientation": <float rad>
        },
        "characters": {
          "num_characters": <int>,
          "spawn_positions": {
            "Character":    {"pos": [x,y,z], "rot": <float>},
            ...
          },
          "commands": {
            "Character": [
              {"cmd": "GoTo", "params": ["x","y","z","_"],
               "path": [[x,y,z], ...]},   ← pre-computed NavMesh path
              ...
            ],
            ...
          }
        }
      }
    }

Usage
-----
    python generate_episode_json.py \
        --usda /workspace/SAGE-3D_Official/SAGE-3D_data/usda/839873.usda \
        --num_people 1 \
        --num_waypoints 3 \
        --episode_id 0

    # Force overwrite an existing episode:
    python generate_episode_json.py \
        --usda /workspace/SAGE-3D_Official/SAGE-3D_data/usda/839873.usda \
        --episode_id 0 \
        --overwrite
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
import numpy as np
from scipy.ndimage import distance_transform_edt
import heapq

# ═══════════════════════════════════════════════════════════════════════════
# CLI  (parsed before SimulationApp so --help works fast)
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate episode JSON with pre-computed GoTo paths."
    )
    p.add_argument("--usda", required=True)
    p.add_argument("--output_dir", default=None,
                   help="Override output directory. By default episodes are "
                        "written to <usda_dir>/../../episodes/<scene_id>/")
    p.add_argument("--num_people",    type=int, default=1)
    p.add_argument("--num_waypoints", type=int, default=3,
                   help="GoTo waypoints per character.")
    p.add_argument("--episode_id",    type=int, default=0)
    p.add_argument("--seed",          type=int, default=0)

    # Overwrite control
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite an existing episode JSON. "
                        "Default behaviour: skip if file already exists.")

    # NavMesh / cache
    p.add_argument("--collision_root",        default="/World/scene_collision")
    p.add_argument("--volume_padding",        type=float, default=1.2)
    p.add_argument("--fallback_size",         type=float, default=100.0)
    p.add_argument("--warmup_frames",         type=int,   default=60)
    p.add_argument("--semantic_map_json", default=None,
                help="2D semantic map JSON (必须提供，用于 2D 路径规划)")
    p.add_argument("--robot_radius_2d", type=float, default=0.3,
                help="2D 地图膨胀半径（米），替代 NavMesh agent_radius")
    p.add_argument("--scale_m_per_px", type=float, default=0.05,
                help="2D 地图分辨率（米/像素）")
    p.add_argument("--cache_dir",             default=None)
    p.add_argument("--force_rebake",          action="store_true")
    p.add_argument("--vis_only", action="store_true",
               help="只重新生成可视化图，不重新采样 episode")
    return p.parse_args()


ARGS = parse_args()

# ═══════════════════════════════════════════════════════════════════════════
# Launch SimulationApp
# ═══════════════════════════════════════════════════════════════════════════
from isaacsim import SimulationApp

_exp = os.environ.get("EXP_PATH")
if _exp is None:
    print("[FATAL] EXP_PATH env var not set.")
    sys.exit(1)

CUSTOM_APP_PATH = os.path.join(
    _exp, "isaacsim.exp.action_and_event_data_generation.base.kit")

simulation_app = SimulationApp(
    launch_config={
        "renderer": "RayTracedLighting",
        "headless": True,          # no display needed for generation
        "enable_cameras": False,
        "crash_reporter/enabled": False,
        "crash_reporter/skip_old_dump_upload": True,
    },
    experience=CUSTOM_APP_PATH,
)

# ═══════════════════════════════════════════════════════════════════════════
# Post-launch imports
# ═══════════════════════════════════════════════════════════════════════════
import numpy as np
import carb
import omni.usd
import omni.timeline
from pxr import Usd, UsdGeom

# Make sage_utils importable regardless of cwd
sys.path.insert(0, os.path.dirname(__file__))

from navmesh_utils import (
    cache_paths, bake_navmesh,
    safe_query_random_point, find_reachable_point,
    spawn_on_navmesh,
    find_safe_navmesh_point, 
)
from people_utils import (
    load_character_assets, spawn_character,
    load_default_skeleton_and_animations,
    bind_animation_graph_to_characters,
    attach_behavior_scripts_to_characters,
    write_commands_to_scriptdata,
    frame_viewport_on,
    compute_goto_path
)

from isaacsim.replicator.agent.core.stage_util import CharacterUtil


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _update(n: int = 1):
    for _ in range(n):
        simulation_app.update()


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
        return False
    print(f"[STAGE] Loaded. Default prim: {stage.GetDefaultPrim().GetPath()}")
    return True


def _scene_id() -> str:
    """Extract the bare filename (without extension) from the USDA path.
    e.g. '.../usda/839873.usda' → '839873'
    """
    return os.path.splitext(os.path.basename(ARGS.usda))[0]


def _output_dir() -> str:
    """Return (and create) the output directory for this scene's episodes.

    Priority:
      1. --output_dir explicitly provided  →  <output_dir>/<scene_id>/
      2. Default                           →  <usda_dir>/../../episodes/<scene_id>/
         (i.e. .../SAGE-3D_data/episodes/<scene_id>/)
    """
    scene_id = _scene_id()

    if ARGS.output_dir:
        d = os.path.join(ARGS.output_dir, scene_id)
    else:
        usda_dir = os.path.dirname(os.path.abspath(ARGS.usda))
        # usda_dir  = .../SAGE-3D_data/usda
        # parent    = .../SAGE-3D_data
        parent = os.path.dirname(usda_dir)
        d = os.path.join(parent, "episodes", scene_id)

    os.makedirs(d, exist_ok=True)
    return d


def _episode_path(out_dir: str) -> str:
    return os.path.join(out_dir, f"episode_{ARGS.episode_id}.json")


# ═══════════════════════════════════════════════════════════════════════════
# Sampling helpers
# ═══════════════════════════════════════════════════════════════════════════
MIN_ROBOT_DIST   = 3.0    # minimum robot start→goal distance (m)
MAX_ROBOT_DIST   = 20.0
MIN_CHAR_DIST    = 2.0    # character spawn must be >= this far from robot endpoints
MIN_CHAR_SPACING = 2.0    # between characters and robot
MIN_CHAR_GOAL_TO_ROBOT_GOAL = 2.0

# waypoint 测地距离降级阈值（依次尝试，适配大小场景）
WAYPOINT_DIST_TIERS = [4.0, 2.5, 1.5]

# character spawn 点最小可达面积（像素数），过滤掉被困在小角落的点
MIN_REACHABLE_PX = 100    

def _navmesh_connected(nm, point_a_isaac, point_b_isaac) -> bool:
    if nm is None:
        return True  # 没有 NavMesh 时跳过检查
    try:
        pa = carb.Float3(float(point_a_isaac[0]),
                         float(point_a_isaac[1]),
                         float(point_a_isaac[2]))
        pb = carb.Float3(float(point_b_isaac[0]),
                         float(point_b_isaac[1]),
                         float(point_b_isaac[2]))
        ra = nm.query_closest_point(pa, 1.0)
        rb = nm.query_closest_point(pb, 1.0)
        if ra is None or ra[0] is None:
            return False
        if rb is None or rb[0] is None:
            return False
        snapped_a = ra[0]
        snapped_b = rb[0]
        da = math.hypot(float(snapped_a[0]) - float(pa[0]),
                        float(snapped_a[1]) - float(pa[1]))
        db = math.hypot(float(snapped_b[0]) - float(pb[0]),
                        float(snapped_b[1]) - float(pb[1]))
        if da > 1.0 or db > 1.0:
            return False
        path = nm.query_shortest_path(snapped_a, snapped_b, agent_radius=0.3)
        return path is not None
    except Exception:
        return False  # 异常也拒绝，不放行

def _reachable_area_px(grid, start_px) -> int:
    """BFS 估算从 start_px 出发可达的自由像素数（上限 MIN_REACHABLE_PX*10 提前退出）。"""
    from collections import deque
    H, W = grid.shape
    sx, sy = start_px
    if not (0 <= sx < W and 0 <= sy < H) or grid[sy, sx] != 0:
        return 0
    limit = MIN_REACHABLE_PX * 10
    visited = {start_px}
    q = deque([start_px])
    count = 0
    while q and count < limit:
        cx, cy = q.popleft()
        count += 1
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (cx+dx, cy+dy)
            nx, ny = nb
            if nb not in visited and 0 <= nx < W and 0 <= ny < H and grid[ny, nx] == 0:
                visited.add(nb)
                q.append(nb)
    return count

def _dist2(a, b) -> float:
    return ((float(a[0])-float(b[0]))**2 + (float(a[1])-float(b[1]))**2)**0.5

# ═══════════════════════════════════════════════════════════════════════════
# 2D map-based planning（官方 pipeline 逻辑）
# ═══════════════════════════════════════════════════════════════════════════

def _load_2d_map(semantic_map_json: str, robot_radius_m: float, scale: float):
    """从 semantic map JSON 构建膨胀后的 occupancy grid。
    
    Returns:
        grid_map: np.ndarray, 1=障碍 0=自由
        min_x, min_y: 世界坐标原点
        scale: 米/像素
    """
    import json as _json
    with open(semantic_map_json, encoding="utf-8") as f:
        sem_data = _json.load(f)

    all_y, all_x = [], []
    for inst in sem_data:
        for y, x in inst.get("mask_coords_m", []):
            try:
                all_y.append(float(y))
                all_x.append(float(x))
            except (ValueError, TypeError):
                continue

    if not all_y:
        return None, None, None, None, None

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

    # ESDF 膨胀
    if robot_radius_m > 0:
        dist_m = distance_transform_edt(grid == 0, sampling=scale)
        grid = (dist_m <= robot_radius_m).astype(np.uint8)

    # 同时计算 ESDF 距离图（用于安全点采样）
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
    """官方 trajectory_2d_to_3d.py 的坐标变换。"""
    px, py = float(x_m), float(y_m)
    if flip_x:
        px = (min_x + max_x) - px
    if flip_y:
        py = (min_y + max_y) - py
    if negate:
        px, py = -px, -py
    return px, py


def _astar(grid, start_px, goal_px):
    """A* 路径规划，返回像素坐标列表 [(px,py), ...] 或 None。"""
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
    """在 2D grid 的自由区域随机采样一个离墙足够远的点（像素坐标）。"""
    H, W = grid.shape
    for _ in range(500):
        py = rng.randint(0, H-1)
        px = rng.randint(0, W-1)
        if grid[py, px] == 0 and esdf[py, px] >= min_clearance_m:
            return (px, py)
    # fallback：放宽要求
    for _ in range(500):
        py = rng.randint(0, H-1)
        px = rng.randint(0, W-1)
        if grid[py, px] == 0:
            return (px, py)
    return None


def _find_safe_point_2d(esdf, grid, center_px, search_r_px=20):
    """在 center_px 周围找 ESDF 值最大（离墙最远）的自由点。"""
    H, W = grid.shape
    cx, cy = center_px
    best_px = center_px
    best_val = esdf[cy, cx] if 0 <= cy < H and 0 <= cx < W else 0.0

    for dy in range(-search_r_px, search_r_px+1):
        for dx in range(-search_r_px, search_r_px+1):
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < W and 0 <= ny < H and grid[ny, nx] == 0:
                v = esdf[ny, nx]
                if v > best_val:
                    best_val = v
                    best_px = (nx, ny)
    return best_px

def _save_episode_vis(out_path, grid, esdf, min_x, min_y, scale,
                      map_min_x, map_max_x, map_min_y, map_max_y,
                      robot_start, robot_goal,
                      spawn_positions_dict, commands_dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[VIS] matplotlib not available, skipping")
        return

    def from_isaac(ix, iy):
        x_m, y_m = -float(ix), -float(iy)
        x_m = (map_min_x + map_max_x) - x_m
        y_m = (map_min_y + map_max_y) - y_m
        return x_m, y_m

    def isaac_to_px(ix, iy):
        x_2d, y_2d = from_isaac(ix, iy)
        px = int(round((x_2d - min_x) / scale))
        py = int(round((y_2d - min_y) / scale))
        return px, py

    H, W = grid.shape
    char_colors = ['#FF6B6B', '#FFD93D', '#6BCB77', '#4D96FF']

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    print(f"[VIS] robot_start px={isaac_to_px(robot_start[0], robot_start[1])}")
    print(f"[VIS] robot_goal  px={isaac_to_px(robot_goal[0],  robot_goal[1])}")
    for char_name, sp_data in spawn_positions_dict.items():
        sp = sp_data["pos"]
        print(f"[VIS] {char_name} spawn px={isaac_to_px(sp[0], sp[1])}")
        for w_idx, cmd in enumerate(commands_dict.get(char_name, [])):
            params = cmd.get("params", [])
            if len(params) >= 2:
                print(f"[VIS]   GoTo#{w_idx} px={isaac_to_px(float(params[0]), float(params[1]))}")

    for ax_idx, ax in enumerate(axes):
        # ── 底图 ──────────────────────────────────────────────────────────
        if ax_idx == 0:
            vis = np.zeros((H, W, 3), dtype=np.uint8)
            vis[grid == 0] = [240, 240, 240]
            vis[grid == 1] = [50,  50,  50]
            ax.imshow(vis, origin="lower")
            ax.set_title("Occupancy Map (inflated)", fontsize=12)
        else:
            esdf_vis = np.clip(esdf, 0, 2.0)
            ax.imshow(esdf_vis, origin="lower", cmap="viridis")
            ax.set_title("ESDF (m, clipped 2m)", fontsize=12)

        # ── 机器人 ────────────────────────────────────────────────────────
        if robot_start is not None:
            rs = isaac_to_px(robot_start[0], robot_start[1])
            ax.plot(rs[0], rs[1], 's', color='cyan',
                    markersize=12, markeredgecolor='black',
                    markeredgewidth=1.5, zorder=5)

        if robot_goal is not None:
            rg = isaac_to_px(robot_goal[0], robot_goal[1])
            ax.plot(rg[0], rg[1], '*', color='cyan',
                    markersize=16, markeredgecolor='black',
                    markeredgewidth=1.5, zorder=5)

        # ── 行人 ─────────────────────────────────────────────────────────
        for c_idx, (char_name, spawn_data) in enumerate(spawn_positions_dict.items()):
            color = char_colors[c_idx % len(char_colors)]
            cmds  = commands_dict.get(char_name, [])

            sp    = spawn_data["pos"]
            sp_px = isaac_to_px(sp[0], sp[1])
            ax.plot(sp_px[0], sp_px[1], 'o', color=color,
                    markersize=10, markeredgecolor='black',
                    markeredgewidth=1.5, zorder=6)

            for w_idx, cmd in enumerate(cmds):
                if cmd.get("cmd") != "GoTo":
                    continue

                # 路径线
                path_pts = cmd.get("path", [])
                if path_pts:
                    pts = [isaac_to_px(p[0], p[1]) for p in path_pts]
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            '-', color=color, linewidth=1.5,
                            alpha=0.7, zorder=4)

                # waypoint marker
                params = cmd.get("params", [])
                if len(params) >= 3:
                    try:
                        wp = isaac_to_px(float(params[0]), float(params[1]))
                        is_last = (w_idx == len(cmds) - 1)
                        ax.plot(wp[0], wp[1],
                                '*' if is_last else 'D',
                                color=color,
                                markersize=14 if is_last else 9,
                                markeredgecolor='black',
                                markeredgewidth=1.2, zorder=7)
                        # ★ 标注序号
                        label = str(w_idx + 1)  # 1-based
                        ax.text(wp[0] + 3, wp[1] + 3, label,
                                fontsize=8, color=color,
                                fontweight='bold', zorder=8)
                    except (ValueError, IndexError):
                        pass

        # 图例放到图外右侧，不遮挡内容
        legend_handles = [
            mpatches.Patch(facecolor='#303030', label='Obstacle'),
            mpatches.Patch(facecolor='#F0F0F0', label='Free space'),
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor='cyan', markersize=9,
                       markeredgecolor='black', label='Robot start'),
            plt.Line2D([0], [0], marker='*', color='w',
                       markerfacecolor='cyan', markersize=13,
                       markeredgecolor='black', label='Robot goal'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='#FF6B6B', markersize=9,
                       markeredgecolor='black', label='Char spawn'),
            plt.Line2D([0], [0], marker='D', color='w',
                       markerfacecolor='#FF6B6B', markersize=8,
                       markeredgecolor='black', label='Waypoint'),
            plt.Line2D([0], [0], marker='*', color='w',
                       markerfacecolor='#FF6B6B', markersize=13,
                       markeredgecolor='black', label='Final goal'),
        ]
        # 只在右图右侧放一个共享图例
        if ax_idx == 1:
            ax.legend(handles=legend_handles,
                      loc='upper left',
                      bbox_to_anchor=(1.02, 1.0),
                      borderaxespad=0,
                      fontsize=8,
                      framealpha=0.9)
        ax.set_xlabel("pixel x")
        ax.set_ylabel("pixel y (origin=lower)")
        ax.grid(False)

    plt.suptitle(f"Episode {ARGS.episode_id}  —  scene {_scene_id()}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    vis_path = out_path.replace(".json", "_vis.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[VIS] Saved → {vis_path}")

def _path_length(path_pts: list) -> float:
    """计算路径点列表的总测地距离（Isaac 坐标，米）。"""
    if path_pts is None or len(path_pts) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(path_pts)):
        dx = path_pts[i][0] - path_pts[i-1][0]
        dy = path_pts[i][1] - path_pts[i-1][1]
        total += math.hypot(dx, dy)
    return total

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> int:
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    rng = random.Random(ARGS.seed)

    # ── Early-exit check ──────────────────────────────────────────────────
    out_dir  = _output_dir()
    out_path = _episode_path(out_dir)


    # ── vis_only 模式：直接读已有 JSON 生成图 ─────────────────────────────
    if ARGS.vis_only:
        if not os.path.exists(out_path):
            print(f"[ERROR] --vis_only 但 JSON 不存在: {out_path}")
            simulation_app.close()
            return 1
        if not ARGS.semantic_map_json or not os.path.exists(ARGS.semantic_map_json):
            print(f"[FATAL] --semantic_map_json 必须提供")
            simulation_app.close()
            return 3

        import json as _json
        with open(out_path, encoding="utf-8") as f:
            ep = _json.load(f)["episode"]

        grid, esdf, min_x, min_y, scale, _ = _load_2d_map(
            ARGS.semantic_map_json, ARGS.robot_radius_2d, ARGS.scale_m_per_px)

        all_y, all_x = [], []
        with open(ARGS.semantic_map_json) as f:
            _sem = _json.load(f)
        for inst in _sem:
            for y, x in inst.get("mask_coords_m", []):
                try: all_y.append(float(y)); all_x.append(float(x))
                except: pass
        map_min_x, map_max_x = min(all_x), max(all_x)
        map_min_y, map_max_y = min(all_y), max(all_y)

        robot_start = ep["robot"]["start_pos"]
        robot_goal  = ep["robot"]["goal_pos"]
        spawn_positions_dict = ep["characters"]["spawn_positions"]
        commands_dict        = ep["characters"]["commands"]

        _save_episode_vis(out_path, grid, esdf, min_x, min_y, scale,
                          map_min_x, map_max_x, map_min_y, map_max_y,
                          robot_start, robot_goal,
                          spawn_positions_dict, commands_dict)
        simulation_app.close()
        return 0

    if os.path.exists(out_path):
        if not ARGS.overwrite:
            print(f"[SKIP] {out_path}")
            simulation_app.close()
            return 0
        print(f"[OVERWRITE] {out_path}")

    # ── 加载 2D map（不依赖 Isaac Sim NavMesh）────────────────────────────
    if not ARGS.semantic_map_json or not os.path.exists(ARGS.semantic_map_json):
        print(f"[FATAL] --semantic_map_json 必须提供且存在")
        simulation_app.close()
        return 3

    print(f"[2D] Loading semantic map: {ARGS.semantic_map_json}")
    grid, esdf, min_x, min_y, scale, sem_data = _load_2d_map(
        ARGS.semantic_map_json,
        robot_radius_m=ARGS.robot_radius_2d,
        scale=ARGS.scale_m_per_px,
    )
    if grid is None:
        print("[FATAL] 2D map load failed")
        simulation_app.close()
        return 3

    # 计算坐标变换所需的边界
    all_y, all_x = [], []
    import json as _json
    with open(ARGS.semantic_map_json) as f:
        _sem = _json.load(f)
    for inst in _sem:
        for y, x in inst.get("mask_coords_m", []):
            try:
                all_y.append(float(y)); all_x.append(float(x))
            except: pass
    map_min_x, map_max_x = min(all_x), max(all_x)
    map_min_y, map_max_y = min(all_y), max(all_y)

    def to_isaac(x_m, y_m):
        return _sm_to_isaac(x_m, y_m, map_min_x, map_max_x, map_min_y, map_max_y)

    def sample_point():
        for _attempt in range(10):   # 最多重试10次找到面积够大的点
            px_py = _sample_free_point_2d(grid, esdf, rng, min_clearance_m=ARGS.robot_radius_2d * 0.5)
            if px_py is None:
                return None
            if _reachable_area_px(grid, px_py) < MIN_REACHABLE_PX:
                continue   # 被困在小角落，换一个
            x_2d, y_2d = _px_to_world(px_py[0], px_py[1], min_x, min_y, scale)
            ix, iy = to_isaac(x_2d, y_2d)
            return (ix, iy, 0.0)
        return None

    def plan_path(start_isaac, goal_isaac):
        """Isaac 坐标 → 2D 像素 → A* → Isaac 坐标路径点列表。"""
        # 逆变换：Isaac → 2D semantic map 坐标
        # _sm_to_isaac 的逆（flip+negate 是对合变换）
        def from_isaac(ix, iy):
            # negate 逆
            x_m, y_m = -ix, -iy
            # flip 逆（flip_x: x = (min+max)-x）
            x_m = (map_min_x + map_max_x) - x_m
            y_m = (map_min_y + map_max_y) - y_m
            return x_m, y_m

        sx_2d, sy_2d = from_isaac(start_isaac[0], start_isaac[1])
        gx_2d, gy_2d = from_isaac(goal_isaac[0], goal_isaac[1])

        spx = _world_to_px(sx_2d, sy_2d, min_x, min_y, scale)
        gpx = _world_to_px(gx_2d, gy_2d, min_x, min_y, scale)

        # snap 到最近自由点
        def snap(px_py):
            H, W = grid.shape
            px, py = px_py
            if 0 <= px < W and 0 <= py < H and grid[py, px] == 0:
                return px_py
            # BFS 找最近自由点
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

        spx = snap(spx)
        gpx = snap(gpx)
        if spx is None or gpx is None:
            return None

        path_px = _astar(grid, spx, gpx)
        if path_px is None:
            return None

        # 转回 Isaac 坐标
        result = []
        for (px, py) in path_px:
            x_2d, y_2d = _px_to_world(px, py, min_x, min_y, scale)
            ix, iy = to_isaac(x_2d, y_2d)
            result.append([ix, iy, 0.0])
        return result

    # ── 此时才需要打开 Isaac Sim（只为了写 JSON，NavMesh 不再用于路径）──
    if not open_stage(ARGS.usda):
        simulation_app.close()
        return 1

    # ── Bake NavMesh（用于连通性验证）────────────────────────────────────
    navvols_usda = cache_paths(ARGS.usda, ARGS.cache_dir)
    nav_ok, inav = bake_navmesh(
        app                    = simulation_app,
        navvols_usda           = navvols_usda,
        force_rebake           = ARGS.force_rebake,
        collision_root         = ARGS.collision_root,
        volume_padding         = ARGS.volume_padding,
        fallback_size          = ARGS.fallback_size,
        warmup_frames          = ARGS.warmup_frames,
        semantic_map_json      = None,
    )
    nm = inav.get_navmesh() if nav_ok else None
    if nm is None:
        print("[WARN] NavMesh bake failed, skipping NavMesh connectivity check")
    else:
        print("[NavMesh] Bake OK, will use for connectivity validation")

    # ── 采样 robot pose ───────────────────────────────────────────────────
    print("\n[GEN] Sampling robot pose (2D map)...")
    robot_start = robot_goal = None
    for _ in range(200):
        s = sample_point()
        g = sample_point()
        if s is None or g is None:
            continue
        if _dist2(s, g) < 2.0:   # 粗筛
            continue
        path_pts = plan_path(s, g)
        if path_pts is None:
            continue
        geo = _path_length(path_pts)
        if not (MIN_ROBOT_DIST <= geo <= MAX_ROBOT_DIST):
            continue
        if not _navmesh_connected(nm, s, g):
            print(f"  [SKIP] NavMesh disconnected: {s[:2]} → {g[:2]}")
            continue
        robot_start, robot_goal = s, g
        print(f"  geodesic start→goal: {geo:.1f}m")
        break
    if robot_start is None:
        robot_start = sample_point()
        robot_goal  = sample_point()
    robot_orientation = rng.uniform(0, 2*math.pi)
    print(f"  start : {robot_start}\n  goal  : {robot_goal}")

    # ── 采样 character positions ──────────────────────────────────────────
    print(f"\n[GEN] Sampling {ARGS.num_people} character position(s) (2D map)...")
    char_positions = []
    occupied = [robot_start, robot_goal]
    for i in range(ARGS.num_people):
        placed = False
        for attempt in range(200):
            p = sample_point()
            if p is None:
                continue
            # 欧氏距离粗筛（下界，够快）
            if _dist2(p, robot_start) < MIN_CHAR_DIST:
                continue
            if _dist2(p, robot_goal) < MIN_CHAR_DIST:
                continue
            # 连通性检查：spawn 点必须和机器人起点连通
            # 用 plan_path 顺便验证，不需要存路径
            if plan_path(p, robot_start) is None:
                continue
            if not _navmesh_connected(nm, p, robot_start):
                continue
            char_positions.append(p)
            occupied.append(p)
            placed = True
            break

        if not placed:
            # 放宽欧氏距离到 2m，但保留连通性检查
            for attempt in range(200):
                p = sample_point()
                if p is None:
                    continue
                if _dist2(p, robot_start) < 2.0:
                    continue
                if _dist2(p, robot_goal) < 2.0:
                    continue
                if plan_path(p, robot_start) is None:
                    continue
                char_positions.append(p)
                placed = True
                break

        if not placed:
            print(f"[WARN] Could not place character {i} with distance constraint")
            char_positions.append(None)

    # ── 采样 waypoints ────────────────────────────────────────────────────
    spawn_positions_dict = {}
    commands_dict = {}

    for i, pos in enumerate(char_positions):
        if pos is None:
            continue
        char_name = CharacterUtil.get_character_name_by_index(i)
        spawn_positions_dict[char_name] = {
            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
            "rot": 0.0,
        }
        print(f"\n[GEN] {char_name}: sampling {ARGS.num_waypoints} waypoints (2D A*)...")
        commands = []
        prev = pos  # spawn 点

        for wi in range(ARGS.num_waypoints):
            placed_wp = False
            for min_dist in WAYPOINT_DIST_TIERS:
                for _ in range(200):
                    target = sample_point()
                    if target is None:
                        continue
                    if _dist2(target, prev) < min_dist * 0.4:   # 欧氏粗筛
                        continue
                    path_pts = plan_path(prev, target)
                    if path_pts is None:
                        continue
                    if _path_length(path_pts) < min_dist:
                        continue
                    if not _navmesh_connected(nm, target, robot_start):
                        continue
                    x, y, z = target
                    cmd = {
                        "cmd": "GoTo",
                        "params": [f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", "_"],
                        "path": path_pts,
                    }
                    commands.append(cmd)
                    geo_dist = _path_length(path_pts)
                    print(f"    GoTo#{wi}: {len(path_pts)} path points, "
                          f"geodesic={geo_dist:.1f}m (min={min_dist}m) → ({x:.2f},{y:.2f})")
                    prev = target
                    placed_wp = True
                    break
                if placed_wp:
                    break
            if not placed_wp:
                print(f"    GoTo#{wi}: failed at all tiers, skipping")
        commands_dict[char_name] = commands

    # ── 写 JSON ───────────────────────────────────────────────────────────
    episode_data = {
        "episode": {
            "episode_id": ARGS.episode_id,
            "robot": {
                "start_pos":         list(robot_start),
                "goal_pos":          list(robot_goal),
                "start_orientation": robot_orientation,
            },
            "characters": {
                "num_characters":  len(spawn_positions_dict),
                "spawn_positions": spawn_positions_dict,
                "commands":        commands_dict,
            },
        }
    }

    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(episode_data, f, indent=4)
    print(f"\n[GEN] Episode saved → {out_path}")

    # ── 可视化 ────────────────────────────────────────────────────────────
    _save_episode_vis(
        out_path,
        grid, esdf, min_x, min_y, scale,
        map_min_x, map_max_x, map_min_y, map_max_y,
        robot_start, robot_goal,
        spawn_positions_dict, commands_dict,
    )

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())