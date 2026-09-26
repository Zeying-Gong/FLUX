"""
generate_episode_json_interaction.py  (accelerated, min_people=1)
────────────────────────────────────────────────────────────────
Episode generator with:
  - optional NavMesh skip (--skip_navmesh_check)
  - reduced retries
  - 4‑directional A*
  - lightweight visualisation
  - adaptive pedestrian avoidance levels
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import heapq
import numpy as np
from collections import deque
from scipy.ndimage import distance_transform_edt

# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--usda",               required=True)
    p.add_argument("--output_dir",         default=None)
    p.add_argument("--num_people",         type=int, default=10)
    p.add_argument("--num_people_for_episode", nargs="+", type=int, default=None)
    p.add_argument("--episode_ids",        nargs="+", type=int, default=[0])
    p.add_argument("--seed_offset",        type=int, default=0)
    p.add_argument("--overwrite",          action="store_true")
    p.add_argument("--vis_only",           action="store_true")
    p.add_argument("--no_vis",             action="store_true")
    p.add_argument("--skip_navmesh_check", action="store_true",
                   help="Disable NavMesh baking and connectivity checks.")
    p.add_argument("--semantic_map_json",  required=True)
    p.add_argument("--robot_radius_2d",    type=float, default=0.3)
    p.add_argument("--scale_m_per_px",     type=float, default=0.05)
    p.add_argument("--collision_root",     default="/World/scene_collision")
    p.add_argument("--volume_padding",     type=float, default=1.2)
    p.add_argument("--fallback_size",      type=float, default=100.0)
    p.add_argument("--warmup_frames",      type=int,   default=60)
    p.add_argument("--cache_dir",          default=None)
    p.add_argument("--force_rebake",       action="store_true")
    return p.parse_args()

ARGS = parse_args()

# ═══════════════════════════════════════════════════════════════════════════
# SimulationApp
# ═══════════════════════════════════════════════════════════════════════════
from isaacsim import SimulationApp

_exp = os.environ.get("EXP_PATH")
if _exp is None:
    print("[FATAL] EXP_PATH not set"); sys.exit(1)

simulation_app = SimulationApp(
    launch_config={
        "renderer": "RayTracedLighting",
        "headless": True,
        "enable_cameras": False,
        "crash_reporter/enabled": False,
        "crash_reporter/skip_old_dump_upload": True,
    },
    experience=os.path.join(
        _exp, "isaacsim.exp.action_and_event_data_generation.base.kit"),
)

import carb
import omni.usd
from isaacsim.replicator.agent.core.stage_util import CharacterUtil
from navmesh_utils import cache_paths, bake_navmesh
sys.path.insert(0, os.path.dirname(__file__))

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
MIN_ROBOT_DIST              = 5.0
MAX_ROBOT_DIST              = 20.0
MIN_CHAR_DIST               = 3.0
MIN_CHAR_SPACING            = 3.0
MIN_CHAR_GOAL_TO_ROBOT_GOAL = 3.0
MIN_ISLAND_AREA_THRESHOLD   = 20.0
MIN_ISLAND_AREA_PER_PERSON  = 20.0
MIN_REACHABLE_PX            = 400
NUM_CHAIN_WPT               = 3
AREA_PER_PERSON_M2          = 20.0
PED_SPAWN_MIN_CLEARANCE_M   = 0.3
PED_RADIUS_M                = 0.3
PED_SEPARATION_M            = 0.8
ROBOT_ESDF_WEIGHT           = 0.12
PED_ESDF_WEIGHT             = 0.0
PED_PSDF_WEIGHT             = 2.0
PSDF_INNER_R_M              = 1.0
PSDF_OUTER_R_M              = 2.5
PSDF_INNER_W                = 3.0
PSDF_OUTER_W                = 1.0
PSDF_CAP                    = 5.0
_PSDF_BLOCK_THR: float      = 0.5
WAYPOINT_MIN_DIST_M         = 1.5
PED_CORRIDOR_BLOCK_R = [0.6, 0.3, 0.0]
_TRANSITION     = {"Idle": 0.4, "LookAround": 0.4, None: 0.2}
_IDLE_RANGE     = (3.0, 10.0)
_LOOKAROUND_RANGE = (3.0, 8.0)

# ═══════════════════════════════════════════════════════════════════════════
# 2D map helpers (unchanged)
# ═══════════════════════════════════════════════════════════════════════════
def _load_2d_map(sem_json, robot_r, scale):
    with open(sem_json, encoding="utf-8") as f:
        sem_data = json.load(f)
    all_y, all_x = [], []
    for inst in sem_data:
        for y, x in inst.get("mask_coords_m", []):
            try: all_y.append(float(y)); all_x.append(float(x))
            except: pass
    if not all_y:
        return None, None, None, None, None, None, None
    min_y, max_y = min(all_y), max(all_y)
    min_x, max_x = min(all_x), max(all_x)
    h = int(np.ceil((max_y - min_y) / scale)) + 1
    w = int(np.ceil((max_x - min_x) / scale)) + 1

    raw_grid = np.zeros((h, w), dtype=np.uint8)
    for inst in sem_data:
        label = str(inst.get("category_label", "")).lower()
        if label in ("wall", "unable area"):
            for y_m, x_m in inst.get("mask_coords_m", []):
                try:
                    py = int(round((float(y_m) - min_y) / scale))
                    px = int(round((float(x_m) - min_x) / scale))
                    if 0 <= py < h and 0 <= px < w:
                        raw_grid[py, px] = 1
                except: pass

    if robot_r > 0:
        dist_m = distance_transform_edt(raw_grid == 0, sampling=scale)
        planning_grid = (dist_m <= robot_r).astype(np.uint8)
    else:
        planning_grid = raw_grid.copy()

    esdf = distance_transform_edt(planning_grid == 0, sampling=scale)
    return planning_grid, esdf, min_x, min_y, scale, sem_data, raw_grid

def _world_to_px(x_m, y_m, min_x, min_y, scale):
    return (int(round((float(x_m) - min_x) / scale)),
            int(round((float(y_m) - min_y) / scale)))

def _px_to_world(px, py, min_x, min_y, scale):
    return min_x + (px + 0.5) * scale, min_y + (py + 0.5) * scale

def _sm_to_isaac(x_m, y_m, map_min_x, map_max_x, map_min_y, map_max_y):
    return -(map_min_x + map_max_x - float(x_m)), -(map_min_y + map_max_y - float(y_m))

def _snap(grid, px_py):
    H, W = grid.shape
    px, py = px_py
    if 0 <= px < W and 0 <= py < H and grid[py, px] == 0:
        return px_py
    visited = set(); q = deque([(px, py, 0)])
    while q:
        cx, cy, d = q.popleft()
        if d > 30: break
        if 0 <= cx < W and 0 <= cy < H and grid[cy, cx] == 0:
            return (cx, cy)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (cx+dx, cy+dy)
            if nb not in visited: visited.add(nb); q.append((nb[0], nb[1], d+1))
    return None

def _reachable_area_px(grid, start_px) -> int:
    H, W = grid.shape
    sx, sy = start_px
    if not (0 <= sx < W and 0 <= sy < H) or grid[sy, sx] != 0:
        return 0
    limit = MIN_REACHABLE_PX * 10
    visited = {start_px}; q = deque([start_px]); count = 0
    while q and count < limit:
        cx, cy = q.popleft(); count += 1
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (cx+dx, cy+dy); nx, ny = nb
            if nb not in visited and 0<=nx<W and 0<=ny<H and grid[ny,nx]==0:
                visited.add(nb); q.append(nb)
    return count

def _dist2(a, b):
    return math.hypot(float(a[0])-float(b[0]), float(a[1])-float(b[1]))

def _path_length(pts):
    if not pts or len(pts) < 2: return 0.0
    return sum(math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
               for i in range(1, len(pts)))

def _navmesh_connected(nm, a, b) -> bool:
    if ARGS.skip_navmesh_check:
        return True
    if nm is None: return True
    try:
        pa = carb.Float3(float(a[0]), float(a[1]), float(a[2]))
        pb = carb.Float3(float(b[0]), float(b[1]), float(b[2]))
        ra = nm.query_closest_point(pa, 1.0)
        rb = nm.query_closest_point(pb, 1.0)
        if ra is None or ra[0] is None or rb is None or rb[0] is None: return False
        sa, sb = ra[0], rb[0]
        if (math.hypot(float(sa[0])-float(pa[0]), float(sa[1])-float(pa[1])) > 1.0 or
            math.hypot(float(sb[0])-float(pb[0]), float(sb[1])-float(pb[1])) > 1.0):
            return False
        return nm.query_shortest_path(sa, sb, agent_radius=0.3) is not None
    except Exception:
        return False

# ═══════════════════════════════════════════════════════════════════════════
# PSDF (unchanged)
# ═══════════════════════════════════════════════════════════════════════════
def _make_psdf_map(H, W):
    return np.zeros((H, W), dtype=np.float32)

def _stamp_path_into_psdf(psdf, grid, path_px, scale):
    from scipy.ndimage import gaussian_filter as _gf
    H, W = grid.shape
    if not path_px:
        return
    inner_sigma = PSDF_INNER_R_M / scale
    outer_sigma = PSDF_OUTER_R_M / scale
    impulse = np.zeros((H, W), dtype=np.float32)
    for cx, cy in path_px:
        if 0 <= cx < W and 0 <= cy < H:
            impulse[cy, cx] = 1.0
    blurred = (PSDF_INNER_W * _gf(impulse, sigma=inner_sigma, mode='constant', cval=0.0) +
               PSDF_OUTER_W * _gf(impulse, sigma=outer_sigma, mode='constant', cval=0.0))
    blurred[grid == 1] = 0.0
    psdf += blurred

def _normalise_psdf(psdf, grid):
    if psdf.max() < 1e-6:
        return np.zeros_like(psdf)
    out = np.clip(psdf, 0.0, PSDF_CAP) / PSDF_CAP
    out[grid == 1] = 0.0
    return out

def _compute_psdf_block_thr(scale):
    d_px    = PED_SEPARATION_M / scale
    inner_s = PSDF_INNER_R_M  / scale
    outer_s = PSDF_OUTER_R_M  / scale
    raw_at_d = (PSDF_INNER_W * math.exp(-0.5 * (d_px / inner_s) ** 2) +
                PSDF_OUTER_W * math.exp(-0.5 * (d_px / outer_s) ** 2))
    return min(raw_at_d / PSDF_CAP, 1.0)

# ═══════════════════════════════════════════════════════════════════════════
# A*  (4‑directional, faster)
# ═══════════════════════════════════════════════════════════════════════════
def _astar(grid, start_px, goal_px,
           esdf=None, min_corridor_m=0.0, esdf_weight=0.0,
           psdf_norm=None, psdf_weight=0.0, psdf_block_thr=0.0):
    H, W = grid.shape
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]   # 4‑directional
    open_set = [(0.0, start_px)]
    came_from = {}
    g = {start_px: 0.0}
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal_px:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]; path.append(cur)
            return path[::-1]
        for dx, dy in dirs:
            nx, ny = cur[0]+dx, cur[1]+dy
            if not (0 <= nx < W and 0 <= ny < H): continue
            if grid[ny, nx] == 1: continue
            if esdf is not None and min_corridor_m > 0:
                if esdf[ny, nx] < min_corridor_m: continue
            if psdf_norm is not None and psdf_block_thr > 0:
                if float(psdf_norm[ny, nx]) >= psdf_block_thr: continue
            nb  = (nx, ny)
            step = 1.0   # 4‑directional cost is always 1
            if esdf is not None and esdf_weight > 0:
                step += esdf_weight / max(esdf[ny, nx], 1e-3)
            if psdf_norm is not None and psdf_weight > 0:
                step += psdf_weight * float(psdf_norm[ny, nx])
            tg = g[cur] + step
            if nb not in g or tg < g[nb]:
                came_from[nb] = cur; g[nb] = tg
                heapq.heappush(open_set,
                    (tg + math.hypot(nx-goal_px[0], ny-goal_px[1]), nb))
    return None

# ═══════════════════════════════════════════════════════════════════════════
# Island allocation (unchanged)
# ═══════════════════════════════════════════════════════════════════════════
def _plan_island_allocation(actual_num, robot_island_idx, valid_islands):
    if actual_num <= 0: return []
    if actual_num == 1: return [robot_island_idx]
    n_isl  = len(valid_islands)
    caps   = [max(1, int(isl["area_m2"] / MIN_ISLAND_AREA_PER_PERSON)) for isl in valid_islands]
    total  = sum(isl["area_m2"] for isl in valid_islands)
    raw    = [isl["area_m2"] / total * actual_num for isl in valid_islands]
    counts = [int(r) for r in raw]
    for idx, _ in sorted(enumerate(raw), key=lambda x: -(x[1]-int(x[1])))[:actual_num-sum(counts)]:
        counts[idx] += 1
    overflow = sum(max(0, c - caps[k]) for k, c in enumerate(counts))
    counts   = [min(c, caps[k]) for k, c in enumerate(counts)]
    for idx in sorted(range(n_isl), key=lambda k: -valid_islands[k]["area_m2"]):
        if overflow <= 0: break
        add = min(caps[idx] - counts[idx], overflow)
        counts[idx] += add; overflow -= add
    if counts[robot_island_idx] == 0:
        donor = max((k for k in range(n_isl) if k != robot_island_idx and counts[k] > 0),
                    key=lambda k: counts[k], default=None)
        if donor is not None:
            counts[donor] -= 1; counts[robot_island_idx] += 1
    allocation = [robot_island_idx]
    remaining  = counts[:]
    remaining[robot_island_idx] -= 1
    for idx in sorted(range(n_isl), key=lambda k: -valid_islands[k]["area_m2"]):
        allocation.extend([idx] * remaining[idx])
    assert len(allocation) == actual_num
    return allocation

def _sample_aux(rng: random.Random):
    k = rng.choices(list(_TRANSITION.keys()), weights=list(_TRANSITION.values()), k=1)[0]
    if k == "Idle":
        return {"cmd": "Idle", "params": [str(round(rng.uniform(*_IDLE_RANGE), 1))]}
    if k == "LookAround":
        return {"cmd": "LookAround", "params": [str(round(rng.uniform(*_LOOKAROUND_RANGE), 1))]}
    return None

def _update(n=1):
    for _ in range(n): simulation_app.update()

def open_stage(usda_path):
    if not os.path.exists(usda_path):
        print(f"[FATAL] USDA not found: {usda_path}"); return False
    omni.usd.get_context().open_stage(usda_path)
    _update(30)
    waited = 0
    while omni.usd.get_context().get_stage_loading_status()[1] > 0:
        simulation_app.update(); waited += 1
        if waited > 3000: print("[WARN] Stage still loading after 30s."); break
    return omni.usd.get_context().get_stage() is not None

def _scene_id():
    return os.path.splitext(os.path.basename(ARGS.usda))[0]

def _output_dir():
    sid = _scene_id()
    d = os.path.join(ARGS.output_dir, sid) if ARGS.output_dir else \
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ARGS.usda))),
                     "episodes", sid)
    os.makedirs(d, exist_ok=True)
    return d

def _sample_free_point(grid, esdf, rng, min_clearance, _free_cache=None):
    if _free_cache is not None:
        key = (id(grid), min_clearance)
        if key not in _free_cache:
            ys, xs = np.where((grid == 0) & (esdf >= min_clearance))
            if len(ys) == 0: ys, xs = np.where(grid == 0)
            _free_cache[key] = list(zip(xs.tolist(), ys.tolist()))
        cands = _free_cache[key]
        if not cands: return None
        return cands[rng.randint(0, len(cands)-1)]
    H, W = grid.shape
    for _ in range(500):
        py = rng.randint(0, H-1); px = rng.randint(0, W-1)
        if grid[py, px] == 0 and esdf[py, px] >= min_clearance: return (px, py)
    for _ in range(500):
        py = rng.randint(0, H-1); px = rng.randint(0, W-1)
        if grid[py, px] == 0: return (px, py)
    return None

# ═══════════════════════════════════════════════════════════════════════════
# Visualisation  (lighter: smaller figure, lower dpi)
# ═══════════════════════════════════════════════════════════════════════════
def _save_vis(out_path, grid, esdf, min_x, min_y, scale,
              map_min_x, map_max_x, map_min_y, map_max_y,
              robot_start, robot_goal, spawn_dict, commands_dict,
              episode_id=0):
    if ARGS.no_vis:
        return
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[VIS] matplotlib not available, skipping"); return

    H, W = grid.shape
    ped_colors = ['#FF6B6B','#FFD93D','#6BCB77','#4D96FF','#FF9F45','#C77DFF']

    def from_isaac(ix, iy):
        x_m, y_m = -float(ix), -float(iy)
        return (map_min_x+map_max_x)-x_m, (map_min_y+map_max_y)-y_m

    def i2px(ix, iy):
        x2, y2 = from_isaac(ix, iy)
        return int(round((x2-min_x)/scale)), int(round((y2-min_y)/scale))

    psdf_vis = _make_psdf_map(H, W)
    for cn, sd in spawn_dict.items():
        _stamp_path_into_psdf(psdf_vis, grid, [i2px(*sd["pos"][:2])], scale)
        for cmd in commands_dict.get(cn, []):
            if cmd.get("cmd") != "GoTo": continue
            path_px = [i2px(p[0], p[1]) for p in cmd.get("path", [])]
            if path_px:
                _stamp_path_into_psdf(psdf_vis, grid, path_px, scale)

    def draw_agents(ax, draw_paths=True):
        if robot_start:
            rs = i2px(*robot_start[:2])
            ax.plot(rs[0], rs[1], 's', color='cyan', markersize=13, zorder=8)
            ax.annotate("R_s", rs, fontsize=7, color='cyan',
                        xytext=(4,4), textcoords='offset points')
        if robot_goal:
            rg = i2px(*robot_goal[:2])
            ax.plot(rg[0], rg[1], '*', color='cyan', markersize=17, zorder=8)
            ax.annotate("R_g", rg, fontsize=7, color='cyan',
                        xytext=(4,4), textcoords='offset points')
        for ci, (cn, sd) in enumerate(spawn_dict.items()):
            c    = ped_colors[ci % len(ped_colors)]
            sp3  = i2px(*sd["pos"][:2])
            ax.plot(sp3[0], sp3[1], 'o', color=c, markersize=9, zorder=7,
                    markeredgecolor='k', markeredgewidth=0.8)
            ax.annotate(f"P{ci}", sp3, fontsize=7, color=c, fontweight='bold',
                        xytext=(3,3), textcoords='offset points')
            goto_cmds = [cmd for cmd in commands_dict.get(cn, [])
                         if cmd.get("cmd") == "GoTo"]
            for wi, cmd in enumerate(goto_cmds):
                params = cmd.get("params", [])
                if len(params) < 2: continue
                wp      = i2px(float(params[0]), float(params[1]))
                is_last = (wi == len(goto_cmds)-1)
                ax.plot(wp[0], wp[1], '*' if is_last else 'D', color=c,
                        markersize=11 if is_last else 7,
                        markeredgecolor='k', markeredgewidth=0.8, zorder=7)
                ax.annotate(f"P{ci}W{wi+1}", wp, fontsize=6, color=c,
                            xytext=(3,-10), textcoords='offset points')
                if draw_paths:
                    pts = [i2px(p[0], p[1]) for p in cmd.get("path", [])]
                    if pts:
                        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                                '-', color=c, lw=1.5, alpha=0.7, zorder=5)

    n_peds = len(spawn_dict)
    # smaller figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    ax_occ, ax_esdf, ax_psdf = axes

    vis_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    vis_rgb[grid==0] = [240,240,240]; vis_rgb[grid==1] = [50,50,50]
    ax_occ.imshow(vis_rgb, origin="lower")
    ax_occ.set_title("① Occupancy + Agents & Paths", fontsize=10, fontweight='bold')
    draw_agents(ax_occ, draw_paths=True)

    esdf_m = np.ma.masked_where(grid==1, np.clip(esdf, 0, 2))
    im1 = ax_esdf.imshow(esdf_m, origin="lower", cmap="viridis",
                          vmin=0, vmax=2, interpolation="bilinear")
    plt.colorbar(im1, ax=ax_esdf, fraction=0.046, pad=0.04,
                 label="ESDF clearance (m, clipped 2 m)")
    ax_esdf.set_title("② ESDF (Static Obstacle Clearance)", fontsize=10, fontweight='bold')
    draw_agents(ax_esdf, draw_paths=False)

    p95 = float(np.percentile(psdf_vis[grid==0], 95)) if (grid==0).any() else 1.0
    psdf_m = np.ma.masked_where(grid==1, np.clip(psdf_vis, 0, p95))
    im2 = ax_psdf.imshow(psdf_m, origin="lower", cmap="hot_r", interpolation="bilinear")
    plt.colorbar(im2, ax=ax_psdf, fraction=0.046, pad=0.04,
                 label="Ped proximity cost (higher = more crowded)")
    if p95 > 0:
        ax_psdf.contour(np.clip(psdf_vis, 0, p95),
                        levels=[p95*0.2, p95*0.6],
                        colors=['orange', 'red'], linewidths=[1.0, 1.5],
                        origin="lower", zorder=6)
    ax_psdf.set_title(
        f"③ PSDF — ped avoidance field  "
        f"(inner={PSDF_INNER_R_M} m  outer={PSDF_OUTER_R_M} m)",
        fontsize=10, fontweight='bold')
    draw_agents(ax_psdf, draw_paths=True)

    plt.suptitle(f"Episode {episode_id}  —  {_scene_id()}   peds={n_peds}",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    vis_path = out_path.replace(".json", "_vis.png")
    plt.savefig(vis_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"[VIS] → {vis_path}")

# ═══════════════════════════════════════════════════════════════════════════
# Core episode generator (with adaptive avoidance & reduced retries)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_one_episode(
    episode_id, seed, out_dir,
    grid, esdf, min_x, min_y, scale,
    map_min_x, map_max_x, map_min_y, map_max_y,
    nm, to_isaac, plan_path_fn,
    nav_area_m2, actual_num,
    valid_islands, px_to_island,
    get_isaac_island,
):
    t0 = time.time()
    random.seed(seed); np.random.seed(seed)
    rng = random.Random(seed)

    out_path = os.path.join(out_dir, f"episode_{episode_id}.json")
    if os.path.exists(out_path) and not ARGS.overwrite:
        print(f"  [SKIP] episode_{episode_id}"); return {"skipped": True}

    H, W = grid.shape
    _free_cache: dict = {}
    psdf = _make_psdf_map(H, W)
    placed_path_px_union: set = set()

    global _PSDF_BLOCK_THR
    _PSDF_BLOCK_THR = _compute_psdf_block_thr(scale)
    print(f"  [ep{episode_id}] PSDF hard-block thr={_PSDF_BLOCK_THR:.3f}")

    def _from_isaac(ix, iy):
        x_m, y_m = -float(ix), -float(iy)
        return (map_min_x+map_max_x)-x_m, (map_min_y+map_max_y)-y_m

    def _isaac_to_px(ix, iy):
        return _world_to_px(*_from_isaac(ix, iy), min_x, min_y, scale)

    def sample_point():
        for _ in range(10):
            px_py = _sample_free_point(grid, esdf, rng,
                                       PED_SPAWN_MIN_CLEARANCE_M, _free_cache)
            if px_py is None: return None
            if _reachable_area_px(grid, px_py) < MIN_REACHABLE_PX: continue
            x_2d, y_2d = _px_to_world(px_py[0], px_py[1], min_x, min_y, scale)
            ix, iy = to_isaac(x_2d, y_2d)
            return (ix, iy, 0.0)
        return None

    # ── Robot sampling (reduced retries) ──────────────────────────────
    robot_start = robot_goal = robot_island_idx = None

    # L1: strict (100 attempts)
    for attempt in range(100):
        s = sample_point(); g = sample_point()
        if s is None or g is None: continue
        if _dist2(s, g) < 2.0: continue
        path_pts = plan_path_fn(s, g)
        if path_pts is None: continue
        geo = _path_length(path_pts)
        if not (MIN_ROBOT_DIST <= geo <= MAX_ROBOT_DIST): continue
        if not _navmesh_connected(nm, s, g): continue
        s_island = get_isaac_island(s)
        if s_island is None: continue
        robot_start, robot_goal, robot_island_idx = s, g, s_island
        print(f"  [ep{episode_id}] robot geo={geo:.1f}m island={s_island} attempt={attempt+1}")
        break

    if robot_start is None:
        print(f"  [ep{episode_id}] WARN: L1 failed → L2 (relax MAX_DIST)")
        for attempt in range(200):  # L2: 200 attempts
            s = sample_point(); g = sample_point()
            if s is None or g is None: continue
            if _dist2(s, g) < MIN_ROBOT_DIST: continue
            if plan_path_fn(s, g) is None: continue
            s_island = get_isaac_island(s)
            if s_island is None: continue
            robot_start, robot_goal, robot_island_idx = s, g, s_island
            print(f"  [ep{episode_id}] L2 OK attempt={attempt+1}")
            break

    if robot_start is None:
        print(f"  [ep{episode_id}] WARN: L2 failed → L3 (largest island)")
        for attempt in range(200):
            s = sample_point(); g = sample_point()
            if s is None or g is None: continue
            if get_isaac_island(s) != 0 or get_isaac_island(g) != 0: continue
            if plan_path_fn(s, g) is None: continue
            robot_start, robot_goal, robot_island_idx = s, g, 0
            print(f"  [ep{episode_id}] L3 OK attempt={attempt+1}")
            break

    if robot_start is None:
        print(f"  [ep{episode_id}] FATAL: robot sampling completely failed")
        return {"skipped": True}

    robot_orientation = rng.uniform(0, 2*math.pi)

    # ── Pedestrian sampling ───────────────────────────────────────────
    island_plan = _plan_island_allocation(actual_num, robot_island_idx, valid_islands)
    print(f"  [ep{episode_id}] island_plan={island_plan}")

    spawn_positions_dict: dict = {}
    commands_dict:        dict = {}

    # Adaptive blocking levels
    if actual_num <= 2:
        active_block_levels = [2]          # only no-block
    elif actual_num <= 4:
        active_block_levels = [1, 2]       # skip strongest block
    else:
        active_block_levels = [0, 1, 2]    # full cascade

    def _build_blocked_grid(block_r_m):
        if block_r_m <= 0 or not placed_path_px_union:
            return grid
        from scipy.ndimage import binary_dilation
        r_px = max(1, int(round(block_r_m / scale)))
        ys, xs = np.ogrid[-r_px:r_px+1, -r_px:r_px+1]
        struct = (xs**2 + ys**2 <= r_px**2)
        mask = np.zeros((H, W), dtype=bool)
        for px, py in placed_path_px_union:
            if 0 <= px < W and 0 <= py < H:
                mask[py, px] = True
        dilated = binary_dilation(mask, structure=struct)
        blocked = grid.copy()
        blocked[dilated] = 1
        return blocked

    def _try_place_ped(i: int) -> bool:
        target_island = island_plan[i] if i < len(island_plan) else None
        same_island   = (target_island == robot_island_idx)

        for level in active_block_levels:
            block_r = PED_CORRIDOR_BLOCK_R[level]
            min_spawn_dist = 2.0 if level > 0 else MIN_CHAR_DIST
            b_grid = _build_blocked_grid(block_r)
            b_esdf = distance_transform_edt(b_grid == 0, sampling=scale) if block_r > 0 and placed_path_px_union else esdf

            def sample_point_blocked():
                for _ in range(10):
                    px_py = _sample_free_point(b_grid, b_esdf, rng,
                                               PED_SPAWN_MIN_CLEARANCE_M,
                                               _free_cache if block_r == 0 else None)
                    if px_py is None: return None
                    if _reachable_area_px(b_grid, px_py) < MIN_REACHABLE_PX: continue
                    x_2d, y_2d = _px_to_world(px_py[0], px_py[1], min_x, min_y, scale)
                    ix, iy = to_isaac(x_2d, y_2d)
                    return (ix, iy, 0.0)
                return None

            def plan_path_ped_blocked(a, b):
                spx = _snap(b_grid, _world_to_px(*_from_isaac(a[0], a[1]), min_x, min_y, scale))
                gpx = _snap(b_grid, _world_to_px(*_from_isaac(b[0], b[1]), min_x, min_y, scale))
                if spx is None or gpx is None: return None
                psdf_norm = _normalise_psdf(psdf, b_grid)
                path_px = _astar(b_grid, spx, gpx,
                                 esdf=b_esdf,
                                 min_corridor_m=0.0,
                                 esdf_weight=PED_ESDF_WEIGHT,
                                 psdf_norm=psdf_norm,
                                 psdf_weight=PED_PSDF_WEIGHT)
                if path_px is None: return None
                isaac_pts = []
                for px, py in path_px:
                    x_2d, y_2d = _px_to_world(px, py, min_x, min_y, scale)
                    ix, iy = to_isaac(x_2d, y_2d)
                    isaac_pts.append([ix, iy, 0.0])
                return isaac_pts, path_px

            # 80 attempts per level
            for attempt in range(80):
                p0 = sample_point_blocked()
                if p0 is None: continue
                if same_island and _dist2(p0, robot_start) < min_spawn_dist: continue
                if any(_dist2(p0, sp["pos"]) < MIN_CHAR_SPACING
                       for sp in spawn_positions_dict.values()): continue
                p0_island = get_isaac_island(p0)
                if target_island is not None and p0_island != target_island: continue
                if p0_island is None: continue
                if same_island:
                    if plan_path_ped_blocked(p0, robot_start) is None: continue
                    if not _navmesh_connected(nm, p0, robot_start): continue

                chain    = [(p0, None)]
                prev     = p0
                chain_ok = True
                for _ in range(NUM_CHAIN_WPT):
                    wpt_ok = False
                    for _ in range(120):  # reduced from 200
                        w = sample_point_blocked()
                        if w is None: continue
                        if _dist2(w, prev) < 2.0: continue
                        if not same_island and get_isaac_island(w) != target_island:
                            continue
                        result = plan_path_ped_blocked(prev, w)
                        if result is None: continue
                        isaac_pts, _ = result
                        if _path_length(isaac_pts) < WAYPOINT_MIN_DIST_M: continue
                        chain.append((w, result)); prev = w; wpt_ok = True; break
                    if not wpt_ok: chain_ok = False; break
                if not chain_ok: continue

                wpts = chain[1:]
                if (len(wpts) >= 2 and
                        _dist2(wpts[-1][0], robot_goal) < MIN_CHAR_GOAL_TO_ROBOT_GOAL):
                    best_k, best_d = None, -1.0
                    for k in range(len(wpts)-1):
                        d = _dist2(wpts[k][0], robot_goal)
                        if d >= MIN_CHAR_GOAL_TO_ROBOT_GOAL and d > best_d:
                            best_d = d; best_k = k
                    if best_k is not None:
                        wpts[best_k], wpts[-1] = wpts[-1], wpts[best_k]
                        r = plan_path_ped_blocked(chain[best_k][0], wpts[best_k][0])
                        if r: wpts[best_k] = (wpts[best_k][0], r)
                        r = plan_path_ped_blocked(wpts[-2][0], wpts[-1][0])
                        if r: wpts[-1] = (wpts[-1][0], r)
                chain = [chain[0]] + wpts

                new_path_px: set = {_isaac_to_px(*p0[:2])}
                for _, plan_result in chain[1:]:
                    if plan_result is None: continue
                    _, path_px = plan_result
                    new_path_px.update(path_px)

                _stamp_path_into_psdf(psdf, grid, list(new_path_px), scale)
                placed_path_px_union.update(new_path_px)

                cn = CharacterUtil.get_character_name_by_index(
                    len(spawn_positions_dict))
                spawn_positions_dict[cn] = {
                    "pos": [float(p0[0]), float(p0[1]), 0.0], "rot": 0.0}
                cmds = []
                for _, plan_result in chain[1:]:
                    if plan_result is None: continue
                    isaac_pts, _ = plan_result
                    if _path_length(isaac_pts) < WAYPOINT_MIN_DIST_M: continue
                    x, y, z = isaac_pts[-1]
                    cmds.append({
                        "cmd": "GoTo",
                        "params": [f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", "_"],
                        "path": isaac_pts,
                    })
                    aux = _sample_aux(rng)
                    if aux: cmds.append(aux)
                commands_dict[cn] = cmds

                seg_info = " | ".join(
                    f"wpt{wi}: {_path_length(r[0]):.1f}m"
                    for wi, (_, r) in enumerate(chain[1:]) if r is not None)
                level_str = ["L0(0.6m)", "L1(0.3m)", "L2(no-block)"][level]
                print(f"  [ep{episode_id}] ped{i} OK {level_str} "
                      f"attempt={attempt+1} island={p0_island} segs=[{seg_info}]")
                return True

            print(f"  [ep{episode_id}] ped{i} level {level} exhausted")

        return False

    for i in range(actual_num):
        if not _try_place_ped(i):
            print(f"  [ep{episode_id}] WARN: could not place pedestrian {i}")

    # ── Write JSON ────────────────────────────────────────────────────
    episode_data = {
        "episode": {
            "episode_id": episode_id,
            "seed":       seed,
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
        json.dump(episode_data, f, indent=4)

    elapsed = round(time.time()-t0, 2)
    print(f"  [ep{episode_id}] done  placed={len(spawn_positions_dict)}  "
          f"elapsed={elapsed}s  → {out_path}")

    _save_vis(out_path, grid, esdf, min_x, min_y, scale,
              map_min_x, map_max_x, map_min_y, map_max_y,
              robot_start, robot_goal,
              spawn_positions_dict, commands_dict,
              episode_id=episode_id)

    return {"episode_id": episode_id, "placed": len(spawn_positions_dict),
            "elapsed_s": elapsed}

# ═══════════════════════════════════════════════════════════════════════════
# Main  (adapted for --skip_navmesh_check)
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    out_dir = _output_dir()
    print(f"[GEN] Output dir: {out_dir}")

    if not os.path.exists(ARGS.semantic_map_json):
        print("[FATAL] --semantic_map_json not found"); simulation_app.close(); return 3

    grid, esdf, min_x, min_y, scale, sem_data, raw_grid = _load_2d_map(
        ARGS.semantic_map_json, ARGS.robot_radius_2d, ARGS.scale_m_per_px)
    if grid is None:
        print("[FATAL] 2D map load failed"); simulation_app.close(); return 3

    all_y, all_x = [], []
    for inst in sem_data:
        for y, x in inst.get("mask_coords_m", []):
            try: all_y.append(float(y)); all_x.append(float(x))
            except: pass
    map_min_x, map_max_x = min(all_x), max(all_x)
    map_min_y, map_max_y = min(all_y), max(all_y)

    def to_isaac(x_m, y_m):
        return _sm_to_isaac(x_m, y_m, map_min_x, map_max_x, map_min_y, map_max_y)

    def plan_path(start_isaac, goal_isaac):
        spx = _snap(grid, _world_to_px(
            *[(map_min_x+map_max_x)-(-float(start_isaac[0])),
              (map_min_y+map_max_y)-(-float(start_isaac[1]))],
            min_x, min_y, scale))
        gpx = _snap(grid, _world_to_px(
            *[(map_min_x+map_max_x)-(-float(goal_isaac[0])),
              (map_min_y+map_max_y)-(-float(goal_isaac[1]))],
            min_x, min_y, scale))
        if spx is None or gpx is None: return None
        path_px = _astar(grid, spx, gpx,
                         esdf=esdf,
                         min_corridor_m=ARGS.robot_radius_2d * 0.5,
                         esdf_weight=ROBOT_ESDF_WEIGHT)
        if path_px is None: return None
        return [[*to_isaac(*_px_to_world(px, py, min_x, min_y, scale)), 0.0]
                for px, py in path_px]

    if ARGS.vis_only:
        for ep_id in ARGS.episode_ids:
            out_path = os.path.join(out_dir, f"episode_{ep_id}.json")
            if not os.path.exists(out_path):
                print(f"[VIS_ONLY] {out_path} not found, skipping"); continue
            with open(out_path, encoding="utf-8") as f:
                ep = json.load(f)["episode"]
            _save_vis(out_path, grid, esdf, min_x, min_y, scale,
                      map_min_x, map_max_x, map_min_y, map_max_y,
                      ep["robot"]["start_pos"], ep["robot"]["goal_pos"],
                      ep["characters"]["spawn_positions"],
                      ep["characters"]["commands"],
                      episode_id=ep_id)
        simulation_app.close(); return 0

    # Island analysis
    def _find_islands(area_grid, plan_grid, scale):
        from scipy.ndimage import label as _label
        free_mask = (area_grid == 0).astype(np.int32)
        labeled, n_labels = _label(free_mask)
        islands = []
        for lbl in range(1, n_labels + 1):
            area_ys, area_xs = np.where(labeled == lbl)
            area_m2 = len(area_ys) * scale ** 2
            plan_ys, plan_xs = np.where((labeled == lbl) & (plan_grid == 0))
            pixels = set(zip(plan_xs.tolist(), plan_ys.tolist()))
            islands.append({"pixels": pixels, "area_m2": area_m2})
        islands.sort(key=lambda x: x["area_m2"], reverse=True)
        return islands

    all_islands   = _find_islands(raw_grid, grid, scale)
    valid_islands = [isl for isl in all_islands
                     if isl["area_m2"] >= MIN_ISLAND_AREA_THRESHOLD]
    print(f"[GEN] Islands total={len(all_islands)} valid={len(valid_islands)}")
    for idx, isl in enumerate(valid_islands):
        print(f"  island[{idx}] area={isl['area_m2']:.1f}m²")

    if not valid_islands:
        all_px = set(zip(*np.where(grid==0)[::-1]))
        valid_islands = [{"pixels": all_px, "area_m2": len(all_px)*scale**2}]

    px_to_island = {}
    for idx, isl in enumerate(valid_islands):
        for pxpy in isl["pixels"]: px_to_island[pxpy] = idx

    def _get_isaac_island(isaac_pos):
        x_2d, y_2d = (map_min_x+map_max_x)-(-float(isaac_pos[0])), \
                     (map_min_y+map_max_y)-(-float(isaac_pos[1]))
        return px_to_island.get(_world_to_px(x_2d, y_2d, min_x, min_y, scale))

    nav_area_m2    = sum(isl["area_m2"] for isl in valid_islands)
    area_based_cap = max(1, int(nav_area_m2 / AREA_PER_PERSON_M2))

    ep_num_people = {}
    if ARGS.num_people_for_episode is not None:
        if len(ARGS.num_people_for_episode) != len(ARGS.episode_ids):
            print("[FATAL] --num_people_for_episode length mismatch")
            simulation_app.close(); return 2
        for ep_id, n in zip(ARGS.episode_ids, ARGS.num_people_for_episode):
            capped = min(n, area_based_cap)
            if capped < n:
                print(f"[WARN] ep{ep_id}: num_people capped {n}->{capped}")
            ep_num_people[ep_id] = capped
    else:
        for ep_id in ARGS.episode_ids:
            capped = min(ARGS.num_people, area_based_cap)
            ep_num_people[ep_id] = capped

    print(f"[GEN] nav_area={nav_area_m2:.1f}m²  area_cap={area_based_cap}")

    # Open stage & optionally bake NavMesh
    if not open_stage(ARGS.usda):
        simulation_app.close(); return 1

    if not ARGS.skip_navmesh_check:
        navvols_usda = cache_paths(ARGS.usda, ARGS.cache_dir)
        nav_ok, inav = bake_navmesh(
            app=simulation_app, navvols_usda=navvols_usda,
            force_rebake=ARGS.force_rebake, collision_root=ARGS.collision_root,
            volume_padding=ARGS.volume_padding, fallback_size=ARGS.fallback_size,
            warmup_frames=ARGS.warmup_frames, semantic_map_json=None,
        )
        nm = inav.get_navmesh() if nav_ok else None
        if nm is None: print("[WARN] NavMesh bake failed — connectivity checks disabled")
    else:
        nm = None
        print("[GEN] NavMesh checks disabled (--skip_navmesh_check).")

    print(f"[GEN] Generating episodes: {ARGS.episode_ids}")
    for ep_id in ARGS.episode_ids:
        seed       = ARGS.seed_offset + ep_id
        actual_num = ep_num_people[ep_id]
        print(f"\n[GEN] ── episode_{ep_id}  seed={seed}  num_people={actual_num} ──")
        _generate_one_episode(
            episode_id=ep_id, seed=seed,
            out_dir=out_dir,
            grid=grid, esdf=esdf, min_x=min_x, min_y=min_y, scale=scale,
            map_min_x=map_min_x, map_max_x=map_max_x,
            map_min_y=map_min_y, map_max_y=map_max_y,
            nm=nm, to_isaac=to_isaac, plan_path_fn=plan_path,
            nav_area_m2=nav_area_m2, actual_num=actual_num,
            valid_islands=valid_islands,
            px_to_island=px_to_island,
            get_isaac_island=_get_isaac_island,
        )

    simulation_app.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())