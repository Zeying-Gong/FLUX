#!/usr/bin/env python3
"""
generate_episode_tracking_modes.py
───────────────────────────────────
Pure‑Python tracking episode generator with auto STT/DT/AT mode assignment.

Mode is assigned automatically based on episode population:
  •  1 person  →  STT  (Single‑Target Tracking)
  •  >1 person →  50% DT  (Distracted Tracking), 50% AT  (Ambiguity Tracking)

When --easy_mode is set:
  • robot distance relaxed to 1.5‑15 m
  • LOS not required
  • island constraint lifted
  • joint attempts increased to 800
  • each episode gets up to 5 retries with different random seeds
"""
from __future__ import annotations

import argparse, json, math, os, random, sys, time, heapq, re
from collections import deque
import numpy as np
from scipy.ndimage import distance_transform_edt
from clothing_appearance import ClothingProfiles
from clothing_appearance_collab import assign_episode_appearance

# ── CLI ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--semantic_map_json", required=True)
    p.add_argument("--output_dir",
                   default="/workspace/SAGE-3D_Official/SAGE-3D_data/v1_tracking_episodes")
    p.add_argument("--episode_ids",       nargs="+", type=int, default=[0])
    p.add_argument("--num_people_for_episode", nargs="+", type=int, default=None)
    p.add_argument("--num_people",        type=int, default=10)
    p.add_argument("--seed_offset",       type=int, default=0)
    p.add_argument("--overwrite",         action="store_true")
    p.add_argument("--no_vis",            action="store_true")
    p.add_argument("--robot_radius_2d",   type=float, default=0.3)
    p.add_argument("--scale_m_per_px",    type=float, default=0.05)
    p.add_argument("--easy_mode",         action="store_true",
                   help="Relax placement constraints for very small/difficult scenes.")
    p.add_argument("--profiles_json",
                   default="/workspace/FLUX/sage_utils/character_clothing_profiles.json",
                   help="Path to character_clothing_profiles.json for appearance assignment.")

    # ── Population control ────────────────────────────────────────────
    p.add_argument("--single_ratio", type=float, default=0.3,
                   help="Fraction of episodes with 1 person (→ STT). "
                        "Remaining episodes get area_cap people, split 50/50 DT/AT. "
                        "(default: 0.3)")
    return p.parse_args()


ARGS = parse_args()

# ── Constants ───────────────────────────────────────────────────────
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
WAYPOINT_MIN_DIST_M         = 1.5
PED_CORRIDOR_BLOCK_R        = [0.6, 0.3, 0.0]
_TRANSITION     = {"Idle": 0.4, "LookAround": 0.4, None: 0.2}
_IDLE_RANGE     = (3.0, 10.0)
_LOOKAROUND_RANGE = (3.0, 8.0)

# ── Character name helper ──────────────────────────────────────────
def get_character_name_by_index(i: int) -> str:
    if i == 0:
        return "Character"
    elif i < 10:
        return "Character_0" + str(i)
    else:
        return "Character_" + str(i)

# ═══════════════════════════════════════════════════════════════════
# 2D map loading (unchanged)
# ═══════════════════════════════════════════════════════════════════
def load_2d_map(sem_json, robot_r, scale):
    """Build occupancy grid + dual ESDFs.

    Synced with the navigation pipeline (`generate_episode_fast.py`):
      - Supports both legacy bare-list JSON and new {meta, instances}.
      - Restricts navigation to inferred interior (Strategy A: floor labels;
        Strategy B: building footprint from walls+furniture).
      - Returns TWO ESDFs:
          esdf_walls    : hard corridor check (walls only).
          esdf_combined : soft steer cost (walls + furniture).
        Single-ESDF prevented planning in furniture-dense rooms (conference,
        dining) because furniture filled the interior and ESDF dropped to 0.
    """
    with open(sem_json, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "instances" in raw:
        meta  = raw.get("meta", {}) or {}
        items = raw["instances"]
        print(f"[load_2d_map] New format JSON, {len(items)} instances.")
    else:
        meta  = {}
        items = raw
        print(f"[load_2d_map] Legacy format JSON, {len(items)} instances.")

    # ── map extent ────────────────────────────────────────────────────
    if all(k in meta for k in ("x_min","x_max","y_min","y_max")):
        min_x, max_x = float(meta["x_min"]), float(meta["x_max"])
        min_y, max_y = float(meta["y_min"]), float(meta["y_max"])
        print(f"[load_2d_map] Extent from meta: "
              f"x∈[{min_x:.2f},{max_x:.2f}] y∈[{min_y:.2f},{max_y:.2f}]")
    else:
        all_y, all_x = [], []
        for inst in items:
            for y, x in inst.get("mask_coords_m", []):
                try: all_y.append(float(y)); all_x.append(float(x))
                except (ValueError, TypeError): pass
        if not all_y:
            print("[load_2d_map] ERROR: no extent info available.")
            return (None,) * 12
        min_y, max_y = min(all_y), max(all_y)
        min_x, max_x = min(all_x), max(all_x)

    h = int(np.ceil((max_y - min_y) / scale)) + 1
    w = int(np.ceil((max_x - min_x) / scale)) + 1
    print(f"[load_2d_map] Grid: {w}×{h} px (scale={scale} m/px)")

    # ── walls + unable areas → raw_grid ───────────────────────────────
    raw_grid = np.zeros((h, w), dtype=np.uint8)
    obstacle_labels = ("wall", "unable area", "unable_area")
    wall_px = 0
    for inst in items:
        label = str(inst.get("category_label", "")).lower()
        if label not in obstacle_labels: continue
        for y_m, x_m in inst.get("mask_coords_m", []):
            try:
                py = int(round((float(y_m) - min_y) / scale))
                px = int(round((float(x_m) - min_x) / scale))
                if 0 <= py < h and 0 <= px < w:
                    raw_grid[py, px] = 1; wall_px += 1
            except (ValueError, TypeError): pass
    print(f"[load_2d_map] Walls/unable-area: {wall_px} px.")

    # ── furniture mask (soft obstacle, ESDF only) ─────────────────────
    furniture_labels = ("table", "chair", "sofa", "bed", "wardrobe",
                        "desk", "counter", "cabinet")
    furniture_mask = np.zeros((h, w), dtype=np.uint8)
    furn_px = 0
    for inst in items:
        label = str(inst.get("category_label", "")).lower()
        if not any(k in label for k in furniture_labels): continue
        for y_m, x_m in inst.get("mask_coords_m", []):
            try:
                py = int(round((float(y_m) - min_y) / scale))
                px = int(round((float(x_m) - min_x) / scale))
                if 0 <= py < h and 0 <= px < w:
                    furniture_mask[py, px] = 1; furn_px += 1
            except (ValueError, TypeError): pass
    print(f"[load_2d_map] Furniture: {furn_px} px (ESDF inflation only).")

    # ── interior mask ──────────────────────────────────────────────────
    from scipy.ndimage import (binary_dilation, binary_fill_holes,
                                binary_closing, label as _scilabel)

    floor_px = 0
    interior_mask = np.zeros((h, w), dtype=np.uint8)
    for inst in items:
        label = str(inst.get("category_label", "")).lower()
        if label != "floor": continue
        for y_m, x_m in inst.get("mask_coords_m", []):
            try:
                py = int(round((float(y_m) - min_y) / scale))
                px = int(round((float(x_m) - min_x) / scale))
                if 0 <= py < h and 0 <= px < w:
                    interior_mask[py, px] = 1; floor_px += 1
            except (ValueError, TypeError): pass

    if floor_px > 50:
        print(f"[load_2d_map] Strategy A: floor labels ({floor_px} px).")
        interior_mask = binary_dilation(interior_mask.astype(bool), iterations=2)
        interior_mask = binary_fill_holes(interior_mask).astype(np.uint8)
    else:
        # Step A: is interior detection needed at all?
        free_now = (raw_grid == 0).astype(np.uint8)
        border_mask = np.zeros_like(free_now, dtype=bool)
        border_mask[0, :] = border_mask[-1, :] = True
        border_mask[:, 0] = border_mask[:, -1] = True
        labeled_free, _ = _scilabel(free_now, structure=np.ones((3,3), dtype=np.int32))
        border_lbls = set(np.unique(labeled_free[border_mask]).tolist()); border_lbls.discard(0)
        border_free_px = sum(int((labeled_free == lbl).sum()) for lbl in border_lbls)
        total_free_px = int(free_now.sum())
        border_ratio = border_free_px / max(total_free_px, 1)
        print(f"[load_2d_map] Strategy B step A: border-free ratio = "
              f"{100.0*border_ratio:.1f}%.")

        if border_ratio < 0.15:
            print("[load_2d_map] Walls already enclose the scene; skip morph.")
            interior_mask = free_now
        else:
            building_seed = ((raw_grid == 1) | (furniture_mask == 1)).astype(bool)
            bridge_px = max(1, int(round(0.4 / scale)))
            struct_bridge = np.ones((2*bridge_px+1, 2*bridge_px+1), dtype=bool)
            building_blob = binary_dilation(building_seed, structure=struct_bridge)
            close_px = max(1, int(round(0.5 / scale)))
            struct_close = np.ones((2*close_px+1, 2*close_px+1), dtype=bool)
            building_solid = binary_closing(building_blob, structure=struct_close)
            building_solid = binary_fill_holes(building_solid)

            outside_candidates = (~building_solid).astype(np.uint8)
            lbl_out, n_out = _scilabel(outside_candidates,
                                       structure=np.ones((3,3), dtype=np.int32))
            exterior_mask = np.zeros_like(outside_candidates, dtype=bool)
            if n_out > 0:
                border_out_lbls = set(np.unique(lbl_out[border_mask]).tolist())
                border_out_lbls.discard(0)
                for lbl in border_out_lbls:
                    exterior_mask |= (lbl_out == lbl)
            interior_mask = ((~exterior_mask) & (raw_grid == 0)).astype(np.uint8)

            labeled, n_lbl = _scilabel(interior_mask,
                                       structure=np.ones((3,3), dtype=np.int32))
            if n_lbl > 1:
                sizes = [(lbl, int((labeled == lbl).sum()))
                         for lbl in range(1, n_lbl+1)]
                sizes.sort(key=lambda x: -x[1])
                threshold = max(50, int(0.1 * sizes[0][1]))
                keep = {lbl for lbl, sz in sizes if sz >= threshold}
                interior_mask = np.isin(labeled, list(keep)).astype(np.uint8)
                print(f"[load_2d_map] Kept {len(keep)} interior components.")

    interior_px = int(interior_mask.sum())
    total_free  = int((raw_grid == 0).sum())
    if total_free == 0 or interior_px / max(total_free, 1) < 0.05:
        print(f"[load_2d_map] WARN: interior too small "
              f"({100.0*interior_px/max(total_free,1):.1f}%); falling back.")
        interior_mask = (raw_grid == 0).astype(np.uint8)

    print(f"[load_2d_map] Interior: {interior_px}/{total_free} free px "
          f"({100.0*interior_px/max(total_free,1):.1f}%).")
    raw_grid[interior_mask == 0] = 1
    print(f"[load_2d_map] After interior masking: "
          f"{int((raw_grid == 1).sum())} obstacle px.")

    # ── planning grid + dual ESDFs ────────────────────────────────────
    if robot_r > 0:
        dist_m = distance_transform_edt(raw_grid == 0, sampling=scale)
        planning_grid = (dist_m <= robot_r).astype(np.uint8)
    else:
        planning_grid = raw_grid.copy()

    esdf_walls = distance_transform_edt(raw_grid == 0, sampling=scale)
    combined = (raw_grid | furniture_mask).astype(np.uint8)
    esdf_combined = distance_transform_edt(combined == 0, sampling=scale)
    print(f"[load_2d_map] ESDF walls max={float(esdf_walls.max()):.2f}m, "
          f"combined max={float(esdf_combined.max()):.2f}m.")

    return (planning_grid, esdf_walls, esdf_combined,
            min_x, min_y, scale, items, raw_grid,
            min_y, max_y, min_x, max_x)

def world_to_px(x_m, y_m, min_x, min_y, scale):
    return (int(round((float(x_m) - min_x) / scale)),
            int(round((float(y_m) - min_y) / scale)))

def px_to_world(px, py, min_x, min_y, scale):
    return min_x + (px + 0.5) * scale, min_y + (py + 0.5) * scale

def sm_to_isaac(x_m, y_m, map_min_x, map_max_x, map_min_y, map_max_y):
    return -(map_min_x + map_max_x - float(x_m)), -(map_min_y + map_max_y - float(y_m))

def snap(grid, px_py):
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

def reachable_area_px(grid, start_px):
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

def dist2(a, b):
    return math.hypot(float(a[0])-float(b[0]), float(a[1])-float(b[1]))

def path_length(pts):
    if not pts or len(pts) < 2: return 0.0
    return sum(math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
               for i in range(1, len(pts)))

# ═══════════════════════════════════════════════════════════════════
# PSDF (unchanged)
# ═══════════════════════════════════════════════════════════════════
def make_psdf_map(H, W):
    return np.zeros((H, W), dtype=np.float32)

def stamp_path_into_psdf(psdf, grid, path_px, scale):
    from scipy.ndimage import gaussian_filter as _gf
    H, W = grid.shape
    if not path_px: return
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

def normalise_psdf(psdf, grid):
    if psdf.max() < 1e-6:
        return np.zeros_like(psdf)
    out = np.clip(psdf, 0.0, PSDF_CAP) / PSDF_CAP
    out[grid == 1] = 0.0
    return out

def compute_psdf_block_thr(scale):
    d_px    = PED_SEPARATION_M / scale
    inner_s = PSDF_INNER_R_M  / scale
    outer_s = PSDF_OUTER_R_M  / scale
    raw_at_d = (PSDF_INNER_W * math.exp(-0.5 * (d_px / inner_s) ** 2) +
                PSDF_OUTER_W * math.exp(-0.5 * (d_px / outer_s) ** 2))
    return min(raw_at_d / PSDF_CAP, 1.0)

# ═══════════════════════════════════════════════════════════════════
# A* (4‑directional)
# ═══════════════════════════════════════════════════════════════════
def astar(grid, start_px, goal_px,
          esdf=None, min_corridor_m=0.0, esdf_weight=0.0,
          esdf_corridor=None,
          psdf_norm=None, psdf_weight=0.0, psdf_block_thr=0.0):
    """`esdf` = steer cost (typically combined). `esdf_corridor` = hard
    `min_corridor_m` check (typically walls-only). When None, falls back
    to `esdf`. Splitting them is critical in furniture-dense rooms."""
    H, W = grid.shape
    corridor_field = esdf_corridor if esdf_corridor is not None else esdf
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
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
            if corridor_field is not None and min_corridor_m > 0:
                if corridor_field[ny, nx] < min_corridor_m: continue
            if psdf_norm is not None and psdf_block_thr > 0:
                if float(psdf_norm[ny, nx]) >= psdf_block_thr: continue
            nb  = (nx, ny)
            step = 1.0
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

# ═══════════════════════════════════════════════════════════════════
# Island analysis
# ═══════════════════════════════════════════════════════════════════
def find_islands(area_grid, plan_grid, scale):
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

def plan_island_allocation(actual_num, robot_island_idx, valid_islands):
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

def sample_aux(rng):
    k = rng.choices(list(_TRANSITION.keys()), weights=list(_TRANSITION.values()), k=1)[0]
    if k == "Idle":
        return {"cmd": "Idle", "params": [str(round(rng.uniform(*_IDLE_RANGE), 1))]}
    if k == "LookAround":
        return {"cmd": "LookAround", "params": [str(round(rng.uniform(*_LOOKAROUND_RANGE), 1))]}
    return None

def sample_free_point(grid, esdf, rng, min_clearance, _free_cache=None):
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

# ═══════════════════════════════════════════════════════════════════
# Visualisation (unchanged)
# ═══════════════════════════════════════════════════════════════════
def save_vis(out_path, grid, esdf, min_x, min_y, scale,
             map_min_x, map_max_x, map_min_y, map_max_y,
             robot_start, robot_goal, spawn_dict, commands_dict,
             episode_id, scene_id, robot_orientation):
    if ARGS.no_vis:
        return
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import RegularPolygon
    except ImportError:
        return
    H, W = grid.shape
    ped_colors = ['#FF6B6B','#FFD93D','#6BCB77','#4D96FF','#FF9F45','#C77DFF']

    def from_isaac(ix, iy):
        x_m, y_m = -float(ix), -float(iy)
        return (map_min_x+map_max_x)-x_m, (map_min_y+map_max_y)-y_m
    def i2px(ix, iy):
        x2, y2 = from_isaac(ix, iy)
        return int(round((x2-min_x)/scale)), int(round((y2-min_y)/scale))

    psdf_vis = make_psdf_map(H, W)
    for cn, sd in spawn_dict.items():
        stamp_path_into_psdf(psdf_vis, grid, [i2px(*sd["pos"][:2])], scale)
        for cmd in commands_dict.get(cn, []):
            if cmd.get("cmd") != "GoTo": continue
            path_px = [i2px(p[0], p[1]) for p in cmd.get("path", [])]
            if path_px: stamp_path_into_psdf(psdf_vis, grid, path_px, scale)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_occ, ax_esdf, ax_psdf = axes

    vis_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    vis_rgb[grid==0] = [240,240,240]; vis_rgb[grid==1] = [50,50,50]
    ax_occ.imshow(vis_rgb, origin="lower")
    ax_occ.set_title("① Occupancy + Agents & Paths", fontsize=10, fontweight='bold')

    if robot_start:
        rs = i2px(*robot_start[:2])
        robot_tri = RegularPolygon(rs, numVertices=3, radius=8,
                                   orientation=robot_orientation,
                                   facecolor='cyan', edgecolor='black', linewidth=1, zorder=10)
        ax_occ.add_patch(robot_tri)
        ax_occ.arrow(rs[0], rs[1],
                     12 * math.cos(robot_orientation),
                     12 * math.sin(robot_orientation),
                     head_width=4, head_length=4, fc='cyan', ec='cyan', zorder=9)
    if robot_goal:
        rg = i2px(*robot_goal[:2])
        ax_occ.plot(rg[0], rg[1], '*', color='cyan', markersize=16, markeredgewidth=1, zorder=8)

    for ci, (cn, sd) in enumerate(spawn_dict.items()):
        is_target = (cn == "Character")
        c = '#FF0000' if is_target else ped_colors[ci % len(ped_colors)]
        sp = i2px(*sd["pos"][:2])
        orient = sd.get("rot", 0.0)

        ped_tri = RegularPolygon(sp, numVertices=3, radius=7,
                                 orientation=orient,
                                 facecolor=c, edgecolor='black',
                                 linewidth=1.5 if is_target else 0.8,
                                 zorder=9 if is_target else 8)
        ax_occ.add_patch(ped_tri)
        label = "T0" if is_target else f"D{ci}"
        ax_occ.annotate(label, sp, fontsize=7, color='white' if is_target else 'black',
                        fontweight='bold', xytext=(4,4), textcoords='offset points')

        if is_target:
            goto_cmds = [cmd for cmd in commands_dict.get(cn, []) if cmd.get("cmd") == "GoTo"]
            for wi, cmd in enumerate(goto_cmds):
                params = cmd.get("params", [])
                if len(params) < 2: continue
                wp = i2px(float(params[0]), float(params[1]))
                is_last = (wi == len(goto_cmds) - 1)
                marker = 'X' if is_last else 'D'
                size = 12 if is_last else 8
                ax_occ.plot(wp[0], wp[1], marker, color=c, markersize=size,
                            markeredgecolor='k', markeredgewidth=1.5, zorder=10)
                ax_occ.annotate(f"T0W{wi+1}", wp, fontsize=6, color=c,
                                xytext=(3,-10), textcoords='offset points')

        for cmd in commands_dict.get(cn, []):
            if cmd.get("cmd") != "GoTo": continue
            path_px = [i2px(p[0], p[1]) for p in cmd.get("path", [])]
            if path_px:
                lw = 2.0 if is_target else 1.2
                ax_occ.plot([p[0] for p in path_px], [p[1] for p in path_px],
                            '-', color=c, lw=lw, alpha=0.8)

    esdf_m = np.ma.masked_where(grid==1, np.clip(esdf, 0, 2))
    ax_esdf.imshow(esdf_m, origin="lower", cmap="viridis", vmin=0, vmax=2)
    ax_esdf.set_title("② ESDF (Static Obstacle Clearance)", fontsize=10, fontweight='bold')

    p95 = float(np.percentile(psdf_vis[grid==0], 95)) if (grid==0).any() else 1.0
    psdf_m = np.ma.masked_where(grid==1, np.clip(psdf_vis, 0, p95))
    ax_psdf.imshow(psdf_m, origin="lower", cmap="hot_r")
    ax_psdf.set_title("③ PSDF — Ped Proximity Field", fontsize=10, fontweight='bold')

    plt.suptitle(f"Tracking Episode {episode_id}  —  {scene_id}   peds={len(spawn_dict)}",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    vis_path = out_path.replace(".json", "_vis.png")
    plt.savefig(vis_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"[VIS] → {vis_path}")

# ═══════════════════════════════════════════════════════════════════
# Core episode generator (tracking) with --easy_mode support
# ═══════════════════════════════════════════════════════════════════

def generate_one_episode(episode_id, seed, out_dir, scene_id,
                         grid, esdf, esdf_walls,
                         min_x, min_y, scale,
                         map_min_x, map_max_x, map_min_y, map_max_y,
                         to_isaac, plan_path_fn,
                         nav_area_m2, actual_num,
                         valid_islands, px_to_island, get_isaac_island, clothing_profiles,
                         mode="dt"):
    t0 = time.time()
    random.seed(seed); np.random.seed(seed)
    rng = random.Random(seed)

    out_path = os.path.join(out_dir, f"episode_{episode_id}.json")
    if os.path.exists(out_path) and not ARGS.overwrite:
        print(f"  [SKIP] episode_{episode_id}"); return {"skipped": True}

    H, W = grid.shape
    _free_cache = {}
    psdf = make_psdf_map(H, W)
    placed_path_px_union = set()
    _PSDF_BLOCK_THR = compute_psdf_block_thr(scale)

    # ---------- 内部辅助函数 ----------
    def from_isaac(ix, iy):
        x_m, y_m = -float(ix), -float(iy)
        return (map_min_x+map_max_x)-x_m, (map_min_y+map_max_y)-y_m
    def isaac_to_px(ix, iy):
        return world_to_px(*from_isaac(ix, iy), min_x, min_y, scale)
    def check_line_clear(start_isaac, end_isaac):
        spx, spy = world_to_px(*from_isaac(*start_isaac[:2]), min_x, min_y, scale)
        epx, epy = world_to_px(*from_isaac(*end_isaac[:2]), min_x, min_y, scale)
        dx = abs(epx - spx); dy = abs(epy - spy)
        x, y = spx, spy
        sx = 1 if epx > spx else -1
        sy = 1 if epy > spy else -1
        err = dx - dy
        while True:
            if not (0 <= x < W and 0 <= y < H): return False
            if grid[y, x] == 1: return False
            if (x, y) == (epx, epy): break
            e2 = 2 * err
            if e2 > -dy: err -= dy; x += sx
            if e2 < dx: err += dx; y += sy
        return True
    def line_to_point_distance(lp1, lp2, p):
        x0, y0 = p[0], p[1]
        x1, y1 = lp1[0], lp1[1]; x2, y2 = lp2[0], lp2[1]
        dx = x2 - x1; dy = y2 - y1
        if dx == 0 and dy == 0: return math.hypot(x0-x1, y0-y1)
        t = ((x0-x1)*dx + (y0-y1)*dy) / (dx*dx + dy*dy)
        t = max(0.0, min(1.0, t))
        projx = x1 + t*dx; projy = y1 + t*dy
        return math.hypot(x0-projx, y0-projy)
    def sample_point():
        for _ in range(10):
            px_py = sample_free_point(grid, esdf_walls, rng,
                                      PED_SPAWN_MIN_CLEARANCE_M, _free_cache)
            if px_py is None: return None
            if reachable_area_px(grid, px_py) < MIN_REACHABLE_PX: continue
            x_2d, y_2d = px_to_world(px_py[0], px_py[1], min_x, min_y, scale)
            return to_isaac(x_2d, y_2d) + (0.0,)
        return None

    def build_blocked_grid(block_r_m):
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
    def sample_point_blocked(b_grid, b_esdf, block_r):
        for _ in range(10):
            px_py = sample_free_point(b_grid, b_esdf, rng, PED_SPAWN_MIN_CLEARANCE_M,
                                      _free_cache if block_r == 0 else None)
            if px_py is None: return None
            if reachable_area_px(b_grid, px_py) < MIN_REACHABLE_PX: continue
            x_2d, y_2d = px_to_world(px_py[0], px_py[1], min_x, min_y, scale)
            return to_isaac(x_2d, y_2d) + (0.0,)
        return None
    def plan_path_ped_blocked(a, b, b_grid, b_esdf):
        spx = snap(b_grid, world_to_px(*from_isaac(a[0], a[1]), min_x, min_y, scale))
        gpx = snap(b_grid, world_to_px(*from_isaac(b[0], b[1]), min_x, min_y, scale))
        if spx is None or gpx is None: return None
        psdf_norm = normalise_psdf(psdf, b_grid)
        path_px = astar(b_grid, spx, gpx,
                        esdf=b_esdf, min_corridor_m=0.0,
                        esdf_weight=PED_ESDF_WEIGHT,
                        psdf_norm=psdf_norm, psdf_weight=PED_PSDF_WEIGHT)
        if path_px is None: return None
        isaac_pts = []
        for px, py in path_px:
            x_2d, y_2d = px_to_world(px, py, min_x, min_y, scale)
            isaac_pts.append([*to_isaac(x_2d, y_2d), 0.0])
        return isaac_pts, path_px

    # ── 根据 easy_mode 调整联合采样参数 ──
    if ARGS.easy_mode:
        max_joint_attempts = 800
        dist_passes = [(1.5, 15.0, 500)]
        los_required = False
        same_island_required = False
    else:
        max_joint_attempts = 400
        dist_passes = [(3.0, 6.0, 200), (2.0, 10.0, 200)]
        los_required = True
        same_island_required = True

    target_spawn = None
    robot_start = None
    robot_island_idx = None
    target_chain = []
    target_path_px = []

    for joint_attempt in range(max_joint_attempts):
        p0 = sample_point()
        if p0 is None: continue
        if reachable_area_px(grid, isaac_to_px(*p0[:2])) < MIN_REACHABLE_PX: continue

        chain = [(p0, None)]
        prev = p0
        chain_ok = True
        for _ in range(NUM_CHAIN_WPT):
            wpt_ok = False
            for _ in range(120):
                w = sample_point()
                if w is None: continue
                if dist2(w, prev) < 2.0: continue
                result = plan_path_ped_blocked(prev, w, grid, esdf)
                if result is None: continue
                isaac_pts, _ = result
                if path_length(isaac_pts) < WAYPOINT_MIN_DIST_M: continue
                chain.append((w, result)); prev = w; wpt_ok = True; break
            if not wpt_ok: chain_ok = False; break
        if not chain_ok: continue

        tx, ty, _ = p0
        x_2d_t, y_2d_t = from_isaac(tx, ty)
        target_px_snap = snap(grid, isaac_to_px(tx, ty))
        if target_px_snap is None: continue

        robot_found = False
        for d_min, d_max, n_tries in dist_passes:
            for _ in range(n_tries):
                angle = rng.uniform(0, 2*math.pi)
                d = rng.uniform(d_min, d_max)
                cx_m = x_2d_t + d * math.cos(angle)
                cy_m = y_2d_t + d * math.sin(angle)
                px, py = world_to_px(cx_m, cy_m, min_x, min_y, scale)
                if not (0 <= px < W and 0 <= py < H): continue
                if grid[py, px] == 1: continue
                start_px = (px, py)
                if astar(grid, start_px, target_px_snap,
                         esdf=esdf,
                         esdf_corridor=esdf_walls,
                         min_corridor_m=ARGS.robot_radius_2d*0.5,
                         esdf_weight=ROBOT_ESDF_WEIGHT) is None:
                    continue
                ix, iy = to_isaac(cx_m, cy_m)
                cand_robot = (ix, iy, 0.0)
                if los_required and not check_line_clear(cand_robot, p0):
                    continue
                cand_island = px_to_island.get(start_px)
                if same_island_required and cand_island is None:
                    continue
                if cand_island is None:
                    cand_island = 0  # fallback for easy_mode
                robot_start = cand_robot
                robot_island_idx = cand_island
                target_spawn = p0
                target_chain = chain
                new_px = {isaac_to_px(*p0[:2])}
                for _, plan_result in chain[1:]:
                    if plan_result is None: continue
                    _, path_px_res = plan_result
                    new_px.update(path_px_res)
                target_path_px = new_px
                robot_found = True
                print(f"  [ep{episode_id}] joint placement OK "
                      f"target_attempt={joint_attempt+1} robot_d={d:.1f}m")
                break
            if robot_found:
                break
        if robot_found:
            break

    if target_spawn is None or robot_start is None:
        print(f"  [ep{episode_id}] FATAL: cannot find target+robot pair after {max_joint_attempts} attempts")
        return {"skipped": True}

    # 成功，stamp PSDF
    stamp_path_into_psdf(psdf, grid, list(target_path_px), scale)
    placed_path_px_union.update(target_path_px)

    target_cmds = []
    for _, plan_result in target_chain[1:]:
        if plan_result is None: continue
        isaac_pts, _ = plan_result
        if path_length(isaac_pts) < WAYPOINT_MIN_DIST_M: continue
        x, y, z = isaac_pts[-1]
        target_cmds.append({"cmd": "GoTo",
                            "params": [f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", "_"],
                            "path": isaac_pts})
        aux = sample_aux(rng)
        if aux: target_cmds.append(aux)

    spawn_dict = {}
    commands_dict = {}
    if target_cmds and target_cmds[0]["cmd"] == "GoTo":
        tx = float(target_cmds[0]["params"][0])
        ty = float(target_cmds[0]["params"][1])
        orient = math.atan2(ty - target_spawn[1], tx - target_spawn[0])
    else:
        orient = 0.0
    spawn_dict["Character"] = {
        "pos": [float(target_spawn[0]), float(target_spawn[1]), 0.0],
        "rot": orient
    }
    commands_dict["Character"] = target_cmds

    dx = target_spawn[0] - robot_start[0]
    dy = target_spawn[1] - robot_start[1]
    robot_orientation = math.atan2(dy, dx)
    robot_goal = target_spawn

    # ── 干扰行人 ──
    distractor_num = max(0, actual_num - 1)
    island_plan = None
    if distractor_num > 0:
        island_plan = plan_island_allocation(distractor_num, robot_island_idx, valid_islands)
        print(f"  [ep{episode_id}] distractor island_plan={island_plan}")

    if distractor_num <= 2:
        active_levels = [2]
    elif distractor_num <= 4:
        active_levels = [1, 2]
    else:
        active_levels = [0, 1, 2]

    def try_place_distractor(i, target_island):
        same_island = (target_island == robot_island_idx)
        for level in active_levels:
            block_r = PED_CORRIDOR_BLOCK_R[level]
            min_spawn_dist = 2.0 if level > 0 else MIN_CHAR_DIST
            b_grid = build_blocked_grid(block_r)
            b_esdf = distance_transform_edt(b_grid == 0, sampling=scale) if block_r > 0 and placed_path_px_union else esdf

            for attempt in range(80):
                p0 = sample_point_blocked(b_grid, b_esdf, block_r)
                if p0 is None: continue
                if same_island and dist2(p0, robot_start) < min_spawn_dist: continue
                if any(dist2(p0, sp["pos"]) < MIN_CHAR_SPACING for sp in spawn_dict.values()): continue
                min_clearance = PED_RADIUS_M + ARGS.robot_radius_2d + 0.3
                if line_to_point_distance(robot_start, robot_goal, p0) < min_clearance:
                    continue
                p0_island = get_isaac_island(p0)
                if target_island is not None and p0_island != target_island: continue
                if p0_island is None: continue
                if same_island and plan_path_ped_blocked(p0, robot_start, b_grid, b_esdf) is None: continue

                chain = [(p0, None)]
                prev = p0
                chain_ok = True
                for _ in range(NUM_CHAIN_WPT):
                    wpt_ok = False
                    for _ in range(120):
                        w = sample_point_blocked(b_grid, b_esdf, block_r)
                        if w is None: continue
                        if dist2(w, prev) < 2.0: continue
                        if not same_island and get_isaac_island(w) != target_island: continue
                        result = plan_path_ped_blocked(prev, w, b_grid, b_esdf)
                        if result is None: continue
                        isaac_pts, _ = result
                        if path_length(isaac_pts) < WAYPOINT_MIN_DIST_M: continue
                        chain.append((w, result)); prev = w; wpt_ok = True; break
                    if not wpt_ok: chain_ok = False; break
                if not chain_ok: continue

                wpts = chain[1:]
                if (len(wpts) >= 2 and dist2(wpts[-1][0], robot_goal) < MIN_CHAR_GOAL_TO_ROBOT_GOAL):
                    best_k, best_d = None, -1.0
                    for k in range(len(wpts)-1):
                        d = dist2(wpts[k][0], robot_goal)
                        if d >= MIN_CHAR_GOAL_TO_ROBOT_GOAL and d > best_d:
                            best_d = d; best_k = k
                    if best_k is not None:
                        wpts[best_k], wpts[-1] = wpts[-1], wpts[best_k]
                        r = plan_path_ped_blocked(chain[best_k][0], wpts[best_k][0], b_grid, b_esdf)
                        if r: wpts[best_k] = (wpts[best_k][0], r)
                        r = plan_path_ped_blocked(wpts[-2][0], wpts[-1][0], b_grid, b_esdf)
                        if r: wpts[-1] = (wpts[-1][0], r)
                chain = [chain[0]] + wpts

                new_path_px = {isaac_to_px(*p0[:2])}
                for _, plan_result in chain[1:]:
                    if plan_result is None: continue
                    _, path_px_res = plan_result
                    new_path_px.update(path_px_res)
                stamp_path_into_psdf(psdf, grid, list(new_path_px), scale)
                placed_path_px_union.update(new_path_px)

                cn = get_character_name_by_index(i + 1)
                cmds = []
                for _, plan_result in chain[1:]:
                    if plan_result is None: continue
                    isaac_pts, _ = plan_result
                    if path_length(isaac_pts) < WAYPOINT_MIN_DIST_M: continue
                    x, y, z = isaac_pts[-1]
                    cmds.append({"cmd": "GoTo", "params": [f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", "_"], "path": isaac_pts})
                    aux = sample_aux(rng)
                    if aux: cmds.append(aux)
                if cmds and cmds[0]["cmd"] == "GoTo":
                    tx = float(cmds[0]["params"][0])
                    ty = float(cmds[0]["params"][1])
                    orient = math.atan2(ty - float(p0[1]), tx - float(p0[0]))
                else:
                    orient = 0.0
                spawn_dict[cn] = {"pos": [float(p0[0]), float(p0[1]), 0.0], "rot": orient}
                commands_dict[cn] = cmds
                print(f"  [ep{episode_id}] distractor{i} OK level={level} attempt={attempt+1}")
                return True
            print(f"  [ep{episode_id}] distractor{i} level {level} exhausted")
        return False

    for d_idx in range(distractor_num):
        target_isl = island_plan[d_idx] if island_plan else robot_island_idx
        if not try_place_distractor(d_idx, target_isl):
            print(f"  [ep{episode_id}] WARN: could not place distractor {d_idx}")

# ── Hard requirement: tracking episodes MUST contain the target. ──
    # If the joint-placement loop somehow returned success but the target
    # didn't end up in spawn_dict (e.g. due to a future refactor), or if
    # the target has no GoTo commands (nothing for the robot to follow),
    # we refuse to write the episode and signal failure to the batch
    # runner, rather than silently producing an unusable episode.
    target_key = "Character"
    if target_key not in spawn_dict:
        print(f"  [ep{episode_id}] FATAL: target '{target_key}' missing "
              f"from spawn_dict; refusing to write tracking episode.")
        return {"skipped": True, "reason": "no_target"}
    target_goto_count = sum(
        1 for c in commands_dict.get(target_key, [])
        if c.get("cmd") == "GoTo"
    )
    if target_goto_count == 0:
        print(f"  [ep{episode_id}] FATAL: target has 0 GoTo commands; "
              f"robot would have nothing to track. Refusing to write.")
        return {"skipped": True, "reason": "no_target_path"}

    # ── 服装外观分配(资产 + 逐部位颜色),写进 episode ──
    appearance_block = None
    if clothing_profiles is not None:
        # spawn_dict 的 key 顺序即角色顺序,第0个是 target "Character"
        char_names = list(spawn_dict.keys())
        try:
            appearance_block = assign_episode_appearance(
                clothing_profiles, episode_id, seed, char_names, mode=mode)
            print(f"  [ep{episode_id}] appearance: target_asset="
                  f"{appearance_block['target_asset']} "
                  f"distinct={appearance_block['distinct_level']}")
        except Exception as e:
            print(f"  [ep{episode_id}] WARN appearance assignment failed: {e}")

    episode_data = {
        "episode": {
            "episode_id": episode_id,
            "mode": mode,
            "seed": seed,
            "robot": {
                "start_pos": list(robot_start),
                "goal_pos": list(robot_goal),
                "start_orientation": robot_orientation,
            },
            "characters": {
                "num_characters": len(spawn_dict),
                "spawn_positions": spawn_dict,
                "commands": commands_dict,
            },
            "tracking": {
                "target_character": target_key,
                "target_num_goto": target_goto_count,
            },
            "appearance": appearance_block,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(episode_data, f, indent=4)

    elapsed = round(time.time()-t0, 2)
    print(f"  [ep{episode_id}] done placed={len(spawn_dict)} elapsed={elapsed}s → {out_path}")

    save_vis(out_path, grid, esdf, min_x, min_y, scale,
             map_min_x, map_max_x, map_min_y, map_max_y,
             robot_start, robot_goal, spawn_dict, commands_dict,
             episode_id, scene_id, robot_orientation)

    return {"episode_id": episode_id, "placed": len(spawn_dict), "elapsed_s": elapsed}

# ═══════════════════════════════════════════════════════════════════
def main():
    if not os.path.exists(ARGS.semantic_map_json):
        print("[FATAL] Semantic map not found"); return 1
    res = load_2d_map(ARGS.semantic_map_json, ARGS.robot_radius_2d, ARGS.scale_m_per_px)
    if res[0] is None:
        print("[FATAL] 2D map load failed"); return 1
    (grid, esdf_walls, esdf_combined,
     min_x, min_y, scale, sem_data, raw_grid,
     map_min_y, map_max_y, map_min_x, map_max_x) = res
    # Alias: existing code references `esdf` for cost-style use → combined.
    esdf = esdf_combined

    def to_isaac(x_m, y_m):
        return sm_to_isaac(x_m, y_m, map_min_x, map_max_x, map_min_y, map_max_y)

    def plan_path(start_isaac, goal_isaac):
        spx = snap(grid, world_to_px(
            *[(map_min_x+map_max_x)-(-float(start_isaac[0])),
              (map_min_y+map_max_y)-(-float(start_isaac[1]))],
            min_x, min_y, scale))
        gpx = snap(grid, world_to_px(
            *[(map_min_x+map_max_x)-(-float(goal_isaac[0])),
              (map_min_y+map_max_y)-(-float(goal_isaac[1]))],
            min_x, min_y, scale))
        if spx is None or gpx is None: return None
        path_px = astar(grid, spx, gpx,
                        esdf=esdf,
                        esdf_corridor=esdf_walls,
                        min_corridor_m=ARGS.robot_radius_2d * 0.5,
                        esdf_weight=ROBOT_ESDF_WEIGHT)
        if path_px is None: return None
        return [[*to_isaac(*px_to_world(px, py, min_x, min_y, scale)), 0.0]
                for px, py in path_px]

    # scene_id parsing — match generate_episode_fast.py:
    #   2D_Semantic_Map_839873_Complete.json        → "839873"
    #   2D_Semantic_Map_0_839873_Complete.json      → "839873"
    #   2D_Semantic_Map_0044_839926_Complete.json   → "0044_839926"
    fname = os.path.basename(ARGS.semantic_map_json)
    m = re.match(r"2D_Semantic_Map_(.+)_Complete\.json$", fname)
    scene_id = m.group(1) if m else os.path.splitext(fname)[0]
    print(f"[main] scene_id = {scene_id!r}")
    out_dir = os.path.join(ARGS.output_dir, scene_id)
    os.makedirs(out_dir, exist_ok=True)
    # 载入服装 profiles(一次,供所有 episode 共用)
    clothing_profiles = None
    if os.path.exists(ARGS.profiles_json):
        try:
            clothing_profiles = ClothingProfiles(ARGS.profiles_json)
            print(f"[main] Loaded clothing profiles: {clothing_profiles.num_assets} assets")
        except Exception as e:
            print(f"[WARN] Failed to load profiles ({e}); episodes will have no appearance.")
    else:
        print(f"[WARN] profiles_json not found: {ARGS.profiles_json}; no appearance.")
    all_islands = find_islands(raw_grid, grid, scale)
    valid_islands = [isl for isl in all_islands if isl["area_m2"] >= MIN_ISLAND_AREA_THRESHOLD]
    if not valid_islands:
        all_px = set(zip(*np.where(grid==0)[::-1]))
        valid_islands = [{"pixels": all_px, "area_m2": len(all_px)*scale**2}]
    px_to_island = {}
    for idx, isl in enumerate(valid_islands):
        for pxpy in isl["pixels"]:
            px_to_island[pxpy] = idx
    def get_isaac_island(isaac_pos):
        x_2d = (map_min_x+map_max_x)-(-float(isaac_pos[0]))
        y_2d = (map_min_y+map_max_y)-(-float(isaac_pos[1]))
        return px_to_island.get(world_to_px(x_2d, y_2d, min_x, min_y, scale))

    nav_area_m2 = sum(isl["area_m2"] for isl in valid_islands)
    # NOTE: for *tracking*, ≥1 person (the target) is mandatory. We do not
    # let area_cap fall below 1 even in tiny scenes — those should fail
    # later in the actual placement loop with a clear FATAL message, not
    # be silently emptied here.
    area_cap = max(1, int(nav_area_m2 / AREA_PER_PERSON_M2))

    ep_num_people = {}
    if ARGS.num_people_for_episode:
        if len(ARGS.num_people_for_episode) != len(ARGS.episode_ids):
            print("[FATAL] length mismatch"); return 2
        for eid, n in zip(ARGS.episode_ids, ARGS.num_people_for_episode):
            capped = min(n, area_cap)
            if capped < 1:
                # Should never happen with area_cap>=1, but be defensive.
                print(f"[WARN] ep{eid}: requested {n} people, area_cap={area_cap}, "
                      f"forcing to 1 (tracking needs a target).")
                capped = 1
            ep_num_people[eid] = capped
    else:
        for eid in ARGS.episode_ids:
            ep_num_people[eid] = max(1, min(ARGS.num_people, area_cap))

    print(f"[GEN] area_cap={area_cap}, episode people: {ep_num_people}")

    # ── Episode 生成 (带 easy_mode 种子重试) ──
    # Mode is auto-assigned per episode based on number of people:
    #   1 person  →  STT
    #   >1 person →  DT (even episode_id) / AT (odd episode_id)  50/50
    #
    # Population randomization (when --num_people_for_episode is NOT used):
    #   Each episode independently gets 1 person (prob = single_ratio) or
    #   area_cap people (prob = 1 - single_ratio) so that scenes naturally
    #   contain a mix of STT / DT / AT episodes.
    failed_episodes: list[tuple[int, str]] = []

    _use_fixed_people = ARGS.num_people_for_episode is not None
    _max_people = max(1, min(ARGS.num_people, area_cap))

    for eid in ARGS.episode_ids:
        base_seed = ARGS.seed_offset + eid
        _rng = random.Random(base_seed ^ 0xBEEF)

        # ── Determine number of people ────────────────────────────────
        if _use_fixed_people:
            actual_num = ep_num_people[eid]
        elif _rng.random() < ARGS.single_ratio:
            actual_num = 1
        else:
            actual_num = max(2, _max_people)   # at least 2 for DT/AT

        if actual_num < 1:
            actual_num = 1

        # ── Assign mode from population ───────────────────────────────
        if actual_num == 1:
            _mode = "stt"
        else:
            _mode = "dt" if eid % 2 == 0 else "at"   # 50/50 across episode_ids

        success = False
        last_reason = "unknown"
        max_retries = 5 if ARGS.easy_mode else 1
        for retry in range(max_retries):
            seed = base_seed + retry * 99999
            print(f"\n[GEN] episode_{eid} mode={_mode} seed={seed} num_people={actual_num}" +
                  (f" retry {retry+1}" if retry > 0 else ""))
            result = generate_one_episode(eid, seed, out_dir, scene_id,
                                                    grid, esdf, esdf_walls,
                                                    min_x, min_y, scale,
                                                    map_min_x, map_max_x, map_min_y, map_max_y,
                                                    to_isaac, plan_path,
                                                    nav_area_m2, actual_num,
                                                    valid_islands, px_to_island, get_isaac_island,
                                                    clothing_profiles,
                                                    mode=_mode)
            if result.get("skipped") is not True:
                success = True
                break
            last_reason = result.get("reason", "placement_failed")
        if not success:
            print(f"  [ep{eid}] mode={_mode} FAILED after {max_retries} attempt(s) "
                  f"reason={last_reason}")
            failed_episodes.append((eid, last_reason))

    if failed_episodes:
        print(f"\n[FATAL] {len(failed_episodes)} episode(s) failed in "
              f"scene {scene_id!r}: {failed_episodes}")
        return 3   # non-zero so batch runner marks this scene FAIL
    return 0

if __name__ == "__main__":
    sys.exit(main())