#!/usr/bin/env python3
"""
generate_episode_fast.py
────────────────────────
Pure‑Python episode generator without Isaac Sim.
Preserves sampling logic, PSDF avoidance, island allocation,
and visualisation.  Suitable for CPU‑only parallel generation.

Usage
-----
    python generate_episode_fast.py \
        --semantic_map_json .../2D_Semantic_Map_839873_Complete.json \
        --output_dir /workspace/.../v1_episodes \
        --episode_ids 0 1 2 \
        --num_people_for_episode 2 3 2 \
        --no_vis         # optional speed boost
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

# ── CLI ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--semantic_map_json", required=True)
    p.add_argument("--output_dir",        required=True,
                   help="Root output dir; episodes go to <output_dir>/<scene_id>")
    p.add_argument("--episode_ids",       nargs="+", type=int, default=[0])
    p.add_argument("--num_people_for_episode", nargs="+", type=int, default=None)
    p.add_argument("--num_people",        type=int, default=10,
                   help="Fallback when --num_people_for_episode is omitted")
    p.add_argument("--seed_offset",       type=int, default=0)
    p.add_argument("--overwrite",         action="store_true")
    p.add_argument("--no_vis",            action="store_true")
    # map / sampling
    p.add_argument("--robot_radius_2d",   type=float, default=0.3)
    p.add_argument("--scale_m_per_px",    type=float, default=0.05)
    return p.parse_args()

ARGS = parse_args()

# ── Constants (same as original) ─────────────────────────────────────
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
PED_CORRIDOR_BLOCK_R = [0.6, 0.3, 0.0]
_TRANSITION     = {"Idle": 0.4, "LookAround": 0.4, None: 0.2}
_IDLE_RANGE     = (3.0, 10.0)
_LOOKAROUND_RANGE = (3.0, 8.0)

# ── Helper: character name ──────────────────────────────────────────
def get_character_name_by_index(i: int) -> str:
    if i == 0:
        return "Character"
    elif i < 10:
        return "Character_0" + str(i)
    else:
        return "Character_" + str(i)

# ── 2D map loading ──────────────────────────────────────────────────
def load_2d_map(sem_json, robot_r, scale):
    """Build an occupancy grid (1 = obstacle) from the semantic map JSON.

    Supports both the legacy bare-list layout and the new
    {"meta": {...}, "instances": [...]} layout written by
    semantic_map_builder.py. When `meta` is present we trust it for the
    map extent (more reliable than computing from instance coords, since
    the occupancy map may have empty borders).

    Obstacles (raw_grid == 1): wall + unable_area only.
    Furniture is INTENTIONALLY NOT marked here as a hard obstacle in
    raw_grid — instead, we inflate it into the ESDF below so that the
    planner's `min_corridor_m` and ESDF-weighted A* steer paths around
    it while still allowing pedestrians to spawn near (but not inside)
    furniture. This matches the policy of NOT using NavMesh exclude
    volumes.
    """
    with open(sem_json, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "instances" in raw:
        meta  = raw.get("meta", {}) or {}
        items = raw["instances"]
        print(f"[load_2d_map] New format JSON, {len(items)} instances, "
              f"meta keys: {list(meta.keys())}")
    else:
        meta  = {}
        items = raw
        print(f"[load_2d_map] Legacy format JSON, {len(items)} instances.")

    # ── map extent ─────────────────────────────────────────────────────
    if all(k in meta for k in ("x_min","x_max","y_min","y_max")):
        min_x, max_x = float(meta["x_min"]), float(meta["x_max"])
        min_y, max_y = float(meta["y_min"]), float(meta["y_max"])
        print(f"[load_2d_map] Extent from meta: "
              f"x∈[{min_x:.2f},{max_x:.2f}]  y∈[{min_y:.2f},{max_y:.2f}]")
    else:
        all_y, all_x = [], []
        for inst in items:
            for y, x in inst.get("mask_coords_m", []):
                try:
                    all_y.append(float(y)); all_x.append(float(x))
                except (ValueError, TypeError):
                    pass
        if not all_y:
            print("[load_2d_map] ERROR: no mask_coords_m and no meta extent.")
            return (None,) * 11
        min_y, max_y = min(all_y), max(all_y)
        min_x, max_x = min(all_x), max(all_x)
        print(f"[load_2d_map] Extent from mask stats: "
              f"x∈[{min_x:.2f},{max_x:.2f}]  y∈[{min_y:.2f},{max_y:.2f}]")

    h = int(np.ceil((max_y - min_y) / scale)) + 1
    w = int(np.ceil((max_x - min_x) / scale)) + 1
    print(f"[load_2d_map] Grid size: {w} × {h} px (scale={scale} m/px)")

    # ── obstacle grid: walls + unable areas ────────────────────────────
    raw_grid = np.zeros((h, w), dtype=np.uint8)
    obstacle_labels = ("wall", "unable area", "unable_area")
    wall_px = 0
    for inst in items:
        label = str(inst.get("category_label", "")).lower()
        if label not in obstacle_labels:
            continue
        for y_m, x_m in inst.get("mask_coords_m", []):
            try:
                py = int(round((float(y_m) - min_y) / scale))
                px = int(round((float(x_m) - min_x) / scale))
                if 0 <= py < h and 0 <= px < w:
                    raw_grid[py, px] = 1
                    wall_px += 1
            except (ValueError, TypeError):
                pass
    print(f"[load_2d_map] Marked {wall_px} wall/unable-area pixels.")

    # ── furniture mask (NOT in raw_grid; used only to inflate ESDF and
    #    to anchor interior detection below). ──────────────────────────
    furniture_labels = ("table", "chair", "sofa", "bed", "wardrobe",
                        "desk", "counter", "cabinet")
    furniture_mask = np.zeros((h, w), dtype=np.uint8)
    furn_px = 0
    for inst in items:
        label = str(inst.get("category_label", "")).lower()
        if not any(k in label for k in furniture_labels):
            continue
        for y_m, x_m in inst.get("mask_coords_m", []):
            try:
                py = int(round((float(y_m) - min_y) / scale))
                px = int(round((float(x_m) - min_x) / scale))
                if 0 <= py < h and 0 <= px < w:
                    furniture_mask[py, px] = 1
                    furn_px += 1
            except (ValueError, TypeError):
                pass
    print(f"[load_2d_map] Marked {furn_px} furniture pixels "
          f"(for ESDF inflation + interior anchoring).")

    # ── interior mask: restrict navigation to indoor area ─────────────
    # InteriorGS occupancy maps often have exterior empty space that's
    # *not* separated from interior floor by any wall pixels — naive
    # flood-fill therefore can't tell them apart. We use a stronger
    # signal: furniture only exists indoors. The procedure:
    #
    #   1. Strategy A: if `floor` instances are present, use their union
    #      (most accurate when available).
    #
    #   2. Strategy B (used when no floor): treat the dilated wall+furniture
    #      union as a coarse "building footprint", then morphologically
    #      close + fill holes to get the building's interior. Subtract the
    #      walls to recover the navigable indoor area.
    #
    # The mask is then ORed into raw_grid so exterior cells become obstacles.
    floor_labels = ("floor",)
    interior_mask = np.zeros((h, w), dtype=np.uint8)
    floor_px = 0
    for inst in items:
        label = str(inst.get("category_label", "")).lower()
        if label not in floor_labels:
            continue
        for y_m, x_m in inst.get("mask_coords_m", []):
            try:
                py = int(round((float(y_m) - min_y) / scale))
                px = int(round((float(x_m) - min_x) / scale))
                if 0 <= py < h and 0 <= px < w:
                    interior_mask[py, px] = 1
                    floor_px += 1
            except (ValueError, TypeError):
                pass

    from scipy.ndimage import (binary_dilation, binary_fill_holes,
                                binary_closing, binary_erosion,
                                label as _scilabel)

    if floor_px > 50:
        print(f"[load_2d_map] Strategy A: using {floor_px} floor pixels "
              f"as interior anchor.")
        interior_mask = binary_dilation(interior_mask.astype(bool),
                                        iterations=2)
        interior_mask = binary_fill_holes(interior_mask).astype(np.uint8)
    else:
        # ── Strategy B: derive interior without floor labels. ─────────
        # The original implementation morphologically dilated + closed +
        # filled + eroded, which has two failure modes:
        #
        #   1. For *already-closed* buildings (walls form a complete
        #      perimeter), the dilate→erode round-trip erodes ~bridge_m
        #      worth of pixels off the inside of every wall, killing
        #      narrow corridors. Such scenes don't need interior detection
        #      at all — the wall mask alone already separates inside
        #      from outside.
        #
        #   2. For *open-perimeter* buildings (InteriorGS crops where the
        #      outside is visually a "courtyard" that shares pixel values
        #      with the floor), interior must be inferred from
        #      furniture+wall geometry, but erosion is still the wrong
        #      tool — it shrinks the interior unnecessarily.
        #
        # New approach:
        #   Step A: decide if interior detection is *needed* at all,
        #           by checking whether the free region leaks to the
        #           image border in significant amount. If not, the wall
        #           mask already does the job → return raw_grid==0.
        #
        #   Step B: if needed, build the building "outline" by dilating
        #           the (walls ∪ furniture) seed by `bridge_m`, then
        #           closing and filling holes. From this we get a solid
        #           building footprint.
        #
        #   Step C: flood-fill the *exterior* of that footprint from the
        #           image border. Interior = footprint minus exterior
        #           minus walls. No erosion — we never shrink the inside.

        # Step A: how much of the free area touches the image border?
        free_now = (raw_grid == 0).astype(np.uint8)
        border_mask = np.zeros_like(free_now, dtype=bool)
        border_mask[0, :] = border_mask[-1, :] = True
        border_mask[:, 0] = border_mask[:, -1] = True

        # Connected components of free space; count those touching border.
        labeled_free, n_free = _scilabel(
            free_now, structure=np.ones((3, 3), dtype=np.int32))
        border_lbls = set(np.unique(labeled_free[border_mask]).tolist())
        border_lbls.discard(0)
        border_free_px = sum(int((labeled_free == lbl).sum())
                             for lbl in border_lbls)
        total_free_px = int(free_now.sum())
        border_ratio = border_free_px / max(total_free_px, 1)

        print(f"[load_2d_map] Strategy B step A: "
              f"free px touching border = {border_free_px}/{total_free_px} "
              f"({100.0*border_ratio:.1f}%).")

        if border_ratio < 0.15:
            # Walls already enclose the scene. No further work needed.
            print(f"[load_2d_map] Strategy B: walls already close the "
                  f"scene; skipping morphological interior detection.")
            interior_mask = free_now
        else:
            # Step B: build a coarse "building footprint" from walls +
            # furniture. Bridge distance defines how big a gap (door, mask
            # artefact) we're willing to seal.
            building_seed = ((raw_grid == 1) |
                             (furniture_mask == 1)).astype(bool)

            bridge_m = 0.4   # smaller than before; just enough for doorways
            bridge_px = max(1, int(round(bridge_m / scale)))
            struct_bridge = np.ones((2 * bridge_px + 1,
                                     2 * bridge_px + 1), dtype=bool)
            building_blob = binary_dilation(building_seed,
                                            structure=struct_bridge)

            close_m = 0.5    # close small wall gaps
            close_px = max(1, int(round(close_m / scale)))
            struct_close = np.ones((2 * close_px + 1,
                                    2 * close_px + 1), dtype=bool)
            building_solid = binary_closing(building_blob,
                                            structure=struct_close)
            building_solid = binary_fill_holes(building_solid)

            # Step C: flood-fill the EXTERIOR from the image border, then
            # interior = ¬exterior ∧ ¬wall. This avoids any erosion.
            # The exterior is the connected component of "not building_solid"
            # that touches the image border.
            outside_candidates = (~building_solid).astype(np.uint8)
            lbl_out, n_out = _scilabel(
                outside_candidates,
                structure=np.ones((3, 3), dtype=np.int32))
            exterior_mask = np.zeros_like(outside_candidates, dtype=bool)
            if n_out > 0:
                border_out_lbls = set(np.unique(lbl_out[border_mask]).tolist())
                border_out_lbls.discard(0)
                for lbl in border_out_lbls:
                    exterior_mask |= (lbl_out == lbl)

            # Interior = inside building footprint AND not wall.
            # Equivalently: not exterior AND not wall.
            interior_mask = ((~exterior_mask) & (raw_grid == 0)).astype(np.uint8)

            ext_px = int(exterior_mask.sum())
            print(f"[load_2d_map] Strategy B: building footprint "
                  f"(bridge={bridge_m}m, close={close_m}m). "
                  f"Exterior flood-fill: {ext_px} px outside.")

            # Drop tiny disconnected interior fragments.
            labeled, n_lbl = _scilabel(
                interior_mask,
                structure=np.ones((3, 3), dtype=np.int32))
            if n_lbl > 1:
                sizes = [(lbl, int((labeled == lbl).sum()))
                         for lbl in range(1, n_lbl + 1)]
                sizes.sort(key=lambda x: -x[1])
                # Keep components ≥ 10% of the largest (don't lose
                # legitimately separated rooms via a doorway).
                threshold = max(50, int(0.1 * sizes[0][1]))
                keep = {lbl for lbl, sz in sizes if sz >= threshold}
                interior_mask = np.isin(labeled,
                                        list(keep)).astype(np.uint8)
                kept_n = len(keep)
                dropped_n = n_lbl - kept_n
                print(f"[load_2d_map] Strategy B: kept {kept_n} interior "
                      f"component(s) ≥{threshold}px, dropped {dropped_n} "
                      f"smaller fragments.")

    interior_px = int(interior_mask.sum())
    total_free  = int((raw_grid == 0).sum())
    if total_free == 0 or interior_px / max(total_free, 1) < 0.05:
        print(f"[load_2d_map] WARN: interior mask too small "
              f"({interior_px}/{total_free} px = "
              f"{100.0*interior_px/max(total_free,1):.1f}%). "
              "Falling back to all-free (no interior restriction).")
        interior_mask = (raw_grid == 0).astype(np.uint8)
        interior_px = int(interior_mask.sum())

    print(f"[load_2d_map] Interior mask: {interior_px}/{total_free} free px "
          f"({100.0*interior_px/max(total_free,1):.1f}% of free area).")

    # Hard-block everything outside the interior by writing into raw_grid.
    raw_grid[interior_mask == 0] = 1
    print(f"[load_2d_map] After interior masking: "
          f"{int((raw_grid == 1).sum())} obstacle px.")
    
    # ── planning_grid: walls inflated by robot radius (hard "no-go") ──
    # NOTE: we deliberately inflate ONLY walls, not furniture. Furniture-
    # cluttered rooms (e.g. conference rooms, dining areas) often have
    # furniture-to-furniture clearance below the robot radius; treating
    # furniture as a hard obstacle would leave no walkable cells at all.
    # Instead, furniture appears only in the ESDF below, which makes the
    # A* prefer wide corridors but still allows tight passes when needed.
    if robot_r > 0:
        dist_m = distance_transform_edt(raw_grid == 0, sampling=scale)
        planning_grid = (dist_m <= robot_r).astype(np.uint8)
    else:
        planning_grid = raw_grid.copy()

    # ── Two ESDFs, used for different purposes by the A* ──────────────
    #   esdf_walls    : distance to nearest WALL only.
    #                   Used as a HARD floor (`min_corridor_m`) so we never
    #                   plan a path that would graze a wall corner.
    #   esdf_combined : distance to nearest WALL or FURNITURE.
    #                   Used as a SOFT cost (`esdf_weight`) so paths
    #                   prefer wider, less cluttered corridors.
    #
    # The previous unified ESDF made `min_corridor_m` apply to furniture
    # too, which is catastrophic in furniture-dense rooms: ESDF max can
    # drop to 0 m and every cell fails the corridor check.
    esdf_walls = distance_transform_edt(raw_grid == 0, sampling=scale)
    combined_obstacles = (raw_grid | furniture_mask).astype(np.uint8)
    esdf_combined = distance_transform_edt(combined_obstacles == 0,
                                           sampling=scale)

    free_px = int((raw_grid == 0).sum())
    furn_in_interior = int(((furniture_mask == 1) & (raw_grid == 0)).sum())
    print(f"[load_2d_map] planning_grid obstacles: "
          f"{int((planning_grid == 1).sum())} px")
    print(f"[load_2d_map] ESDF (walls only)    max = "
          f"{float(esdf_walls.max()):.2f} m")
    print(f"[load_2d_map] ESDF (walls+furniture) max = "
          f"{float(esdf_combined.max()):.2f} m")
    print(f"[load_2d_map] Furniture inside interior: "
          f"{furn_in_interior}/{free_px} px "
          f"({100.0*furn_in_interior/max(free_px,1):.1f}% of interior).")

    # NOTE: returned tuple is 12 elements now (was 11). Callers must
    # unpack both ESDFs. The 5th element name `sem_data` is the instance
    # list, kept for backward compat with old visualisation code.
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

# ── PSDF ────────────────────────────────────────────────────────────
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

# ── A* (4‑directional) ──────────────────────────────────────────────
def astar(grid, start_px, goal_px,
          esdf=None, min_corridor_m=0.0, esdf_weight=0.0,
          esdf_corridor=None,
          psdf_norm=None, psdf_weight=0.0, psdf_block_thr=0.0):
    """A* on a 4-connected grid.

    Args:
        esdf:           Field used as a STEER COST (lower clearance →
                        higher step cost). Typically walls+furniture.
        esdf_corridor:  Optional field used for the HARD `min_corridor_m`
                        check. If None, falls back to `esdf`. Pass a
                        walls-only ESDF here so that furniture clutter
                        doesn't make every cell illegal in dense rooms.
        min_corridor_m: Minimum clearance (m) required vs `esdf_corridor`.
    """
    H, W = grid.shape
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    open_set = [(0.0, start_px)]
    came_from = {}
    g = {start_px: 0.0}
    corridor_field = esdf_corridor if esdf_corridor is not None else esdf
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

# ── Island analysis ─────────────────────────────────────────────────
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

# ── Visualisation (lightweight) ─────────────────────────────────────
def save_vis(out_path, grid, esdf, min_x, min_y, scale,
             map_min_x, map_max_x, map_min_y, map_max_y,
             robot_start, robot_goal, spawn_dict, commands_dict,
             episode_id, scene_id):
    if ARGS.no_vis:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[VIS] matplotlib not available, skipping")
        return

    H, W = grid.shape
    ped_colors = ['#FF6B6B', '#FFD93D', '#6BCB77', '#4D96FF', '#FF9F45', '#C77DFF']

    def from_isaac(ix, iy):
        x_m, y_m = -float(ix), -float(iy)
        return (map_min_x + map_max_x) - x_m, (map_min_y + map_max_y) - y_m

    def i2px(ix, iy):
        x2, y2 = from_isaac(ix, iy)
        return int(round((x2 - min_x) / scale)), int(round((y2 - min_y) / scale))

    # 构建用于显示的 PSDF（所有行人路径）
    psdf_vis = make_psdf_map(H, W)
    for cn, sd in spawn_dict.items():
        stamp_path_into_psdf(psdf_vis, grid, [i2px(*sd["pos"][:2])], scale)
        for cmd in commands_dict.get(cn, []):
            if cmd.get("cmd") != "GoTo":
                continue
            path_px = [i2px(p[0], p[1]) for p in cmd.get("path", [])]
            if path_px:
                stamp_path_into_psdf(psdf_vis, grid, path_px, scale)

    def draw_agents(ax, draw_paths=True):
        # 机器人起点 / 终点
        if robot_start:
            rs = i2px(*robot_start[:2])
            ax.plot(rs[0], rs[1], 's', color='cyan', markersize=11, zorder=8)
            ax.annotate("R_s", rs, fontsize=6, color='cyan',
                        xytext=(4, 4), textcoords='offset points')
        if robot_goal:
            rg = i2px(*robot_goal[:2])
            ax.plot(rg[0], rg[1], '*', color='cyan', markersize=15, zorder=8)
            ax.annotate("R_g", rg, fontsize=6, color='cyan',
                        xytext=(4, 4), textcoords='offset points')

        for ci, (cn, sd) in enumerate(spawn_dict.items()):
            color = ped_colors[ci % len(ped_colors)]
            # 行人出生点
            sp = i2px(*sd["pos"][:2])
            ax.plot(sp[0], sp[1], 'o', color=color, markersize=8, zorder=7,
                    markeredgecolor='k', markeredgewidth=0.8)
            ax.annotate(f"P{ci}", sp, fontsize=6, color=color, fontweight='bold',
                        xytext=(3, 3), textcoords='offset points')

            # 绘制该行人的所有 GoTo 命令及其路径
            goto_cmds = [cmd for cmd in commands_dict.get(cn, []) if cmd.get("cmd") == "GoTo"]
            for wi, cmd in enumerate(goto_cmds):
                params = cmd.get("params", [])
                if len(params) < 2:
                    continue
                # 航点坐标
                wp = i2px(float(params[0]), float(params[1]))
                is_last = (wi == len(goto_cmds) - 1)
                marker = '*' if is_last else 'D'
                size = 10 if is_last else 6
                ax.plot(wp[0], wp[1], marker, color=color, markersize=size,
                        markeredgecolor='k', markeredgewidth=0.8, zorder=7)
                ax.annotate(f"P{ci}W{wi+1}", wp, fontsize=5, color=color,
                            xytext=(3, -8), textcoords='offset points')
                if draw_paths:
                    pts = [i2px(p[0], p[1]) for p in cmd.get("path", [])]
                    if pts:
                        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                                '-', color=color, lw=1.2, alpha=0.8, zorder=5)

    n_peds = len(spawn_dict)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    ax_occ, ax_esdf, ax_psdf = axes

    # ── Panel ①: Occupancy + Agents ──
    vis_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    vis_rgb[grid == 0] = [240, 240, 240]
    vis_rgb[grid == 1] = [50, 50, 50]
    ax_occ.imshow(vis_rgb, origin="lower")
    ax_occ.set_title("① Occupancy + Agents & Paths", fontsize=10, fontweight='bold')
    draw_agents(ax_occ, draw_paths=True)

    # ── Panel ②: ESDF ──
    esdf_m = np.ma.masked_where(grid == 1, np.clip(esdf, 0, 2))
    im1 = ax_esdf.imshow(esdf_m, origin="lower", cmap="viridis",
                         vmin=0, vmax=2, interpolation="bilinear")
    plt.colorbar(im1, ax=ax_esdf, fraction=0.046, pad=0.04,
                 label="ESDF clearance (m, clipped 2 m)")
    ax_esdf.set_title("② ESDF (Static Obstacle Clearance)", fontsize=10, fontweight='bold')
    draw_agents(ax_esdf, draw_paths=False)

    # ── Panel ③: PSDF ──
    p95 = float(np.percentile(psdf_vis[grid == 0], 95)) if (grid == 0).any() else 1.0
    psdf_m = np.ma.masked_where(grid == 1, np.clip(psdf_vis, 0, p95))
    im2 = ax_psdf.imshow(psdf_m, origin="lower", cmap="hot_r", interpolation="bilinear")
    plt.colorbar(im2, ax=ax_psdf, fraction=0.046, pad=0.04,
                 label="Ped proximity cost (higher = more crowded)")
    if p95 > 0:
        ax_psdf.contour(np.clip(psdf_vis, 0, p95),
                        levels=[p95 * 0.2, p95 * 0.6],
                        colors=['orange', 'red'], linewidths=[1.0, 1.5],
                        origin="lower", zorder=6)
    ax_psdf.set_title(
        f"③ PSDF — ped avoidance field  "
        f"(inner={PSDF_INNER_R_M} m  outer={PSDF_OUTER_R_M} m)",
        fontsize=10, fontweight='bold')
    draw_agents(ax_psdf, draw_paths=True)

    plt.suptitle(f"Episode {episode_id}  —  {scene_id}   peds={n_peds}",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    vis_path = out_path.replace(".json", "_vis.png")
    plt.savefig(vis_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"[VIS] → {vis_path}")

# ── Core episode generator ──────────────────────────────────────────
def generate_one_episode(episode_id, seed, out_dir, scene_id,
                         grid, esdf, esdf_walls,
                         min_x, min_y, scale,
                         map_min_x, map_max_x, map_min_y, map_max_y,
                         to_isaac, plan_path_fn,
                         nav_area_m2, actual_num,
                         valid_islands, px_to_island, get_isaac_island):
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

    def from_isaac(ix, iy):
        x_m, y_m = -float(ix), -float(iy)
        return (map_min_x+map_max_x)-x_m, (map_min_y+map_max_y)-y_m

    def isaac_to_px(ix, iy):
        return world_to_px(*from_isaac(ix, iy), min_x, min_y, scale)

    def sample_point():
        for _ in range(10):
            px_py = sample_free_point(grid, esdf_walls, rng,
                                      PED_SPAWN_MIN_CLEARANCE_M, _free_cache)
            if px_py is None: return None
            if reachable_area_px(grid, px_py) < MIN_REACHABLE_PX: continue
            x_2d, y_2d = px_to_world(px_py[0], px_py[1], min_x, min_y, scale)
            ix, iy = to_isaac(x_2d, y_2d)
            return (ix, iy, 0.0)
        return None

    # ── Robot sampling (reduced retries) ────────────
    robot_start = robot_goal = robot_island_idx = None
    for attempt in range(100):
        s = sample_point(); g = sample_point()
        if s is None or g is None: continue
        if dist2(s, g) < 2.0: continue
        path_pts = plan_path_fn(s, g)
        if path_pts is None: continue
        geo = path_length(path_pts)
        if not (MIN_ROBOT_DIST <= geo <= MAX_ROBOT_DIST): continue
        s_island = get_isaac_island(s)
        if s_island is None: continue
        robot_start, robot_goal, robot_island_idx = s, g, s_island
        print(f"  [ep{episode_id}] robot geo={geo:.1f}m island={s_island} attempt={attempt+1}")
        break
    if robot_start is None:
        print(f"  [ep{episode_id}] L2 relax dist")
        for attempt in range(200):
            s = sample_point(); g = sample_point()
            if s is None or g is None: continue
            if dist2(s, g) < MIN_ROBOT_DIST: continue
            if plan_path_fn(s, g) is None: continue
            s_island = get_isaac_island(s)
            if s_island is None: continue
            robot_start, robot_goal, robot_island_idx = s, g, s_island
            break
    if robot_start is None:
        print(f"  [ep{episode_id}] L3 largest island")
        for attempt in range(200):
            s = sample_point(); g = sample_point()
            if s is None or g is None: continue
            if get_isaac_island(s) != 0 or get_isaac_island(g) != 0: continue
            if plan_path_fn(s, g) is None: continue
            robot_start, robot_goal, robot_island_idx = s, g, 0
            break
    if robot_start is None:
        print(f"  [ep{episode_id}] FATAL robot sampling failed")
        return {"skipped": True}

    robot_orientation = rng.uniform(0, 2*math.pi)

    # ── Pedestrians ────────────────────────────────
    island_plan = plan_island_allocation(actual_num, robot_island_idx, valid_islands)
    spawn_dict = {}
    commands_dict = {}

    # adaptive levels
    if actual_num <= 2:
        active_levels = [2]
    elif actual_num <= 4:
        active_levels = [1, 2]
    else:
        active_levels = [0, 1, 2]

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

    def try_place_ped(i):
        target_island = island_plan[i] if i < len(island_plan) else None
        same_island = (target_island == robot_island_idx)

        for level in active_levels:
            block_r = PED_CORRIDOR_BLOCK_R[level]
            min_spawn_dist = 2.0 if level > 0 else MIN_CHAR_DIST
            b_grid = build_blocked_grid(block_r)
            b_esdf = distance_transform_edt(b_grid == 0, sampling=scale) if block_r > 0 and placed_path_px_union else esdf

            def sample_point_blocked():
                for _ in range(10):
                    px_py = sample_free_point(b_grid, b_esdf, rng, PED_SPAWN_MIN_CLEARANCE_M,
                                              _free_cache if block_r == 0 else None)
                    if px_py is None: return None
                    if reachable_area_px(b_grid, px_py) < MIN_REACHABLE_PX: continue
                    x_2d, y_2d = px_to_world(px_py[0], px_py[1], min_x, min_y, scale)
                    return to_isaac(x_2d, y_2d) + (0.0,)
                return None

            def plan_path_ped_blocked(a, b):
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

            for attempt in range(80):
                p0 = sample_point_blocked()
                if p0 is None: continue
                if same_island and dist2(p0, robot_start) < min_spawn_dist: continue
                if any(dist2(p0, sp["pos"]) < MIN_CHAR_SPACING for sp in spawn_dict.values()): continue
                p0_island = get_isaac_island(p0)
                if target_island is not None and p0_island != target_island: continue
                if p0_island is None: continue
                if same_island and plan_path_ped_blocked(p0, robot_start) is None: continue

                chain = [(p0, None)]
                prev = p0
                chain_ok = True
                for _ in range(NUM_CHAIN_WPT):
                    wpt_ok = False
                    for _ in range(120):
                        w = sample_point_blocked()
                        if w is None: continue
                        if dist2(w, prev) < 2.0: continue
                        if not same_island and get_isaac_island(w) != target_island: continue
                        result = plan_path_ped_blocked(prev, w)
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
                        r = plan_path_ped_blocked(chain[best_k][0], wpts[best_k][0])
                        if r: wpts[best_k] = (wpts[best_k][0], r)
                        r = plan_path_ped_blocked(wpts[-2][0], wpts[-1][0])
                        if r: wpts[-1] = (wpts[-1][0], r)
                chain = [chain[0]] + wpts

                new_path_px = {isaac_to_px(*p0[:2])}
                for _, plan_result in chain[1:]:
                    if plan_result is None: continue
                    _, path_px = plan_result
                    new_path_px.update(path_px)
                stamp_path_into_psdf(psdf, grid, list(new_path_px), scale)
                placed_path_px_union.update(new_path_px)

                cn = get_character_name_by_index(len(spawn_dict))
                spawn_dict[cn] = {"pos": [float(p0[0]), float(p0[1]), 0.0], "rot": 0.0}
                cmds = []
                for _, plan_result in chain[1:]:
                    if plan_result is None: continue
                    isaac_pts, _ = plan_result
                    if path_length(isaac_pts) < WAYPOINT_MIN_DIST_M: continue
                    x, y, z = isaac_pts[-1]
                    cmds.append({"cmd": "GoTo", "params": [f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", "_"], "path": isaac_pts})
                    aux = sample_aux(rng)
                    if aux: cmds.append(aux)
                commands_dict[cn] = cmds
                print(f"  [ep{episode_id}] ped{i} OK level={level} attempt={attempt+1} island={p0_island}")
                return True
            print(f"  [ep{episode_id}] ped{i} level {level} exhausted")
        return False

    for i in range(actual_num):
        if not try_place_ped(i):
            print(f"  [ep{episode_id}] WARN could not place ped {i}")

    # ── Write JSON ──────────────────────────────
    episode_data = {
        "episode": {
            "episode_id": episode_id,
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
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(episode_data, f, indent=4)
    elapsed = round(time.time()-t0, 2)
    print(f"  [ep{episode_id}] done placed={len(spawn_dict)} elapsed={elapsed}s → {out_path}")
    save_vis(out_path, grid, esdf, min_x, min_y, scale,
             map_min_x, map_max_x, map_min_y, map_max_y,
             robot_start, robot_goal, spawn_dict, commands_dict,
             episode_id, scene_id)
    return {"episode_id": episode_id, "placed": len(spawn_dict), "elapsed_s": elapsed}

# ═════════════════════════════════════════════════════════════════════
def main():
    if not os.path.exists(ARGS.semantic_map_json):
        print("[FATAL] Semantic map not found"); return 1
    res = load_2d_map(ARGS.semantic_map_json, ARGS.robot_radius_2d, ARGS.scale_m_per_px)
    if res[0] is None:
        print("[FATAL] 2D map load failed"); return 1
    (grid, esdf_walls, esdf_combined,
     min_x, min_y, scale, sem_data, raw_grid,
     map_min_y, map_max_y, map_min_x, map_max_x) = res

    # `esdf` retained as alias for the combined field (used for cost).
    # Hard corridor checks now use `esdf_walls`.
    esdf = esdf_combined
    # Isaac coordinate helpers
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

    scene_id = os.path.splitext(os.path.basename(ARGS.semantic_map_json))[0]
    scene_id = scene_id.replace("2D_Semantic_Map_", "")
    # typical filename: 2D_Semantic_Map_0_839873_Complete.json → scene_id = "839873"
    # better parsing:
    import re
    m = re.match(r"2D_Semantic_Map_\d+_(\d+)_Complete", os.path.basename(ARGS.semantic_map_json))
    if m:
        scene_id = m.group(1)
    out_dir = os.path.join(ARGS.output_dir, scene_id)
    os.makedirs(out_dir, exist_ok=True)

    # islands
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
    area_cap = max(1, int(nav_area_m2 / AREA_PER_PERSON_M2))

    ep_num_people = {}
    if ARGS.num_people_for_episode:
        if len(ARGS.num_people_for_episode) != len(ARGS.episode_ids):
            print("[FATAL] length mismatch"); return 2
        for eid, n in zip(ARGS.episode_ids, ARGS.num_people_for_episode):
            capped = min(n, area_cap)
            ep_num_people[eid] = capped
    else:
        for eid in ARGS.episode_ids:
            ep_num_people[eid] = min(ARGS.num_people, area_cap)

    for eid in ARGS.episode_ids:
        seed = ARGS.seed_offset + eid
        print(f"\n[GEN] episode_{eid} seed={seed} num_people={ep_num_people[eid]}")
        generate_one_episode(eid, seed, out_dir, scene_id,
                             grid, esdf, esdf_walls,
                             min_x, min_y, scale,
                             map_min_x, map_max_x, map_min_y, map_max_y,
                             to_isaac, plan_path,
                             nav_area_m2, ep_num_people[eid],
                             valid_islands, px_to_island, get_isaac_island)

    return 0

if __name__ == "__main__":
    sys.exit(main())