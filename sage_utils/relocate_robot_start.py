#!/usr/bin/env python3
"""Post-process existing tracking episodes: relocate robot start to 1-3m
from the target character, keeping all other episode data unchanged."""
from __future__ import annotations

import argparse, heapq, json, math, os, sys, random

# ═══ Inlined from generate_episode_fast_tracking.py ═══════════════════
# (Module-level arg-parse in that file prevents clean import)
import numpy as np
from collections import deque
from scipy.ndimage import distance_transform_edt


def world_to_px(x_m, y_m, min_x, min_y, scale):
    return (int(round((float(x_m) - min_x) / scale)),
            int(round((float(y_m) - min_y) / scale)))

def sm_to_isaac(x_m, y_m, map_min_x, map_max_x, map_min_y, map_max_y):
    return -(map_min_x + map_max_x - float(x_m)), -(map_min_y + map_max_y - float(y_m))

def snap(grid, px_py):
    H, W = grid.shape
    px, py = px_py
    if 0 <= px < W and 0 <= py < H and grid[py, px] == 0:
        return px_py
    for r in range(1, 100):
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                nx, ny = px+dx, py+dy
                if 0 <= nx < W and 0 <= ny < H and grid[ny, nx] == 0:
                    return (nx, ny)
    return None

def astar(grid, start, goal, esdf=None, esdf_corridor=None,
          min_corridor_m=0.15, esdf_weight=0.2, max_steps=50000):
    H, W = grid.shape
    sy, sx = start[1], start[0]
    gy, gx = goal[1], goal[0]
    if not (0 <= sx < W and 0 <= sy < H) or not (0 <= gx < W and 0 <= gy < H):
        return None
    if grid[sy, sx] == 1 or grid[gy, gx] == 1:
        return None
    start_w = 0.0
    parent = {}
    g_cost = {start: 0.0}
    f_cost = {start: math.hypot(sx-gx, sy-gy)}
    heap = [(f_cost[start], start)]
    visited = 0
    while heap and visited < max_steps:
        _, cur = heapq.heappop(heap)
        visited += 1
        if cur == goal:
            path = []
            while cur in parent:
                path.append(cur)
                cur = parent[cur]
            path.append(start)
            path.reverse()
            return [(x, y) for y, x in path]
        for dy, dx in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]:
            ny, nx = cur[0]+dy, cur[1]+dx
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if grid[ny, nx] == 1:
                continue
            nxt = (ny, nx)
            step_cost = 1.414 if dx and dy else 1.0
            if esdf_corridor is not None:
                corridor_val = float(esdf_corridor[ny, nx])
                if corridor_val < min_corridor_m:
                    continue
                step_cost += 20.0 * (1.0 - min(corridor_val / (min_corridor_m*3), 1.0))
            if esdf is not None and esdf_weight > 0:
                ed_val = float(esdf[ny, nx])
                step_cost += esdf_weight * max(15.0 - ed_val, 0.0)
            ng = g_cost[cur] + step_cost
            if nxt not in g_cost or ng < g_cost[nxt]:
                g_cost[nxt] = ng
                f = ng + math.hypot(nx-gx, ny-gy)
                f_cost[nxt] = f
                parent[nxt] = cur
                heapq.heappush(heap, (f, nxt))
    return None

def px_to_world(px, py, min_x, min_y, scale):
    return min_x + (px + 0.5) * scale, min_y + (py + 0.5) * scale

# ── load_2d_map — synced from generate_episode_fast_tracking.py ────
def load_2d_map(sem_json, robot_r, scale):
    """Build occupancy grid + dual ESDFs from semantic map JSON."""
    with open(sem_json, encoding="utf-8") as f:
        raw = json.load(f)
    items = raw["instances"] if isinstance(raw, dict) and "instances" in raw else raw
    meta  = raw.get("meta", {}) if isinstance(raw, dict) else {}
    if all(k in meta for k in ("x_min", "x_max", "y_min", "y_max")):
        min_x, max_x = float(meta["x_min"]), float(meta["x_max"])
        min_y, max_y = float(meta["y_min"]), float(meta["y_max"])
    else:
        coords = [(float(y), float(x)) for inst in items for y, x in inst.get("mask_coords_m", [])]
        if not coords:
            print("[load_2d_map] ERROR: no extent info available.")
            return (None,) * 12
        min_y = min(c[0] for c in coords); max_y = max(c[0] for c in coords)
        min_x = min(c[1] for c in coords); max_x = max(c[1] for c in coords)
    h = int(np.ceil((max_y - min_y) / scale)) + 1
    w = int(np.ceil((max_x - min_x) / scale)) + 1
    raw_grid = np.zeros((h, w), dtype=np.uint8)
    for inst in items:
        label = str(inst.get("category_label", "")).lower()
        if label not in ("wall", "unable area", "unable_area"):
            continue
        for y_m, x_m in inst.get("mask_coords_m", []):
            try:
                py = int(round((float(y_m) - min_y) / scale))
                px = int(round((float(x_m) - min_x) / scale))
                if 0 <= py < h and 0 <= px < w:
                    raw_grid[py, px] = 1
            except (ValueError, TypeError):
                pass
    furniture_labels = ("table", "chair", "sofa", "bed", "wardrobe", "desk", "counter", "cabinet")
    furniture_mask = np.zeros((h, w), dtype=np.uint8)
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
            except (ValueError, TypeError):
                pass
    floor_px = 0
    interior_mask = np.zeros((h, w), dtype=np.uint8)
    for inst in items:
        label = str(inst.get("category_label", "")).lower()
        if label != "floor":
            continue
        for y_m, x_m in inst.get("mask_coords_m", []):
            try:
                py = int(round((float(y_m) - min_y) / scale))
                px = int(round((float(x_m) - min_x) / scale))
                if 0 <= py < h and 0 <= px < w:
                    interior_mask[py, px] = 1; floor_px += 1
            except (ValueError, TypeError):
                pass
    from scipy.ndimage import binary_dilation, binary_fill_holes, binary_closing
    if floor_px > 50:
        interior_mask = binary_dilation(interior_mask.astype(bool), iterations=2)
        interior_mask = binary_fill_holes(interior_mask).astype(np.uint8)
    else:
        free_now = (raw_grid == 0).astype(np.uint8)
        border_mask = np.zeros_like(free_now, dtype=bool)
        border_mask[0, :] = border_mask[-1, :] = True
        border_mask[:, 0] = border_mask[:, -1] = True
        from scipy.ndimage import label as scilabel
        labeled_free, _ = scilabel(free_now, structure=np.ones((3,3)))
        border_lbls = set(np.unique(labeled_free[border_mask]).tolist()); border_lbls.discard(0)
        border_free_px = sum(int((labeled_free == lbl).sum()) for lbl in border_lbls)
        total_free_px = int(free_now.sum())
        border_ratio = border_free_px / max(total_free_px, 1)
        if border_ratio < 0.15:
            interior_mask = free_now
        else:
            building_seed = ((raw_grid == 1) | (furniture_mask == 1)).astype(bool)
            bridge_px = max(1, int(round(0.4 / scale)))
            building_blob = binary_dilation(building_seed, structure=np.ones((2*bridge_px+1, 2*bridge_px+1)))
            close_px = max(1, int(round(0.5 / scale)))
            building_solid = binary_closing(building_blob, structure=np.ones((2*close_px+1, 2*close_px+1)))
            building_solid = binary_fill_holes(building_solid)
            outside_cands = (~building_solid).astype(np.uint8)
            lbl_out, n_out = scilabel(outside_cands, structure=np.ones((3,3)))
            exterior_mask = np.zeros_like(outside_cands, dtype=bool)
            if n_out > 0:
                border_out_lbls = set(np.unique(lbl_out[border_mask]).tolist())
                border_out_lbls.discard(0)
                for lbl in border_out_lbls:
                    exterior_mask |= (lbl_out == lbl)
            interior_mask = ((~exterior_mask) & (raw_grid == 0)).astype(np.uint8)
            labeled, n_lbl = scilabel(interior_mask, structure=np.ones((3,3)))
            if n_lbl > 1:
                sizes = [(lbl, int((labeled == lbl).sum())) for lbl in range(1, n_lbl+1)]
                sizes.sort(key=lambda x: -x[1])
                threshold = max(50, int(0.1 * sizes[0][1]))
                keep = {lbl for lbl, sz in sizes if sz >= threshold}
                interior_mask = np.isin(labeled, list(keep)).astype(np.uint8)
    interior_px = int(interior_mask.sum())
    total_free = int((raw_grid == 0).sum())
    if total_free == 0 or interior_px / max(total_free, 1) < 0.05:
        interior_mask = (raw_grid == 0).astype(np.uint8)
    raw_grid[interior_mask == 0] = 1
    if robot_r > 0:
        dist_m = distance_transform_edt(raw_grid == 0, sampling=scale)
        planning_grid = (dist_m <= robot_r).astype(np.uint8)
    else:
        planning_grid = raw_grid.copy()
    esdf_combined = distance_transform_edt(planning_grid == 0, sampling=scale)
    walls_only = (raw_grid == 1).astype(np.uint8)
    esdf_walls = distance_transform_edt(walls_only == 0, sampling=scale)
    return (planning_grid, esdf_walls, esdf_combined,
            min_x, min_y, scale, items, raw_grid,
            min_y, max_y, min_x, max_x)
# ═══ End inlined functions ═══════════════════════════════════════════


def _line_clear_on_grid(grid, start_px, end_px):
    """Bresenham-style check: return True if all cells are free (0)."""
    sx, sy = start_px
    ex, ey = end_px
    dx, dy = abs(ex - sx), abs(ey - sy)
    nx, ny = 1 if ex > sx else -1, 1 if ey > sy else -1
    err = dx - dy
    cx, cy = sx, sy
    while True:
        if not (0 <= cx < grid.shape[1] and 0 <= cy < grid.shape[0]):
            return False
        if grid[cy, cx] == 1:
            return False
        if (cx, cy) == (ex, ey):
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; cx += nx
        if e2 < dx:
            err += dx; cy += ny


def find_robot_start(target_pos_isaac, grid, esdf_walls, esdf_combined,
                     min_x, min_y, scale,
                     map_min_x, map_max_x, map_min_y, map_max_y,
                     rng, d_min=1.0, d_max=3.0, robot_r=0.3, los=False):
    """Sample a valid robot start d_min–d_max from target (returns None if fail)."""
    from_isaac = lambda ix, iy: (
        (map_min_x + map_max_x) + float(ix),
        (map_min_y + map_max_y) + float(iy),
    )
    isaac_to_px = lambda ix, iy: world_to_px(
        *from_isaac(ix, iy), min_x, min_y, scale)

    tx, ty, _ = target_pos_isaac
    target_px = isaac_to_px(tx, ty)
    H, W = grid.shape
    if not (0 <= target_px[0] < W and 0 <= target_px[1] < H):
        return None

    # Convert target to raw coords for sampling offset
    target_raw_x, target_raw_y = from_isaac(tx, ty)

    for _ in range(800):
        angle = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(d_min, d_max)
        cand_raw_x = target_raw_x + d * math.cos(angle)
        cand_raw_y = target_raw_y + d * math.sin(angle)
        px, py = world_to_px(cand_raw_x, cand_raw_y, min_x, min_y, scale)
        if not (0 <= px < W and 0 <= py < H):
            continue
        if grid[py, px] == 1:
            continue
        # Convert candidate to Isaac coords
        cand_isaac_x, cand_isaac_y = sm_to_isaac(
            cand_raw_x, cand_raw_y, map_min_x, map_max_x, map_min_y, map_max_y)
        cand = (cand_isaac_x, cand_isaac_y, 0.0)
        if los:
            spx, spy = world_to_px(
                *from_isaac(*target_pos_isaac[:2]), min_x, min_y, scale)
            epx, epy = world_to_px(
                *from_isaac(cand_isaac_x, cand_isaac_y), min_x, min_y, scale)
            if not _line_clear_on_grid(grid, (spx, spy), (epx, epy)):
                continue
        robot_yaw = math.atan2(ty - cand_isaac_y, tx - cand_isaac_x)
        return cand, robot_yaw
    return None


def main():
    p = argparse.ArgumentParser(
        description="Relocate robot start positions 1-3m from target.")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--scene_id", required=True)
    p.add_argument("--semantic_maps_root",
                   default="/workspace/SAGE-3D_Official/SAGE-3D_data/semantic_maps_v2")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--min_dist", type=float, default=1.0,
                   help="Min robot distance from target (m)")
    p.add_argument("--max_dist", type=float, default=3.0,
                   help="Max robot distance from target (m)")
    p.add_argument("--los", action="store_true", default=False,
                   help="Require line-of-sight between robot and target")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    sem_json = os.path.join(
        args.semantic_maps_root,
        f"2D_Semantic_Map_{args.scene_id}_Complete.json")
    if not os.path.exists(sem_json):
        print(f"[FATAL] Semantic map not found: {sem_json}")
        return 1

    input_scene = os.path.join(args.input_dir, args.scene_id)
    if not os.path.isdir(input_scene):
        print(f"[FATAL] Input scene dir not found: {input_scene}")
        return 1

    output_scene = os.path.join(args.output_dir, args.scene_id)
    os.makedirs(output_scene, exist_ok=True)

    # Load map — returns 12 values
    result = load_2d_map(sem_json, robot_r=0.3, scale=0.05)
    if result is None or result[0] is None:
        print("[FATAL] load_2d_map failed")
        return 1
    grid, esdf_walls, esdf_combined, min_x, min_y, scale = result[:6]
    map_min_x, map_max_x, map_min_y, map_max_y = result[3], result[11], result[4], result[9]

    print(f"  Grid: {grid.shape[1]}x{grid.shape[0]}, scale={scale}m, "
          f"obstacles={int(grid.sum())}/{grid.size}")

    ep_files = sorted(
        [f for f in os.listdir(input_scene)
         if f.startswith("episode_") and f.endswith(".json")],
        key=lambda x: int(x.split("_")[1].split(".")[0]))
    if not ep_files:
        print(f"[WARN] No episode_*.json in {input_scene}")
        return 0

    rng = random.Random(args.seed)
    modified = 0
    skipped = 0

    for ep_fn in ep_files:
        with open(os.path.join(input_scene, ep_fn)) as f:
            data = json.load(f)
        episode = data["episode"] if "episode" in data else data

        target_spawn = (episode["characters"]["spawn_positions"]
                        .get("Character", {}).get("pos"))
        if target_spawn is None:
            skipped += 1
            with open(os.path.join(output_scene, ep_fn), "w") as f:
                json.dump(data, f, indent=2)
            continue

        target_pos = (target_spawn[0], target_spawn[1], 0.0)
        robot_info = find_robot_start(
            target_pos, grid, esdf_walls, esdf_combined,
            min_x, min_y, scale,
            map_min_x, map_max_x, map_min_y, map_max_y,
            rng, d_min=args.min_dist, d_max=args.max_dist,
            los=args.los)

        if robot_info is None:
            print(f"  [SKIP] {ep_fn}: no valid robot start found")
            skipped += 1
            with open(os.path.join(output_scene, ep_fn), "w") as f:
                json.dump(data, f, indent=2)
            continue

        robot_start, robot_yaw = robot_info
        episode["robot"]["start_pos"] = [float(robot_start[0]),
                                         float(robot_start[1]),
                                         float(robot_start[2])]
        episode["robot"]["start_orientation"] = float(robot_yaw)

        dist = math.hypot(target_spawn[0] - robot_start[0],
                          target_spawn[1] - robot_start[1])
        print(f"  [ OK] {ep_fn}: robot {dist:.2f}m from target, "
              f"yaw={math.degrees(robot_yaw):.0f}°")

        out_data = data if "episode" in data else {"episode": episode}
        with open(os.path.join(output_scene, ep_fn), "w") as f:
            json.dump(out_data, f, indent=2)
        modified += 1

    print(f"\nDone: {modified} modified, {skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
