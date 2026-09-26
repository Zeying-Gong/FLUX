"""
fix_occupancy_and_gen.py
────────────────────────
1. 读 occupancy.png，flood-fill 从四角找"场景外部"灰色区域
2. 内部(非外部) & 非障碍物 → 全部设为可导航
3. 把修复后的 mask 写回一个新的 occupancy_fixed.png（供对比）
4. 同时生成一个 patched semantic map JSON，把外部区域的坐标
   追加为 "unable area" instance，这样 generate_episode_json_interaction.py
   直接用 patched JSON 就能正确建图
5. 顺带打印场景统计（岛屿、可走面积、geodesic 距离分布）
   帮你判断 MIN_ROBOT_DIST 该设多少

Usage:
  python fix_occupancy_and_gen.py \
    --occ_png   /path/to/0026_839976/occupancy.png \
    --occ_json  /path/to/0026_839976/occupancy.json \
    --sem_json  /path/to/semantic_maps/2D_Semantic_Map_0026_839976_Complete.json \
    --out_dir   /tmp/0026_fixed \
    [--scale 0.05] [--robot_r 0.3] [--min_robot_dist 5.0]
"""
import argparse, json, math, heapq, os, shutil
import numpy as np
from PIL import Image
from collections import deque
from scipy.ndimage import distance_transform_edt, label as nd_label

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--occ_png",          required=True)
    p.add_argument("--occ_json",         required=True)
    p.add_argument("--sem_json",         required=True)
    p.add_argument("--out_dir",          required=True)
    p.add_argument("--scale",            type=float, default=0.05)
    p.add_argument("--robot_r",          type=float, default=0.3)
    p.add_argument("--min_robot_dist",   type=float, default=5.0)
    p.add_argument("--obstacle_thresh",  type=int,   default=50,
                   help="Pixels darker than this are obstacles (0=black wall)")
    p.add_argument("--exterior_thresh",  type=int,   default=200,
                   help="Pixels brighter than this AND outside boundary = exterior gray")
    return p.parse_args()


# ── Step 1: parse occupancy.png ──────────────────────────────────────────
def load_occ_png(png_path, obstacle_thresh, exterior_thresh):
    """
    Returns:
      raw_gray   : H×W uint8 grayscale
      obstacle   : H×W bool  (dark pixels = walls/furniture)
      white_free : H×W bool  (bright white = explicitly free in original)
    """
    img = Image.open(png_path).convert("L")
    raw = np.array(img, dtype=np.uint8)
    obstacle   = raw < obstacle_thresh          # black = obstacle
    white_free = raw > exterior_thresh          # white = free interior
    return raw, obstacle, white_free


# ── Step 2: flood-fill exterior ──────────────────────────────────────────
def find_exterior(raw_gray, obstacle, obstacle_thresh, exterior_thresh):
    """
    BFS from all 4-border pixels that are NOT obstacle (i.e. gray or white).
    Everything reachable = exterior.
    Interior = everything NOT exterior.
    """
    H, W = raw_gray.shape
    exterior = np.zeros((H, W), dtype=bool)

    # seed: border pixels that are not obstacle
    seeds = []
    for x in range(W):
        for y in [0, H-1]:
            if not obstacle[y, x]:
                seeds.append((y, x))
    for y in range(H):
        for x in [0, W-1]:
            if not obstacle[y, x]:
                seeds.append((y, x))

    q = deque()
    for s in seeds:
        if not exterior[s]:
            exterior[s] = True
            q.append(s)

    while q:
        cy, cx = q.popleft()
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = cy+dy, cx+dx
            if 0<=ny<H and 0<=nx<W and not exterior[ny,nx] and not obstacle[ny,nx]:
                exterior[ny,nx] = True
                q.append((ny,nx))

    interior = ~exterior  # inside the boundary (including obstacle pixels)
    return exterior, interior


# ── Step 3: build corrected navigable grid ───────────────────────────────
def build_fixed_grid(obstacle, interior):
    """
    navigable = interior AND NOT obstacle
    """
    navigable = interior & ~obstacle          # True = free to walk
    grid = (~navigable).astype(np.uint8)      # 0=free, 1=blocked (same convention as gen script)
    return grid


# ── Step 4: get real-world coords from occ_json + grid ───────────────────
def grid_to_world_coords(occ_json_path, grid_shape):
    """
    occupancy.json stores: lower, upper, scale
    Returns functions px→world and world→px, plus bounds.
    """
    with open(occ_json_path) as f:
        occ = json.load(f)

    # 3D bounds → use X,Y plane (index 0,1)
    lower = occ["lower"]   # [x_min, y_min, z_min]
    upper = occ["upper"]   # [x_max, y_max, z_max]
    scale = occ["scale"]

    x_min, y_min = lower[0], lower[1]
    x_max, y_max = upper[0], upper[1]

    H, W = grid_shape
    # pixels span (x_min..x_max) × (y_min..y_max)
    # px col → x,  px row → y
    def px_to_world(col, row):
        x = x_min + col * scale
        y = y_min + row * scale
        return x, y

    def world_to_px(x, y):
        col = int(round((x - x_min) / scale))
        row = int(round((y - y_min) / scale))
        return col, row

    return px_to_world, world_to_px, x_min, y_min, x_max, y_max, scale


# ── Step 5: patch semantic map JSON ─────────────────────────────────────
def patch_sem_json(sem_json_path, exterior_mask, px_to_world, out_path):
    """
    Adds one new instance with category_label='unable area'
    containing all exterior pixel coords in real-world meters.
    This lets generate_episode_json_interaction.py treat them as obstacles.
    """
    with open(sem_json_path, encoding="utf-8") as f:
        sem_data = json.load(f)

    ys, xs = np.where(exterior_mask)
    coords = []
    # subsample to keep JSON size manageable (every 2nd pixel)
    step = 2
    for row, col in zip(ys[::step], xs[::step]):
        wx, wy = px_to_world(col, row)
        coords.append([float(wy), float(wx)])   # stored as [y_m, x_m] per gen script

    exterior_instance = {
        "instance_id": 99999,
        "category_label": "unable area",
        "mask_coords_m": coords,
    }
    patched = sem_data + [exterior_instance]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(patched, f)
    print(f"[PATCH] Added exterior unable-area instance with {len(coords)} coords → {out_path}")
    return patched


# ── Step 6: diagnostics ──────────────────────────────────────────────────
def astar_dist(grid, start, goal):
    H, W = grid.shape
    open_set = [(0.0, start)]
    g = {start: 0.0}
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            return g[cur]
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = cur[0]+dx, cur[1]+dy
            if not (0<=nx<W and 0<=ny<H): continue
            if grid[ny,nx] == 1: continue
            nb = (nx,ny); tg = g[cur]+1.0
            if nb not in g or tg < g[nb]:
                g[nb] = tg
                heapq.heappush(open_set, (tg+math.hypot(nx-goal[0],ny-goal[1]), nb))
    return None

def run_diagnostics(grid, scale, robot_r, min_robot_dist):
    H, W = grid.shape

    # erode by robot radius
    esdf_raw = distance_transform_edt(grid == 0, sampling=scale)
    plan_grid = ((esdf_raw <= robot_r) | (grid == 1)).astype(np.uint8)
    esdf = distance_transform_edt(plan_grid == 0, sampling=scale)

    total_walk = int((plan_grid == 0).sum())
    print(f"\n{'='*55}")
    print(f"FIXED GRID DIAGNOSTICS  (robot_r={robot_r}m, scale={scale}m)")
    print(f"{'='*55}")
    print(f"Grid size        : {W}×{H} px")
    print(f"Walkable pixels  : {total_walk}  ({total_walk*scale**2:.1f} m²)")

    # island analysis
    free_mask = (grid == 0).astype(np.int32)
    labeled, n_labels = nd_label(free_mask)
    islands = []
    for lbl in range(1, n_labels+1):
        ys_i, xs_i = np.where(labeled == lbl)
        area_m2 = len(ys_i) * scale**2
        walk_px = int(((labeled==lbl) & (plan_grid==0)).sum())
        cx, cy = int(xs_i.mean()), int(ys_i.mean())
        islands.append({"lbl": lbl, "area_m2": area_m2,
                         "walk_px": walk_px, "centroid": (cx, cy)})
    islands.sort(key=lambda x: -x["area_m2"])
    print(f"Islands (raw)    : {n_labels}")
    for i, isl in enumerate(islands[:8]):
        print(f"  [{i}] {isl['area_m2']:.1f}m²  walk_px={isl['walk_px']}  c={isl['centroid']}")

    # geodesic distance sample
    walk_ys, walk_xs = np.where(plan_grid == 0)
    if len(walk_ys) < 10:
        print("Too few walkable pixels for distance sampling.")
        return plan_grid, esdf, islands

    rng = np.random.default_rng(42)
    n = min(300, len(walk_ys))
    idx = rng.choice(len(walk_ys), size=n*2, replace=True)
    dists = []
    for k in range(n):
        s = (int(walk_xs[idx[k]]),         int(walk_ys[idx[k]]))
        g = (int(walk_xs[idx[k+n]]),       int(walk_ys[idx[k+n]]))
        d = astar_dist(plan_grid, s, g)
        if d is not None:
            dists.append(d * scale)

    if dists:
        dists = np.array(dists)
        print(f"\nGeodesic distance ({len(dists)}/{n} pairs connected):")
        print(f"  min={dists.min():.2f}  median={np.median(dists):.2f}  "
              f"max={dists.max():.2f}  mean={dists.mean():.2f} m")
        frac = (dists >= min_robot_dist).mean()
        print(f"  Fraction >= {min_robot_dist}m : {frac*100:.1f}%")
        if frac < 0.10:
            suggested = float(np.percentile(dists, 40))
            print(f"  ⚠  Too few valid pairs!  Suggest MIN_ROBOT_DIST ≤ {suggested:.1f}m")
        else:
            print(f"  ✓  Enough valid pairs for episode generation.")
    else:
        print("  No connected pairs found.")

    return plan_grid, esdf, islands


# ── Step 7: visualise ────────────────────────────────────────────────────
def save_vis(out_dir, raw_gray, obstacle, exterior, fixed_grid,
             plan_grid, esdf, scene_id):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.ndimage import label as nd_label
    except ImportError:
        print("[VIS] matplotlib not available"); return

    H, W = raw_gray.shape
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes = axes.flatten()

    # 0: original occupancy
    axes[0].imshow(raw_gray, cmap="gray", origin="lower")
    axes[0].set_title("Original occupancy.png", fontsize=10)

    # 1: exterior mask
    ext_vis = np.zeros((H,W,3), dtype=np.uint8)
    ext_vis[~exterior & ~obstacle] = [200,230,200]   # interior free = green
    ext_vis[exterior]               = [180,180,220]   # exterior = blue-gray
    ext_vis[obstacle]               = [40, 40, 40]    # obstacle = dark
    axes[1].imshow(ext_vis, origin="lower")
    from matplotlib.patches import Patch
    axes[1].legend(handles=[
        Patch(color=[c/255 for c in [200,230,200]], label="interior free"),
        Patch(color=[c/255 for c in [180,180,220]], label="exterior (excluded)"),
        Patch(color=[c/255 for c in [40,40,40]],   label="obstacle"),
    ], fontsize=7, loc="lower right")
    axes[1].set_title("Exterior flood-fill result", fontsize=10)

    # 2: fixed grid (0=free white, 1=blocked dark)
    fix_vis = np.zeros((H,W,3), dtype=np.uint8)
    fix_vis[fixed_grid==0] = [240,240,240]
    fix_vis[fixed_grid==1] = [40, 40, 40]
    axes[2].imshow(fix_vis, origin="lower")
    axes[2].set_title("Fixed navigable grid (before robot erosion)", fontsize=10)

    # 3: planning grid after erosion
    plan_vis = np.zeros((H,W,3), dtype=np.uint8)
    plan_vis[plan_grid==0] = [180,230,180]
    plan_vis[plan_grid==1] = [60, 30, 30]
    axes[3].imshow(plan_vis, origin="lower")
    axes[3].set_title(f"Planning grid (after {0.3}m robot erosion)", fontsize=10)

    # 4: ESDF
    esdf_m = np.ma.masked_where(plan_grid==1, np.clip(esdf,0,2))
    im = axes[4].imshow(esdf_m, origin="lower", cmap="viridis", vmin=0, vmax=2)
    plt.colorbar(im, ax=axes[4], fraction=0.04, label="clearance (m)")
    axes[4].set_title("ESDF clearance", fontsize=10)

    # 5: island coloring on planning grid
    free_mask = (plan_grid == 0).astype(np.int32)
    labeled, n_labels = nd_label(free_mask)
    isl_vis = plan_vis.copy()
    cmap = plt.cm.get_cmap("tab20", max(n_labels,1))
    for lbl in range(1, min(n_labels+1, 21)):
        col = (np.array(cmap(lbl-1)[:3])*255).astype(np.uint8)
        isl_vis[labeled==lbl] = col
    axes[5].imshow(isl_vis, origin="lower")
    axes[5].set_title(f"Islands on planning grid (n={n_labels})", fontsize=10)

    plt.suptitle(f"Scene: {scene_id}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{scene_id}_fixed_diag.png")
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIS] → {out_path}")

    # also save fixed occupancy PNG with 3 levels:
    #   black (0)     = obstacle
    #   gray (128)    = exterior / unable area
    #   white (255)   = free interior
    fixed_img_arr = np.where(exterior, 128,            # 外部→灰色
                     np.where(obstacle, 0, 255))      # 内部障碍→黑，自由→白
    Image.fromarray(fixed_img_arr.astype(np.uint8)).save(
        os.path.join(out_dir, f"{scene_id}_occupancy_fixed.png"))


# ═══════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    scene_id = os.path.basename(os.path.dirname(args.occ_png))
    print(f"[FIX] Scene: {scene_id}")

    # 1. Load occupancy image
    raw_gray, obstacle, white_free = load_occ_png(
        args.occ_png, args.obstacle_thresh, args.exterior_thresh)
    H, W = raw_gray.shape
    print(f"[FIX] Image: {W}×{H}  "
          f"obstacle_px={obstacle.sum()}  white_free_px={white_free.sum()}")

    # 2. Flood-fill exterior
    exterior, interior = find_exterior(raw_gray, obstacle,
                                       args.obstacle_thresh, args.exterior_thresh)
    print(f"[FIX] Exterior px={exterior.sum()}  Interior px={interior.sum()}")

    # 3. Build fixed grid
    fixed_grid = build_fixed_grid(obstacle, interior)
    free_px = int((fixed_grid==0).sum())
    print(f"[FIX] Fixed grid free px={free_px}  ({free_px*args.scale**2:.1f} m²)")

    # 4. World coord functions from occ_json
    px_to_world, world_to_px, x_min, y_min, x_max, y_max, occ_scale = \
        grid_to_world_coords(args.occ_json, (H, W))
    print(f"[FIX] World bounds: x=[{x_min:.2f},{x_max:.2f}]  y=[{y_min:.2f},{y_max:.2f}]  scale={occ_scale}")

    # 5. Patch semantic map JSON
    patched_json_path = os.path.join(args.out_dir,
        os.path.basename(args.sem_json).replace(".json", "_fixed.json"))
    patch_sem_json(args.sem_json, exterior, px_to_world, patched_json_path)

    # 6. Diagnostics
    plan_grid, esdf, islands = run_diagnostics(
        fixed_grid, args.scale, args.robot_r, args.min_robot_dist)

    # 7. Visualise
    save_vis(args.out_dir, raw_gray, obstacle, exterior,
             fixed_grid, plan_grid, esdf, scene_id)

    # 8. Print recommended episode gen command
    walk_m2 = int((plan_grid==0).sum()) * args.scale**2
    # suggest MIN_ROBOT_DIST based on scene size
    longest_dim = max((x_max-x_min), (y_max-y_min))
    suggested_min = max(2.0, min(args.min_robot_dist, longest_dim * 0.25))
    print(f"\n{'='*55}")
    print(f"RECOMMENDED EPISODE GENERATION COMMAND:")
    print(f"  --semantic_map_json {patched_json_path}")
    print(f"  --skip_navmesh_check  (test first without NavMesh)")
    print(f"  --episode_ids 0 1 2")
    print(f"  --num_people_for_episode 2 2 2")
    if suggested_min < args.min_robot_dist:
        print(f"\n  ⚠  Consider reducing MIN_ROBOT_DIST to {suggested_min:.1f}m "
              f"(scene longest dim={longest_dim:.1f}m)")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()