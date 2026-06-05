# sage_utils/occupancy_utils.py
"""
Occupancy grid generation via PhysX raycast on collision mesh.
Independent of NavMesh internal APIs (which don't expose geometry in Python).
"""
from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Optional


def occupancy_from_collision_mesh(
    stage,
    collision_root: str = "/World/scene_collision",
    cell_size: float = 0.1,
    bounds_margin: float = 2.0,
    height_range: Tuple[float, float] = (0.10, 1.2),
    obstacle_probe_radius: float = 0.08,
    floor_ray_extra: float = 5.0,
    floor_tolerance: float = 0.5,
    n_height_probes: int = 3,
) -> Optional[Tuple[np.ndarray, dict]]:
    """
    Build 2D occupancy via PhysX raycast.
      - 0 = free, 1 = occupied
      - Uses stage up-axis automatically.
      - Estimates ground height to reject "fake floors" (tables, beds, upper levels).
    """
    import carb
    from pxr import UsdGeom, Usd
    import omni.physx

    # --- detect up axis ---
    up_token = UsdGeom.GetStageUpAxis(stage)   # "Y" or "Z"
    if up_token == "Z":
        h0, h1, up = 0, 1, 2
    else:
        h0, h1, up = 0, 2, 1
    print(f"[OCC] up-axis={up_token}, horiz=({h0},{h1}), up={up}")

    # --- compute scene bounds ---
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    root_prim = stage.GetPrimAtPath(collision_root)
    if not root_prim or not root_prim.IsValid():
        print(f"[OCC] collision_root invalid: {collision_root}")
        return None
    rng = cache.ComputeWorldBound(root_prim).ComputeAlignedRange()
    if rng.IsEmpty():
        print("[OCC] bbox is empty")
        return None

    bmin, bmax = rng.GetMin(), rng.GetMax()
    min_u = float(bmin[h0]) - bounds_margin
    max_u = float(bmax[h0]) + bounds_margin
    min_v = float(bmin[h1]) - bounds_margin
    max_v = float(bmax[h1]) + bounds_margin
    floor_low  = float(bmin[up])
    floor_high = float(bmax[up])
    ray_top = floor_high + floor_ray_extra
    ray_len = (ray_top - floor_low) + 1.0

    W = int(math.ceil((max_u - min_u) / cell_size))
    H = int(math.ceil((max_v - min_v) / cell_size))
    grid = np.ones((H, W), dtype=np.uint8)

    pxq = omni.physx.get_physx_scene_query_interface()
    if pxq is None:
        print("[OCC] PhysX scene query interface unavailable")
        return None

    # --- helpers ---
    def world_point(u, v, up_val):
        p = [0.0, 0.0, 0.0]
        p[h0] = u; p[h1] = v; p[up] = up_val
        return carb.Float3(p[0], p[1], p[2])

    def world_down():
        d = [0.0, 0.0, 0.0]
        d[up] = -1.0
        return carb.Float3(d[0], d[1], d[2])

    down_dir = world_down()

    # --- sanity ray at scene center ---
    cu0 = 0.5 * (min_u + max_u)
    cv0 = 0.5 * (min_v + max_v)
    sanity = pxq.raycast_closest(world_point(cu0, cv0, ray_top), down_dir, ray_len)
    print(f"[OCC] sanity ray at center: hit={bool(sanity and sanity.get('hit'))}"
          + (f", up={float(sanity['position'][up]):.3f}" if (sanity and sanity.get('hit')) else ""))

    # --- estimate ground height by sampling 50 random cells ---
    rng_local = np.random.default_rng(0)
    sample_hs = []
    for _ in range(80):
        su = float(rng_local.uniform(min_u, max_u))
        sv = float(rng_local.uniform(min_v, max_v))
        h = pxq.raycast_closest(world_point(su, sv, ray_top), down_dir, ray_len)
        if h and h.get("hit"):
            sample_hs.append(float(h["position"][up]))
    if sample_hs:
        ground_h = float(np.percentile(sample_hs, 10))
        print(f"[OCC] ground height ≈ {ground_h:.3f} "
              f"(from {len(sample_hs)} hits, range "
              f"[{min(sample_hs):.2f}, {max(sample_hs):.2f}])")
    else:
        ground_h = floor_low
        print(f"[OCC] no floor hits in sampling, fallback ground_h={ground_h:.3f}")

    hmin, hmax = height_range
    probe_heights = np.linspace(hmin, hmax, n_height_probes)
    print(f"[OCC] probe radius={obstacle_probe_radius}, heights={probe_heights.tolist()}, "
          f"floor_tol={floor_tolerance}")

    # --- main rasterization ---
    free_count = 0
    no_floor = 0
    wrong_floor = 0
    for iv in range(H):
        cv = min_v + (iv + 0.5) * cell_size
        for iu in range(W):
            cu = min_u + (iu + 0.5) * cell_size

            hit = pxq.raycast_closest(world_point(cu, cv, ray_top), down_dir, ray_len)
            if not hit or not hit.get("hit"):
                no_floor += 1
                continue
            floor_h = float(hit["position"][up])
            if abs(floor_h - ground_h) > floor_tolerance:
                wrong_floor += 1
                continue

            blocked = False
            for dh in probe_heights:
                n = pxq.overlap_sphere(
                    obstacle_probe_radius,
                    world_point(cu, cv, floor_h + dh),
                    lambda h: True,
                    True,   # any_hit
                )
                if n > 0:
                    blocked = True
                    break
            if not blocked:
                grid[iv, iu] = 0
                free_count += 1

    total = grid.size
    print(f"[OCC] grid {grid.shape}: free={free_count} ({100*free_count/total:.1f}%), "
          f"no_floor={no_floor} ({100*no_floor/total:.1f}%), "
          f"wrong_floor={wrong_floor} ({100*wrong_floor/total:.1f}%)")

    meta = {
        "up_axis": up_token,
        "horizontal_axes": [int(h0), int(h1)],
        "min_u": min_u, "min_v": min_v,
        "max_u": max_u, "max_v": max_v,
        "cell_size": cell_size,
        "ground_h": ground_h,
        "floor_tolerance": floor_tolerance,
        "probe_radius": obstacle_probe_radius,
        "height_range": list(height_range),
    }
    return grid, meta