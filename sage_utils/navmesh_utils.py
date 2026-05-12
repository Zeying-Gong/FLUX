"""
navmesh_utils.py
────────────────
NavMesh bake + delta-layer caching helpers.

Cache strategy
--------------
After the first bake two files are written to <usda_parent>/../usda_processed/:

  <stem>_navvols.usda   – minimal delta layer containing only the
                          NavMeshVolume prims (written via Sdf.CopySpec,
                          loaded back as a root sublayer so the original
                          scene is never touched).

  Isaac Sim's internal navmesh cache (CACHE_ENABLED=True) handles the
  binary NavMesh data automatically – no save/load API needed.

Second run:
  1. Open original USDA normally.
  2. Insert _navvols.usda as a sublayer (session-layer-safe).
  3. Make collision geometry visible, call start_navmesh_baking_and_wait()
     → internal cache hits → completes in < 1 s.
  4. Hide collision geometry again.
"""
from __future__ import annotations

import os
import sys

import carb
import omni.kit.commands
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom


# ── helpers ────────────────────────────────────────────────────────────────

def _update(app, n: int = 1):
    for _ in range(n):
        app.update()


def cache_paths(usda_path: str, cache_dir: str | None) -> str:
    """Return the navvols delta-USDA path for *usda_path*.

    Default location: <usda_parent>/../usda_processed/<stem>_navvols.usda
    Override with *cache_dir*.
    """
    stem = os.path.splitext(os.path.basename(usda_path))[0]
    if cache_dir:
        base = cache_dir
    else:
        usda_dir = os.path.dirname(os.path.abspath(usda_path))
        parent   = os.path.dirname(usda_dir)
        base     = os.path.join(parent, "usda_processed")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{stem}_navvols.usda")


# ── visibility ─────────────────────────────────────────────────────────────

def set_subtree_visibility(stage, root_path: str, visible: bool) -> str | None:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        print(f"[Vis] {root_path} not found.")
        return None
    imageable = UsdGeom.Imageable(root)
    if not imageable:
        print(f"[Vis] {root_path} is not Imageable.")
        return None
    attr = imageable.GetVisibilityAttr()
    prev = attr.Get() if attr and attr.HasAuthoredValue() else "inherited"
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()
    print(f"[Vis] {root_path}: {prev} → {'inherited' if visible else 'invisible'}")
    return prev


# ── bbox ───────────────────────────────────────────────────────────────────

def compute_scene_bbox(stage):
    try:
        root = stage.GetDefaultPrim() if stage.HasDefaultPrim() \
               else stage.GetPseudoRoot()
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        bbox = cache.ComputeWorldBound(root)
        rng  = bbox.ComputeAlignedRange()
        if rng.IsEmpty():
            return None
        return rng.GetMin(), rng.GetMax()
    except Exception as e:
        print(f"[BBox] failed: {e}")
        return None


# ── NavMeshVolume sizing ────────────────────────────────────────────────────

def size_navmesh_volume(stage, volume_path: Sdf.Path,
                        padding: float, fallback_size: float):
    prim = stage.GetPrimAtPath(volume_path)
    if not prim.IsValid():
        print(f"[NavMesh] Volume prim not found at {volume_path}")
        return
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    half_extent_stage_units = 0.5 / mpu
    bbox = compute_scene_bbox(stage)
    if bbox is None:
        print(f"[NavMesh] Using fallback {fallback_size}m cube.")
        center   = Gf.Vec3d(0.0, 0.0, fallback_size * 0.25)
        half_ext = Gf.Vec3d(fallback_size*0.5, fallback_size*0.5, fallback_size*0.5)
    else:
        bmin, bmax = bbox
        center   = Gf.Vec3d((bmin[0]+bmax[0])*0.5, (bmin[1]+bmax[1])*0.5,
                             (bmin[2]+bmax[2])*0.5)
        half_ext = Gf.Vec3d((bmax[0]-bmin[0])*0.5*padding,
                             (bmax[1]-bmin[1])*0.5*padding,
                             (bmax[2]-bmin[2])*0.5*padding)
        print(f"[NavMesh] Scene bbox min={tuple(bmin)} max={tuple(bmax)}")
        print(f"[NavMesh] Volume center={tuple(center)} "
              f"half_extent(padded)={tuple(half_ext)}")
    scale = Gf.Vec3d(half_ext[0]/half_extent_stage_units,
                     half_ext[1]/half_extent_stage_units,
                     half_ext[2]/half_extent_stage_units)
    mat = Gf.Matrix4d(1.0)
    mat.SetScale(scale)
    mat = mat * Gf.Matrix4d(1.0).SetTranslate(center)
    omni.kit.commands.execute("TransformPrim",
                              path=volume_path, new_transform_matrix=mat)
    print(f"[NavMesh] NavMeshVolume resized. scale={tuple(scale)}")


# ── exclude volumes from semantic map ──────────────────────────────────────

def add_exclude_volumes_from_semantic_map(app, stage, semantic_map_json: str,
                                          flip_x=True, flip_y=True,
                                          negate_xy=True) -> int:
    import json as _json
    with open(semantic_map_json) as f:
        items = _json.load(f)

    exclude_label_set = ["table", "chair", "sofa", "bed", "wardrobe",
                         "desk", "counter", "cabinet"]
    all_y, all_x = [], []
    for inst in items:
        for y, x in inst.get("mask_coords_m", []):
            try:
                all_y.append(float(y)); all_x.append(float(x))
            except (ValueError, TypeError):
                continue
    if not all_x:
        print("[Exclude] ERROR: no mask_coords_m found")
        return 0

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    print(f"[Exclude] map bounds: x∈[{min_x:.3f},{max_x:.3f}] y∈[{min_y:.3f},{max_y:.3f}]")

    def sm_to_isaac(sm_x, sm_y):
        px, py = sm_x, sm_y
        if flip_x:  px = (min_x + max_x) - px
        if flip_y:  py = (min_y + max_y) - py
        if negate_xy: px, py = -px, -py
        return px, py

    mpu  = UsdGeom.GetStageMetersPerUnit(stage)
    heu  = 0.5 / mpu
    bbox = compute_scene_bbox(stage)
    created = skipped = 0

    for item in items:
        label = str(item.get("category_label", "")).lower()
        if not any(k in label for k in exclude_label_set):
            continue
        x_l, y_b, x_r, y_t = [float(v) for v in item["bbox_m"]]
        z_min, z_max = float(item["min_z_m"]), float(item["max_z_m"])
        corners_isaac = [sm_to_isaac(cx, cy)
                         for cx, cy in [(x_l,y_b),(x_l,y_t),(x_r,y_b),(x_r,y_t)]]
        ix_vals = [c[0] for c in corners_isaac]
        iy_vals = [c[1] for c in corners_isaac]
        cx = 0.5*(min(ix_vals)+max(ix_vals))
        cy = 0.5*(min(iy_vals)+max(iy_vals))
        hx = 0.5*(max(ix_vals)-min(ix_vals)) + 0.10
        hy = 0.5*(max(iy_vals)-min(iy_vals)) + 0.10
        if bbox:
            bmin, bmax = bbox
            if not (bmin[0]-1 <= cx <= bmax[0]+1 and bmin[1]-1 <= cy <= bmax[1]+1):
                skipped += 1; continue
        hz  = 0.5*(z_max+0.05 - max(z_min-0.05,-0.01))
        cz  = 0.5*(z_max+0.05 + max(z_min-0.05,-0.01))

        existing = {p.GetPath() for p in stage.Traverse()
                    if p.GetName().startswith("NavMeshVolume")}
        omni.kit.commands.execute("CreateNavMeshVolumeCommand",
                                  parent_prim_path=Sdf.Path.emptyPath,
                                  volume_type=1, usd_context_name="", layer=None)
        _update(app, 1)
        new_prims = [p.GetPath() for p in stage.Traverse()
                     if p.GetName().startswith("NavMeshVolume")
                     and p.GetPath() not in existing]
        if not new_prims:
            continue
        scale = Gf.Vec3d(hx/heu, hy/heu, hz/heu)
        mat   = Gf.Matrix4d(1.0)
        mat.SetScale(scale)
        mat = mat * Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(cx, cy, cz))
        omni.kit.commands.execute("TransformPrim", path=new_prims[0],
                                  new_transform_matrix=mat)
        created += 1

    _update(app, 10)
    print(f"[Exclude] Created {created}, skipped {skipped}.")
    return created


# ── delta layer save / restore ─────────────────────────────────────────────

def save_navmesh_volumes_as_delta(stage, navvols_usda: str):
    """Export only NavMeshVolume prims into a minimal delta USDA."""
    vol_paths = [p.GetPath() for p in stage.Traverse()
                 if p.GetName().startswith("NavMeshVolume")]
    if not vol_paths:
        print("[Cache] No NavMeshVolume prims found, nothing to save.")
        return

    layer_stack = stage.GetLayerStack(includeSessionLayers=False)

    # Remove stale file so CreateNew doesn't complain
    if os.path.exists(navvols_usda):
        os.remove(navvols_usda)

    delta = Sdf.Layer.CreateNew(navvols_usda)
    delta.Clear()
    Sdf.PrimSpec(delta, "World", Sdf.SpecifierOver)   # parent container

    for vp in vol_paths:
        src_layer = next((l for l in layer_stack if l.GetPrimAtPath(vp)), None)
        if src_layer is None:
            print(f"[Cache] WARNING: no layer has spec for {vp}, skipping.")
            continue
        vol_name = vp.name
        Sdf.CopySpec(src_layer, vp, delta, Sdf.Path(f"/World/{vol_name}"))

    delta.Save()
    print(f"[Cache] Saved {len(vol_paths)} NavMeshVolume(s) → {navvols_usda}")


def restore_navmesh_volumes_as_sublayer(app, stage, navvols_usda: str) -> bool:
    """Insert the delta layer as the strongest sublayer of the root layer."""
    root_layer = stage.GetRootLayer()
    if navvols_usda in root_layer.subLayerPaths:
        print("[Cache] Delta layer already inserted, skipping.")
        return True
    root_layer.subLayerPaths.insert(0, navvols_usda)
    _update(app, 5)
    vol_count = sum(1 for p in stage.Traverse()
                    if p.GetName().startswith("NavMeshVolume"))
    print(f"[Cache] Delta sublayer inserted, {vol_count} NavMeshVolume(s) active.")
    return vol_count > 0


# ── settings ───────────────────────────────────────────────────────────────

def apply_navmesh_settings(app):
    from omni.anim.navigation.core import NavMeshSettings
    from omni.anim.people.settings import PeopleSettings

    s = carb.settings.get_settings()
    s.set("/persistent/exts/omni.anim.navigation.core/navMesh/config/agentMinRadius", 0.3)
    s.set("/persistent/exts/omni.anim.navigation.core/navMesh/config/cellSize", 0.15)
    s.set("/persistent/exts/omni.anim.navigation.core/navMesh/config/regionMergeSize", 20)
    s.set(PeopleSettings.CHARACTER_FINAL_TARGET_DISTANCE, 0.5)
    s.set("/persistent/exts/omni.anim.people/minDistanceToIntermediateTarget", 0.8)
    s.set("/persistent/exts/omni.anim.navigation.core/navMesh/config/agentMinHeight", 1.8)
    s.set("/persistent/exts/omni.anim.navigation.core/navMesh/config/agentMinIslandRadius", 0.5)
    s.set("/persistent/exts/omni.anim.navigation.core/navMesh/config/autoRebakeOnChanges", False)
    s.set(PeopleSettings.NAVMESH_ENABLED, True)
    s.set(PeopleSettings.DYNAMIC_AVOIDANCE_ENABLED, False)
    s.set(PeopleSettings.NUMBER_OF_LOOP, "0")
    s.set(NavMeshSettings.CACHE_ENABLED_SETTING_PATH, True)
    _update(app, 5)


# ── random point helpers ────────────────────────────────────────────────────

def safe_query_random_point(nm, max_z: float = 0.1, retries: int = 30):
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(500)
    try:
        best = None
        for _ in range(retries):
            try:
                p = nm.query_random_point()
                if float(p[2]) < max_z:
                    return p
                if best is None:
                    best = p
            except RecursionError:
                print("[WARN] query_random_point recursion")
                break
            except Exception as e:
                print(f"[WARN] query_random_point: {e}")
                break
        return best
    finally:
        sys.setrecursionlimit(old)


def _path_is_valid(path) -> bool:
    if path is None:
        return False
    if hasattr(path, 'get_points'):
        pts = path.get_points()
        return pts is not None and len(pts) > 0
    if hasattr(path, 'points'):
        return path.points is not None and len(path.points) > 0
    try:
        return len(path) > 0
    except TypeError:
        return True


def find_reachable_point(nm, from_xyz, max_attempts: int = 50, min_dist: float = 3.0):
    from_pt = carb.Float3(float(from_xyz[0]), float(from_xyz[1]), float(from_xyz[2]))
    for _ in range(max_attempts):
        c = safe_query_random_point(nm)
        if c is None:
            continue
        dx = float(c[0]) - float(from_xyz[0])
        dy = float(c[1]) - float(from_xyz[1])
        if (dx*dx + dy*dy)**0.5 < min_dist:
            continue
        try:
            if _path_is_valid(nm.query_shortest_path(
                    from_pt, carb.Float3(float(c[0]), float(c[1]), float(c[2])),
                    agent_radius=0.5)):
                return c
        except Exception:
            continue

    print("[NavMesh] WARNING: Relaxing min_dist to 1.0m")
    for _ in range(max_attempts):
        c = safe_query_random_point(nm)
        if c is None:
            continue
        dx = float(c[0]) - float(from_xyz[0])
        dy = float(c[1]) - float(from_xyz[1])
        if (dx*dx + dy*dy)**0.5 < 1.0:
            continue
        try:
            if _path_is_valid(nm.query_shortest_path(
                    from_pt, carb.Float3(float(c[0]), float(c[1]), float(c[2])),
                    agent_radius=0.5)):
                return c
        except Exception:
            continue
    return safe_query_random_point(nm)


def spawn_on_navmesh(nm) -> tuple:
    """Return a spawn point that has at least one reachable neighbour."""
    for _ in range(50):
        p = safe_query_random_point(nm)
        if p is None:
            continue
        try:
            result  = nm.query_closest_point(p, 0.5)
            snapped = result[0] if result else None
        except Exception:
            snapped = p
        if snapped is None:
            continue
        if find_reachable_point(nm, snapped, max_attempts=20, min_dist=1.0) is not None:
            return snapped
    return safe_query_random_point(nm)


# ── main bake entry point ───────────────────────────────────────────────────

def bake_navmesh(app, navvols_usda: str, force_rebake: bool,
                 collision_root: str,
                 volume_padding: float, fallback_size: float,
                 warmup_frames: int,
                 semantic_map_json: str | None = None):
    """Bake (or restore from cache) the NavMesh.

    Returns (success: bool, inav).
    """
    import omni.anim.navigation.core as nav

    apply_navmesh_settings(app)
    inav  = nav.acquire_interface()
    print(f"[NavMesh] Internal cache dir: {inav.get_cache_dir()}")

    has_cache = not force_rebake and os.path.exists(navvols_usda)

    if has_cache:
        print(f"[Cache] Restoring NavMeshVolumes from delta: {navvols_usda}")
        stage = omni.usd.get_context().get_stage()
        ok = restore_navmesh_volumes_as_sublayer(app, stage, navvols_usda)
        if ok:
            prev_vis = set_subtree_visibility(stage, collision_root, visible=True)
            _update(app, 5)
            print("[NavMesh] Baking (volumes restored, expect cache hit)...")
            inav.start_navmesh_baking_and_wait()
            nm = inav.get_navmesh()
            rp = safe_query_random_point(nm)
            if prev_vis == "invisible":
                set_subtree_visibility(stage, collision_root, visible=False)
                _update(app, 2)
            if rp is not None:
                print(f"[Cache] NavMesh OK. Sample point: {tuple(rp)}")
                return True, inav
        print("[Cache] Restore failed — falling back to full bake.")

    # ── full bake ──────────────────────────────────────────────────────────
    print("\n[NavMesh] Running full bake...")
    stage    = omni.usd.get_context().get_stage()
    prev_vis = set_subtree_visibility(stage, collision_root, visible=True)
    _update(app, 5)

    existing_vols = {p.GetPath() for p in stage.Traverse()
                     if p.GetName().startswith("NavMeshVolume")}
    try:
        omni.kit.commands.execute("CreateNavMeshVolumeCommand",
                                  parent_prim_path=Sdf.Path.emptyPath,
                                  volume_type=0, usd_context_name="", layer=None)
    except Exception as e:
        print(f"[NavMesh] CreateNavMeshVolumeCommand failed: {e}")
        return False, None
    _update(app, 5)

    new_vols = [p.GetPath() for p in stage.Traverse()
                if p.GetName().startswith("NavMeshVolume")
                and p.GetPath() not in existing_vols]
    if not new_vols:
        print("[NavMesh] Could not find new NavMeshVolume.")
        return False, None

    size_navmesh_volume(stage, new_vols[0], padding=volume_padding,
                        fallback_size=fallback_size)

    if semantic_map_json and os.path.exists(semantic_map_json):
        add_exclude_volumes_from_semantic_map(app, stage, semantic_map_json)
    elif semantic_map_json:
        print(f"[Exclude] semantic_map_json not found: {semantic_map_json}")

    print(f"[NavMesh] Warming up {warmup_frames} frames...")
    _update(app, warmup_frames)

    print("[NavMesh] Baking...")
    inav.start_navmesh_baking_and_wait()
    nm = inav.get_navmesh()
    rp = safe_query_random_point(nm)
    print(f"[NavMesh] OK. Sample point: {tuple(rp)}")

    stage = omni.usd.get_context().get_stage()
    save_navmesh_volumes_as_delta(stage, navvols_usda)

    if prev_vis == "invisible":
        set_subtree_visibility(stage, collision_root, visible=False)
        _update(app, 2)

    return True, inav


def find_safe_navmesh_point(nm, target, min_clearance: float = 0.6,
                             search_radii=None, num_angles: int = 8) -> object:
    """在 target 附近找离障碍物至少 min_clearance 的 NavMesh 点。

    用多方向射线 + 二分查找估算每个候选点的最小 clearance，
    返回 clearance 最大的点。若找不到则返回原始 target。

    Args:
        nm:             NavMesh 接口（inav.get_navmesh()）
        target:         原始目标点，carb.Float3 或 [x,y,z]
        min_clearance:  期望的最小离墙距离（米）
        search_radii:   候选点搜索半径列表，默认 [0.0, 0.3, 0.5, 0.7, 1.0]
        num_angles:     方向射线数量

    Returns:
        carb.Float3，离墙最远的安全点
    """
    import math

    if search_radii is None:
        search_radii = [0.0, 0.3, 0.5, 0.7, 1.0]

    angles_rad = [i * (2 * math.pi / num_angles) for i in range(num_angles)]
    ray_len = 1.5

    # 构建候选点列表（原始点 + 周围采样）
    candidates = [carb.Float3(float(target[0]), float(target[1]), float(target[2]))]
    for r in search_radii:
        if r == 0.0:
            continue
        for a in angles_rad:
            candidates.append(carb.Float3(
                float(target[0]) + r * math.cos(a),
                float(target[1]) + r * math.sin(a),
                float(target[2]),
            ))

    def _clearance(pt) -> float:
        """估算 pt 到最近障碍的距离（用射线二分）。"""
        min_d = ray_len  # 默认无障碍时返回 ray_len
        for a in angles_rad:
            ray_end = carb.Float3(
                float(pt[0]) + ray_len * math.cos(a),
                float(pt[1]) + ray_len * math.sin(a),
                float(pt[2]),
            )
            try:
                hit = nm.query_closest_point(ray_end, 0.1)
                on_mesh = hit and hit[0] is not None
            except Exception:
                on_mesh = False

            if not on_mesh:
                # 二分找边界
                lo, hi = 0.0, ray_len
                for _ in range(6):
                    mid = (lo + hi) / 2
                    mid_pt = carb.Float3(
                        float(pt[0]) + mid * math.cos(a),
                        float(pt[1]) + mid * math.sin(a),
                        float(pt[2]),
                    )
                    try:
                        r2 = nm.query_closest_point(mid_pt, 0.1)
                        on_mesh2 = r2 and r2[0] is not None
                    except Exception:
                        on_mesh2 = False
                    if on_mesh2:
                        lo = mid
                    else:
                        hi = mid
                min_d = min(min_d, lo)

        return min_d

    best_point = None
    best_clearance = -1.0

    for pt in candidates:
        # snap 到 NavMesh
        try:
            result = nm.query_closest_point(pt, 0.8)
            snapped = result[0] if result and result[0] is not None else None
        except Exception:
            snapped = None
        if snapped is None:
            continue

        c = _clearance(snapped)
        if c > best_clearance:
            best_clearance = c
            best_point = snapped
            if c >= min_clearance:
                break  # 够好了，不用继续找

    if best_point is not None:
        print(f"[SafePoint] clearance={best_clearance:.2f}m "
              f"at ({float(best_point[0]):.2f},{float(best_point[1]):.2f}) "
              f"(target was ({float(target[0]):.2f},{float(target[1]):.2f}))")
        return best_point

    print(f"[SafePoint] WARNING: no snappable point near target, returning original")
    return carb.Float3(float(target[0]), float(target[1]), float(target[2]))