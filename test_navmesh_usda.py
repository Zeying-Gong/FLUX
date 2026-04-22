"""Open a USDA scene, bake NavMesh, bring up physics, spawn N pedestrians
on the NavMesh, and have them randomly walk forever until Ctrl+C.

Pipeline (ORDER matters):
    1. open_stage(usda)
    2. temporarily make collision geometry visible to the navmesh baker
    3. create + size NavMeshVolume to cover the scene
    4. BAKE NAVMESH   ← before World(), or PhysX hides meshes from baker
    5. restore collision visibility
    6. World(...) + initialize_physics + reset
    7. spawn N characters at random NavMesh points
    8. timeline.play()
    9. send initial GoTo commands
   10. hold loop: world.step(render=True) + reissue waypoints on timeout

Usage:
    python check_scene_navmesh.py --usda /path/to/scene.usda
    python check_scene_navmesh.py --usda /path/to/scene.usda --num_people 3
    python check_scene_navmesh.py --usda /path/to/scene.usda --num_people 0
"""
from __future__ import annotations
import argparse
import os
import random
import sys
import time


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Bake NavMesh, start physics, spawn random-walking pedestrians."
    )
    p.add_argument("--usda", type=str, required=True)

    # NavMesh / bake
    p.add_argument("--collision_root", type=str, default="/World/scene_collision")
    p.add_argument("--keep_collision_visible", action="store_true")
    p.add_argument("--volume_padding", type=float, default=1.2)
    p.add_argument("--fallback_size",  type=float, default=100.0)
    p.add_argument("--warmup_frames",  type=int,   default=60)

    # People
    p.add_argument("--num_people", type=int, default=1)
    p.add_argument("--waypoint_timeout", type=float, default=15.0)

    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--navmesh_probe_interval", type=float, default=10.0)

    # CLI: 加两行
    p.add_argument("--semantic_map_json", type=str, default=None,
                help="Optional 2D semantic map JSON — used to carve out "
                        "Exclude volumes above furniture so characters don't "
                        "walk onto tables/sofas/beds.")
    p.add_argument("--exclude_labels", type=str, nargs="*",
                default=["table", "chair", "sofa", "bed", "wardrobe",
                            "desk", "counter", "cabinet"],
                help="Category labels from semantic map JSON that should "
                        "become NavMesh Exclude volumes.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Launch SimulationApp FIRST
# ═══════════════════════════════════════════════════════════════════════════
ARGS = parse_args()

from isaacsim import SimulationApp

_exp = os.environ.get("EXP_PATH")
if _exp is None:
    print("[FATAL] EXP_PATH env var not set.")
    sys.exit(1)

CUSTOM_APP_PATH = os.path.join(_exp, "isaacsim.exp.action_and_event_data_generation.base.kit")

simulation_app = SimulationApp(
    launch_config={
        "renderer": "RayTracedLighting",
        "headless": False,
        "enable_cameras": True,
        "crash_reporter/enabled": False,
        "crash_reporter/skip_old_dump_upload": True,
    },
    experience=CUSTOM_APP_PATH,
)

# ═══════════════════════════════════════════════════════════════════════════
# Post-launch imports  (NOT World yet — that's deferred past bake)
# ═══════════════════════════════════════════════════════════════════════════
import numpy as np
import carb
import omni
import omni.client
import omni.usd
import omni.kit.app
import omni.kit.commands
import omni.timeline
from pxr import Sdf, Usd, UsdGeom, Gf

from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.replicator.agent.core.settings import AssetPaths, PrimPaths, BehaviorScriptPaths, Settings
from isaacsim.replicator.agent.core.stage_util import CharacterUtil
from isaacsim.core.utils import prims
from omni.anim.people.scripts.custom_command.populate_anim_graph import populate_anim_graph

# ═══════════════════════════════════════════════════════════════════════════
# Stage / bbox / visibility helpers
# ═══════════════════════════════════════════════════════════════════════════
def _update(n: int = 1):
    for _ in range(n):
        simulation_app.update()

import sys

def safe_query_random_point(nm, max_z=0.3, retries=30):
    """安全的 NavMesh 随机点采样，过滤家具面高度点，防递归崩溃"""
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(500)  # 提前截断，避免真正崩溃
    try:
        best = None
        for _ in range(retries):
            try:
                p = nm.query_random_point()
                z = float(p[2])
                if z < max_z:
                    return p
                if best is None:
                    best = p  # 备用：至少返回个点
            except RecursionError:
                print("[WARN] query_random_point recursion, NavMesh may be damaged")
                break
            except Exception as e:
                print(f"[WARN] query_random_point failed: {e}")
                break
        return best
    finally:
        sys.setrecursionlimit(old_limit)

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
            print("[WARN] Stage still loading after 30s, continuing anyway.")
            break
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[FATAL] Stage failed to open.")
        return False
    print(f"[STAGE] Loaded. Default prim: {stage.GetDefaultPrim().GetPath()}")
    return True


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
    print(f"[Vis] {root_path}: {prev} → "
          f"{'inherited' if visible else 'invisible'}")
    return prev


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

def verify_one_furniture_alignment(stage, semantic_map_json: str,
                                   flip_x=True, flip_y=True):
    """找 semantic_map 里第一个家具，在 stage 里搜索对应 prim，对比坐标"""
    import json as _json
    with open(semantic_map_json) as f:
        items = _json.load(f)
    
    furn_labels = {"table","chair","sofa","bed","wardrobe","desk","counter","cabinet"}
    
    # 取第一个家具
    target = None
    for it in items:
        if it.get("category_label","").lower() in furn_labels:
            target = it
            break
    if target is None:
        print("[AlignCheck] No furniture found in semantic_map")
        return
    
    x_l, y_b, x_r, y_t = [float(v) for v in target["bbox_m"]]
    cx_raw = 0.5*(x_l+x_r)
    cy_raw = 0.5*(y_b+y_t)
    cx_isaac = -cx_raw if flip_x else cx_raw
    cy_isaac = -cy_raw if flip_y else cy_raw
    
    print(f"\n[AlignCheck] Target: {target['item_id']}")
    print(f"[AlignCheck]   semantic_map bbox (raw): x∈[{x_l},{x_r}] y∈[{y_b},{y_t}]")
    print(f"[AlignCheck]   computed Isaac center: ({cx_isaac:.3f}, {cy_isaac:.3f})")
    print(f"[AlignCheck]   z_max={target['max_z_m']}")
    
    # 在 stage 里搜索所有 prim，找 bbox 中心最接近的
    label = target.get("category_label","").lower()
    print(f"[AlignCheck] Searching stage for prims matching label '{label}'...")
    
    best_prim = None
    best_dist = float('inf')
    
    for prim in stage.Traverse():
        prim_name = prim.GetName().lower()
        if label not in prim_name:
            continue
        if not prim.IsA(UsdGeom.Xformable):
            continue
        try:
            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                useExtentsHint=True
            )
            bbox = cache.ComputeWorldBound(prim)
            rng = bbox.ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            mn, mx = rng.GetMin(), rng.GetMax()
            pcx = (float(mn[0])+float(mx[0]))*0.5
            pcy = (float(mn[1])+float(mx[1]))*0.5
            pcz_max = float(mx[2])
            dist = ((pcx-cx_isaac)**2 + (pcy-cy_isaac)**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_prim = (str(prim.GetPath()), pcx, pcy, pcz_max)
        except Exception:
            continue
    
    if best_prim:
        path, pcx, pcy, pcz = best_prim
        print(f"[AlignCheck] Best match prim: {path}")
        print(f"[AlignCheck]   Isaac prim center: ({pcx:.3f}, {pcy:.3f}), z_max={pcz:.3f}")
        print(f"[AlignCheck]   Distance from computed: {best_dist:.3f}m")
        print(f"[AlignCheck]   ★ Offset needed: dx={pcx-cx_isaac:.3f}, dy={pcy-cy_isaac:.3f}")
        return pcx-cx_isaac, pcy-cy_isaac
    else:
        print(f"[AlignCheck] No prim with '{label}' in name found in stage")
        # fallback: 列出所有可能相关的prim名
        names = set()
        for prim in stage.Traverse():
            n = prim.GetName().lower()
            if any(l in n for l in furn_labels):
                names.add(prim.GetName())
        print(f"[AlignCheck] Stage prims with furniture keywords: {sorted(names)[:20]}")
        return 0.0, 0.0

# ═══════════════════════════════════════════════════════════════════════════
# NavMesh bake
# ═══════════════════════════════════════════════════════════════════════════
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

def add_exclude_volumes_from_semantic_map(stage,
                                          semantic_map_json: str,
                                          exclude_labels: list[str],
                                          flip_x: bool = True,
                                          flip_y: bool = True,
                                          offset_x = 0.0,
                                          offset_y = 0.0
                                          ) -> int:   # ← 新增
    import json as _json
    with open(semantic_map_json, "r", encoding="utf-8") as f:
        items = _json.load(f)

    label_set = set(l.lower() for l in exclude_labels)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    half_extent_stage_units = 0.5 / mpu

    bbox = compute_scene_bbox(stage)
    if bbox:
        bmin, bmax = bbox
        print(f"[Exclude] Live scene bbox: x∈[{bmin[0]:.2f},{bmax[0]:.2f}] "
              f"y∈[{bmin[1]:.2f},{bmax[1]:.2f}]")

    # 诊断变换前后坐标
    sample_items = [it for it in items
                    if it.get("category_label","").lower() in label_set][:3]
    for it in sample_items:
        x_l, y_b, x_r, y_t = [float(v) for v in it["bbox_m"]]
        cx_raw, cy_raw = 0.5*(x_l+x_r), 0.5*(y_b+y_t)
        cy_xfm = -cy_raw if flip_y else cy_raw
        print(f"[Exclude] {it['item_id']}: raw_cx={cx_raw:.2f} raw_cy={cy_raw:.2f} "
              f"→ isaac_cy={cy_xfm:.2f}")

    created = 0
    skipped = 0

    for item in items:
        label = str(item.get("category_label", "")).lower()
        if label not in label_set:
            continue
        try:
            x_l, y_b, x_r, y_t = [float(v) for v in item["bbox_m"]]
            z_min = float(item["min_z_m"])
            z_max = float(item["max_z_m"])
        except Exception as e:
            print(f"[Exclude] skip {item.get('item_id')}: parse failed ({e})")
            continue

        cx_raw = 0.5 * (x_l + x_r)
        cy_raw = 0.5 * (y_b + y_t)

        # ★ 两轴都取反
        cx = (-cx_raw if flip_x else cx_raw) + offset_x
        cy = (-cy_raw if flip_y else cy_raw) + offset_y

        # bbox 边界也跟着翻
        if flip_x:
            new_x_l = -x_r
            new_x_r = -x_l
        else:
            new_x_l, new_x_r = x_l, x_r

        if flip_y:
            new_y_b = -y_t
            new_y_t = -y_b
        else:
            new_y_b, new_y_t = y_b, y_t

        PAD = 0.10
        hx = 0.5 * (new_x_r - new_x_l) + PAD
        hy = 0.5 * (new_y_t - new_y_b) + PAD

        # in-bounds 检查（变换后）
        if bbox:
            bmin, bmax = bbox
            margin = 1.0
            if not (bmin[0]-margin <= cx    <= bmax[0]+margin and
                    bmin[1]-margin <= cy    <= bmax[1]+margin):
                skipped += 1
                print(f"[Exclude] SKIP {item.get('item_id')} "
                      f"cx={cx:.2f} cy={cy:.2f} still out of bounds after flip")
                continue
        FLOOR_CLEARANCE = 0.30
        bottom = max(z_max - 0.15, FLOOR_CLEARANCE)
        top    = z_max + 1.80   # 顶面上方1.8m（人体高度）
        hz     = 0.5 * (top - bottom)
        cz     = 0.5 * (top + bottom)

        existing = {p.GetPath() for p in stage.Traverse()
                    if p.GetName().startswith("NavMeshVolume")}
        omni.kit.commands.execute(
            "CreateNavMeshVolumeCommand",
            parent_prim_path=Sdf.Path.emptyPath,
            volume_type=1,
            usd_context_name="",
            layer=None,
        )
        _update(1)
        new_prims = [p.GetPath() for p in stage.Traverse()
                     if p.GetName().startswith("NavMeshVolume")
                     and p.GetPath() not in existing]
        if not new_prims:
            print(f"[Exclude] failed to locate new volume for {item.get('item_id')}")
            continue
        vol_path = new_prims[0]

        scale  = Gf.Vec3d(hx / half_extent_stage_units,
                          hy / half_extent_stage_units,
                          hz / half_extent_stage_units)
        center = Gf.Vec3d(cx, cy, cz)
        mat = Gf.Matrix4d(1.0)
        mat.SetScale(scale)
        mat = mat * Gf.Matrix4d(1.0).SetTranslate(center)
        omni.kit.commands.execute("TransformPrim",
                                  path=vol_path, new_transform_matrix=mat)
        created += 1

    _update(10)

    # 验证最终位置
    print(f"[Exclude] Created {created}, skipped {skipped}. Final positions:")
    for p in stage.Traverse():
        if "NavMeshVolume" in p.GetName() and p.GetPath() != Sdf.Path("/World/NavMeshVolume"):
            xf = UsdGeom.Xformable(p)
            if xf:
                t = xf.ComputeLocalToWorldTransform(0).ExtractTranslation()
                print(f"  {p.GetPath()}  center=({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})")

    verify_exclude_volume_alignment(stage, semantic_map_json, flip_x=flip_x, flip_y=flip_y)

    return created

def verify_exclude_volume_alignment(stage, semantic_map_json: str,
                                    flip_x=True, flip_y=True, n_check=5):
    """对比已创建的 NavMeshVolume 和 stage 里实际家具 prim 的中心，量化偏差"""
    import json as _json
    with open(semantic_map_json) as f:
        items = _json.load(f)

    furn_labels = {"table","chair","sofa","bed","wardrobe","desk","counter","cabinet"}
    
    # 预先建立 stage 里所有家具相关 prim 的 bbox 中心表
    print("\n[AlignCheck] Building stage furniture prim index...")
    stage_furns = []   # list of (prim_path, cx, cy, cz_max, label_guess)
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True
    )
    for prim in stage.Traverse():
        prim_name = prim.GetName().lower()
        matched_label = next((l for l in furn_labels if l in prim_name), None)
        if matched_label is None:
            continue
        if not prim.IsA(UsdGeom.Xformable):
            continue
        try:
            bbox = bbox_cache.ComputeWorldBound(prim)
            rng = bbox.ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            mn, mx = rng.GetMin(), rng.GetMax()
            pcx = (float(mn[0])+float(mx[0]))*0.5
            pcy = (float(mn[1])+float(mx[1]))*0.5
            pcz_max = float(mx[2])
            stage_furns.append((str(prim.GetPath()), pcx, pcy, pcz_max, matched_label))
        except Exception:
            continue
    print(f"[AlignCheck] Found {len(stage_furns)} furniture prims in stage")

    # 取 semantic_map 里前 n_check 个家具，找对应的 NavMeshVolume，再找最近的 stage prim
    checked = 0
    vol_index = 1  # NavMeshVolume_01, _02, ...
    
    for item in items:
        if checked >= n_check:
            break
        label = item.get("category_label","").lower()
        if label not in furn_labels:
            continue
        if "bbox_m" not in item:
            continue

        x_l, y_b, x_r, y_t = [float(v) for v in item["bbox_m"]]
        cx_raw = 0.5*(x_l+x_r)
        cy_raw = 0.5*(y_b+y_t)
        cx_computed = -cx_raw if flip_x else cx_raw
        cy_computed = -cy_raw if flip_y else cy_raw

        # 找对应的 NavMeshVolume（按创建顺序，_01对应第一个家具）
        vol_name = f"/World/NavMeshVolume_{vol_index:02d}"
        vol_prim = stage.GetPrimAtPath(vol_name)
        vol_index += 1

        if not vol_prim or not vol_prim.IsValid():
            print(f"[AlignCheck] {vol_name} not found, skipping")
            continue

        # 读 NavMeshVolume 的实际世界坐标
        xf = UsdGeom.Xformable(vol_prim)
        t = xf.ComputeLocalToWorldTransform(0).ExtractTranslation()
        vol_cx, vol_cy = float(t[0]), float(t[1])

        # 找 stage 里最近的同类家具 prim
        candidates = [(p, px, py, pz) for p, px, py, pz, pl in stage_furns if pl == label]
        if not candidates:
            candidates = [(p, px, py, pz) for p, px, py, pz, pl in stage_furns]
        
        best = min(candidates, key=lambda c: (c[1]-vol_cx)**2 + (c[2]-vol_cy)**2)
        best_path, best_px, best_py, best_pz = best
        
        print(f"\n[AlignCheck] {item['item_id']} ({label})")
        print(f"  semantic_map raw:      cx={cx_raw:+.3f}  cy={cy_raw:+.3f}")
        print(f"  computed (after flip): cx={cx_computed:+.3f}  cy={cy_computed:+.3f}")
        print(f"  NavMeshVolume actual:  cx={vol_cx:+.3f}  cy={vol_cy:+.3f}  ← should match computed")
        print(f"  Stage prim ({best_path}):")
        print(f"    prim center:         cx={best_px:+.3f}  cy={best_py:+.3f}")
        print(f"  ★ Volume→Prim offset:  dx={best_px-vol_cx:+.3f}  dy={best_py-vol_cy:+.3f}")
        
        checked += 1

    print(f"\n[AlignCheck] Done. If dx/dy are consistent across items, "
          f"add that as a fixed offset in add_exclude_volumes_from_semantic_map.\n")

def try_bake_navmesh(offset_x=0.0, offset_y=0.0):
    """Returns (success, nav_interface). Toggles collision_root visibility."""
    try:
        import omni.anim.navigation.core as nav
    except Exception as e:
        print(f"[NavMesh] omni.anim.navigation.core not available: {e}")
        return False, None

    stage = omni.usd.get_context().get_stage()

    prev_vis = set_subtree_visibility(stage, ARGS.collision_root, visible=True)
    _update(5)

    existing = {p.GetPath() for p in stage.Traverse()
                if p.GetName().startswith("NavMeshVolume")}
    try:
        omni.kit.commands.execute(
            "CreateNavMeshVolumeCommand",
            parent_prim_path=Sdf.Path.emptyPath,
            volume_type=0,
            usd_context_name="",
            layer=None,
        )
    except Exception as e:
        print(f"[NavMesh] CreateNavMeshVolumeCommand failed: {e}")
        return False, None
    _update(5)

    new_volumes = [p.GetPath() for p in stage.Traverse()
                   if p.GetName().startswith("NavMeshVolume")
                   and p.GetPath() not in existing]
    if not new_volumes:
        print("[NavMesh] Could not find new NavMeshVolume.")
        return False, None
    volume_path = new_volumes[0]
    print(f"[NavMesh] Created {volume_path}")

    size_navmesh_volume(stage, volume_path,
                        padding=ARGS.volume_padding,
                        fallback_size=ARGS.fallback_size)

    # Carve out furniture tops so characters don't climb onto tables/sofas
    if ARGS.semantic_map_json and os.path.exists(ARGS.semantic_map_json):
        add_exclude_volumes_from_semantic_map(
            stage, ARGS.semantic_map_json, ARGS.exclude_labels, offset_x=offset_x, offset_y=offset_y,
        )
    elif ARGS.semantic_map_json:
        print(f"[Exclude] semantic_map_json not found: {ARGS.semantic_map_json}")

    print(f"[NavMesh] Warming up for {ARGS.warmup_frames} frames before bake...")
    _update(ARGS.warmup_frames)

    inav = nav.acquire_interface()
    print("[NavMesh] Baking...")
    try:
        inav.start_navmesh_baking_and_wait()
        nm = inav.get_navmesh()
        diagnose_navmesh_connectivity(nm)

        if ARGS.semantic_map_json and os.path.exists(ARGS.semantic_map_json):
            stage = omni.usd.get_context().get_stage()
            diagnose_coordinate_offset(ARGS.semantic_map_json, stage, flip_y=True)
            
    except Exception as e:
        print(f"[NavMesh] Baking raised: {e}")
        nm = None

    if nm is None:
        print("[NavMesh] FAILED.")
        success = False
    else:
        try:
            rp = safe_query_random_point(nm)
            print(f"[NavMesh] OK. Sample random point: {tuple(rp)}")
        except Exception as e:
            print(f"[NavMesh] OK but query_random_point raised: {e}")
        success = True

    if prev_vis == "invisible" and not ARGS.keep_collision_visible:
        set_subtree_visibility(stage, ARGS.collision_root, visible=False)
        _update(2)

    return success, inav

def diagnose_navmesh_connectivity(nm, n_samples=20):
    """采样 NavMesh 点，测试连通性，打印孤岛分布"""
    print("\n[ConnDiag] ── NavMesh Connectivity Diagnosis ──")
    
    # 1. 采样点
    samples = []
    for i in range(n_samples):
        p = safe_query_random_point(nm, max_z=0.5)
        if p is not None:
            samples.append(p)
            print(f"  sample_{i:02d}: ({float(p[0]):+.3f}, {float(p[1]):+.3f}, {float(p[2]):+.5f})")
    
    if len(samples) < 2:
        print("[ConnDiag] Not enough samples.")
        return
    
    # x/y 范围
    xs = [float(p[0]) for p in samples]
    ys = [float(p[1]) for p in samples]
    print(f"[ConnDiag] Sample x range: [{min(xs):.2f}, {max(xs):.2f}]")
    print(f"[ConnDiag] Sample y range: [{min(ys):.2f}, {max(ys):.2f}]")
    print(f"[ConnDiag] Sample centroid: ({sum(xs)/len(xs):.2f}, {sum(ys)/len(ys):.2f})")
    
    # 2. 连通性矩阵：前5个点两两测试
    print("[ConnDiag] Path reachability matrix (first 5 samples):")
    test = samples[:5]
    for i, a in enumerate(test):
        row = []
        for j, b in enumerate(test):
            if i == j:
                row.append(" ·")
                continue
            try:
                fa = carb.Float3(float(a[0]), float(a[1]), float(a[2]))
                fb = carb.Float3(float(b[0]), float(b[1]), float(b[2]))
                path = nm.query_shortest_path(fa, fb)
                row.append(" ✓" if _path_is_valid(path) else " ✗")
            except Exception as e:
                row.append(" E")
                if j == 1 and i == 0:  # 只打印一次
                    print(f"[ConnDiag] query_shortest_path exception: {type(e).__name__}: {e}")
        print(f"  [{i}] ({float(test[i][0]):+.2f},{float(test[i][1]):+.2f}) → {''.join(row)}")
    
    print("[ConnDiag] ── End ──\n")


def diagnose_coordinate_offset(semantic_map_json: str, stage, flip_y: bool = True):
    """对比 semantic_map 坐标原点和 Isaac 场景 bbox，计算偏移"""
    import json as _json
    
    print("\n[OffsetDiag] ── Coordinate Offset Diagnosis ──")
    
    # 读 semantic_map meta（需要 occupancy.json，从 semantic_map_json 路径推断）
    sm_path = os.path.dirname(semantic_map_json)
    # semantic_map_json 通常形如 .../2D_Semantic_Map_839873_Complete.json
    # 对应场景 .../839873/occupancy.json
    scene_id = os.path.basename(semantic_map_json).replace(
        "2D_Semantic_Map_", "").replace("_Complete.json", "")
    
    # 尝试几个常见路径
    occ_candidates = [
        os.path.join(sm_path, scene_id, "occupancy.json"),
        os.path.join(sm_path, "..", scene_id, "occupancy.json"),
        os.path.join(sm_path, "..", "..", scene_id, "occupancy.json"),
    ]
    meta = None
    for cand in occ_candidates:
        cand = os.path.normpath(cand)
        if os.path.exists(cand):
            with open(cand) as f:
                meta = _json.load(f)
            print(f"[OffsetDiag] Loaded occupancy meta from: {cand}")
            break
    
    if meta:
        sm_origin_x = float(meta["min"][0])
        sm_origin_y = float(meta["min"][1])
        scale       = float(meta["scale"])
        print(f"[OffsetDiag] semantic_map origin (min): x={sm_origin_x:.3f}, y={sm_origin_y:.3f}")
        print(f"[OffsetDiag] semantic_map scale: {scale:.4f} m/pixel")
        
        if flip_y:
            # flip_y: isaac_y = -sm_y，所以原点在 isaac_y = -sm_origin_y
            print(f"[OffsetDiag] After flip_y: isaac_y_origin = {-sm_origin_y:.3f}")
    else:
        print("[OffsetDiag] Could not find occupancy.json, reading from semantic_map items directly")
        with open(semantic_map_json) as f:
            items = _json.load(f)
        all_x = []
        all_y = []
        for it in items:
            if "bbox_m" in it:
                x_l, y_b, x_r, y_t = [float(v) for v in it["bbox_m"]]
                all_x += [x_l, x_r]
                all_y += [y_b, y_t]
        if all_x:
            sm_xmin, sm_xmax = min(all_x), max(all_x)
            sm_ymin, sm_ymax = min(all_y), max(all_y)
            print(f"[OffsetDiag] semantic_map all-items x range: [{sm_xmin:.2f}, {sm_xmax:.2f}]")
            print(f"[OffsetDiag] semantic_map all-items y range (raw): [{sm_ymin:.2f}, {sm_ymax:.2f}]")
            if flip_y:
                print(f"[OffsetDiag] semantic_map all-items y range (flipped): [{-sm_ymax:.2f}, {-sm_ymin:.2f}]")
    
    # Isaac scene bbox
    bbox = compute_scene_bbox(stage)
    if bbox:
        bmin, bmax = bbox
        isaac_cx = (float(bmin[0]) + float(bmax[0])) * 0.5
        isaac_cy = (float(bmin[1]) + float(bmax[1])) * 0.5
        print(f"[OffsetDiag] Isaac scene bbox: x∈[{bmin[0]:.2f},{bmax[0]:.2f}] y∈[{bmin[1]:.2f},{bmax[1]:.2f}]")
        print(f"[OffsetDiag] Isaac scene center: ({isaac_cx:.2f}, {isaac_cy:.2f})")
        
        if meta:
            sm_cx = (sm_origin_x + float(meta.get("max", [0,0])[0])) * 0.5 if "max" in meta else None
            # 直接算偏移
            # 假设 semantic_map x 对应 isaac x（方向一致），y 翻转
            # 则理论上：isaac_x = sm_x + offset_x
            #           isaac_y = -sm_y + offset_y
            # 如果家具中心在 semantic_map 里是 (sm_cx, sm_cy)
            # 在 isaac 里对应 (sm_cx + offset_x, -sm_cy + offset_y)
            # 我们知道 isaac scene center ≈ 家具分布中心
            # 从 semantic_map items 估算家具中心
            with open(semantic_map_json) as f:
                items = _json.load(f)
            furn_labels = {"table","chair","sofa","bed","wardrobe","desk","counter","cabinet"}
            fxs, fys = [], []
            for it in items:
                if it.get("category_label","").lower() in furn_labels and "bbox_m" in it:
                    x_l, y_b, x_r, y_t = [float(v) for v in it["bbox_m"]]
                    fxs.append(0.5*(x_l+x_r))
                    fys.append(0.5*(y_b+y_t))
            if fxs:
                sm_furn_cx = sum(fxs)/len(fxs)
                sm_furn_cy = sum(fys)/len(fys)
                print(f"[OffsetDiag] semantic_map furniture centroid (raw): ({sm_furn_cx:.2f}, {sm_furn_cy:.2f})")
                print(f"[OffsetDiag] semantic_map furniture centroid (flip_y): ({sm_furn_cx:.2f}, {-sm_furn_cy:.2f})")
                
                # 估算需要的额外平移offset
                est_offset_x = isaac_cx - sm_furn_cx
                est_offset_y = isaac_cy - (-sm_furn_cy)
                print(f"[OffsetDiag] ★ Estimated missing offset: dx={est_offset_x:.3f}, dy={est_offset_y:.3f}")
                print(f"[OffsetDiag]   (if non-zero, add to cx/cy in add_exclude_volumes_from_semantic_map)")
    
    print("[OffsetDiag] ── End ──\n")
# ═══════════════════════════════════════════════════════════════════════════
# Character discovery (uses AssetPaths.default_character_path())
# ═══════════════════════════════════════════════════════════════════════════
EXCLUDED_CHARACTER_FOLDERS = {"biped_demo"}


def load_character_assets() -> list[str]:
    """List every character .usd file under AssetPaths.default_character_path().

    AssetPaths resolves to a local path on Isaac Sim 5.x (bundled assets),
    so this does NOT hit Nucleus and won't hang.
    """
    assets_root_path = AssetPaths.default_character_path()
    print(f"[PEOPLE] Listing characters under {assets_root_path}")

    result, folder_list = omni.client.list(f"{assets_root_path}/")
    if result != omni.client.Result.OK:
        print(f"[PEOPLE] omni.client.list failed: {result}")
        return []

    character_assets: list[str] = []
    for folder in folder_list:
        if not (folder.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN):
            continue
        if folder.relative_path.startswith("."):
            continue
        if folder.relative_path in EXCLUDED_CHARACTER_FOLDERS:
            continue

        folder_path = f"{assets_root_path}/{folder.relative_path}"
        res2, file_list = omni.client.list(f"{folder_path}/")
        if res2 != omni.client.Result.OK:
            continue
        for f in file_list:
            if f.relative_path.endswith((".usd", ".usda")):
                character_assets.append(f"{folder_path}/{f.relative_path}")
                break
    return character_assets


def spawn_character(idx: int, usd_path: str, spawn_xyz) -> str:
    """Use CharacterUtil so the character gets loaded with the structure
    omni.anim.people expects (SkelRoot at the right place, default xform ops,
    correct name under PrimPaths.characters_parent_path())."""
    spawn_location = carb.Float3(float(spawn_xyz[0]),
                                 float(spawn_xyz[1]),
                                 float(spawn_xyz[2]))
    char_name = CharacterUtil.get_character_name_by_index(idx)
    prim = CharacterUtil.load_character_usd_to_stage(
        usd_path, spawn_location, 0.0, char_name
    )
    _update(5)
    parent_path = PrimPaths.characters_parent_path()
    prim_path   = f"{parent_path}/{char_name}"
    print(f"[PEOPLE] Spawned {prim_path}  usd={usd_path}  at={tuple(spawn_xyz)}")
    return prim_path

def load_default_skeleton_and_animations():
    """Create /World/Characters parent + biped template + populate_anim_graph.
    Without this, SkelRoot can't find an Animation Graph to bind to."""
    # Make sure CharacterBehavior can derive agent name from prim path
    from omni.anim.people.settings import PeopleSettings
    carb.settings.get_settings().set(
        PeopleSettings.CHARACTER_PRIM_PATH,
        str(PrimPaths.characters_parent_path()),
    )

    stage = omni.usd.get_context().get_stage()
    parent_path = PrimPaths.characters_parent_path()
    if not stage.GetPrimAtPath(parent_path):
        prims.create_prim(parent_path, "Xform")

    if Settings.skip_biped_setup():
        return

    biped_path = PrimPaths.biped_prim_path()
    biped_prim = stage.GetPrimAtPath(biped_path)
    if not biped_prim or not biped_prim.IsValid():
        print(f"[PEOPLE] Creating biped template at {biped_path}")
        p = prims.create_prim(
            biped_path, "Xform",
            usd_path=AssetPaths.default_biped_asset_path(),
        )
        p.GetAttribute("visibility").Set("invisible")
    populate_anim_graph()


def bind_animation_graph_to_characters():
    stage = omni.usd.get_context().get_stage()
    biped_prim = stage.GetPrimAtPath(PrimPaths.biped_prim_path())
    anim_graph = CharacterUtil.get_anim_graph_from_character(biped_prim)
    if anim_graph is None:
        print("[PEOPLE] Could not find animation graph on biped template.")
        return

    ag_path = anim_graph.GetPrimPath()
    skelroots = CharacterUtil.get_characters_in_stage()
    for prim in skelroots:
        skel_path = prim.GetPrimPath()
        try:
            omni.kit.commands.execute(
                "RemoveAnimationGraphAPICommand",
                paths=[Sdf.Path(skel_path)])
        except Exception:
            pass

        omni.kit.commands.execute(
            "ApplyAnimationGraphAPICommand",
            paths=[Sdf.Path(skel_path)],
            animation_graph_path=Sdf.Path(ag_path),
        )

        # ★ Explicitly point skel:animationGraph rel at the AG. Some
        #   kit versions don't set the target through ApplyAnimationGraphAPICommand.
        rel = prim.GetRelationship("skel:animationGraph")
        if not rel:
            rel = prim.CreateRelationship("skel:animationGraph", custom=False)
        rel.SetTargets([ag_path])

        # Verify
        targets = rel.GetTargets()
        print(f"[PEOPLE]   {skel_path}  skel:animationGraph → {list(targets)}")

    print(f"[PEOPLE] Animation graph bound to {len(skelroots)} SkelRoot(s).")
    _update(30)

def attach_behavior_scripts_to_characters():
    """Attach the omni.anim.people CharacterBehavior script so that GoTo
    commands we push via CommandTextAPI are actually consumed.
    Without this, every character just stands in its T-Pose."""
    stage = omni.usd.get_context().get_stage()
    skelroots = CharacterUtil.get_characters_in_stage()
    script_path = BehaviorScriptPaths.behavior_script_path()
    print(f"[PEOPLE] CharacterBehavior script path: {script_path}")   # ← 加这行
    for prim in skelroots:
        skel_path = str(prim.GetPrimPath())
        if not prim.HasAttribute("omni:scripting:scripts"):
            omni.kit.commands.execute(
                "ApplyScriptingAPICommand",
                paths=[Sdf.Path(skel_path)],
            )
        attr = prim.GetAttribute("omni:scripting:scripts")
        attr.Set([script_path])
    print(f"[PEOPLE] CharacterBehavior attached to {len(skelroots)} SkelRoot(s).")
    _update(30)
    stage = omni.usd.get_context().get_stage()
    for prim in CharacterUtil.get_characters_in_stage():
        rel = prim.GetRelationship("skel:animationGraph")
        ag_targets = list(rel.GetTargets()) if rel else []
        has_ag = len(ag_targets) > 0
        print(f"[Diag] {prim.GetPath()}  typeName={prim.GetTypeName()}  "
              f"AG_targets={ag_targets}  applied={prim.GetAppliedSchemas()}")

def _path_is_valid(path) -> bool:
    """兼容 INavMeshPath 对象和列表两种返回形式"""
    if path is None:
        return False
    # INavMeshPath 对象：用 get_points() 或 points 属性
    if hasattr(path, 'get_points'):
        pts = path.get_points()
        return pts is not None and len(pts) > 0
    if hasattr(path, 'points'):
        return path.points is not None and len(path.points) > 0
    # 兼容直接返回列表的情况
    try:
        return len(path) > 0
    except TypeError:
        # 对象存在但无 len，保守认为有效（说明路径找到了）
        return True

def find_reachable_point(nm, from_xyz, max_attempts=50):
    """从 from_xyz 出发，找一个 NavMesh 上可达的随机目标点"""
    from_pt = carb.Float3(float(from_xyz[0]), float(from_xyz[1]), float(from_xyz[2]))
    for i in range(max_attempts):
        candidate = safe_query_random_point(nm)
        if candidate is None:
            continue
        to_pt = carb.Float3(float(candidate[0]), float(candidate[1]), float(candidate[2]))
        try:
            path = nm.query_shortest_path(from_pt, to_pt)
            if _path_is_valid(path):
                return candidate
        except Exception:
            continue
    print(f"[NavMesh] WARNING: No reachable point found from {from_xyz[:2]} "
          f"after {max_attempts} attempts — NavMesh may be disconnected")
    return safe_query_random_point(nm)  # fallback

def write_initial_commands_to_scriptdata(char_prims: list[str],
                                        nm,
                                        n_waypoints: int = 20):
    """Write a sequence of GoTo commands into each SkelRoot's
    omni:scripting:scriptData. CharacterBehavior reads these in
    init_character() when the timeline starts playing.

    Important: CharacterBehavior strips the agent name prefix itself,
    so the entries in scriptData should NOT include the name — it reads:
        cmd_with_name = f"{self.character_name} {cmd.strip()}"
    i.e., it prepends the name. So we just write "GoTo x y z _" here.
    """
    stage = omni.usd.get_context().get_stage()
    for char_prim_path in char_prims:
        # Find the SkelRoot under /World/Characters/Character/...
        char_prim = stage.GetPrimAtPath(char_prim_path)
        skelroot = None
        for desc in Usd.PrimRange(char_prim):
            if desc.GetTypeName() == "SkelRoot":
                skelroot = desc
                break
        if skelroot is None:
            print(f"[CMD] No SkelRoot under {char_prim_path}, skipping")
            continue

        # Build N random waypoints on the ground (low-z navmesh points)
        commands = []
        for _ in range(n_waypoints):
            spawn_pos = safe_query_random_point(nm)  # 或者用实际 spawn 坐标
            p = find_reachable_point(nm, spawn_pos)
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            commands.append(f"GoTo {x} {y} {z} _")

        # Write into scriptData on the SkelRoot (that's where the behavior
        # script is attached, so that's where it reads from).
        attr = skelroot.GetAttribute("omni:scripting:scriptData")
        if not attr:
            attr = skelroot.CreateAttribute(
                "omni:scripting:scriptData",
                Sdf.ValueTypeNames.StringArray,
            )
        attr.Set(commands)
        # Sanity: read it back exactly the way CharacterBehavior will.
        readback = skelroot.GetAttribute("omni:scripting:scriptData").Get()
        print(f"[CMD] Readback length = "
              f"{len(readback) if readback else 0}, "
              f"first = {readback[0] if readback else None}")
        print(f"[CMD] Wrote {len(commands)} GoTo commands to {skelroot.GetPath()}")
        
# ═══════════════════════════════════════════════════════════════════════════
# omni.anim.people command API
# ═══════════════════════════════════════════════════════════════════════════
def _get_command_api():
    try:
        from omni.anim.people.scripts.command_text_api import CommandTextAPI
        api = CommandTextAPI.get_instance()
        if api is not None and hasattr(api, "execute_command"):
            return lambda cmd: api.execute_command(cmd)
    except Exception:
        pass
    try:
        settings = carb.settings.get_settings()
        def _via_settings(cmd: str):
            settings.set("/exts/omni.anim.people/command_text", cmd)
        return _via_settings
    except Exception:
        pass
    return None


def send_goto(cmd_api, char_prim: str, target_xyz) -> bool:
    agent_name = char_prim.rstrip("/").split("/")[-1]   # ← 加这一行
    cmd = f"{agent_name} GoTo {float(target_xyz[0])} {float(target_xyz[1])} {float(target_xyz[2])} _"
    try:
        cmd_api(cmd)
        return True
    except Exception as e:
        print(f"[CMD] Failed '{cmd}': {e}")
        return False

def get_character_position(char_prim: str):
    try:
        from omni.anim.people.scripts.global_character_position_manager import (
            GlobalCharacterPositionManager,
        )
        mgr = GlobalCharacterPositionManager.get_instance()
        if mgr is not None:
            pos = mgr.get_character_current_pos(char_prim)
            if pos is not None:
                return np.array([float(pos[0]), float(pos[1]), float(pos[2])])
    except Exception:
        pass
    try:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(char_prim)
        if not prim.IsValid():
            return None
        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        t  = xf.ExtractTranslation()
        return np.array([float(t[0]), float(t[1]), float(t[2])])
    except Exception:
        return None

def frame_viewport_on(prim_path: str):
    """Frame the active viewport camera on the given prim."""
    try:
        from omni.kit.viewport.utility import get_active_viewport
        vp = get_active_viewport()
        if vp is None:
            print("[View] no active viewport")
            return
        cam_path = vp.camera_path.pathString
        omni.kit.commands.execute(
            "FramePrimsCommand",
            prim_to_move=cam_path,
            prims_to_frame=[prim_path],
            zoom=0.45,
        )
        print(f"[View] framed on {prim_path}")
    except Exception as e:
        print(f"[View] frame failed: {e}")
# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> int:
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)

    if not open_stage(ARGS.usda):
        return 1

    stage = omni.usd.get_context().get_stage()
    # offset_x, offset_y = verify_one_furniture_alignment(stage, ARGS.semantic_map_json)

    nav_ok, inav = try_bake_navmesh() # offset_x=offset_x, offset_y=offset_y
    if not nav_ok:
        print("\n[FATAL-ish] NavMesh bake failed — holding scene for inspection.")
        try:
            while simulation_app.is_running():
                simulation_app.update()
        except KeyboardInterrupt:
            pass
        simulation_app.close()
        return 2

    nm = inav.get_navmesh()

    print("\n[World] Creating isaacsim World + physicsScene...")
    from isaacsim.core.api import World
    world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 30.0)
    world.initialize_physics()
    world.reset()
    _update(10)
    print("[World] Physics initialized.")

    try:
        rp = safe_query_random_point(nm)
        print(f"[Check] navmesh post-World: random point {tuple(rp)}")
    except Exception as e:
        print(f"[Check] navmesh post-World probe failed: {e}")
    # ── 5. Hold ──────────────────────────────────────────────────────────
    print(f"\n[HOLD] Scene is open. "
          f"{'Physics ON' if world is not None else 'Physics OFF'}. "
          f"Ctrl+C to quit.\n")

    last_probe = time.time()
    try:
        while simulation_app.is_running():
            if world is not None:
                world.step(render=True)
            else:
                simulation_app.update()

    except KeyboardInterrupt:
        print("\n[HOLD] Ctrl+C received, shutting down.")

    simulation_app.close()
    return 0

    # Spawn people
    # char_prims: list[str] = []
    # if ARGS.num_people > 0:
    #     char_pool = load_character_assets()
    #     if not char_pool:
    #         print("[PEOPLE] No character USDs found.")
    #     else:
    #         print(f"[PEOPLE] {len(char_pool)} character asset(s) available.")
    #         for i in range(ARGS.num_people):
    #             usd = random.choice(char_pool)
    #             try:
    #                 spawn = safe_query_random_point(nm)
    #             except Exception as e:
    #                 print(f"[PEOPLE] query_random_point failed for spawn {i}: {e}")
    #                 break
    #             try:
    #                 prim = spawn_character(i, usd, spawn)
    #                 char_prims.append(prim)
    #             except Exception as e:
    #                 print(f"[PEOPLE] spawn_character failed for {i}: {e}")

    # load_default_skeleton_and_animations()
    # _update(30)
    # bind_animation_graph_to_characters()
    # _update(30)

    # # ★ Write scriptData FIRST, THEN attach scripts. Otherwise
    # #   CharacterBehavior.on_init runs before scriptData exists and
    # #   caches self.commands = [] permanently.
    # if char_prims:
    #     write_initial_commands_to_scriptdata(char_prims, nm, n_waypoints=30)
    #     _update(10)

    # attach_behavior_scripts_to_characters()
    # _update(60)

    # if char_prims:
    #     frame_viewport_on(char_prims[0])
    #     _update(5)

    # tl = omni.timeline.get_timeline_interface()
    # tl.set_current_time(0.0)
    # tl.play()
    # _update(10)

    # cmd_api = _get_command_api()
    # if cmd_api is None:
    #     print("[INFO] CommandTextAPI not available; characters will follow "
    #           "their initial scriptData commands only (no runtime waypoint updates).")
    # elif char_prims:
    #     # scriptData already has the initial waypoints; optionally add an
    #     # extra live one so movement kicks off as soon as possible.
    #     for p in char_prims:
    #         send_goto(cmd_api, p, safe_query_random_point(nm))

    # print("\n[HOLD] Press Ctrl+C to quit.\n")
    # last_waypoint_t = {p: time.time() for p in char_prims}
    # last_pos        = {p: None        for p in char_prims}
    # last_probe      = time.time()
    # MOVE_EPS        = 0.05

    # try:
    #     total_dist: dict[str, float] = {p: 0.0 for p in char_prims}
    #     last_report = time.time()
    #     REPORT_EVERY = 2.0   # seconds
        
    #     while simulation_app.is_running():
    #         world.step(render=True)

    #         if cmd_api is not None:
    #             for p in char_prims:
    #                 pos = get_character_position(p)
    #                 if pos is not None:
    #                     prev = last_pos[p]
    #                     if prev is not None:
    #                         step = np.linalg.norm(pos - prev)
    #                         if step > MOVE_EPS:
    #                             last_waypoint_t[p] = time.time()
    #                         # accumulate distance per character for reporting
    #                         total_dist[p] = total_dist.get(p, 0.0) + step
    #                     last_pos[p] = pos

    #                 if (time.time() - last_waypoint_t[p]) > ARGS.waypoint_timeout:
    #                     try:
    #                         pos = last_pos.get(p_char)
    #                         if pos is not None:
    #                             new_target = find_reachable_point(nm, pos)
    #                         else:
    #                             new_target = safe_query_random_point(nm)
    #                         send_goto(cmd_api, p_char, new_target)
    #                         last_waypoint_t[p] = time.time()
    #                         print(f"[RUN] {p} → new waypoint {tuple(new_target)}")
    #                     except Exception as e:
    #                         print(f"[WARN] waypoint reissue for {p}: {e}")

    #         if (time.time() - last_probe) > ARGS.navmesh_probe_interval:
    #             try:
    #                 rp = safe_query_random_point(nm)
    #                 print(f"[probe] navmesh random point: {tuple(rp)}")
    #             except Exception as e:
    #                 print(f"[probe] failed: {e}")
    #             last_probe = time.time()

    #         if (time.time() - last_report) > REPORT_EVERY:
    #             for p in char_prims:
    #                 pos = last_pos.get(p)
    #                 if pos is not None:
    #                     print(f"[POS] {p}  pos=({pos[0]:+.2f},{pos[1]:+.2f},"
    #                           f"{pos[2]:+.2f})  total_walked={total_dist[p]:.2f}m")
    #             last_report = time.time()
    # except KeyboardInterrupt:
    #     print("\n[HOLD] Ctrl+C received, shutting down.")

    # tl.stop()
    # simulation_app.close()
    # return 0


if __name__ == "__main__":
    sys.exit(main())