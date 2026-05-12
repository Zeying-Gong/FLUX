"""Open a USDA scene, bake NavMesh, bring up physics, spawn N pedestrians
on the NavMesh, and have them randomly walk forever until Ctrl+C.

NavMesh caching
---------------
For a given USDA scene, two cache files are written beside the USDA:

  <scene_stem>.navmesh          – binary NavMesh data  (inav.save/load_navmesh)
  <scene_stem>_navvols.usda     – NavMeshVolume prims  (USD sublayer)

On the second run both are restored and baking is skipped entirely.
Pass --force_rebake to ignore the cache and redo everything.
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

    p.add_argument("--semantic_map_json", type=str, default=None,
                   help="Optional 2D semantic map JSON — used to carve out "
                        "Exclude volumes above furniture so characters don't "
                        "walk onto tables/sofas/beds.")

    # ── Cache control ──────────────────────────────────────────────────────
    p.add_argument("--force_rebake", action="store_true",
                   help="Ignore existing NavMesh / NavMeshVolume cache and "
                        "redo bake from scratch.")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Directory to store cache files. "
                        "Defaults to the same directory as --usda.")
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
# Post-launch imports
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
from omni.anim.people.settings import PeopleSettings

CHARACTER_ASSET_PATH = "/workspace/FLUX/assets/isaacsim_assets/Characters"


# ═══════════════════════════════════════════════════════════════════════════
# Cache path helpers
# ═══════════════════════════════════════════════════════════════════════════
def _cache_paths(usda_path: str, cache_dir: str | None) -> str:
    """Return navvols_usda path. NavMesh binary is managed by Isaac Sim internally."""
    stem = os.path.splitext(os.path.basename(usda_path))[0]
    if cache_dir:
        base = cache_dir
    else:
        usda_dir = os.path.dirname(os.path.abspath(usda_path))
        parent   = os.path.dirname(usda_dir)
        base     = os.path.join(parent, "usda_processed")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{stem}_navvols.usda")


# ═══════════════════════════════════════════════════════════════════════════
# Stage / bbox / visibility helpers
# ═══════════════════════════════════════════════════════════════════════════
def _update(n: int = 1):
    for _ in range(n):
        simulation_app.update()


def safe_query_random_point(nm, max_z=0.1, retries=30):
    """Safely sample a random NavMesh point, filtering furniture-top heights."""
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(500)
    try:
        best = None
        for _ in range(retries):
            try:
                p = nm.query_random_point()
                z = float(p[2])
                if z < max_z:
                    return p
                if best is None:
                    best = p
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


# ═══════════════════════════════════════════════════════════════════════════
# NavMesh Volume helpers
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


def add_exclude_volumes_from_semantic_map(stage, semantic_map_json,
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
                all_y.append(float(y))
                all_x.append(float(x))
            except (ValueError, TypeError):
                continue

    if not all_x:
        print("[Exclude] ERROR: no mask_coords_m found")
        return 0

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    print(f"[Exclude] map bounds from mask_coords_m: "
          f"x∈[{min_x:.3f},{max_x:.3f}] y∈[{min_y:.3f},{max_y:.3f}]")

    def sm_to_isaac(sm_x, sm_y):
        px, py = sm_x, sm_y
        if flip_x:
            px = (min_x + max_x) - px
        if flip_y:
            py = (min_y + max_y) - py
        if negate_xy:
            px = -px
            py = -py
        return px, py

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    half_extent_stage_units = 0.5 / mpu
    bbox = compute_scene_bbox(stage)
    created = 0
    skipped = 0

    for item in items:
        label = str(item.get("category_label", "")).lower()
        if not any(keyword in label for keyword in exclude_label_set):
            continue
        x_l, y_b, x_r, y_t = [float(v) for v in item["bbox_m"]]
        z_min = float(item["min_z_m"])
        z_max = float(item["max_z_m"])

        corners_sm = [(x_l, y_b), (x_l, y_t), (x_r, y_b), (x_r, y_t)]
        corners_isaac = [sm_to_isaac(cx, cy) for cx, cy in corners_sm]
        ix_vals = [c[0] for c in corners_isaac]
        iy_vals = [c[1] for c in corners_isaac]
        ix_l, ix_r = min(ix_vals), max(ix_vals)
        iy_b, iy_t = min(iy_vals), max(iy_vals)
        cx = 0.5*(ix_l + ix_r)
        cy = 0.5*(iy_b + iy_t)

        PAD = 0.10
        hx = 0.5*(ix_r - ix_l) + PAD
        hy = 0.5*(iy_t - iy_b) + PAD

        if bbox:
            bmin, bmax = bbox
            margin = 1.0
            if not (bmin[0]-margin <= cx <= bmax[0]+margin and
                    bmin[1]-margin <= cy <= bmax[1]+margin):
                skipped += 1
                continue

        bottom = max(z_min - 0.05, -0.01)
        top    = z_max + 0.05
        hz     = 0.5 * (top - bottom)
        cz     = 0.5 * (top + bottom)

        existing = {p.GetPath() for p in stage.Traverse()
                    if p.GetName().startswith("NavMeshVolume")}
        omni.kit.commands.execute("CreateNavMeshVolumeCommand",
                                  parent_prim_path=Sdf.Path.emptyPath,
                                  volume_type=1, usd_context_name="", layer=None)
        _update(1)
        new_prims = [p.GetPath() for p in stage.Traverse()
                     if p.GetName().startswith("NavMeshVolume")
                     and p.GetPath() not in existing]
        if not new_prims:
            continue
        vol_path = new_prims[0]

        scale  = Gf.Vec3d(hx / half_extent_stage_units,
                          hy / half_extent_stage_units,
                          hz / half_extent_stage_units)
        mat = Gf.Matrix4d(1.0)
        mat.SetScale(scale)
        mat = mat * Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(cx, cy, cz))
        omni.kit.commands.execute("TransformPrim", path=vol_path,
                                  new_transform_matrix=mat)
        created += 1

    _update(10)
    print(f"[Exclude] Created {created}, skipped {skipped}.")
    return created


# ═══════════════════════════════════════════════════════════════════════════
# NavMeshVolume USD cache  save / restore
# ═══════════════════════════════════════════════════════════════════════════
def _collect_navmesh_volume_paths(stage) -> list[Sdf.Path]:
    """Return USD paths of all NavMeshVolume prims currently in the stage."""
    return [p.GetPath() for p in stage.Traverse()
            if p.GetName().startswith("NavMeshVolume")]

def restore_navmesh_volumes(stage, navvols_usda: str) -> int:
    """Re-add NavMeshVolume prims from a cached USDA into the live stage.

    We use SdfLayer merging so that the prims land in the session layer
    (non-destructive, not written back to the original USDA).
    """
    cached = Sdf.Layer.FindOrOpen(navvols_usda)
    if cached is None:
        print(f"[Cache] Could not open {navvols_usda}")
        return 0

    session = stage.GetSessionLayer()
    edit_target = Usd.EditTarget(session)
    stage.SetEditTarget(edit_target)

    restored = 0
    for spec in cached.rootPrims:
        vp = Sdf.Path(f"/{spec.name}")
        if not stage.GetPrimAtPath(vp).IsValid():
            Sdf.CopySpec(cached, vp, session, vp)
            restored += 1
        else:
            print(f"[Cache]   {vp} already present, skipping.")

    # Restore edit target to default layer
    stage.SetEditTarget(Usd.EditTarget(stage.GetRootLayer()))

    _update(5)
    print(f"[Cache] Restored {restored} NavMeshVolume(s) from {navvols_usda}")
    return restored


# ═══════════════════════════════════════════════════════════════════════════
# NavMesh binary cache  save / restore
# ═══════════════════════════════════════════════════════════════════════════
def save_navmesh_binary(inav, navmesh_bin: str) -> bool:
    """Persist the baked NavMesh to disk using the navigation interface."""
    try:
        # Isaac Sim ≥ 4.x exposes save_navmesh(path) on the nav interface.
        inav.save_navmesh(navmesh_bin)
        print(f"[Cache] NavMesh binary saved → {navmesh_bin}")
        return True
    except AttributeError:
        print("[Cache] inav.save_navmesh() not available on this Isaac Sim version.")
    except Exception as e:
        print(f"[Cache] save_navmesh failed: {e}")
    return False


def load_navmesh_binary(inav, navmesh_bin: str) -> bool:
    """Restore a previously baked NavMesh from disk."""
    try:
        inav.load_navmesh(navmesh_bin)
        print(f"[Cache] NavMesh binary loaded ← {navmesh_bin}")
        return True
    except AttributeError:
        print("[Cache] inav.load_navmesh() not available on this Isaac Sim version.")
    except Exception as e:
        print(f"[Cache] load_navmesh failed: {e}")
    return False


# ═══════════════════════════════════════════════════════════════════════════
# NavMesh bake  (full or cached)
# ═══════════════════════════════════════════════════════════════════════════
def _apply_navmesh_settings():
    settings = carb.settings.get_settings()
    settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/config/agentMinRadius", 0.3)
    settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/config/cellSize", 0.15)
    settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/config/regionMergeSize", 20)
    settings.set(PeopleSettings.CHARACTER_FINAL_TARGET_DISTANCE, 0.5)
    settings.set("/persistent/exts/omni.anim.people/minDistanceToIntermediateTarget", 0.8)
    settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/config/agentMinHeight", 1.8)
    settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/config/agentMinIslandRadius", 0.5)
    settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/config/autoRebakeOnChanges", False)
    settings.set(PeopleSettings.NAVMESH_ENABLED, True)
    settings.set(PeopleSettings.DYNAMIC_AVOIDANCE_ENABLED, False)
    settings.set(PeopleSettings.NUMBER_OF_LOOP, "0")
    from omni.anim.navigation.core import NavMeshSettings
    settings.set(NavMeshSettings.CACHE_ENABLED_SETTING_PATH, True)
    _update(5)

def save_navmesh_volumes_as_delta(stage, navvols_usda: str):
    """把所有 NavMeshVolume specs 单独存成一个最小 delta layer。
    不碰原始场景，不 flatten，材质/灯光完全不受影响。
    """
    # 找到实际持有这些 spec 的 layer（通常是 edit target layer）
    vol_paths = [p.GetPath() for p in stage.Traverse()
                 if p.GetName().startswith("NavMeshVolume")]
    if not vol_paths:
        print("[Cache] No NavMeshVolume prims found, nothing to save.")
        return

    layer_stack = stage.GetLayerStack(includeSessionLayers=False)

    # 创建一个新的空 layer，手动写入 Volume specs
    delta = Sdf.Layer.CreateNew(navvols_usda)
    delta.Clear()

    # /World spec 必须存在（作为父节点容器，但不带任何其他内容）
    world_spec = Sdf.PrimSpec(delta, "World", Sdf.SpecifierOver)

    for vp in vol_paths:
        # 找持有该 spec 的最强 layer
        src_layer = None
        for lyr in layer_stack:
            if lyr.GetPrimAtPath(vp):
                src_layer = lyr
                break
        if src_layer is None:
            print(f"[Cache] WARNING: no layer has spec for {vp}, skipping.")
            continue
        vol_name = vp.name  # e.g. "NavMeshVolume", "NavMeshVolume_01"
        Sdf.CopySpec(src_layer, vp, delta, Sdf.Path(f"/World/{vol_name}"))

    delta.Save()
    print(f"[Cache] Saved {len(vol_paths)} NavMeshVolume(s) as delta → {navvols_usda}")


def restore_navmesh_volumes_as_sublayer(stage, navvols_usda: str) -> bool:
    """把 delta layer 作为 sublayer 插入原始场景最顶层（强于 root layer）。
    原始场景结构、材质、灯光完全不变。
    """
    root_layer = stage.GetRootLayer()

    # 避免重复插入
    if navvols_usda in root_layer.subLayerPaths:
        print(f"[Cache] Delta layer already in subLayerPaths, skipping.")
        return True

    # 插到 subLayerPaths[0]（最高优先级）
    root_layer.subLayerPaths.insert(0, navvols_usda)
    _update(5)

    # 验证
    vol_count = sum(1 for p in stage.Traverse()
                    if p.GetName().startswith("NavMeshVolume"))
    print(f"[Cache] Delta sublayer inserted, {vol_count} NavMeshVolume(s) active.")
    return vol_count > 0

def try_bake_navmesh(navvols_usda: str, force_rebake: bool):
    import omni.anim.navigation.core as nav

    _apply_navmesh_settings()
    inav = nav.acquire_interface()
    print(f"[NavMesh] Internal cache dir: {inav.get_cache_dir()}")

    has_cached_delta = (
        not force_rebake
        and os.path.exists(navvols_usda)
    )

    if has_cached_delta:
        print(f"[Cache] Restoring NavMeshVolumes from delta: {navvols_usda}")
        stage = omni.usd.get_context().get_stage()
        ok = restore_navmesh_volumes_as_sublayer(stage, navvols_usda)
        if ok:
            # ★ 让 collision geometry 可见，NavMesh bake 才能采样到它
            prev_vis = set_subtree_visibility(stage, ARGS.collision_root, visible=True)
            _update(5)

            print("[NavMesh] Baking (volumes restored, expect cache hit)...")
            inav.start_navmesh_baking_and_wait()
            nm = inav.get_navmesh()
            rp = safe_query_random_point(nm)

            # bake 完恢复原来的 visibility
            if prev_vis == "invisible" and not ARGS.keep_collision_visible:
                set_subtree_visibility(stage, ARGS.collision_root, visible=False)
                _update(2)

            if rp is not None:
                print(f"[Cache] NavMesh OK. Sample point: {tuple(rp)}")
                return True, inav
        print("[Cache] Restore failed — falling back to full bake.")

    # ── 完整 bake ──────────────────────────────────────────────────────────
    print("\n[NavMesh] Running full bake...")
    stage = omni.usd.get_context().get_stage()

    prev_vis = set_subtree_visibility(stage, ARGS.collision_root, visible=True)
    _update(5)

    existing_vols = {p.GetPath() for p in stage.Traverse()
                     if p.GetName().startswith("NavMeshVolume")}
    try:
        omni.kit.commands.execute("CreateNavMeshVolumeCommand",
                                  parent_prim_path=Sdf.Path.emptyPath,
                                  volume_type=0, usd_context_name="", layer=None)
    except Exception as e:
        print(f"[NavMesh] CreateNavMeshVolumeCommand failed: {e}")
        return False, None
    _update(5)

    new_volumes = [p.GetPath() for p in stage.Traverse()
                   if p.GetName().startswith("NavMeshVolume")
                   and p.GetPath() not in existing_vols]
    if not new_volumes:
        print("[NavMesh] Could not find new NavMeshVolume.")
        return False, None

    size_navmesh_volume(stage, new_volumes[0],
                        padding=ARGS.volume_padding,
                        fallback_size=ARGS.fallback_size)

    if ARGS.semantic_map_json and os.path.exists(ARGS.semantic_map_json):
        add_exclude_volumes_from_semantic_map(stage, ARGS.semantic_map_json)
    elif ARGS.semantic_map_json:
        print(f"[Exclude] semantic_map_json not found: {ARGS.semantic_map_json}")

    print(f"[NavMesh] Warming up {ARGS.warmup_frames} frames...")
    _update(ARGS.warmup_frames)

    print("[NavMesh] Baking...")
    inav.start_navmesh_baking_and_wait()
    nm = inav.get_navmesh()
    rp = safe_query_random_point(nm)
    print(f"[NavMesh] OK. Sample point: {tuple(rp)}")

    # ── 保存 delta（只存 Volume specs，不碰原始场景）──────────────────────
    stage = omni.usd.get_context().get_stage()
    save_navmesh_volumes_as_delta(stage, navvols_usda)

    if prev_vis == "invisible" and not ARGS.keep_collision_visible:
        set_subtree_visibility(stage, ARGS.collision_root, visible=False)
        _update(2)

    return True, inav

# ═══════════════════════════════════════════════════════════════════════════
# Character discovery
# ═══════════════════════════════════════════════════════════════════════════
EXCLUDED_CHARACTER_FOLDERS = {"biped_demo"}


def load_character_assets() -> list[str]:
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

        rel = prim.GetRelationship("skel:animationGraph")
        if not rel:
            rel = prim.CreateRelationship("skel:animationGraph", custom=False)
        rel.SetTargets([ag_path])

        targets = rel.GetTargets()
        print(f"[PEOPLE]   {skel_path}  skel:animationGraph → {list(targets)}")

    print(f"[PEOPLE] Animation graph bound to {len(skelroots)} SkelRoot(s).")
    _update(30)


def attach_behavior_scripts_to_characters():
    stage = omni.usd.get_context().get_stage()
    skelroots = CharacterUtil.get_characters_in_stage()
    script_path = BehaviorScriptPaths.behavior_script_path()
    print(f"[PEOPLE] CharacterBehavior script path: {script_path}")
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
        print(f"[Diag] {prim.GetPath()}  typeName={prim.GetTypeName()}  "
              f"AG_targets={ag_targets}  applied={prim.GetAppliedSchemas()}")


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

def spawn_on_navmesh(nm) -> tuple:
    """找一个离墙足够远、agent 能站稳的 spawn 点。"""
    for _ in range(50):
        p = safe_query_random_point(nm)
        if p is None:
            continue
        # 用 query_closest_point 做 snap，确认落在可行走面上
        try:
            result = nm.query_closest_point(p, 0.5)
            snapped = result[0] if result else None
        except Exception:
            snapped = p
        if snapped is None:
            continue
        # 验证从 snapped 出发能找到至少一个可达目标
        target = find_reachable_point(nm, snapped, max_attempts=20, min_dist=1.0)
        if target is not None:
            return snapped
    return safe_query_random_point(nm)  # fallback

def find_reachable_point(nm, from_xyz, max_attempts=50, min_dist=3.0):
    from_pt = carb.Float3(float(from_xyz[0]), float(from_xyz[1]), float(from_xyz[2]))
    for i in range(max_attempts):
        candidate = safe_query_random_point(nm)
        if candidate is None:
            continue
        dx = float(candidate[0]) - float(from_xyz[0])
        dy = float(candidate[1]) - float(from_xyz[1])
        dist = (dx*dx + dy*dy) ** 0.5
        if dist < min_dist:
            continue
        to_pt = carb.Float3(float(candidate[0]), float(candidate[1]), float(candidate[2]))
        try:
            path = nm.query_shortest_path(from_pt, to_pt, agent_radius=0.3)
            if _path_is_valid(path):
                return candidate
        except Exception:
            continue

    print(f"[NavMesh] WARNING: Relaxing min_dist to 1.0m")
    for i in range(max_attempts):
        candidate = safe_query_random_point(nm)
        if candidate is None:
            continue
        dx = float(candidate[0]) - float(from_xyz[0])
        dy = float(candidate[1]) - float(from_xyz[1])
        if (dx*dx + dy*dy) ** 0.5 < 1.0:
            continue
        to_pt = carb.Float3(float(candidate[0]), float(candidate[1]), float(candidate[2]))
        try:
            path = nm.query_shortest_path(from_pt, to_pt, agent_radius=0.3)
            if _path_is_valid(path):
                return candidate
        except Exception:
            continue
    return safe_query_random_point(nm)


def write_initial_commands_to_scriptdata(char_prims: list[str],
                                         nm,
                                         spawn_positions: dict,
                                         n_waypoints: int = 20):
    stage = omni.usd.get_context().get_stage()
    for char_prim_path in char_prims:
        char_prim = stage.GetPrimAtPath(char_prim_path)
        skelroot = None
        for desc in Usd.PrimRange(char_prim):
            if desc.GetTypeName() == "SkelRoot":
                skelroot = desc
                break
        if skelroot is None:
            print(f"[CMD] No SkelRoot under {char_prim_path}, skipping")
            continue

        current_pos = spawn_positions.get(char_prim_path)
        if current_pos is None:
            current_pos = safe_query_random_point(nm)
        # ★ snap 到 NavMesh 最近点，防止 spawn 落点略微偏离可行走面
        try:
            result = nm.query_closest_point(current_pos, 0.5)
            if result and result[0] is not None:
                current_pos = result[0]
        except Exception:
            pass

        commands = []
        prev_pos = current_pos
        for _ in range(n_waypoints):
            p = find_reachable_point(nm, prev_pos)
            if p is None:
                continue
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            commands.append(f"GoTo {x} {y} {z} _")
            prev_pos = p

        attr = skelroot.GetAttribute("omni:scripting:scriptData")
        if not attr:
            attr = skelroot.CreateAttribute(
                "omni:scripting:scriptData",
                Sdf.ValueTypeNames.StringArray,
            )
        attr.Set(commands)
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
    agent_name = char_prim.rstrip("/").split("/")[-1]
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

    # Resolve cache file paths
    navvols_usda = _cache_paths(ARGS.usda, ARGS.cache_dir)

    if ARGS.force_rebake:
        print("[Cache] --force_rebake set: ignoring existing cache.")

    nav_ok, inav = try_bake_navmesh(navvols_usda, ARGS.force_rebake)
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

    # ── Spawn people ──────────────────────────────────────────────────────
    char_prims: list[str] = []
    char_spawn_positions: dict[str, tuple] = {}

    char_pool = load_character_assets()
    if not char_pool:
        print("[PEOPLE] No character USDs found.")
    else:
        print(f"[PEOPLE] {len(char_pool)} character asset(s) available.")

    for i in range(ARGS.num_people):
        usd = random.choice(char_pool)
        spawn = spawn_on_navmesh(nm)   # ← 替代 safe_query_random_point(nm)
        prim = spawn_character(i, usd, spawn)
        char_prims.append(prim)
        char_spawn_positions[prim] = spawn

    if char_prims:
        frame_viewport_on(char_prims[0])
        _update(5)

    load_default_skeleton_and_animations()
    _update(30)
    bind_animation_graph_to_characters()
    _update(30)

    if char_prims:
        write_initial_commands_to_scriptdata(char_prims, nm, char_spawn_positions, n_waypoints=3)
        _update(10)

    attach_behavior_scripts_to_characters()
    _update(60)

    tl = omni.timeline.get_timeline_interface()
    tl.set_current_time(0.0)
    tl.play()
    _update(300)

    print("\n[HOLD] Press Ctrl+C to quit.\n")

    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        print("\n[HOLD] Ctrl+C received, shutting down.")

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())