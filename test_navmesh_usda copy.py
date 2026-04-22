"""Open a USDA scene, bake NavMesh BEFORE initializing any PhysX World,
then bring up a World and verify the navmesh query still works while
physics is running. Hold viewport until Ctrl+C.

Correct order of operations (this matters!):
    1. open_stage(usda)
    2. make collision geometry visible to the baker
    3. create + size NavMeshVolume
    4. bake navmesh   ← must happen BEFORE World(), because otherwise
                         PhysX takes over the collision meshes and the
                         baker sees nothing
    5. World(...) + initialize_physics + reset
    6. Stream world.step() in the hold loop — navmesh is already in
       memory, so query_random_point() still works

Usage:
    python check_scene_navmesh.py --usda /path/to/scene.usda
    python check_scene_navmesh.py --usda /path/to/scene.usda --skip_world
"""
from __future__ import annotations
import argparse
import os
import sys
import time


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Bake NavMesh, then start physics World, then hold."
    )
    p.add_argument("--usda", type=str, required=True,
                   help="Absolute path to the scene .usda / .usd file.")
    p.add_argument("--collision_root", type=str, default="/World/scene_collision",
                   help="Prim path whose visibility must be ON for bake. "
                        "SAGE-3D scenes keep this invisible by default.")
    p.add_argument("--keep_collision_visible", action="store_true",
                   help="After bake, keep collision_root visible.")
    p.add_argument("--volume_padding", type=float, default=1.2)
    p.add_argument("--fallback_size",  type=float, default=100.0)
    p.add_argument("--warmup_frames",  type=int,   default=60)
    p.add_argument("--skip_world",     action="store_true",
                   help="Skip initializing isaacsim World / physicsScene. "
                        "Useful if you only want to eyeball the scene.")
    p.add_argument("--navmesh_probe_interval", type=float, default=3.0,
                   help="Seconds between test navmesh queries during hold.")
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
import omni
import omni.usd
import omni.kit.app
import omni.kit.commands
from pxr import Sdf, Usd, UsdGeom, Gf


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
    new_val = "inherited" if visible else "invisible"
    print(f"[Vis] {root_path}: {prev} → {new_val}")
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
        half_ext = Gf.Vec3d(fallback_size * 0.5,) * 3
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


def try_bake_navmesh():
    """Returns (success: bool, navmesh_interface or None)."""
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
        print("[NavMesh] Could not find new NavMeshVolume prim.")
        return False, None
    volume_path = new_volumes[0]
    print(f"[NavMesh] Created {volume_path}")

    size_navmesh_volume(stage, volume_path,
                        padding=ARGS.volume_padding,
                        fallback_size=ARGS.fallback_size)

    print(f"[NavMesh] Warming up for {ARGS.warmup_frames} frames before bake...")
    _update(ARGS.warmup_frames)

    inav = nav.acquire_interface()
    print("[NavMesh] Baking...")
    nm = None
    try:
        inav.start_navmesh_baking_and_wait()
        nm = inav.get_navmesh()
    except Exception as e:
        print(f"[NavMesh] Baking raised: {e}")

    if nm is None:
        print("[NavMesh] FAILED.")
        success = False
    else:
        try:
            rp = nm.query_random_point()
            print(f"[NavMesh] OK. Sample random point: {tuple(rp)}")
        except Exception as e:
            print(f"[NavMesh] OK but query_random_point raised: {e}")
        success = True

    if prev_vis == "invisible" and not ARGS.keep_collision_visible:
        set_subtree_visibility(stage, ARGS.collision_root, visible=False)
        _update(2)

    return success, inav


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> int:
    # ── 1. Open stage ────────────────────────────────────────────────────
    if not open_stage(ARGS.usda):
        return 1

    # ── 2. BAKE NAVMESH FIRST  (before any physics World) ────────────────
    nav_ok, inav = try_bake_navmesh()
    if not nav_ok:
        print("\n[WARN] NavMesh bake failed; will still hold scene for inspection.")

    # ── 3. Now it's safe to bring up PhysX World ─────────────────────────
    world = None
    if not ARGS.skip_world:
        print("\n[World] Creating isaacsim World + physicsScene...")
        # Import here (after bake) so the mere import doesn't affect bake
        from isaacsim.core.api import World
        world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 30.0)
        world.initialize_physics()
        world.reset()
        _update(10)
        print("[World] Physics initialized. World.step() will now advance sim.")
    else:
        print("\n[World] Skipped (--skip_world).")

    # ── 4. Sanity-check: does the navmesh still answer queries? ─────────
    if nav_ok:
        print("\n[Check] Probing navmesh after physics init...")
        try:
            for i in range(3):
                rp = inav.get_navmesh().query_random_point()
                print(f"  probe {i}: {tuple(rp)}")
        except Exception as e:
            print(f"  probe failed: {e}")

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

            # periodic navmesh probe so we can see it's still alive
            if nav_ok and (time.time() - last_probe) > ARGS.navmesh_probe_interval:
                try:
                    rp = inav.get_navmesh().query_random_point()
                    print(f"[probe] random navmesh point: {tuple(rp)}")
                except Exception as e:
                    print(f"[probe] failed: {e}")
                last_probe = time.time()

    except KeyboardInterrupt:
        print("\n[HOLD] Ctrl+C received, shutting down.")

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())