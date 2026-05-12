"""
people_utils.py
───────────────
Character spawning, animation-graph wiring, scriptData command injection,
and GoTo path pre-computation helpers.

All functions accept `app` (SimulationApp) so they can call app.update()
without importing it themselves.
"""
from __future__ import annotations

import omni
import omni.client
import omni.kit.commands
import omni.usd
import carb
from pxr import Sdf, Usd, UsdGeom

from isaacsim.replicator.agent.core.settings import (
    AssetPaths, PrimPaths, BehaviorScriptPaths, Settings,
)
from isaacsim.replicator.agent.core.stage_util import CharacterUtil
from isaacsim.core.utils import prims
from omni.anim.people.scripts.custom_command.populate_anim_graph import populate_anim_graph

from navmesh_utils import (
    safe_query_random_point, find_reachable_point, spawn_on_navmesh,
)


EXCLUDED_CHARACTER_FOLDERS = {"biped_demo"}


# ── helpers ────────────────────────────────────────────────────────────────

def _update(app, n: int = 1):
    for _ in range(n):
        app.update()


# ── asset discovery ─────────────────────────────────────────────────────────

def load_character_assets() -> list[str]:
    assets_root = AssetPaths.default_character_path() # "/workspace/FLUX/assets/isaacsim_assets/Assets/Isaac/4.5/Isaac/People/Characters" # 
    print(f"[PEOPLE] Listing characters under {assets_root}")
    result, folder_list = omni.client.list(f"{assets_root}/")
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
        folder_path = f"{assets_root}/{folder.relative_path}"
        res2, file_list = omni.client.list(f"{folder_path}/")
        if res2 != omni.client.Result.OK:
            continue
        for f in file_list:
            if f.relative_path.endswith((".usd", ".usda")):
                character_assets.append(f"{folder_path}/{f.relative_path}")
                break
    return character_assets


# ── spawn ───────────────────────────────────────────────────────────────────

def _disable_character_physics(app, char_prim_path: str):
    """移除角色 subtree 下所有 PhysicsAPI，防止被场景碰撞体阻挡。"""
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root  = stage.GetPrimAtPath(char_prim_path)
    if not root.IsValid():
        return

    removed = 0
    for prim in Usd.PrimRange(root):
        # 移除 RigidBody
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            removed += 1
        # 移除 Collider
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            removed += 1
        # 移除 CharacterController（如果有）
        try:
            from pxr import PhysxSchema
            if prim.HasAPI(PhysxSchema.PhysxCharacterControllerAPI):
                prim.RemoveAPI(PhysxSchema.PhysxCharacterControllerAPI)
                removed += 1
        except ImportError:
            pass

    if removed:
        print(f"[PEOPLE] Removed {removed} Physics API(s) from {char_prim_path}")
    _update(app, 2)

def spawn_character(app, idx: int, usd_path: str, spawn_xyz) -> str:
    loc  = carb.Float3(float(spawn_xyz[0]), float(spawn_xyz[1]), float(spawn_xyz[2]))
    name = CharacterUtil.get_character_name_by_index(idx)
    CharacterUtil.load_character_usd_to_stage(usd_path, loc, 0.0, name)
    _update(app, 5)
    prim_path = f"{PrimPaths.characters_parent_path()}/{name}"
    
    # ★ 禁用角色物理碰撞，让 animation graph 纯运动学驱动
    # _disable_character_physics(app, prim_path)

    print(f"[PEOPLE] Spawned {prim_path}  at={tuple(spawn_xyz)}")
    return prim_path


# ── skeleton / animation graph ──────────────────────────────────────────────

def load_default_skeleton_and_animations(app):
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
        p = prims.create_prim(biped_path, "Xform",
                              usd_path=AssetPaths.default_biped_asset_path())
        p.GetAttribute("visibility").Set("invisible")
    populate_anim_graph()


def bind_animation_graph_to_characters(app):
    stage    = omni.usd.get_context().get_stage()
    biped    = stage.GetPrimAtPath(PrimPaths.biped_prim_path())
    anim_graph = CharacterUtil.get_anim_graph_from_character(biped)
    if anim_graph is None:
        print("[PEOPLE] Could not find animation graph on biped template.")
        return
    ag_path   = anim_graph.GetPrimPath()
    skelroots = CharacterUtil.get_characters_in_stage()
    for prim in skelroots:
        skel_path = prim.GetPrimPath()
        try:
            omni.kit.commands.execute("RemoveAnimationGraphAPICommand",
                                      paths=[Sdf.Path(skel_path)])
        except Exception:
            pass
        omni.kit.commands.execute("ApplyAnimationGraphAPICommand",
                                  paths=[Sdf.Path(skel_path)],
                                  animation_graph_path=Sdf.Path(ag_path))
        rel = prim.GetRelationship("skel:animationGraph")
        if not rel:
            rel = prim.CreateRelationship("skel:animationGraph", custom=False)
        rel.SetTargets([ag_path])
        print(f"[PEOPLE]   {skel_path}  skel:animationGraph → {list(rel.GetTargets())}")
    print(f"[PEOPLE] Animation graph bound to {len(skelroots)} SkelRoot(s).")
    _update(app, 30)


def attach_behavior_scripts_to_characters(app):
    stage      = omni.usd.get_context().get_stage()
    skelroots  = CharacterUtil.get_characters_in_stage()
    script_path = BehaviorScriptPaths.behavior_script_path()
    print(f"[PEOPLE] CharacterBehavior script path: {script_path}")
    for prim in skelroots:
        skel_path = str(prim.GetPrimPath())
        if not prim.HasAttribute("omni:scripting:scripts"):
            omni.kit.commands.execute("ApplyScriptingAPICommand",
                                      paths=[Sdf.Path(skel_path)])
        prim.GetAttribute("omni:scripting:scripts").Set([script_path])
    print(f"[PEOPLE] CharacterBehavior attached to {len(skelroots)} SkelRoot(s).")
    _update(app, 30)
    for prim in CharacterUtil.get_characters_in_stage():
        rel = prim.GetRelationship("skel:animationGraph")
        ag_targets = list(rel.GetTargets()) if rel else []
        print(f"[Diag] {prim.GetPath()}  typeName={prim.GetTypeName()}  "
              f"AG_targets={ag_targets}  applied={prim.GetAppliedSchemas()}")


# ── path pre-computation ────────────────────────────────────────────────────

def compute_goto_path(nm, from_xyz, to_xyz) -> list[list[float]] | None:
    """Query NavMesh for a shortest path and return it as [[x,y,z], ...].

    Returns None if no path was found.
    """
    from_pt = carb.Float3(float(from_xyz[0]), float(from_xyz[1]), float(from_xyz[2]))
    to_pt   = carb.Float3(float(to_xyz[0]),   float(to_xyz[1]),   float(to_xyz[2]))
    try:
        path = nm.query_shortest_path(from_pt, to_pt, agent_radius=0.3)
    except Exception:
        return None
    if path is None:
        return None
    # extract points – handle both INavMeshPath objects and plain lists
    if hasattr(path, 'get_points'):
        pts = path.get_points()
    elif hasattr(path, 'points'):
        pts = path.points
    else:
        pts = path
    if pts is None or len(pts) == 0:
        return None
    return [[float(p[0]), float(p[1]), float(p[2])] for p in pts]


# ── scriptData command writing ──────────────────────────────────────────────

def write_commands_to_scriptdata(app, char_prims: list[str], nm,
                                 spawn_positions: dict,
                                 n_waypoints: int = 3):
    """Write GoTo command strings + pre-computed path data to each character's
    SkelRoot omni:scripting:scriptData and omni:scripting:pathData attributes.

    scriptData  → ["GoTo x y z _", ...]          (consumed by CharacterBehavior)
    pathData    → JSON string {"0": [[x,y,z],...], "1": [...], ...}
                  (consumed by the modified GoTo command class)
    """
    import json as _json
    stage = omni.usd.get_context().get_stage()

    for char_prim_path in char_prims:
        char_prim = stage.GetPrimAtPath(char_prim_path)
        skelroot  = None
        for desc in Usd.PrimRange(char_prim):
            if desc.GetTypeName() == "SkelRoot":
                skelroot = desc
                break
        if skelroot is None:
            print(f"[CMD] No SkelRoot under {char_prim_path}, skipping")
            continue

        # Snap starting position onto the NavMesh
        current_pos = spawn_positions.get(char_prim_path)
        if current_pos is None:
            current_pos = safe_query_random_point(nm)
        try:
            result = nm.query_closest_point(current_pos, 0.5)
            if result and result[0] is not None:
                current_pos = result[0]
        except Exception:
            pass

        command_strings: list[str] = []
        path_data: dict[int, list[list[float]]] = {}   # goto_index → path points
        goto_index = 0
        prev_pos   = current_pos

        for _ in range(n_waypoints):
            target = find_reachable_point(nm, prev_pos)
            if target is None:
                continue
            x, y, z = float(target[0]), float(target[1]), float(target[2])
            command_strings.append(f"GoTo {x} {y} {z} _")

            # Pre-compute path for this GoTo
            path_pts = compute_goto_path(nm, prev_pos, target)
            if path_pts:
                path_data[goto_index] = path_pts
                print(f"[CMD] GoTo#{goto_index} path: {len(path_pts)} points")
            else:
                print(f"[CMD] GoTo#{goto_index} no pre-computed path (will plan on-the-fly)")

            goto_index += 1
            prev_pos    = target

        # Write scriptData
        sd_attr = skelroot.GetAttribute("omni:scripting:scriptData")
        if not sd_attr:
            sd_attr = skelroot.CreateAttribute("omni:scripting:scriptData",
                                               Sdf.ValueTypeNames.StringArray)
        sd_attr.Set(command_strings)

        # Write pathData
        if path_data:
            pd_attr = skelroot.GetAttribute("omni:scripting:pathData")
            if not pd_attr:
                pd_attr = skelroot.CreateAttribute("omni:scripting:pathData",
                                                   Sdf.ValueTypeNames.String)
            pd_attr.Set(_json.dumps(path_data))
            print(f"[CMD] Saved {len(path_data)} pre-computed paths for {char_prim_path}")

        readback = skelroot.GetAttribute("omni:scripting:scriptData").Get()
        print(f"[CMD] {char_prim_path}: {len(readback) if readback else 0} commands, "
              f"first={readback[0] if readback else None}")


# ── viewport framing ────────────────────────────────────────────────────────

def frame_viewport_on(prim_path: str):
    try:
        from omni.kit.viewport.utility import get_active_viewport
        vp = get_active_viewport()
        if vp is None:
            return
        omni.kit.commands.execute("FramePrimsCommand",
                                  prim_to_move=vp.camera_path.pathString,
                                  prims_to_frame=[prim_path],
                                  zoom=0.45)
        print(f"[View] framed on {prim_path}")
    except Exception as e:
        print(f"[View] frame failed: {e}")