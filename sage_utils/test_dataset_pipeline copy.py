"""
test_dataset_pipeline.py
────────────────────────
Load a USDA scene + a pre-generated episode JSON and run the people
simulation interactively.

Episode resolution
------------------
Given  --usda /workspace/SAGE-3D_Official/SAGE-3D_data/usda/839873.usda

The script looks for episodes in:
    /workspace/SAGE-3D_Official/SAGE-3D_data/episodes/839873/

  1. If --episode_json is given explicitly → use that file.
  2. Otherwise scan the above directory for episode_*.json, pick the
     lowest-numbered one (episode_0.json first, then episode_1.json, …).
  3. If the directory is empty / doesn't exist → run without episode data
     (characters walk to random NavMesh points, same as test_navmesh_usda.py).

Usage
-----
    # Minimal – auto-pick first episode
    python test_dataset_pipeline.py \
        --usda /workspace/SAGE-3D_Official/SAGE-3D_data/usda/839873.usda

    # Explicit episode file
    python test_dataset_pipeline.py \
        --usda /workspace/SAGE-3D_Official/SAGE-3D_data/usda/839873.usda \
        --episode_json /workspace/SAGE-3D_Official/SAGE-3D_data/episodes/839873/episode_0.json

    # Custom episodes root (overrides the auto-derived path)
    python test_dataset_pipeline.py \
        --usda ... --episodes_root /my/episodes
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import random
import sys


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--usda", required=True)

    # Episode selection
    p.add_argument("--episode_json",  default=None,
                   help="Explicit path to an episode JSON file. "
                        "If omitted the script auto-picks from episodes_root.")
    p.add_argument("--episodes_root", default=None,
                   help="Root directory that contains per-scene sub-folders "
                        "(e.g. .../episodes/).  Defaults to "
                        "<usda_parent>/../episodes/.")

    # NavMesh
    p.add_argument("--collision_root",          default="/World/scene_collision")
    p.add_argument("--volume_padding",          type=float, default=1.2)
    p.add_argument("--fallback_size",           type=float, default=100.0)
    p.add_argument("--warmup_frames",           type=int,   default=60)

    # Fallback random-walk (used when no episode JSON is found)
    p.add_argument("--num_people",    type=int, default=1)
    p.add_argument("--num_waypoints", type=int, default=3)

    # Misc
    p.add_argument("--seed",              type=int, default=0)
    p.add_argument("--semantic_map_json", default=None)
    p.add_argument("--cache_dir",         default=None)
    p.add_argument("--force_rebake",      action="store_true")
    return p.parse_args()


ARGS = parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# SimulationApp
# ═══════════════════════════════════════════════════════════════════════════
from isaacsim import SimulationApp

_exp = os.environ.get("EXP_PATH")
if _exp is None:
    print("[FATAL] EXP_PATH env var not set.")
    sys.exit(1)

CUSTOM_APP_PATH = os.path.join(
    _exp, "isaacsim.exp.action_and_event_data_generation.base.kit")

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
import omni.usd
import omni.timeline
from pxr import Sdf, Usd

# Make sage_utils importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # /workspace/FLUX

from sage_utils.navmesh_utils import (
    cache_paths, bake_navmesh,
    safe_query_random_point,
)
from sage_utils.people_utils import (
    load_character_assets, spawn_character,
    load_default_skeleton_and_animations,
    bind_animation_graph_to_characters,
    attach_behavior_scripts_to_characters,
    write_commands_to_scriptdata,
    frame_viewport_on,
    spawn_on_navmesh,
)
from isaacsim.replicator.agent.core.settings import PrimPaths, BehaviorScriptPaths
from isaacsim.replicator.agent.core.stage_util import CharacterUtil


# ═══════════════════════════════════════════════════════════════════════════
# Episode resolution
# ═══════════════════════════════════════════════════════════════════════════
def _resolve_episode_json() -> str | None:
    """Return path to the episode JSON to use, or None if not found."""
    # 1. Explicit override
    if ARGS.episode_json:
        if os.path.exists(ARGS.episode_json):
            return ARGS.episode_json
        print(f"[Episode] --episode_json not found: {ARGS.episode_json}")
        return None

    # 2. Auto-derive directory from scene stem
    scene_id = os.path.splitext(os.path.basename(ARGS.usda))[0]   # "839873"
    if ARGS.episodes_root:
        ep_dir = os.path.join(ARGS.episodes_root, scene_id)
    else:
        usda_dir = os.path.dirname(os.path.abspath(ARGS.usda))
        parent   = os.path.dirname(usda_dir)
        ep_dir   = os.path.join(parent, "episodes", scene_id)

    print(f"[Episode] Looking for episodes in: {ep_dir}")

    if not os.path.isdir(ep_dir):
        print(f"[Episode] Directory not found — will run random-walk fallback.")
        return None

    # Collect episode_*.json and sort by episode id number
    candidates = glob.glob(os.path.join(ep_dir, "episode_*.json"))
    if not candidates:
        print(f"[Episode] No episode_*.json files found — will run random-walk fallback.")
        return None

    def _ep_id(path: str) -> int:
        try:
            return int(os.path.splitext(os.path.basename(path))[0].split("_")[1])
        except (IndexError, ValueError):
            return 99999

    candidates.sort(key=_ep_id)
    chosen = candidates[0]
    print(f"[Episode] Auto-selected: {chosen}  "
          f"(total available: {len(candidates)})")
    return chosen


# ═══════════════════════════════════════════════════════════════════════════
# Stage helpers
# ═══════════════════════════════════════════════════════════════════════════
def _update(n: int = 1):
    for _ in range(n):
        simulation_app.update()

def _hide_navmesh_volumes():
    """Make all NavMeshVolume prims invisible in the viewport."""
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    hidden = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() == "NavMeshVolume":
            imageable = UsdGeom.Imageable(prim)
            imageable.MakeInvisible()
            hidden += 1
    print(f"[NavMesh] Hidden {hidden} NavMeshVolume prim(s).")

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
            print("[WARN] Stage still loading after 30s.")
            break
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[FATAL] Stage failed to open.")
        return False
    print(f"[STAGE] Loaded. Default prim: {stage.GetDefaultPrim().GetPath()}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Episode-based character setup
# ═══════════════════════════════════════════════════════════════════════════
def _setup_characters_from_episode(episode: dict, char_pool: list[str]) -> list[str]:
    """Spawn characters at positions defined in the episode JSON and write
    GoTo commands + pre-computed paths into scriptData / pathData."""
    import json as _json

    chars_data      = episode["characters"]
    spawn_positions = chars_data["spawn_positions"]
    commands_dict   = chars_data["commands"]
    char_prims: list[str] = []

    stage = omni.usd.get_context().get_stage()

    for i, (char_name, spawn_data) in enumerate(spawn_positions.items()):
        pos = spawn_data["pos"]
        usd = char_pool[i % len(char_pool)]
        prim_path = spawn_character(simulation_app, i, usd, pos)
        char_prims.append(prim_path)
        print(f"[Episode] Spawned {char_name} at {pos}")

    # Skeleton + animation graph must come before scriptData
    load_default_skeleton_and_animations(simulation_app)
    _update(30)
    bind_animation_graph_to_characters(simulation_app)
    _update(60)

    # Write commands + pre-computed paths per character
    _write_episode_commands(stage, spawn_positions, commands_dict)
    _update(10)

    attach_behavior_scripts_to_characters(simulation_app)
    _update(60)

    return char_prims


def _write_episode_commands(stage, spawn_positions: dict, commands_dict: dict):
    """Write scriptData and pathData from episode JSON into each SkelRoot."""
    import json as _json

    parent_path = str(PrimPaths.characters_parent_path())

    for char_name, cmds in commands_dict.items():
        # Locate the SkelRoot for this character
        char_prim_path = f"{parent_path}/{char_name}"
        char_prim = stage.GetPrimAtPath(char_prim_path)
        if not char_prim.IsValid():
            print(f"[Episode] WARNING: prim not found for {char_name}, skipping.")
            continue

        skelroot = None
        for desc in Usd.PrimRange(char_prim):
            if desc.GetTypeName() == "SkelRoot":
                skelroot = desc
                break
        if skelroot is None:
            print(f"[Episode] WARNING: no SkelRoot under {char_prim_path}, skipping.")
            continue

        command_strings: list[str] = []
        path_data: dict[int, list]  = {}
        goto_index = 0

        for cmd in cmds:
            cmd_name = cmd.get("cmd", "")
            params   = cmd.get("params", [])
            command_strings.append(f"{cmd_name} " + " ".join(str(p) for p in params))

            if cmd_name == "GoTo":
                if "path" in cmd and len(cmd["path"]) > 0:
                    path_data[goto_index] = cmd["path"]
                    print(f"[Episode]   {char_name} GoTo#{goto_index}: "
                          f"{len(cmd['path'])} path points")
                else:
                    print(f"[Episode]   {char_name} GoTo#{goto_index}: "
                          f"no pre-computed path (on-the-fly)")
                goto_index += 1

        # scriptData
        sd_attr = skelroot.GetAttribute("omni:scripting:scriptData")
        if not sd_attr:
            sd_attr = skelroot.CreateAttribute("omni:scripting:scriptData",
                                               Sdf.ValueTypeNames.StringArray)
        sd_attr.Set(command_strings)

        # pathData
        if path_data:
            pd_attr = skelroot.GetAttribute("omni:scripting:pathData")
            if not pd_attr:
                pd_attr = skelroot.CreateAttribute("omni:scripting:pathData",
                                                   Sdf.ValueTypeNames.String)
            pd_attr.Set(_json.dumps(path_data))

        print(f"[Episode] {char_name}: {len(command_strings)} commands, "
              f"{len(path_data)} pre-computed paths written.")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> int:
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)

    # ── Resolve episode ────────────────────────────────────────────────────
    episode_json_path = _resolve_episode_json()
    episode: dict | None = None

    if episode_json_path:
        with open(episode_json_path, encoding="utf-8") as f:
            raw = json.load(f)
        episode = raw.get("episode")
        if episode is None:
            print(f"[Episode] WARNING: 'episode' key missing in {episode_json_path}, "
                  "falling back to random walk.")
            episode = None
        else:
            ep_id      = episode.get("episode_id", "?")
            num_chars  = episode["characters"].get("num_characters", "?")
            print(f"[Episode] Loaded episode {ep_id}  "
                  f"({num_chars} character(s))  ← {episode_json_path}")

    # ── Open scene ─────────────────────────────────────────────────────────
    if not open_stage(ARGS.usda):
        return 1

    # ── NavMesh ────────────────────────────────────────────────────────────
    navvols_usda = cache_paths(ARGS.usda, ARGS.cache_dir)
    print(f"[Cache] NavVols USDA : {navvols_usda}")

    nav_ok, inav = bake_navmesh(
        app                    = simulation_app,
        navvols_usda           = navvols_usda,
        force_rebake           = ARGS.force_rebake,
        collision_root         = ARGS.collision_root,
        volume_padding         = ARGS.volume_padding,
        fallback_size          = ARGS.fallback_size,
        warmup_frames          = ARGS.warmup_frames,
        semantic_map_json      = ARGS.semantic_map_json,
    )
    if not nav_ok:
        print("\n[FATAL-ish] NavMesh bake failed — holding for inspection.")
        try:
            while simulation_app.is_running():
                simulation_app.update()
        except KeyboardInterrupt:
            pass
        simulation_app.close()
        return 2

    nm = inav.get_navmesh()

    # ── Hide NavMesh volumes ───────────────────────────────────────────────
    _hide_navmesh_volumes()
    _update(10)

    # ── Physics world ──────────────────────────────────────────────────────
    print("\n[World] Creating World + physicsScene...")
    from isaacsim.core.api import World
    world = World(physics_dt=1.0/60.0, rendering_dt=1.0/30.0)
    world.initialize_physics()
    world.reset()
    _update(10)
    print("[World] Physics initialized.")

    rp = safe_query_random_point(nm)
    if rp:
        print(f"[Check] NavMesh post-World: {tuple(rp)}")

    # ── Character pool ─────────────────────────────────────────────────────
    char_pool = load_character_assets()
    print(f"[PEOPLE] {len(char_pool)} asset(s) available.")

    # ── Spawn + command setup ──────────────────────────────────────────────
    if episode is not None:

        spawn_positions = episode["characters"]["spawn_positions"]
        for char_name, sp_data in spawn_positions.items():
            pos = sp_data["pos"]
            test_pt = carb.Float3(float(pos[0]), float(pos[1]), float(pos[2]))
            result = nm.query_closest_point(test_pt, 1.0)
            snapped = result[0] if result else None
            print(f"[NavMesh Check] {char_name} spawn {pos} → "
                  f"closest NavMesh point: {snapped}")
            if snapped:
                # 测试能否从 snapped 点到场景中心
                center = carb.Float3(0.0, 5.0, 0.0)
                path = nm.query_shortest_path(snapped, center, agent_radius=0.5)
                print(f"[NavMesh Check] path to center: "
                      f"{'OK' if path else 'FAILED'}")

        # ── Episode mode ──────────────────────────────────────────────────
        print("\n[Mode] EPISODE — spawning from JSON data.")
        char_prims = _setup_characters_from_episode(episode, char_pool)

    else:
        # ── Random-walk fallback ──────────────────────────────────────────
        print("\n[Mode] RANDOM WALK — no episode JSON available.")
        char_prims:           list[str]        = []
        char_spawn_positions: dict[str, tuple] = {}

        for i in range(ARGS.num_people):
            usd   = random.choice(char_pool)
            spawn = spawn_on_navmesh(nm)
            prim  = spawn_character(simulation_app, i, usd, spawn)
            char_prims.append(prim)
            char_spawn_positions[prim] = spawn

        load_default_skeleton_and_animations(simulation_app)
        _update(30)
        bind_animation_graph_to_characters(simulation_app)
        _update(30)

        write_commands_to_scriptdata(
            simulation_app, char_prims, nm, char_spawn_positions,
            n_waypoints=ARGS.num_waypoints,
        )
        _update(10)
        attach_behavior_scripts_to_characters(simulation_app)
        _update(60)

    # ── Frame viewport on first character ─────────────────────────────────
    if char_prims:
        frame_viewport_on(char_prims[0])
        _update(5)

    # ── Play ───────────────────────────────────────────────────────────────
    tl = omni.timeline.get_timeline_interface()
    tl.set_current_time(0.0)
    tl.play()
    # _update(300)

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