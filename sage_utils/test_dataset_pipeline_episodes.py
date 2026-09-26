"""
test_dataset_pipeline.py
────────────────────────
Load a USDA scene + a pre-generated episode JSON and run the people
simulation.  After all characters finish their commands, the simulation
ends automatically.

Supports batch processing of multiple episodes via:
    --episode_start_index  START
    --episode_end_index    END

Episode resolution (single episode mode, without --episode_start_index/--episode_end_index)
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

    # Batch run episodes 0 to 5 (inclusive)
    python test_dataset_pipeline.py \
        --usda /workspace/SAGE-3D_Official/SAGE-3D_data/usda/839873.usda \
        --episode_start_index 0 --episode_end_index 5

    # Run only episode 3
    python test_dataset_pipeline.py \
        --usda /workspace/SAGE-3D_Official/SAGE-3D_data/usda/839873.usda \
        --episode_start_index 3

    # Custom episodes root (overrides the auto-derived path)
    python test_dataset_pipeline.py \
        --usda ... --episodes_root /my/episodes --episode_start_index 0 --episode_end_index 2
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

    # Batch range
    p.add_argument("--episode_start_index", type=int, default=None,
                   help="Start index for batch episode execution (inclusive).")
    p.add_argument("--episode_end_index", type=int, default=None,
                   help="End index for batch episode execution (inclusive). "
                        "If omitted and start is given, only that episode is run.")

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
# SimulationApp – created once, shared across episodes
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
import carb.settings
from omni.anim.people.settings import PeopleSettings
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
# Episode resolution helpers
# ═══════════════════════════════════════════════════════════════════════════
def _get_episode_dir() -> str | None:
    """Return the directory where episode JSONs are expected."""
    scene_id = os.path.splitext(os.path.basename(ARGS.usda))[0]
    if ARGS.episodes_root:
        ep_dir = os.path.join(ARGS.episodes_root, scene_id)
    else:
        usda_dir = os.path.dirname(os.path.abspath(ARGS.usda))
        parent   = os.path.dirname(usda_dir)
        ep_dir   = os.path.join(parent, "episodes", scene_id)
    return ep_dir


def _resolve_single_episode() -> str | None:
    """Original single‑episode resolution (no batch mode)."""
    if ARGS.episode_json:
        if os.path.exists(ARGS.episode_json):
            return ARGS.episode_json
        print(f"[Episode] --episode_json not found: {ARGS.episode_json}")
        return None

    ep_dir = _get_episode_dir()
    print(f"[Episode] Looking for episodes in: {ep_dir}")

    if not os.path.isdir(ep_dir):
        print(f"[Episode] Directory not found — will run random-walk fallback.")
        return None

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
    chosen = candidates[2]  # original behavior
    print(f"[Episode] Auto-selected: {chosen}  "
          f"(total available: {len(candidates)})")
    return chosen


def _resolve_batch_episodes() -> list[str]:
    """Return a sorted list of episode JSON paths for the given index range."""
    start = ARGS.episode_start_index
    end   = ARGS.episode_end_index

    # Determine actual start/end
    if start is None and end is None:
        return []   # should not happen, caller checks
    if start is None:
        start = 0
    if end is None:
        end = start

    if start > end:
        print(f"[Batch] Invalid range: start={start} > end={end}")
        return []

    ep_dir = _get_episode_dir()
    if not os.path.isdir(ep_dir):
        print(f"[Batch] Episode directory not found: {ep_dir}")
        return []

    all_jsons = glob.glob(os.path.join(ep_dir, "episode_*.json"))
    # Filter by index
    selected = []
    for path in all_jsons:
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]   # episode_XX
        try:
            idx = int(stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if start <= idx <= end:
            selected.append((idx, path))

    selected.sort(key=lambda x: x[0])
    return [p for _, p in selected]


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
# Auto‑finish helper
# ═══════════════════════════════════════════════════════════════════════════
def _all_characters_finished(char_prims: list[str]) -> bool:
    """Return True if every character's behavior script has finished."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return False

    for p in char_prims:
        prim = stage.GetPrimAtPath(p)
        if not prim.IsValid():
            continue

        skelroot = None
        for desc in Usd.PrimRange(prim):
            if desc.GetTypeName() == "SkelRoot":
                skelroot = desc
                break
        if skelroot is None:
            return False

        state_attr = skelroot.GetAttribute("omni:scripting:scriptState")
        if not state_attr:
            return False
        state = state_attr.Get()
        if state != "finished":
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Single episode runner (called once per episode, may be reused)
# ═══════════════════════════════════════════════════════════════════════════
def run_one_episode(episode_json_path: str | None) -> bool:
    """
    Run a single episode (or random walk if json_path is None).
    Returns True if characters finished normally, False on error/timeout.
    """
    # ── Load episode data ─────────────────────────────────────────────────
    episode: dict | None = None
    if episode_json_path:
        with open(episode_json_path, encoding="utf-8") as f:
            raw = json.load(f)
        episode = raw.get("episode")
        if episode is None:
            print(f"[Episode] WARNING: 'episode' key missing in {episode_json_path}, "
                  "falling back to random walk.")
        else:
            ep_id      = episode.get("episode_id", "?")
            num_chars  = episode["characters"].get("num_characters", "?")
            print(f"[Episode] Loaded episode {ep_id}  "
                  f"({num_chars} character(s))  ← {episode_json_path}")

    # ── Open scene ────────────────────────────────────────────────────────
    if not open_stage(ARGS.usda):
        return False

    # ── NavMesh ───────────────────────────────────────────────────────────
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
        print("[FATAL-ish] NavMesh bake failed — cannot run this episode.")
        return False

    nm = inav.get_navmesh()
    _hide_navmesh_volumes()
    _update(10)

    # ── Physics world ─────────────────────────────────────────────────────
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

    # ── Character pool ────────────────────────────────────────────────────
    char_pool = load_character_assets()
    print(f"[PEOPLE] {len(char_pool)} asset(s) available.")

    # ── Spawn + command setup ─────────────────────────────────────────────
    if episode is not None:
        # Optional: check spawn points vs navmesh
        spawn_positions = episode["characters"]["spawn_positions"]
        for char_name, sp_data in spawn_positions.items():
            pos = sp_data["pos"]
            test_pt = carb.Float3(float(pos[0]), float(pos[1]), float(pos[2]))
            result = nm.query_closest_point(test_pt, 1.0)
            snapped = result[0] if result else None
            print(f"[NavMesh Check] {char_name} spawn {pos} → "
                  f"closest NavMesh point: {snapped}")
            if snapped:
                center = carb.Float3(0.0, 5.0, 0.0)
                path = nm.query_shortest_path(snapped, center, agent_radius=0.5)
                print(f"[NavMesh Check] path to center: "
                      f"{'OK' if path else 'FAILED'}")

        print("\n[Mode] EPISODE — spawning from JSON data.")
        char_prims = _setup_characters_from_episode(episode, char_pool)
    else:
        print("\n[Mode] RANDOM WALK — no episode JSON available.")
        char_prims: list[str] = []
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

    # ── Frame viewport on first character ────────────────────────────────
    if char_prims:
        frame_viewport_on(char_prims[0])
        _update(5)

    # ── Play and wait for finish ──────────────────────────────────────────
    tl = omni.timeline.get_timeline_interface()
    tl.set_current_time(0.0)
    tl.play()
    _update(60)   # let scripts start

    print("\n[HOLD] Waiting for all characters to finish their commands...\n")
    timeout_frames = 3600   # ~60 seconds
    finished_normally = False

    for _ in range(timeout_frames):
        if _all_characters_finished(char_prims):
            print("[Done] All characters have finished their commands.")
            finished_normally = True
            break
        simulation_app.update()
    else:
        print("[Timeout] Characters did not finish within the time limit.")

    tl.stop()
    return finished_normally


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> int:
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)

    # Determine if batch mode is requested
    batch_mode = (ARGS.episode_start_index is not None) or (ARGS.episode_end_index is not None)

    if batch_mode:
        episode_paths = _resolve_batch_episodes()
        if not episode_paths:
            print("[Batch] No episodes found in the specified range. Exiting.")
            simulation_app.close()
            return 1

        total = len(episode_paths)
        print(f"\n[Batch] Running {total} episode(s).\n")
        success_count = 0

        for idx, ep_path in enumerate(episode_paths, start=1):
            print(f"\n{'='*60}")
            print(f"[Batch] Episode {idx}/{total}: {os.path.basename(ep_path)}")
            print(f"{'='*60}\n")
            ok = run_one_episode(ep_path)
            if ok:
                success_count += 1
                print(f"[Batch] Episode {os.path.basename(ep_path)} finished successfully.")
            else:
                print(f"[Batch] Episode {os.path.basename(ep_path)} FAILED or timed out.")

        print(f"\n[Batch] Summary: {success_count}/{total} episode(s) succeeded.\n")
        simulation_app.close()
        return 0 if success_count == total else 1

    else:
        # Single episode mode (original behavior)
        episode_json_path = _resolve_single_episode()
        ok = run_one_episode(episode_json_path)
        simulation_app.close()
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())