"""
batch_test_episodes.py
──────────────────────
Run multiple pre-generated episode JSONs inside a *single* Isaac Sim process.

Behaviour:
  • Same scene across episodes → reuse open stage + NavMesh (no re-bake).
  • Different scene           → open new stage + re-bake NavMesh.
  • After all characters finish their command queues (scriptData cleared),
    wait INTER_EPISODE_PAUSE seconds, then tear down characters and load
    the next episode.

Episode completion detection
────────────────────────────
omni.anim.people's CharacterBehavior script drains the SkelRoot attribute
  omni:scripting:scriptData
as it executes commands.  When the list reaches length 0 (or the attribute
disappears), that character is done.  We poll this every frame; once ALL
spawned characters are done we consider the episode finished.

Fix for "characters don't move after episode 1"
────────────────────────────────────────────────
Root cause: when a character prim is deleted and a new one spawned,
the new SkelRoot has no OmniScriptingAPI schema yet.  Calling
attach_behavior_scripts_to_characters() triggers ApplyScriptingAPICommand
on the fresh SkelRoot, which initialises omni:scripting:scriptData to []
as part of schema creation — wiping any data written beforehand.

Fix: always attach_behavior_scripts first (schema apply), then write
scriptData/pathData.  This is safe because attach only resets the
attribute when it has to *create* it via ApplyScriptingAPICommand.
After the first call the attribute exists and subsequent Set() calls
are stable.

Usage
-----
    # Run episodes 0-4 for one scene
    /isaac-sim/python.sh batch_test_episodes.py \\
        --episodes_dir  /workspace/SAGE-3D_Official/SAGE-3D_data/episodes/839873 \\
        --usda          /workspace/SAGE-3D_Official/SAGE-3D_data/usda/839873.usda \\
        --episode_ids   0 1 2 3 4

    # Run ALL episodes found in the directory
    /isaac-sim/python.sh batch_test_episodes.py \\
        --episodes_dir  /workspace/SAGE-3D_Official/SAGE-3D_data/episodes/839873 \\
        --usda          /workspace/SAGE-3D_Official/SAGE-3D_data/usda/839873.usda

    # Multiple scenes (same process, stage re-opened between scenes)
    /isaac-sim/python.sh batch_test_episodes.py \\
        --episodes_dir  /workspace/SAGE-3D_Official/SAGE-3D_data/episodes \\
        --usda_dir      /workspace/SAGE-3D_Official/SAGE-3D_data/usda \\
        --scene_ids     839873 839926
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time


# ─── timing constants ────────────────────────────────────────────────────────
INTER_EPISODE_PAUSE   = 3.0    # seconds to wait after all chars finish
EPISODE_TIMEOUT_S     = 300.0  # hard timeout per episode (seconds wall-clock)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Batch episode test runner (single Isaac Sim process).")

    # ── Single-scene mode ──────────────────────────────────────────────────
    p.add_argument("--usda",          default=None,
                   help="Path to a single .usda scene file.")
    p.add_argument("--episodes_dir",  default=None,
                   help="Directory containing episode_*.json for a single scene.")
    p.add_argument("--episode_ids",   nargs="*", type=int, default=None,
                   help="Episode IDs to run (default: all found).")

    # ── Multi-scene mode ───────────────────────────────────────────────────
    p.add_argument("--usda_dir",      default=None,
                   help="Directory of .usda files (multi-scene mode).")
    p.add_argument("--episodes_root", default=None,
                   help="Root of episodes/ tree; sub-dirs named by scene_id.")
    p.add_argument("--scene_ids",     nargs="*", default=None,
                   help="Scene IDs to process (default: all matched).")

    # ── NavMesh ────────────────────────────────────────────────────────────
    p.add_argument("--collision_root",  default="/World/scene_collision")
    p.add_argument("--volume_padding",  type=float, default=1.2)
    p.add_argument("--fallback_size",   type=float, default=100.0)
    p.add_argument("--warmup_frames",   type=int,   default=60)
    p.add_argument("--cache_dir",       default=None)
    p.add_argument("--force_rebake",    action="store_true")

    # ── Misc ───────────────────────────────────────────────────────────────
    p.add_argument("--inter_episode_pause", type=float, default=INTER_EPISODE_PAUSE,
                   help=f"Seconds to wait after episode ends before starting next "
                        f"(default {INTER_EPISODE_PAUSE}).")
    p.add_argument("--episode_timeout",    type=float, default=EPISODE_TIMEOUT_S,
                   help=f"Hard timeout per episode in seconds "
                        f"(default {EPISODE_TIMEOUT_S}).")
    p.add_argument("--headless",           action="store_true",
                   help="Run without a display window.")
    return p.parse_args()


ARGS = parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# SimulationApp  (must be created before any omni imports)
# ═══════════════════════════════════════════════════════════════════════════
from isaacsim import SimulationApp

_exp = os.environ.get("EXP_PATH")
if _exp is None:
    print("[FATAL] EXP_PATH env var not set."); sys.exit(1)

simulation_app = SimulationApp(
    launch_config={
        "renderer":               "RayTracedLighting",
        "headless":               ARGS.headless,
        "enable_cameras":         True,
        "crash_reporter/enabled": False,
        "crash_reporter/skip_old_dump_upload": True,
    },
    experience=os.path.join(
        _exp, "isaacsim.exp.action_and_event_data_generation.base.kit"),
)

# ─── post-launch imports ──────────────────────────────────────────────────
import carb
import omni.usd
import omni.timeline
from pxr import Sdf, Usd, UsdGeom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sage_utils.navmesh_utils import cache_paths, bake_navmesh
from sage_utils.people_utils import (
    load_character_assets, spawn_character,
    load_default_skeleton_and_animations,
    bind_animation_graph_to_characters,
    attach_behavior_scripts_to_characters,
    frame_viewport_on,
)
from isaacsim.replicator.agent.core.settings import PrimPaths
from isaacsim.replicator.agent.core.stage_util import CharacterUtil


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _update(n: int = 1):
    for _ in range(n):
        simulation_app.update()


def _hide_navmesh_volumes():
    stage = omni.usd.get_context().get_stage()
    if stage is None: return
    hidden = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() == "NavMeshVolume":
            UsdGeom.Imageable(prim).MakeInvisible(); hidden += 1
    print(f"[NavMesh] Hidden {hidden} NavMeshVolume prim(s).")


def _open_stage(usda_path: str) -> bool:
    if not os.path.exists(usda_path):
        print(f"[FATAL] USDA not found: {usda_path}"); return False
    print(f"\n[STAGE] Opening: {usda_path}")
    omni.usd.get_context().open_stage(usda_path)
    _update(30)
    waited = 0
    while omni.usd.get_context().get_stage_loading_status()[1] > 0:
        simulation_app.update(); waited += 1
        if waited > 3000: print("[WARN] Stage loading timeout."); break
    return omni.usd.get_context().get_stage() is not None


def _ep_sort_key(path: str) -> int:
    m = re.search(r"episode_(\d+)\.json$", os.path.basename(path))
    return int(m.group(1)) if m else 99999


def _collect_episodes(episodes_dir: str,
                      episode_ids: list[int] | None) -> list[str]:
    """Return sorted list of episode JSON paths."""
    all_eps = sorted(glob.glob(os.path.join(episodes_dir, "episode_*.json")),
                     key=_ep_sort_key)
    if episode_ids is not None:
        id_set = set(episode_ids)
        all_eps = [p for p in all_eps
                   if _ep_sort_key(p) in id_set]
    return all_eps


# ─── character deletion ───────────────────────────────────────────────────
def _delete_all_characters():
    """Remove all character prims under the characters parent path."""
    stage = omni.usd.get_context().get_stage()
    if stage is None: return
    parent = str(PrimPaths.characters_parent_path())
    parent_prim = stage.GetPrimAtPath(parent)
    if not parent_prim or not parent_prim.IsValid(): return
    children = list(parent_prim.GetChildren())
    for child in children:
        stage.RemovePrim(child.GetPath())
        print(f"[CLEANUP] Removed {child.GetPath()}")
    _update(10)
    print(f"[CLEANUP] Deleted {len(children)} character(s).")


# ─── episode duration estimation ──────────────────────────────────────────
WALK_SPEED_M_S   = 1.4   # omni.anim.people default walking speed
GOTO_TURN_SLACK  = 3.0   # extra seconds per GoTo for turning / deceleration
EPISODE_BUFFER_S = 15.0  # flat buffer added on top of estimated total

def _estimate_episode_duration(episode: dict) -> float:
    """
    Estimate wall-clock seconds needed for an episode to finish.

    For each character:
      - GoTo       : path_length / WALK_SPEED + GOTO_TURN_SLACK
      - Idle       : params[0] seconds
      - LookAround : params[0] seconds
    Returns max over all characters + EPISODE_BUFFER_S.
    """
    chars_data    = episode.get("characters", {})
    commands_dict = chars_data.get("commands", {})
    max_dur = 0.0

    for char_name, cmds in commands_dict.items():
        char_dur = 0.0
        for cmd in cmds:
            name   = cmd.get("cmd", "")
            params = cmd.get("params", [])
            if name == "GoTo":
                path = cmd.get("path", [])
                length = 0.0
                for i in range(1, len(path)):
                    dx = path[i][0] - path[i-1][0]
                    dy = path[i][1] - path[i-1][1]
                    length += (dx*dx + dy*dy) ** 0.5
                char_dur += length / WALK_SPEED_M_S + GOTO_TURN_SLACK
            elif name in ("Idle", "LookAround"):
                try:
                    char_dur += float(params[0])
                except (IndexError, ValueError):
                    char_dur += 5.0   # fallback if param missing
        max_dur = max(max_dur, char_dur)

    estimated = max_dur + EPISODE_BUFFER_S
    return max(estimated, 30.0)   # at least 30 s


# ─── completion detection (position-stillness based) ──────────────────────
_STILL_THRESHOLD_M  = 0.05   # movement < this in one interval → "still"
_STILL_REQUIRED_S   = 5.0    # must be still for this long to count as done
_STILL_POLL_S       = 1.0    # position poll interval (wall-clock seconds)

def _get_char_pos(prim_path: str):
    """Return (x, y, z) world position of a character prim, or None."""
    try:
        stage = omni.usd.get_context().get_stage()
        if stage is None: return None
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid(): return None
        xf = UsdGeom.Xformable(prim)
        if xf:
            m = xf.ComputeLocalToWorldTransform(0)
            t = m.ExtractTranslation()
            return (float(t[0]), float(t[1]), float(t[2]))
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Episode setup  ← KEY FIX HERE
# ═══════════════════════════════════════════════════════════════════════════
def _setup_episode(episode: dict, char_pool: list[str]) -> list[str]:
    """Spawn characters and write commands from an episode dict.

    Correct order (fixes "characters freeze after episode 0"):
    ──────────────────────────────────────────────────────────
    The problem: when character prims are deleted between episodes and new
    ones are spawned, the SkelRoot has no OmniScriptingAPI schema yet.
    attach_behavior_scripts_to_characters() calls ApplyScriptingAPICommand
    on that fresh SkelRoot, which *creates* omni:scripting:scriptData and
    initialises it to [] — wiping any data we wrote before this call.

    Solution: always call attach_behavior_scripts BEFORE writing
    scriptData/pathData.  After the schema is applied, Set() on the
    existing attribute is stable.

      1. spawn
      2. load_default_skeleton_and_animations
      3. bind_animation_graph_to_characters
      4. ★ attach_behavior_scripts_to_characters   ← schema apply / reset
      5. ★ write scriptData + pathData             ← safe now
      6. update frames so CharacterBehavior picks up the queue

    Returns list of character prim paths.
    """
    import json as _json

    chars_data      = episode["characters"]
    spawn_positions = chars_data["spawn_positions"]
    commands_dict   = chars_data["commands"]
    char_prims: list[str] = []

    # 1. Spawn ----------------------------------------------------------------
    for i, (char_name, spawn_data) in enumerate(spawn_positions.items()):
        pos = spawn_data["pos"]
        usd = char_pool[i % len(char_pool)]
        prim_path = spawn_character(simulation_app, i, usd, pos)
        char_prims.append(prim_path)
        print(f"[Episode] Spawned {char_name} at {pos}")

    # 2 & 3. Skeleton + AnimGraph --------------------------------------------
    load_default_skeleton_and_animations(simulation_app); _update(30)
    bind_animation_graph_to_characters(simulation_app);   _update(60)

    # 4. ★ Attach behavior scripts FIRST ------------------------------------
    # ApplyScriptingAPICommand runs here and resets scriptData → []
    # on any SkelRoot that didn't already have the schema.
    attach_behavior_scripts_to_characters(simulation_app); _update(60)

    # 5. ★ Write scriptData + pathData AFTER schema is applied ---------------
    stage       = omni.usd.get_context().get_stage()
    parent_path = str(PrimPaths.characters_parent_path())

    for char_name, cmds in commands_dict.items():
        char_prim_path = f"{parent_path}/{char_name}"
        char_prim = stage.GetPrimAtPath(char_prim_path)
        if not char_prim.IsValid():
            print(f"[Episode] WARNING: prim not found for {char_name}, skipping.")
            continue

        skelroot = None
        for desc in Usd.PrimRange(char_prim):
            if desc.GetTypeName() == "SkelRoot":
                skelroot = desc; break
        if skelroot is None:
            print(f"[Episode] WARNING: no SkelRoot under {char_prim_path}.")
            continue

        command_strings: list[str] = []
        path_data: dict[int, list] = {}
        goto_index = 0

        for cmd in cmds:
            cmd_name = cmd.get("cmd", "")
            params   = cmd.get("params", [])
            command_strings.append(f"{cmd_name} " + " ".join(str(p) for p in params))
            if cmd_name == "GoTo":
                if "path" in cmd and len(cmd["path"]) > 0:
                    path_data[goto_index] = cmd["path"]
                goto_index += 1

        sd_attr = skelroot.GetAttribute("omni:scripting:scriptData")
        if not sd_attr:
            sd_attr = skelroot.CreateAttribute(
                "omni:scripting:scriptData", Sdf.ValueTypeNames.StringArray)
        sd_attr.Set(command_strings)

        if path_data:
            pd_attr = skelroot.GetAttribute("omni:scripting:pathData")
            if not pd_attr:
                pd_attr = skelroot.CreateAttribute(
                    "omni:scripting:pathData", Sdf.ValueTypeNames.String)
            pd_attr.Set(_json.dumps(path_data))

        # Readback sanity check
        readback = skelroot.GetAttribute("omni:scripting:scriptData").Get()
        n_read = len(readback) if readback else 0
        if n_read != len(command_strings):
            print(f"[Episode] !! scriptData mismatch for {char_name}: "
                  f"wrote {len(command_strings)}, read back {n_read}")
        else:
            print(f"[Episode] {char_name}: {n_read} commands OK, "
                  f"{len(path_data)} pre-computed paths.")

    # 6. Let CharacterBehavior pick up the queue -----------------------------
    _update(30)

    if char_prims:
        frame_viewport_on(char_prims[0]); _update(5)

    return char_prims


# ─── wait for episode completion ──────────────────────────────────────────
def _run_until_done(char_prims: list[str],
                    estimated_s: float,
                    hard_timeout_s: float,
                    pause_s: float) -> str:
    """
    Step the simulation until the episode is considered done, then wait
    pause_s seconds before returning.

    Completion is declared when EITHER:
      (A) estimated_s wall-clock time has elapsed (primary signal), OR
      (B) all characters have been position-still for _STILL_REQUIRED_S
          seconds (early-finish detection), OR
      (C) hard_timeout_s elapsed (safety net).

    Returns "done" | "early_done" | "timeout" | "aborted".
    """
    tl = omni.timeline.get_timeline_interface()
    tl.set_current_time(0.0)
    tl.play()

    t_start      = time.time()
    t_last_poll  = time.time()
    n_chars      = len(char_prims)

    # Per-character stillness tracker: {prim_path: (last_pos, still_since)}
    still_state: dict[str, tuple] = {p: (None, None) for p in char_prims}

    print(f"[RUN] Episode running  ({n_chars} character(s)  "
          f"estimated={estimated_s:.0f}s  hard_timeout={hard_timeout_s:.0f}s) …")

    while simulation_app.is_running():
        simulation_app.update()
        now     = time.time()
        elapsed = now - t_start

        # ── (A) Estimated duration elapsed ────────────────────────────────
        if elapsed >= estimated_s:
            tl.stop()
            print(f"\n[RUN] ✓ Estimated duration reached "
                  f"(elapsed={elapsed:.1f}s / estimated={estimated_s:.0f}s).")
            _do_pause(pause_s)
            return "done"

        # ── Poll every _STILL_POLL_S seconds ──────────────────────────────
        if now - t_last_poll >= _STILL_POLL_S:
            t_last_poll = now
            all_still   = True

            for p in char_prims:
                pos = _get_char_pos(p)
                last_pos, still_since = still_state[p]

                if pos is None or last_pos is None:
                    still_state[p] = (pos, None)
                    all_still = False
                    continue

                dx = pos[0] - last_pos[0]
                dy = pos[1] - last_pos[1]
                moved = (dx*dx + dy*dy) ** 0.5

                if moved > _STILL_THRESHOLD_M:
                    still_state[p] = (pos, None)
                    all_still = False
                else:
                    if still_since is None:
                        still_state[p] = (pos, now)
                        all_still = False
                    else:
                        still_for = now - still_since
                        still_state[p] = (pos, still_since)
                        if still_for < _STILL_REQUIRED_S:
                            all_still = False

            still_info = []
            for p in char_prims:
                _, ss = still_state[p]
                still_for = (now - ss) if ss is not None else 0.0
                still_info.append(f"{still_for:.1f}s")
            print(f"[RUN]  t={elapsed:.1f}s/{estimated_s:.0f}s  "
                  f"still_for={still_info}")

            # ── (B) All characters still for required duration ────────────
            if all_still and n_chars > 0:
                tl.stop()
                print(f"\n[RUN] ✓ Early finish: all characters still for "
                      f"≥{_STILL_REQUIRED_S}s (elapsed={elapsed:.1f}s).")
                _do_pause(pause_s)
                return "early_done"

            # ── (C) Hard timeout ──────────────────────────────────────────
            if elapsed >= hard_timeout_s:
                tl.stop()
                print(f"[RUN] ✗ Hard timeout after {hard_timeout_s:.0f}s")
                return "timeout"

    return "aborted"


def _do_pause(pause_s: float):
    """Run the app for pause_s wall-clock seconds (keeps viewport alive)."""
    print(f"[RUN] Pausing {pause_s:.0f}s before next episode …")
    t0 = time.time()
    while time.time() - t0 < pause_s:
        simulation_app.update()


# ═══════════════════════════════════════════════════════════════════════════
# Scene-level runner
# ═══════════════════════════════════════════════════════════════════════════
def _run_scene(usda_path: str,
               episodes_dir: str,
               episode_paths: list[str],
               char_pool: list[str],
               current_usda: list[str],
               current_nm:   list,
               ) -> dict:
    """Run all episodes for one scene.  Reopens stage only if usda changed."""

    results = {"ok": 0, "timeout": 0, "aborted": 0, "load_fail": 0}
    scene_id = os.path.splitext(os.path.basename(usda_path))[0]

    # ── Open stage (only when scene changes) ──────────────────────────────
    if current_usda[0] != usda_path:
        if not _open_stage(usda_path):
            print(f"[FATAL] Could not open {usda_path}"); return results
        current_usda[0] = usda_path

        navvols_usda = cache_paths(usda_path, ARGS.cache_dir)
        nav_ok, inav = bake_navmesh(
            app=simulation_app, navvols_usda=navvols_usda,
            force_rebake=ARGS.force_rebake,
            collision_root=ARGS.collision_root,
            volume_padding=ARGS.volume_padding,
            fallback_size=ARGS.fallback_size,
            warmup_frames=ARGS.warmup_frames,
            semantic_map_json=None,
        )
        nm = inav.get_navmesh() if nav_ok else None
        if nm is None:
            print("[WARN] NavMesh bake failed — continuing without connectivity checks.")
        current_nm[0] = nm
        _hide_navmesh_volumes()
        _update(10)

        from isaacsim.core.api import World
        world = World(physics_dt=1.0/60.0, rendering_dt=1.0/30.0)
        world.initialize_physics()
        world.reset()
        _update(10)
        print("[World] Physics initialized.")

    n_total = len(episode_paths)
    for ep_idx, ep_path in enumerate(episode_paths, 1):
        ep_id = _ep_sort_key(ep_path)
        print(f"\n{'─'*60}")
        print(f"[SCENE {scene_id}] Episode {ep_idx}/{n_total}  "
              f"(id={ep_id})  ← {os.path.basename(ep_path)}")
        print(f"{'─'*60}")

        # Load episode JSON
        try:
            with open(ep_path, encoding="utf-8") as f:
                raw = json.load(f)
            episode = raw["episode"]
        except Exception as e:
            print(f"[ERROR] Could not load {ep_path}: {e}")
            results["load_fail"] += 1
            continue

        num_chars = episode["characters"].get("num_characters", "?")
        print(f"[Episode] id={ep_id}  num_chars={num_chars}")

        # Clean up previous episode's characters
        _delete_all_characters()

        # Spawn + wire commands (attach_behavior_scripts runs BEFORE scriptData write)
        char_prims = _setup_episode(episode, char_pool)

        if not char_prims:
            print(f"[Episode] No characters spawned (num_people=0) — "
                  f"robot-only episode, pausing {ARGS.inter_episode_pause}s.")
            _do_pause(ARGS.inter_episode_pause)
            results["ok"] += 1
            continue

        estimated_s = _estimate_episode_duration(episode)
        print(f"[Episode] Estimated duration: {estimated_s:.0f}s  "
              f"(hard timeout: {ARGS.episode_timeout:.0f}s)")

        outcome = _run_until_done(
            char_prims,
            estimated_s=estimated_s,
            hard_timeout_s=ARGS.episode_timeout,
            pause_s=ARGS.inter_episode_pause,
        )
        results[outcome] = results.get(outcome, 0) + 1
        ok_label = outcome in ("done", "early_done")
        print(f"[Episode] id={ep_id}  outcome={outcome}  "
              f"{'✓' if ok_label else '✗'}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> int:
    work: list[tuple[str, str, list[str]]] = []

    if ARGS.usda and ARGS.episodes_dir:
        ep_paths = _collect_episodes(ARGS.episodes_dir, ARGS.episode_ids)
        if not ep_paths:
            print(f"[FATAL] No episodes found in {ARGS.episodes_dir}"); return 1
        work.append((ARGS.usda, ARGS.episodes_dir, ep_paths))

    elif ARGS.usda_dir and ARGS.episodes_root:
        filter_ids = set(ARGS.scene_ids) if ARGS.scene_ids else None
        for usda_fp in sorted(glob.glob(os.path.join(ARGS.usda_dir, "*.usda"))):
            scene_id = os.path.splitext(os.path.basename(usda_fp))[0]
            if filter_ids and scene_id not in filter_ids:
                continue
            ep_dir = os.path.join(ARGS.episodes_root, scene_id)
            if not os.path.isdir(ep_dir):
                print(f"[SKIP] No episodes directory for scene {scene_id}"); continue
            ep_paths = _collect_episodes(ep_dir, ARGS.episode_ids)
            if not ep_paths:
                print(f"[SKIP] No episodes found for scene {scene_id}"); continue
            work.append((usda_fp, ep_dir, ep_paths))

    else:
        print("[FATAL] Provide either (--usda + --episodes_dir) "
              "or (--usda_dir + --episodes_root).")
        simulation_app.close(); return 1

    if not work:
        print("[FATAL] No work to do."); simulation_app.close(); return 1

    total_scenes = len(work)
    print(f"\n[BATCH] {total_scenes} scene(s)  "
          f"total episodes={sum(len(w[2]) for w in work)}")

    _open_stage(work[0][0])
    char_pool = load_character_assets()
    print(f"[PEOPLE] {len(char_pool)} asset(s) available.")

    current_usda = [work[0][0]]
    current_nm   = [None]
    current_usda[0] = "__none__"

    grand = {"ok": 0, "timeout": 0, "aborted": 0, "load_fail": 0}
    try:
        for sc_idx, (usda_fp, ep_dir, ep_paths) in enumerate(work, 1):
            scene_id = os.path.splitext(os.path.basename(usda_fp))[0]
            print(f"\n{'═'*60}")
            print(f"[BATCH] Scene {sc_idx}/{total_scenes}: {scene_id}  "
                  f"({len(ep_paths)} episodes)")
            print(f"{'═'*60}")
            res = _run_scene(usda_fp, ep_dir, ep_paths,
                             char_pool, current_usda, current_nm)
            for k, v in res.items():
                grand[k] = grand.get(k, 0) + v

    except KeyboardInterrupt:
        print("\n[BATCH] Ctrl+C — shutting down early.")

    print(f"\n{'═'*60}")
    print(f"[BATCH] All done.  Results: {grand}")
    print(f"{'═'*60}")

    simulation_app.close()
    return 0 if grand.get("aborted", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())