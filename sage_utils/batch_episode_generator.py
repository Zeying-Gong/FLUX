#!/usr/bin/env python3
"""
batch_episode_generator.py  (accelerated, min_people=1)
──────────────────────────────────────────────────────
Batch scheduler for generate_episode_json_interaction.py.
Now skips NavMesh checks when --skip_navmesh_check is used,
and guarantees at least 1 person per episode.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# ─── defaults ─────────────────────────────────────────────────────────────
ISAAC_PYTHON   = "/isaac-sim/python.sh"
GENERATOR_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "generate_episode_json_interaction.py")
ENV_VARS = {
    "DISPLAY":        ":1",
    "EXP_PATH":       "/isaac-sim/apps",
    "CARB_LOG_LEVEL": "info",
}

def parse_args():
    p = argparse.ArgumentParser(description="Batch episode generator for SAGE-3D.")
    p.add_argument("--usda_dir",   required=True)
    p.add_argument("--sem_dir",    required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--scene_ids",  nargs="*", default=None)
    p.add_argument("--total_episodes", type=int, default=100)
    p.add_argument("--max_people", type=int, default=10,
                   help="Maximum people per episode (minimum is 1).")
    p.add_argument("--seed_offset", type=int, default=0)
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--overwrite",  action="store_true")
    p.add_argument("--skip_navmesh_check", action="store_true",
                   help="Disable NavMesh connectivity checks (much faster).")
    p.add_argument("--no_vis",     action="store_true",
                   help="Pass --no_vis to generator.")
    p.add_argument("--isaac_python", default=ISAAC_PYTHON)
    p.add_argument("--generator",    default=GENERATOR_SCRIPT)
    p.add_argument("--dry_run",    action="store_true")
    return p.parse_args()

def discover_scenes(usda_dir, sem_dir, filter_ids):
    sem_index = {}
    for fp in glob.glob(os.path.join(sem_dir, "2D_Semantic_Map_*_Complete.json")):
        m = re.match(r"2D_Semantic_Map_\d+_(\d+)_Complete\.json$", os.path.basename(fp))
        if m:
            sem_index[m.group(1)] = fp
    scenes = []
    for usda_fp in sorted(glob.glob(os.path.join(usda_dir, "*.usda"))):
        scene_id = os.path.splitext(os.path.basename(usda_fp))[0]
        if filter_ids and scene_id not in filter_ids:
            continue
        if scene_id not in sem_index:
            print(f"[SKIP] {scene_id}: no semantic map")
            continue
        scenes.append({"scene_id": scene_id, "usda": usda_fp, "sem_map": sem_index[scene_id]})
    return scenes

def plan_episode_distribution(total: int, max_people: int) -> list[int]:
    """
    Return a list of length `total` where each entry is the num_people
    assigned to that episode.  People counts are 1 .. max_people inclusive.
    """
    n_groups = max_people          # 1 .. max_people
    base     = total // n_groups
    extra    = total % n_groups
    counts = [base] * n_groups
    # remainder to highest-people groups first
    for k in range(n_groups - 1, n_groups - 1 - extra, -1):
        counts[k] += 1
    assignment = []
    for idx, n in enumerate(counts):
        people_count = idx + 1
        assignment.extend([people_count] * n)
    assert len(assignment) == total
    return assignment

def already_done(output_dir, scene_id, episode_id):
    return os.path.exists(os.path.join(output_dir, scene_id, f"episode_{episode_id}.json"))

def run_scene(scene, episode_ids, num_people_list, args):
    """Launch generator for one scene; stdout/stderr → log file."""
    cmd = [
        args.isaac_python, args.generator,
        "--usda",             scene["usda"],
        "--semantic_map_json", scene["sem_map"],
        "--output_dir",       args.output_dir,
        "--episode_ids",      *[str(i) for i in episode_ids],
        "--num_people_for_episode", *[str(n) for n in num_people_list],
        "--seed_offset",      str(args.seed_offset),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.skip_navmesh_check:
        cmd.append("--skip_navmesh_check")
    if args.no_vis:
        cmd.append("--no_vis")

    env = os.environ.copy()
    env.update(ENV_VARS)

    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{scene['scene_id']}.log")

    print(f"[LAUNCH] {scene['scene_id']}  episodes={len(episode_ids)}  "
          f"log={log_path}")
    if args.dry_run:
        print(f"  CMD: {' '.join(cmd)}")
        return True

    t0 = time.time()
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    elapsed = round(time.time() - t0, 1)
    ok = (result.returncode == 0)
    status = "OK" if ok else f"FAILED (rc={result.returncode})"
    print(f"[SCENE] {scene['scene_id']}  {status}  elapsed={elapsed}s")
    return ok

def main():
    args = parse_args()
    if not os.path.isdir(args.usda_dir):
        print("[FATAL] --usda_dir not found"); return 1
    if not os.path.isdir(args.sem_dir):
        print("[FATAL] --sem_dir not found"); return 1
    if not os.path.exists(args.generator):
        print("[FATAL] generator script not found"); return 1

    os.makedirs(args.output_dir, exist_ok=True)

    filter_ids = [str(s) for s in args.scene_ids] if args.scene_ids else None
    scenes = discover_scenes(args.usda_dir, args.sem_dir, filter_ids)
    if not scenes:
        print("[FATAL] No matching scenes"); return 1
    print(f"[BATCH] Found {len(scenes)} scene(s)")

    assignment = plan_episode_distribution(args.total_episodes, args.max_people)
    dist = Counter(assignment)
    print(f"[BATCH] Distribution (total={args.total_episodes}, max_people={args.max_people}):")
    for k in sorted(dist):
        print(f"  {k} people → {dist[k]} episode(s)")

    total_scenes = len(scenes)
    ok_scenes = 0
    fail_scenes = 0
    skipped_total = 0

    for scene_idx, scene in enumerate(scenes, 1):
        print(f"\n[BATCH] ── Scene {scene_idx}/{total_scenes}: {scene['scene_id']} ──")

        if args.resume and not args.overwrite:
            todo_ids, todo_people = [], []
            for ep_id, n_ppl in enumerate(assignment):
                if not already_done(args.output_dir, scene["scene_id"], ep_id):
                    todo_ids.append(ep_id)
                    todo_people.append(n_ppl)
            if not todo_ids:
                print(f"  [RESUME] All {args.total_episodes} episodes already exist, skipping scene.")
                ok_scenes += 1
                continue
            skipped_total += args.total_episodes - len(todo_ids)
            print(f"  [RESUME] {len(todo_ids)} episode(s) remaining "
                  f"({args.total_episodes - len(todo_ids)} already done)")
        else:
            todo_ids = list(range(args.total_episodes))
            todo_people = list(assignment)

        success = run_scene(scene, todo_ids, todo_people, args)
        if success:
            ok_scenes += 1
        else:
            fail_scenes += 1

    print(f"\n{'='*70}")
    print(f"[BATCH] Done.  scenes={total_scenes}  ok={ok_scenes}  "
          f"failed={fail_scenes}  skipped_episodes={skipped_total}")
    print(f"{'='*70}")
    return 0 if fail_scenes == 0 else 1

if __name__ == "__main__":
    sys.exit(main())