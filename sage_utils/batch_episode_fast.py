#!/usr/bin/env python3
"""
batch_episode_fast.py
─────────────────────
CPU‑parallel batch scheduler for fast episode generator.

Usage
-----
    python batch_episode_fast.py \
        --sem_dir /workspace/.../semantic_maps \
        --output_dir /workspace/.../v1_episodes \
        --total_episodes 100 --max_people 10 \
        --resume --workers 16
"""
import argparse, glob, os, re, subprocess, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

GENERATOR_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "generate_episode_fast.py")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sem_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--total_episodes", type=int, default=100)
    p.add_argument("--max_people", type=int, default=10)
    p.add_argument("--seed_offset", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_vis", action="store_true")
    p.add_argument("--workers", type=int, default=4,
                   help="Number of parallel CPU workers (default 4).")
    return p.parse_args()

def plan_distribution(total, max_people):
    n_groups = max_people
    base = total // n_groups
    extra = total % n_groups
    counts = [base] * n_groups
    for k in range(n_groups - 1, n_groups - 1 - extra, -1):
        counts[k] += 1
    assignment = []
    for idx, n in enumerate(counts):
        assignment.extend([idx+1] * n)
    return assignment

def already_done(output_dir, scene_id, ep_id):
    return os.path.exists(os.path.join(output_dir, scene_id, f"episode_{ep_id}.json"))

def run_scene(sem_json, scene_id, todo_ids, todo_people, args):
    cmd = [
        sys.executable, GENERATOR_SCRIPT,
        "--semantic_map_json", sem_json,
        "--output_dir", args.output_dir,
        "--episode_ids", *[str(i) for i in todo_ids],
        "--num_people_for_episode", *[str(n) for n in todo_people],
        "--seed_offset", str(args.seed_offset),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.no_vis:
        cmd.append("--no_vis")

    # Per-scene log so we can debug interior_mask, A* failures, etc.
    log_dir = os.path.join(args.output_dir, "_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{scene_id}.log")

    t0 = time.time()
    with open(log_path, "w") as log_f:
        log_f.write(f"# cmd: {' '.join(cmd)}\n\n")
        log_f.flush()
        result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        # Print tail of log so the batch console shows what went wrong.
        try:
            with open(log_path) as f:
                lines = f.readlines()
            tail = "".join(lines[-20:])
            print(f"[FAIL] {scene_id} (exit={result.returncode}, "
                  f"{elapsed:.0f}s)  tail:\n{tail}")
        except Exception:
            print(f"[FAIL] {scene_id} exit={result.returncode}")
    else:
        print(f"[OK]   {scene_id} ({elapsed:.0f}s)")
    return scene_id, result.returncode == 0

def main():
    args = parse_args()
    sem_files = glob.glob(os.path.join(args.sem_dir, "2D_Semantic_Map_*_Complete.json"))
    scene_map = {}
    for fp in sem_files:
        m = re.match(r"2D_Semantic_Map_\d+_(\d+)_Complete\.json", os.path.basename(fp))
        if m:
            scene_map[m.group(1)] = fp
    if not scene_map:
        print("[FATAL] No semantic maps found"); return 1
    scene_ids = sorted(scene_map.keys())
    print(f"[BATCH] Found {len(scene_ids)} scenes")

    assignment = plan_distribution(args.total_episodes, args.max_people)
    dist = Counter(assignment)
    print(f"[BATCH] Distribution: {dict(sorted(dist.items()))}")

    tasks = []
    for sid in scene_ids:
        if args.resume and not args.overwrite:
            todo_ids, todo_people = [], []
            for ep_id, n_ppl in enumerate(assignment):
                if not already_done(args.output_dir, sid, ep_id):
                    todo_ids.append(ep_id)
                    todo_people.append(n_ppl)
            if not todo_ids:
                print(f"[SKIP] {sid} all done")
                continue
        else:
            todo_ids = list(range(args.total_episodes))
            todo_people = list(assignment)
        tasks.append((scene_map[sid], sid, todo_ids, todo_people))

    if not tasks:
        print("[BATCH] Nothing to do"); return 0

    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_scene, *t, args): t[1] for t in tasks}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                _, success = future.result()
            except Exception as e:
                print(f"[EXCEPTION] {sid}: {e}")
                success = False
            if success:
                ok += 1
            else:
                fail += 1
    print(f"[BATCH] Done. ok={ok} fail={fail}")
    return 0 if fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())