#!/usr/bin/env python3
"""
batch_episode_fast_tracking.py
──────────────────────────────
CPU‑parallel batch scheduler for tracking episodes.
Supports --easy_mode and --scene_ids for selective fix‑up.
"""
import argparse, glob, os, re, subprocess, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

GENERATOR_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "generate_episode_fast_tracking.py") # _hard for hard scenes

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sem_dir", required=True)
    p.add_argument("--output_dir",
                   default="/workspace/SAGE-3D_Official/SAGE-3D_data/v1_tracking_episodes_test")
    p.add_argument("--total_episodes", type=int, default=100)
    p.add_argument("--max_people", type=int, default=10)
    p.add_argument("--seed_offset", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_vis", action="store_true")
    p.add_argument("--easy_mode", action="store_true",
                   help="Pass --easy_mode to the generator for difficult scenes.")
    p.add_argument("--scene_ids", nargs="*", default=None,
                   help="Only process these scene IDs (useful for fixing failures).")
    p.add_argument("--workers", type=int, default=16)
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
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{scene_id}.log")

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
    if args.easy_mode:
        cmd.append("--easy_mode")

    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n{'='*60}\n")
        lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START scene={scene_id}\n")
        lf.write(f"  episodes : {todo_ids}\n")
        lf.write(f"  n_people : {todo_people}\n")
        lf.write(f"  cmd      : {' '.join(cmd)}\n")
        lf.write(f"{'='*60}\n")
        lf.flush()

        result = subprocess.run(cmd, stdout=lf, stderr=lf, text=True)

        elapsed = round(time.time() - t0, 1)
        status = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
        lf.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] END scene={scene_id}  status={status}  elapsed={elapsed}s\n")

    if result.returncode == 3:
        print(f"[INCOMPLETE] {scene_id}  some tracking episodes had no valid "
            f"target → log {log_path}")
    elif result.returncode != 0:
        print(f"[FAIL] {scene_id}  rc={result.returncode}  log → {log_path}")
    else:
        print(f"[OK]   {scene_id}  ({elapsed}s)")
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

    # 若指定了 --scene_ids，只处理这些场景
    if args.scene_ids:
        scene_map = {sid: scene_map[sid] for sid in args.scene_ids if sid in scene_map}
        if not scene_map:
            print("[FATAL] No matching scenes after --scene_ids filter"); return 1

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
            print(f"[PROGRESS] {ok+fail}/{len(tasks)} scenes done")
    print(f"[BATCH] Done. ok={ok} fail={fail}")
    return 0 if fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())