"""
batch_episode_generator.py
──────────────────────────
Batch scheduler for generate_episode_json_interaction.py.

Features
--------
- Discover all scenes from USDA + semantic-map directories
- Generate N episodes per scene with configurable seeds
- Skip existing episodes (resume-safe)
- Per-scene + global progress tracking
- Quality report table (CSV + pretty print) after completion
- 7:2:1 train/val/test split assignment

Directory layout:
    SAGE-3D_data/
    ├── usda/                      <scene_id>.usda
    ├── semantic_maps/             2D_Semantic_Map_<folder_id>_Complete.json
    ├── scene_text/                semantic_map_<folder_id>.txt
    ├── InteriorGS/                <folder_id>/   (used for scene discovery)
    └── episodes/                  <scene_id>/episode_<N>.json   ← output

folder_id format: "0001_839920"  →  scene_id: "839920"

Usage
-----
    # Quick test: 2 scenes × 5 episodes
    python batch_episode_generator.py \
        --num_scenes 2 --episodes_per_scene 5 --use_llm --dry_run

    # Full run
    python batch_episode_generator.py --episodes_per_scene 100 --use_llm

    # Resume (skips already-generated episodes)
    python batch_episode_generator.py --episodes_per_scene 100 --use_llm
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
import datetime
from collections import defaultdict
from pathlib import Path

# ── Default paths ──────────────────────────────────────────────────────────
DATA_ROOT       = Path("/workspace/SAGE-3D_Official/SAGE-3D_data")
USDA_DIR        = DATA_ROOT / "usda"
SEM_MAP_DIR     = DATA_ROOT / "semantic_maps"
SCENE_TEXT_DIR  = DATA_ROOT / "scene_text"
EPISODES_DIR    = DATA_ROOT / "episodes"
REPORT_DIR      = DATA_ROOT / "reports"

# InteriorGS is used only for scene discovery (folder enumeration)
INTERIOR_GS_DIR = DATA_ROOT / "InteriorGS"

GENERATOR_SCRIPT = Path(__file__).parent / "generate_episode_json_interaction.py"
ISAAC_PYTHON     = Path("/isaac-sim/python.sh")

# ── People count ───────────────────────────────────────────────────────────
# Aligned with generate_episode_json_interaction.py and debug_sampling.py:
#   actual_num = floor(nav_area_m2 / AREA_PER_PERSON_M2), capped by MAX_PEOPLE_CAP
AREA_PER_PERSON_M2 = 5.0    # keep in sync with generator constant
MAX_PEOPLE_CAP     = 10     # absolute upper limit

# ── Train / val / test split ──────────────────────────────────────────────
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}


# ═══════════════════════════════════════════════════════════════════════════
# Scene discovery
# ═══════════════════════════════════════════════════════════════════════════

def _compute_people_cap(sem_map_path: Path,
                        robot_radius_m: float = 0.3,
                        scale: float = 0.05) -> int:
    """
    Compute the maximum number of people for this scene based on navigable area.
    Formula: floor(nav_area_m2 / AREA_PER_PERSON_M2), capped at MAX_PEOPLE_CAP.
    This is the UPPER LIMIT; individual episodes may use fewer people.
    """
    import numpy as _np
    from scipy.ndimage import distance_transform_edt as _edt
    try:
        with open(sem_map_path, encoding="utf-8") as f:
            sem_data = json.load(f)
        all_y, all_x = [], []
        for inst in sem_data:
            for y, x in inst.get("mask_coords_m", []):
                try:
                    all_y.append(float(y)); all_x.append(float(x))
                except Exception:
                    pass
        if not all_y:
            return 2
        min_y, max_y = min(all_y), max(all_y)
        min_x, max_x = min(all_x), max(all_x)
        h = int(_np.ceil((max_y - min_y) / scale)) + 1
        w = int(_np.ceil((max_x - min_x) / scale)) + 1
        grid = _np.zeros((h, w), dtype=_np.uint8)
        for inst in sem_data:
            label = str(inst.get("category_label", "")).lower()
            if label in ("wall", "unable area"):
                for y_m, x_m in inst.get("mask_coords_m", []):
                    try:
                        py = int(round((float(y_m) - min_y) / scale))
                        px = int(round((float(x_m) - min_x) / scale))
                        if 0 <= py < h and 0 <= px < w:
                            grid[py, px] = 1
                    except Exception:
                        pass
        if robot_radius_m > 0:
            dist_m = _edt(grid == 0, sampling=scale)
            grid = (dist_m <= robot_radius_m).astype(_np.uint8)
        nav_area_m2 = float(int((grid == 0).sum()) * scale ** 2)
        n = max(1, int(nav_area_m2 / AREA_PER_PERSON_M2))
        return min(n, MAX_PEOPLE_CAP)
    except Exception as e:
        print(f"    [WARN] _compute_people_cap failed for {sem_map_path.name}: {e}")
        return 2


def _distribute_num_people(episode_ids: list[int], cap: int) -> dict[int, int]:
    """
    Distribute num_people across episodes as evenly as possible over [0..cap].

    The (cap+1) distinct values {0, 1, ..., cap} are cycled across episodes
    using largest-remainder so that each value appears as equal a number of
    times as possible.  Remainder episodes (when total is not divisible by
    cap+1) are assigned starting from 0.

    Example: cap=2, episodes=100
      n_values = 3  (0, 1, 2)
      base = 100 // 3 = 33,  remainder = 100 % 3 = 1
      → 0 appears 34 times, 1 appears 33 times, 2 appears 33 times
      (remainder goes to the smallest values first so 0-person episodes
       are slightly over-represented, which is fine for a baseline split)

    Returns: {episode_id: num_people}
    """
    n_ep     = len(episode_ids)
    n_values = cap + 1          # {0, 1, ..., cap}
    base     = n_ep // n_values
    leftover = n_ep % n_values  # leftover episodes go to values 0..leftover-1

    # counts[v] = how many episodes get v people
    counts = [base + (1 if v < leftover else 0) for v in range(n_values)]

    # Build the full sequence: [0]*counts[0] + [1]*counts[1] + ...
    sequence: list[int] = []
    for v, c in enumerate(counts):
        sequence.extend([v] * c)

    # Map episode_id in order → value
    return {ep_id: sequence[i] for i, ep_id in enumerate(episode_ids)}


def _discover_scenes(usda_dir: Path, sem_map_dir: Path,
                     interior_gs_dir: Path,
                     cache_dir: Path | None = None) -> list[dict]:
    """
    Discover scenes by iterating InteriorGS sub-directories.
    Each folder_id (e.g. "0001_839920") maps to:
      - scene_id: "839920"
      - usda:     usda_dir / "839920.usda"
      - sem_map:  sem_map_dir / "2D_Semantic_Map_0001_839920_Complete.json"
      - txt:      scene_text_dir / "semantic_map_0001_839920*.txt"

    num_people is computed once and cached in <cache_dir>/scene_num_people.json.
    """
    if cache_dir is None:
        cache_dir = DATA_ROOT / "scene_meta"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "scene_num_people.json"

    num_people_cache: dict[str, int] = {}
    if cache_path.exists():
        try:
            num_people_cache = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"[DISCOVER] Loaded people_cap cache: {len(num_people_cache)} entries")
        except Exception:
            pass

    if not interior_gs_dir.exists():
        print(f"[WARN] InteriorGS dir not found: {interior_gs_dir}")
        print(f"[DISCOVER] Falling back to scanning sem_map_dir for json files")
        return _discover_scenes_from_sem_map(sem_map_dir, usda_dir,
                                             cache_dir, num_people_cache, cache_path)

    scenes = []
    cache_updated = False

    for folder in sorted(interior_gs_dir.iterdir()):
        if not folder.is_dir():
            continue
        folder_id = folder.name                    # e.g. "0001_839920"
        scene_id  = folder_id.split("_", 1)[-1]   # e.g. "839920"

        usda_path = usda_dir / f"{scene_id}.usda"
        sem_map   = sem_map_dir / f"2D_Semantic_Map_{folder_id}_Complete.json"

        if not usda_path.exists() or not sem_map.exists():
            continue

        txt_candidates = list(SCENE_TEXT_DIR.glob(f"semantic_map_{folder_id}*.txt"))
        scene_text     = txt_candidates[0] if txt_candidates else None

        if folder_id in num_people_cache:
            people_cap = num_people_cache[folder_id]
        else:
            print(f"[DISCOVER] Computing people_cap for {folder_id} ...")
            people_cap = _compute_people_cap(sem_map)
            num_people_cache[folder_id] = people_cap
            cache_updated = True

        scenes.append({
            "scene_id":        scene_id,
            "folder_id":       folder_id,
            "usda_path":       usda_path,
            "sem_map_path":    sem_map,
            "scene_text_path": scene_text,
            "people_cap":      people_cap,   # upper limit; per-episode count set by batch
            "split":           "?",
        })

    if cache_updated:
        cache_path.write_text(
            json.dumps(num_people_cache, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"[DISCOVER] Cache saved → {cache_path} ({len(num_people_cache)} entries)")

    return scenes


def _discover_scenes_from_sem_map(sem_map_dir: Path, usda_dir: Path,
                                   cache_dir: Path,
                                   num_people_cache: dict, cache_path: Path) -> list[dict]:
    """Fallback discovery when InteriorGS dir is absent: scan semantic_maps/ directly."""
    scenes = []
    cache_updated = False
    for sem_map in sorted(sem_map_dir.glob("2D_Semantic_Map_*_Complete.json")):
        stem      = sem_map.stem
        folder_id = stem.replace("2D_Semantic_Map_", "").replace("_Complete", "")
        scene_id  = folder_id.split("_", 1)[-1]
        usda_path = usda_dir / f"{scene_id}.usda"
        if not usda_path.exists():
            continue
        txt_candidates = list(SCENE_TEXT_DIR.glob(f"semantic_map_{folder_id}*.txt"))
        scene_text     = txt_candidates[0] if txt_candidates else None
        if folder_id in num_people_cache:
            people_cap = num_people_cache[folder_id]
        else:
            print(f"[DISCOVER] Computing people_cap for {folder_id} ...")
            people_cap = _compute_people_cap(sem_map)
            num_people_cache[folder_id] = people_cap
            cache_updated = True
        scenes.append({
            "scene_id":        scene_id,
            "folder_id":       folder_id,
            "usda_path":       usda_path,
            "sem_map_path":    sem_map,
            "scene_text_path": scene_text,
            "people_cap":      people_cap,
            "split":           "?",
        })
    if cache_updated:
        cache_path.write_text(
            json.dumps(num_people_cache, indent=2, ensure_ascii=False),
            encoding="utf-8")
    return scenes


def _assign_splits(scenes: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    shuffled = scenes[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val   = int(n * SPLIT_RATIOS["val"])
    for i, s in enumerate(shuffled):
        if   i < n_train:             s["split"] = "train"
        elif i < n_train + n_val:     s["split"] = "val"
        else:                         s["split"] = "test"
    split_map = {s["scene_id"]: s["split"] for s in shuffled}
    for s in scenes:
        s["split"] = split_map[s["scene_id"]]
    return scenes


# ═══════════════════════════════════════════════════════════════════════════
# Episode existence check
# ═══════════════════════════════════════════════════════════════════════════

def _existing_episodes(scene_id: str, episodes_dir: Path) -> set[int]:
    d = episodes_dir / scene_id
    if not d.exists():
        return set()
    return {
        int(p.stem.split("_")[1])
        for p in d.glob("episode_*.json")
        if "_log" not in p.stem and "_vis" not in p.stem
    }


# ═══════════════════════════════════════════════════════════════════════════
# Run one scene (all episodes in a single Isaac Sim launch)
# ═══════════════════════════════════════════════════════════════════════════

def _run_scene(
    scene: dict,
    episode_ids: list[int],
    args: argparse.Namespace,
    dry_run: bool,
) -> list[dict]:
    """
    Launch generate_episode_json_interaction.py ONCE for all episode_ids.
    Isaac Sim starts, processes all episodes, then exits.

    num_people per episode is distributed evenly over [0..people_cap] using
    _distribute_num_people, so the dataset has a balanced human-density
    distribution across episodes.
    """
    cap = scene["people_cap"]

    # Distribute num_people for each episode: {ep_id: n}
    ep_npeople = _distribute_num_people(episode_ids, cap)
    npeople_list = [str(ep_npeople[ep_id]) for ep_id in episode_ids]
    ep_strs      = [str(i) for i in episode_ids]

    print(f"  people_cap={cap}  per-episode distribution: "
          + " ".join(f"ep{ep_id}→{ep_npeople[ep_id]}" for ep_id in episode_ids[:8])
          + (" ..." if len(episode_ids) > 8 else ""))

    cmd = [
        str(args.isaac_python),
        str(args.generator_script),
        "--usda",                    str(scene["usda_path"]),
        "--semantic_map_json",       str(scene["sem_map_path"]),
        "--output_dir",              str(args.output_dir),
        "--episode_ids",             *ep_strs,
        "--num_people_for_episode",  *npeople_list,
        "--num_people",              str(cap),   # kept as reference / fallback
        "--seed_offset",             str(args.seed_offset),
        "--num_waypoints",           str(args.num_waypoints),
        "--robot_radius_2d",         str(args.robot_radius_2d),
        "--scale_m_per_px",          str(args.scale_m_per_px),
    ]
    if args.use_llm and scene.get("scene_text_path"):
        cmd += [
            "--use_llm",
            "--scene_text_file", str(scene["scene_text_path"]),
            "--llm_base_url",    args.llm_base_url,
            "--llm_model",       args.llm_model,
            "--llm_temperature", str(args.llm_temperature),
        ]
    if args.overwrite:
        cmd.append("--overwrite")

    print(f"  cmd: {' '.join(cmd[:6])} ... "
          f"episode_ids={episode_ids[:6]}{'...' if len(episode_ids)>6 else ''}")

    results = []
    if dry_run:
        for ep_id in episode_ids:
            print(f"  [DRY] episode_{ep_id}  num_people={ep_npeople[ep_id]}")
            results.append(_make_result(scene, ep_id, args,
                                        status="dry_run",
                                        num_people=ep_npeople[ep_id]))
        return results

    t0      = time.time()
    timeout = args.timeout_per_episode * len(episode_ids)
    try:
        proc          = subprocess.run(cmd, timeout=timeout)
        elapsed_total = round(time.time() - t0, 1)

        for ep_id in episode_ids:
            log_path = (Path(args.output_dir) / scene["scene_id"] /
                        f"episode_{ep_id}_log.json")
            r = _make_result(scene, ep_id, args,
                             elapsed_s=round(elapsed_total / len(episode_ids), 1),
                             num_people=ep_npeople[ep_id])
            if log_path.exists():
                with open(log_path) as f:
                    log = json.load(f)
                r.update({
                    "status":     "ok",
                    "mode":       log.get("mode", "?"),
                    "num_placed": log.get("num_placed", 0),
                    "nav_m2":     log.get("navigable_m2", 0.0),
                    "narrative":  log.get("narrative", "")[:80],
                })
            elif proc.returncode == 0:
                r["status"] = "ok"
            else:
                r["status"] = "fail"
                r["error"]  = f"returncode={proc.returncode}"
            results.append(r)

    except subprocess.TimeoutExpired:
        for ep_id in episode_ids:
            r = _make_result(scene, ep_id, args,
                             elapsed_s=float(timeout),
                             status="timeout", error="timeout",
                             num_people=ep_npeople[ep_id])
            results.append(r)
        print(f"  [TIMEOUT] all {len(episode_ids)} episodes after {timeout}s")

    except Exception as e:
        for ep_id in episode_ids:
            results.append(_make_result(scene, ep_id, args,
                                        status="error", error=str(e),
                                        num_people=ep_npeople[ep_id]))
        print(f"  [ERROR] {e}")

    return results


def _make_result(scene: dict, ep_id: int, args: argparse.Namespace,
                 status: str = "pending", elapsed_s: float = 0.0,
                 error: str = "", num_people: int = 0) -> dict:
    return {
        "scene_id":   scene["scene_id"],
        "folder_id":  scene.get("folder_id", ""),
        "split":      scene.get("split", "?"),
        "episode_id": ep_id,
        "seed":       args.seed_offset + ep_id,
        "people_cap": scene.get("people_cap", 0),
        "num_people": num_people,          # actual value for this episode
        "llm_mode":   args.use_llm and bool(scene.get("scene_text_path")),
        "status":     status,
        "mode":       "?",
        "num_placed": 0,
        "nav_m2":     0.0,
        "elapsed_s":  elapsed_s,
        "narrative":  "",
        "error":      error,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Quality report
# ═══════════════════════════════════════════════════════════════════════════

def _write_report(results: list[dict], report_dir: Path, timestamp: str):
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"quality_report_{timestamp}.csv"
    fields = ["scene_id", "folder_id", "split", "episode_id", "seed",
              "people_cap", "num_people", "llm_mode", "status", "mode",
              "num_placed", "nav_m2", "elapsed_s", "narrative", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(results)
    print(f"\n[REPORT] CSV → {csv_path}")

    # Per-scene summary
    per_scene: dict[str, dict] = defaultdict(lambda: {
        "split": "?", "total": 0, "ok": 0, "fail": 0, "timeout": 0,
        "llm_ok": 0, "random": 0, "nav_m2": 0.0,
    })
    for r in results:
        sid = r["scene_id"]
        per_scene[sid]["split"]  = r.get("split", "?")
        per_scene[sid]["total"] += 1
        st = r.get("status", "fail")
        per_scene[sid][st] = per_scene[sid].get(st, 0) + 1
        if r.get("mode") == "llm":    per_scene[sid]["llm_ok"] += 1
        if r.get("mode") == "random": per_scene[sid]["random"] += 1
        if r.get("nav_m2", 0):        per_scene[sid]["nav_m2"] = r["nav_m2"]

    hdr = (f"{'scene_id':>12} {'split':>6} {'ok':>5} {'fail':>5} "
           f"{'timeout':>8} {'llm':>5} {'rand':>5} {'nav_m2':>8} {'cap':>4}")
    sep = "─" * len(hdr)
    print("\n" + sep)
    print(hdr)
    print(sep)
    scene_cap = {r["scene_id"]: r.get("people_cap", 0) for r in results}
    for sid, s in sorted(per_scene.items()):
        print(f"{sid:>12} {s['split']:>6} {s['ok']:>5} {s.get('fail',0):>5} "
              f"{s.get('timeout',0):>8} {s['llm_ok']:>5} {s['random']:>5} "
              f"{s['nav_m2']:>8.0f} {scene_cap.get(sid,0):>4}")
    print(sep)

    total_ok   = sum(1 for r in results if r.get("status") == "ok")
    total_fail = sum(1 for r in results
                     if r.get("status") in ("fail", "error", "timeout"))
    print(f"\n  Total episodes: {len(results)}  ✓ {total_ok}  ✗ {total_fail}")

    summary_path = report_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "generated_at": timestamp,
            "total": len(results), "ok": total_ok, "fail": total_fail,
            "area_per_person_m2": AREA_PER_PERSON_M2,
            "per_scene": {k: dict(v) for k, v in per_scene.items()},
        }, f, indent=2)
    print(f"[REPORT] Summary → {summary_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch episode generator for SAGE-3D dataset.")

    # Scene selection
    p.add_argument("--num_scenes",          type=int,   default=None,
                   help="Limit number of scenes (default: all).")
    p.add_argument("--only_scenes",         nargs="+",  default=None,
                   help="Process only these scene_ids.")

    # Episode config
    p.add_argument("--episodes_per_scene",  type=int,   default=5)
    p.add_argument("--seed_offset",         type=int,   default=0)
    p.add_argument("--num_waypoints",       type=int,   default=3)
    p.add_argument("--overwrite",           action="store_true")

    # Paths
    p.add_argument("--usda_dir",            type=Path,  default=USDA_DIR)
    p.add_argument("--sem_map_dir",         type=Path,  default=SEM_MAP_DIR)
    p.add_argument("--scene_text_dir",      type=Path,  default=SCENE_TEXT_DIR)
    p.add_argument("--interior_gs_dir",     type=Path,  default=INTERIOR_GS_DIR,
                   help="Root dir for scene folder discovery (InteriorGS).")
    p.add_argument("--output_dir",          type=Path,  default=EPISODES_DIR)
    p.add_argument("--report_dir",          type=Path,  default=REPORT_DIR)
    p.add_argument("--cache_dir",           type=Path,  default=None,
                   help="Cache dir for scene_num_people.json "
                        "(default: <DATA_ROOT>/scene_meta/).")
    p.add_argument("--generator_script",    type=Path,  default=GENERATOR_SCRIPT)
    p.add_argument("--isaac_python",        type=Path,  default=ISAAC_PYTHON)

    # 2D map params (forwarded to generator)
    p.add_argument("--robot_radius_2d",     type=float, default=0.3)
    p.add_argument("--scale_m_per_px",      type=float, default=0.05)

    # LLM
    p.add_argument("--use_llm",             action="store_true")
    p.add_argument("--llm_base_url",        default="http://localhost:8000/v1")
    p.add_argument("--llm_model",
                   default="/workspace/SAGE-3D_Official/Qwen/Qwen/Qwen3-8B")
    p.add_argument("--llm_temperature",     type=float, default=0.8)

    # Execution
    p.add_argument("--timeout_per_episode", type=int,   default=600,
                   help="Timeout (s) per episode for the subprocess.")
    p.add_argument("--dry_run",             action="store_true",
                   help="Print commands without executing.")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args      = parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    t_global  = time.time()

    # ── Discover scenes ────────────────────────────────────────────────
    print(f"[BATCH] Discovering scenes ...")
    print(f"  usda_dir:        {args.usda_dir}")
    print(f"  sem_map_dir:     {args.sem_map_dir}")
    print(f"  interior_gs_dir: {args.interior_gs_dir}")

    all_scenes = _discover_scenes(
        usda_dir=args.usda_dir,
        sem_map_dir=args.sem_map_dir,
        interior_gs_dir=args.interior_gs_dir,
        cache_dir=args.cache_dir,
    )

    # Override scene_text_dir if custom
    if args.scene_text_dir != SCENE_TEXT_DIR:
        for s in all_scenes:
            cands = list(args.scene_text_dir.glob(
                f"semantic_map_*{s['scene_id']}*.txt"))
            s["scene_text_path"] = cands[0] if cands else None

    if not all_scenes:
        print("[FATAL] No scenes found."); sys.exit(1)

    # Filter
    if args.only_scenes:
        all_scenes = [s for s in all_scenes if s["scene_id"] in args.only_scenes]
    if args.num_scenes:
        all_scenes = all_scenes[:args.num_scenes]

    # Assign splits
    all_scenes = _assign_splits(all_scenes)

    print(f"[BATCH] {len(all_scenes)} scenes selected")
    txt_ok = sum(1 for s in all_scenes if s["scene_text_path"])
    print(f"[BATCH] Scene text available: {txt_ok}/{len(all_scenes)}")
    print(f"[BATCH] AREA_PER_PERSON_M2={AREA_PER_PERSON_M2}  MAX_PEOPLE_CAP={MAX_PEOPLE_CAP}")

    # Print people_cap distribution
    cap_counts: dict[int, int] = defaultdict(int)
    for s in all_scenes:
        cap_counts[s["people_cap"]] += 1
    print(f"[BATCH] people_cap distribution (cap→scenes): "
          + " | ".join(f"cap{c}→{n}scenes" for c, n in sorted(cap_counts.items())))

    target_total = len(all_scenes) * args.episodes_per_scene
    print(f"[BATCH] Target: {target_total} episodes "
          f"({len(all_scenes)} scenes × {args.episodes_per_scene})\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-scene loop ─────────────────────────────────────────────────
    all_results: list[dict] = []

    for si, scene in enumerate(all_scenes, 1):
        sid = scene["scene_id"]
        print(f"\n{'═'*60}")
        print(f"[SCENE {si}/{len(all_scenes)}] {sid}  "
              f"folder={scene.get('folder_id','')}  "
              f"split={scene['split']}")
        if scene["scene_text_path"]:
            print(f"  scene_text: {scene['scene_text_path'].name}")
        else:
            print(f"  scene_text: (not found — LLM skipped for this scene)")
        print(f"  people_cap={scene['people_cap']}  "
              f"(nav_area / {AREA_PER_PERSON_M2}m2, max {MAX_PEOPLE_CAP}; "
              f"episodes distributed over [0..{scene['people_cap']}])")

        existing = _existing_episodes(sid, args.output_dir)
        needed   = [i for i in range(args.episodes_per_scene) if i not in existing]

        if not needed:
            print(f"  [SKIP] All {args.episodes_per_scene} episodes already exist")
            # 加载现有日志到结果中
            for ep_id in sorted(existing):
                log_path = args.output_dir / sid / f"episode_{ep_id}_log.json"
                
                # 1. 预设默认值，防止文件丢失时报错
                log = {}
                if log_path.exists():
                    with open(log_path) as f:
                        try:
                            log = json.load(f)
                        except json.JSONDecodeError:
                            print(f"  [WARN] Failed to decode {log_path.name}")

                # 2. 现在 log 变量已经存在（即使是空的 {}），可以安全调用了
                r = _make_result(
                    scene, 
                    ep_id, 
                    args, 
                    status="ok (cached)",
                    num_people=log.get("num_requested", 0)
                )

                # 3. 更新剩余字段
                r.update({
                    "mode":       log.get("mode", "?"),
                    "num_placed": log.get("num_placed", 0),
                    "nav_m2":     log.get("navigable_m2", 0.0),
                    "elapsed_s":  log.get("elapsed_s", 0.0),
                    "narrative":  log.get("narrative", "")[:80],
                })
                all_results.append(r)
            continue

        print(f"  episodes to generate: {needed}")
        scene_results = _run_scene(
            scene=scene, episode_ids=needed, args=args, dry_run=args.dry_run)
        all_results.extend(scene_results)

        ep_dir = args.output_dir / sid
        print(f"  output dir: {ep_dir}")
        for r in scene_results:
            icon = "✓" if r["status"] == "ok" else "✗"
            print(f"  {icon} ep_{r['episode_id']} [{r['status']}] "
                  f"{r['elapsed_s']}s mode={r['mode']} placed={r['num_placed']}")

    # ── Report ─────────────────────────────────────────────────────────
    total_elapsed = round(time.time() - t_global, 1)
    print(f"\n{'═'*60}")
    print(f"[BATCH] All done in {total_elapsed}s")
    _write_report(all_results, args.report_dir, timestamp)

    # Split manifest
    manifest_path = args.report_dir / f"split_manifest_{timestamp}.json"
    manifest = {s["scene_id"]: s["split"] for s in all_scenes}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[BATCH] Split manifest → {manifest_path}")


if __name__ == "__main__":
    main()