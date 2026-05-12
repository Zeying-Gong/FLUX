#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_episode_generator.py
──────────────────────────
Batch scheduler for generate_episode_json_interaction.py.

Features
--------
- Discover all scenes automatically from USDA / semantic-map directories
- Generate N episodes per scene with configurable seeds
- Skip existing episodes (resume-safe)
- Per-scene + global progress tracking
- Quality report table (CSV + pretty print) after completion
- 7:2:1 train/val/test split assignment

Directory layout assumed:
    SAGE-3D_data/
    ├── usda/              <scene_id>.usda
    ├── 2D_Semantic_Map/   2D_Semantic_Map_<scene_id>_Complete.json
    ├── scene_text/        semantic_map_<folder_id>.txt   (e.g. 0001_839920)
    └── episodes/          <scene_id>/episode_<N>.json    ← output

The mapping between folder_id (e.g. 0001_839920) and scene_id (e.g. 839920)
is handled automatically.

Usage
-----
    # Quick test: 2 scenes × 5 episodes
    python batch_episode_generator.py \\
        --num_scenes 2 --episodes_per_scene 5 \\
        --use_llm --dry_run

    # Full run
    python batch_episode_generator.py \\
        --episodes_per_scene 100 --use_llm

    # Resume (skips already-generated episodes)
    python batch_episode_generator.py --episodes_per_scene 100 --use_llm
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
import datetime
from pathlib import Path

# ── Default paths ──────────────────────────────────────────────────────────
DATA_ROOT        = Path("/workspace/SAGE-3D_Official/SAGE-3D_data")
USDA_DIR         = DATA_ROOT / "usda"
SEM_MAP_DIR      = DATA_ROOT / "semantic_maps"
SCENE_TEXT_DIR   = DATA_ROOT / "scene_text"
EPISODES_DIR     = DATA_ROOT / "episodes"
REPORT_DIR       = DATA_ROOT / "reports"

GENERATOR_SCRIPT = Path(__file__).parent / "generate_episode_json_interaction.py"
ISAAC_PYTHON     = Path("/isaac-sim/python.sh")

# ── People count: capped at MAX_PEOPLE_CAP, derived from navigable area ────────
MAX_PEOPLE_CAP   = 10   # absolute upper limit ("one family")

# ── Train / val / test split ──────────────────────────────────────────────
SPLIT_RATIOS     = {"train": 0.7, "val": 0.2, "test": 0.1}


# ═══════════════════════════════════════════════════════════════════════════
# Scene discovery
# ═══════════════════════════════════════════════════════════════════════════

def _discover_scenes(usda_dir: Path, sem_map_dir: Path,
                     cache_dir: Path | None = None) -> list[dict]:
    """
    Discover all valid scenes. num_people is read from cache if available,
    otherwise computed and cached for future runs.

    Cache file: <cache_dir>/scene_num_people.json
    Default cache location: <sem_map_dir>/../scene_meta/
    """
    if cache_dir is None:
        cache_dir = sem_map_dir.parent / "scene_meta"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "scene_num_people.json"

    # Load existing cache
    num_people_cache: dict[str, int] = {}
    if cache_path.exists():
        try:
            num_people_cache = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"[DISCOVER] Loaded num_people cache: {len(num_people_cache)} entries")
        except Exception:
            pass

    interiorgs_dir = sem_map_dir.parent / "InteriorGS"
    scenes = []
    cache_updated = False

    for folder in sorted(interiorgs_dir.iterdir()):
        if not folder.is_dir():
            continue
        folder_id = folder.name                   # e.g. "0033_839873"
        scene_id  = folder_id.split("_", 1)[-1]  # e.g. "839873"
        usda_path = usda_dir / f"{scene_id}.usda"
        sem_map   = sem_map_dir / f"2D_Semantic_Map_{folder_id}_Complete.json"
        if not usda_path.exists() or not sem_map.exists():
            continue
        txt_candidates = list(SCENE_TEXT_DIR.glob(f"semantic_map_{folder_id}*.txt"))
        scene_text = txt_candidates[0] if txt_candidates else None

        # Use cache if available, otherwise compute and cache
        if folder_id in num_people_cache:
            num_people = num_people_cache[folder_id]
        else:
            print(f"[DISCOVER] Computing num_people for {folder_id} ...")
            num_people = _compute_num_people(sem_map)
            num_people_cache[folder_id] = num_people
            cache_updated = True

        scenes.append({
            "scene_id":        scene_id,
            "folder_id":       folder_id,
            "usda_path":       usda_path,
            "sem_map_path":    sem_map,
            "scene_text_path": scene_text,
            "num_people":      num_people,
        })

    # Save updated cache
    if cache_updated:
        cache_path.write_text(
            json.dumps(num_people_cache, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"[DISCOVER] Cache saved → {cache_path} ({len(num_people_cache)} entries)")

    return scenes


def _compute_num_people(sem_map_path: Path,
                        robot_radius_m: float = 0.3,
                        scale: float = 0.05) -> int:
    """Derive num_people from navigable area: floor(area_m2 / 20), capped at MAX_PEOPLE_CAP."""
    import numpy as _np
    from scipy.ndimage import distance_transform_edt as _edt
    try:
        with open(sem_map_path, encoding="utf-8") as f:
            sem_data = json.load(f)
        all_y, all_x = [], []
        for inst in sem_data:
            for y, x in inst.get("mask_coords_m", []):
                try: all_y.append(float(y)); all_x.append(float(x))
                except: pass
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
                    except: pass
        if robot_radius_m > 0:
            dist_m = _edt(grid == 0, sampling=scale)
            grid = (dist_m <= robot_radius_m).astype(_np.uint8)
        nav_area_m2 = int((grid == 0).sum()) * scale ** 2
        return max(1, min(int(nav_area_m2 / 20), MAX_PEOPLE_CAP))
    except Exception:
        return 2


def _assign_splits(scenes: list[dict], seed: int = 42) -> list[dict]:
    """Assign train/val/test split deterministically."""
    rng = random.Random(seed)
    shuffled = scenes[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val   = int(n * SPLIT_RATIOS["val"])
    for i, s in enumerate(shuffled):
        if   i < n_train:        s["split"] = "train"
        elif i < n_train + n_val: s["split"] = "val"
        else:                    s["split"] = "test"
    # Restore original order with split assigned
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
    return {int(p.stem.split("_")[1])
            for p in d.glob("episode_*.json")
            if "_log" not in p.stem and "_vis" not in p.stem}


# ═══════════════════════════════════════════════════════════════════════════
# Single episode generation
# ═══════════════════════════════════════════════════════════════════════════

def _run_scene(
    scene: dict,
    episode_ids: list[int],
    num_people: int,
    args: argparse.Namespace,
    dry_run: bool,
) -> list[dict]:
    """
    Run generate_episode_json_interaction.py ONCE for all episode_ids of a scene.
    Isaac Sim starts once, loops over all episodes, then exits.
    Returns list of result dicts (one per episode).
    """
    episode_ids_str = [str(i) for i in episode_ids]
    cmd = [
        str(args.isaac_python),
        str(args.generator_script),
        "--usda",              str(scene["usda_path"]),
        "--semantic_map_json", str(scene["sem_map_path"]),
        "--output_dir",        str(args.output_dir),
        "--episode_ids",       *episode_ids_str,
        "--seed_offset",       str(args.seed_offset),
        "--num_people",        str(num_people),
        "--num_waypoints",     str(args.num_waypoints),
        "--robot_radius_2d",   str(args.robot_radius_2d),
        "--scale_m_per_px",    str(args.scale_m_per_px),
    ]
    if args.use_llm and scene.get("scene_text_path"):
        cmd += ["--use_llm",
                "--scene_text_file", str(scene["scene_text_path"]),
                "--llm_base_url",    args.llm_base_url,
                "--llm_model",       args.llm_model,
                "--llm_temperature", str(args.llm_temperature)]
    if args.overwrite:
        cmd.append("--overwrite")

    print(f"  cmd: {' '.join(cmd[:6])} ... episode_ids={episode_ids}")

    results = []
    if dry_run:
        for ep_id in episode_ids:
            print(f"  [DRY] episode_{ep_id}")
            results.append({
                "scene_id": scene["scene_id"], "split": scene.get("split","?"),
                "episode_id": ep_id, "seed": args.seed_offset + ep_id,
                "num_people": num_people, "llm_mode": False,
                "status": "dry_run", "elapsed_s": 0.0,
                "mode": "?", "num_placed": 0, "nav_m2": 0.0,
                "narrative": "", "error": "",
            })
        return results

    t0 = time.time()
    timeout = args.timeout_per_episode * len(episode_ids)
    try:
        # capture_output=False: Isaac Sim subprocess prints flow to terminal in real time
        proc = subprocess.run(cmd, timeout=timeout)
        elapsed_total = round(time.time() - t0, 1)

        for ep_id in episode_ids:
            log_path = (Path(args.output_dir) / scene["scene_id"] /
                        f"episode_{ep_id}_log.json")
            r = {
                "scene_id":   scene["scene_id"],
                "split":      scene.get("split", "?"),
                "episode_id": ep_id,
                "seed":       args.seed_offset + ep_id,
                "num_people": num_people,
                "llm_mode":   args.use_llm and bool(scene.get("scene_text_path")),
                "elapsed_s":  round(elapsed_total / len(episode_ids), 1),
                "mode": "?", "num_placed": 0, "nav_m2": 0.0,
                "narrative": "", "error": "",
            }
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
            results.append({
                "scene_id": scene["scene_id"], "split": scene.get("split","?"),
                "episode_id": ep_id, "seed": args.seed_offset + ep_id,
                "num_people": num_people, "status": "timeout",
                "elapsed_s": timeout, "mode": "?", "num_placed": 0,
                "nav_m2": 0.0, "narrative": "", "error": "timeout", "llm_mode": False,
            })
        print(f"  [TIMEOUT] all episodes after {timeout}s")
    except Exception as e:
        for ep_id in episode_ids:
            results.append({
                "scene_id": scene["scene_id"], "split": scene.get("split","?"),
                "episode_id": ep_id, "seed": args.seed_offset + ep_id,
                "num_people": num_people, "status": "error",
                "elapsed_s": 0, "mode": "?", "num_placed": 0,
                "nav_m2": 0.0, "narrative": "", "error": str(e), "llm_mode": False,
            })
        print(f"  [ERROR] {e}")

    return results

# ═══════════════════════════════════════════════════════════════════════════
# Quality report
# ═══════════════════════════════════════════════════════════════════════════

def _write_report(results: list[dict], report_dir: Path, timestamp: str):
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"quality_report_{timestamp}.csv"
    fields = ["scene_id", "split", "episode_id", "seed", "num_people",
              "llm_mode", "status", "mode", "num_placed", "nav_m2",
              "elapsed_s", "narrative", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(results)
    print(f"\n[REPORT] CSV → {csv_path}")

    # Per-scene summary table
    from collections import defaultdict
    per_scene: dict[str, dict] = defaultdict(lambda: {
        "split": "?", "total": 0, "ok": 0, "fail": 0, "timeout": 0,
        "llm_ok": 0, "random": 0, "nav_m2": 0.0,
    })
    for r in results:
        sid = r["scene_id"]
        per_scene[sid]["split"]  = r.get("split", "?")
        per_scene[sid]["total"] += 1
        per_scene[sid][r.get("status","fail")] = \
            per_scene[sid].get(r.get("status","fail"), 0) + 1
        if r.get("mode") == "llm":  per_scene[sid]["llm_ok"] += 1
        if r.get("mode") == "random": per_scene[sid]["random"] += 1
        if r.get("nav_m2", 0): per_scene[sid]["nav_m2"] = r["nav_m2"]

    # Pretty table
    hdr = f"{'scene_id':>12} {'split':>6} {'ok':>5} {'fail':>5} "     \
          f"{'timeout':>8} {'llm':>5} {'rand':>5} {'nav_m2':>8}"
    sep = "─" * len(hdr)
    print("\n" + sep)
    print(hdr)
    print(sep)
    for sid, s in sorted(per_scene.items()):
        print(f"{sid:>12} {s['split']:>6} {s['ok']:>5} {s['fail']:>5} "
              f"{s.get('timeout',0):>8} {s['llm_ok']:>5} {s['random']:>5} "
              f"{s['nav_m2']:>8.0f}")
    print(sep)

    total_ok   = sum(1 for r in results if r.get("status") == "ok")
    total_fail = sum(1 for r in results if r.get("status") in ("fail","error","timeout"))
    print(f"\n  Total episodes: {len(results)}  ✓ {total_ok}  ✗ {total_fail}")

    # Summary JSON
    summary_path = report_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "generated_at": timestamp,
            "total": len(results), "ok": total_ok, "fail": total_fail,
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
                   help="Limit number of scenes (e.g. 2 for testing). "
                        "Default: all discovered scenes.")
    p.add_argument("--only_scenes",         nargs="+",  default=None,
                   help="Process only these scene_ids.")
    # Episode config
    p.add_argument("--episodes_per_scene",  type=int,   default=5)
    p.add_argument("--seed_offset",         type=int,   default=0,
                   help="episode_id N uses seed = seed_offset + N.")
    p.add_argument("--num_waypoints",       type=int,   default=3)
    p.add_argument("--overwrite",           action="store_true")

    # Paths
    p.add_argument("--usda_dir",            type=Path,  default=USDA_DIR)
    p.add_argument("--sem_map_dir",         type=Path,  default=SEM_MAP_DIR)
    p.add_argument("--scene_text_dir",      type=Path,  default=SCENE_TEXT_DIR)
    p.add_argument("--output_dir",          type=Path,  default=EPISODES_DIR)
    p.add_argument("--report_dir",          type=Path,  default=REPORT_DIR)
    p.add_argument("--cache_dir",           type=Path,  default=None,
                   help="Cache dir for scene_num_people.json. "
                        "Default: <SAGE-3D_data>/scene_meta/")
    p.add_argument("--generator_script",    type=Path,  default=GENERATOR_SCRIPT)
    p.add_argument("--isaac_python",        type=Path,  default=ISAAC_PYTHON)

    # 2D map params (passed through to generator)
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
                   help="Timeout in seconds per episode subprocess.")
    p.add_argument("--dry_run",             action="store_true",
                   help="Print commands without executing.")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    t_global  = time.time()

    # ── Discover scenes ────────────────────────────────────────────────
    print(f"[BATCH] Discovering scenes under {args.usda_dir} ...")
    all_scenes = _discover_scenes(args.usda_dir, args.sem_map_dir,
                                    cache_dir=args.cache_dir)

    # Override scene_text_dir if custom
    if args.scene_text_dir != SCENE_TEXT_DIR:
        for s in all_scenes:
            candidates = list(args.scene_text_dir.glob(
                f"semantic_map_*{s['scene_id']}*.txt"))
            s["scene_text_path"] = candidates[0] if candidates else None

    if not all_scenes:
        print("[FATAL] No scenes found."); sys.exit(1)

    # Filter by --only_scenes
    if args.only_scenes:
        all_scenes = [s for s in all_scenes if s["scene_id"] in args.only_scenes]

    # Limit
    if args.num_scenes:
        all_scenes = all_scenes[:args.num_scenes]

    print(f"[BATCH] {len(all_scenes)} scenes selected")
    txt_count = sum(1 for s in all_scenes if s["scene_text_path"])
    print(f"[BATCH] Scene text available: {txt_count}/{len(all_scenes)}")
    for s in all_scenes:
        s["split"] = "?"   # splits assigned manually later

    target_total = len(all_scenes) * args.episodes_per_scene
    print(f"[BATCH] Target: {target_total} episodes "
          f"({len(all_scenes)} scenes × {args.episodes_per_scene})\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-scene loop ─────────────────────────────────────────────────
    all_results: list[dict] = []

    for si, scene in enumerate(all_scenes, 1):
        sid = scene["scene_id"]
        print(f"\n{'═'*60}")
        print(f"[SCENE {si}/{len(all_scenes)}] {sid}  split={scene['split']}")
        if scene["scene_text_path"]:
            print(f"  scene_text: {scene['scene_text_path'].name}")
        else:
            print(f"  scene_text: (not found — LLM will be skipped for this scene)")

        num_people = scene["num_people"]
        print(f"  num_people: {num_people} (from navigable area)")

        existing = _existing_episodes(sid, args.output_dir)
        needed   = [i for i in range(args.episodes_per_scene) if i not in existing]
        if not needed:
            print(f"  [SKIP] All {args.episodes_per_scene} episodes already exist")
            # Load existing logs into results for the report
            for ep_id in existing:
                log_path = args.output_dir / sid / f"episode_{ep_id}_log.json"
                if log_path.exists():
                    with open(log_path) as f:
                        log = json.load(f)
                    all_results.append({
                        "scene_id":   sid, "split": scene["split"],
                        "episode_id": ep_id, "seed": log.get("seed", -1),
                        "num_people": log.get("num_requested", 0),
                        "llm_mode":   log.get("mode") == "llm",
                        "status":     "ok (cached)",
                        "mode":       log.get("mode", "?"),
                        "num_placed": log.get("num_placed", 0),
                        "nav_m2":     log.get("navigable_m2", 0.0),
                        "elapsed_s":  log.get("elapsed_s", 0.0),
                        "narrative":  log.get("narrative", "")[:80],
                        "error":      "",
                    })
            continue

        print(f"  episodes to generate: {needed}")

        scene_results = _run_scene(
            scene=scene, episode_ids=needed,
            num_people=num_people,
            args=args, dry_run=args.dry_run,
        )
        all_results.extend(scene_results)
        ep_dir = args.output_dir / sid
        print(f"  output dir: {ep_dir}")
        for r in scene_results:
            icon = "✓" if r["status"] == "ok" else "✗"
            print(f"  {icon} ep_{r['episode_id']} [{r['status']}] "
                  f"{r['elapsed_s']}s mode={r['mode']} placed={r['num_placed']}")

    # ── Global summary & report ────────────────────────────────────────
    total_elapsed = round(time.time() - t_global, 1)
    print(f"\n{'═'*60}")
    print(f"[BATCH] All done in {total_elapsed}s")
    _write_report(all_results, args.report_dir, timestamp)

    # Save split manifest
    manifest_path = args.report_dir / f"split_manifest_{timestamp}.json"
    manifest = {s["scene_id"]: s["split"] for s in all_scenes}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[BATCH] Split manifest → {manifest_path}")


if __name__ == "__main__":
    main()