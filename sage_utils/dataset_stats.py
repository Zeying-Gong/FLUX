#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_stats.py
────────────────
Read all episode_*_log.json files under the episodes directory and produce:
  1. A quality summary table (printed + saved as CSV)
  2. Distribution plots (people count, nav area, mode, split)
  3. A per-scene health table highlighting scenes with issues

Usage
-----
    python dataset_stats.py
    python dataset_stats.py --episodes_dir .../episodes --output_dir .../reports
    python dataset_stats.py --no_plot        # table only, no matplotlib
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

DATA_ROOT    = Path("/workspace/SAGE-3D_Official/SAGE-3D_data")
EPISODES_DIR = DATA_ROOT / "episodes"
OUTPUT_DIR   = DATA_ROOT / "reports"


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_all_logs(episodes_dir: Path) -> list[dict]:
    logs = []
    for log_path in sorted(episodes_dir.rglob("episode_*_log.json")):
        try:
            with open(log_path, encoding="utf-8") as f:
                d = json.load(f)
            d["_log_path"] = str(log_path)
            logs.append(d)
        except Exception as e:
            print(f"[WARN] Cannot read {log_path}: {e}")
    return logs


def load_split_manifest(reports_dir: Path) -> dict[str, str]:
    """Load the most recent split manifest if available."""
    manifests = sorted(reports_dir.glob("split_manifest_*.json"))
    if not manifests:
        return {}
    with open(manifests[-1]) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════

def compute_stats(logs: list[dict], split_map: dict[str, str]) -> dict:
    total = len(logs)
    if total == 0:
        return {}

    scene_stats: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "llm": 0, "random": 0,
        "placed_ok": 0, "placed_partial": 0,
        "nav_m2_list": [],
        "elapsed_list": [],
        "split": "?",
    })

    people_counts  = []
    nav_areas      = []
    modes          = []
    elapsed_list   = []
    llm_log_issues = 0

    for d in logs:
        sid     = d.get("scene_id", "unknown")
        mode    = d.get("mode", "random")
        placed  = d.get("num_placed", 0)
        req     = d.get("num_requested", placed)
        nav_m2  = d.get("navigable_m2", 0.0)
        elapsed = d.get("elapsed_s", 0.0)
        llm_log = d.get("llm_log", [])

        scene_stats[sid]["total"] += 1
        scene_stats[sid]["split"] = split_map.get(sid, "?")
        if mode == "llm":
            scene_stats[sid]["llm"]    += 1
        else:
            scene_stats[sid]["random"] += 1

        if placed >= req:
            scene_stats[sid]["placed_ok"] += 1
        else:
            scene_stats[sid]["placed_partial"] += 1

        scene_stats[sid]["nav_m2_list"].append(nav_m2)
        scene_stats[sid]["elapsed_list"].append(elapsed)

        people_counts.append(req)
        nav_areas.append(nav_m2)
        modes.append(mode)
        elapsed_list.append(elapsed)

        # Count LLM issues
        llm_log_issues += sum(
            1 for line in llm_log
            if any(kw in line for kw in ("FAILED", "fail", "invalid", "fallback"))
        )

    return {
        "total_episodes": total,
        "total_scenes":   len(scene_stats),
        "people_counts":  people_counts,
        "nav_areas":      nav_areas,
        "modes":          modes,
        "elapsed_list":   elapsed_list,
        "llm_log_issues": llm_log_issues,
        "scene_stats":    dict(scene_stats),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Pretty-print table
# ═══════════════════════════════════════════════════════════════════════════

def print_scene_table(stats: dict):
    if not stats:
        print("[STATS] No data."); return

    ss = stats["scene_stats"]
    hdr = (f"{'scene_id':>12} {'split':>6} {'eps':>5} "
           f"{'llm':>5} {'rand':>5} {'partial':>8} "
           f"{'nav_m2':>8} {'avg_s':>7}  {'health':>8}")
    sep = "─" * len(hdr)
    print("\n" + sep)
    print(hdr)
    print(sep)

    issues = []
    for sid, s in sorted(ss.items()):
        avg_nav = (sum(s["nav_m2_list"]) / len(s["nav_m2_list"])
                   if s["nav_m2_list"] else 0)
        avg_s   = (sum(s["elapsed_list"]) / len(s["elapsed_list"])
                   if s["elapsed_list"] else 0)
        partial = s["placed_partial"]
        health  = "⚠ partial" if partial > s["total"] * 0.3 else "OK"
        if health != "OK":
            issues.append(sid)
        print(f"{sid:>12} {s['split']:>6} {s['total']:>5} "
              f"{s['llm']:>5} {s['random']:>5} {partial:>8} "
              f"{avg_nav:>8.0f} {avg_s:>7.1f}  {health:>8}")
    print(sep)

    modes_count = defaultdict(int)
    for m in stats["modes"]: modes_count[m] += 1
    print(f"\n  Total episodes : {stats['total_episodes']}")
    print(f"  Total scenes   : {stats['total_scenes']}")
    print(f"  Mode LLM       : {modes_count['llm']}  "
          f"({100*modes_count['llm']/max(1,stats['total_episodes']):.1f}%)")
    print(f"  Mode random    : {modes_count['random']}")
    print(f"  LLM log issues : {stats['llm_log_issues']}")
    if issues:
        print(f"\n  ⚠  Scenes with >30% partial placement: {', '.join(issues)}")


def save_csv(stats: dict, output_path: Path):
    ss = stats["scene_stats"]
    fields = ["scene_id", "split", "total", "llm", "random",
              "placed_ok", "placed_partial", "avg_nav_m2", "avg_elapsed_s"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sid, s in sorted(ss.items()):
            avg_nav = (sum(s["nav_m2_list"]) / len(s["nav_m2_list"])
                       if s["nav_m2_list"] else 0)
            avg_s   = (sum(s["elapsed_list"]) / len(s["elapsed_list"])
                       if s["elapsed_list"] else 0)
            w.writerow({
                "scene_id":        sid,
                "split":           s["split"],
                "total":           s["total"],
                "llm":             s["llm"],
                "random":          s["random"],
                "placed_ok":       s["placed_ok"],
                "placed_partial":  s["placed_partial"],
                "avg_nav_m2":      round(avg_nav, 1),
                "avg_elapsed_s":   round(avg_s, 1),
            })
    print(f"[CSV] → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════

def make_plots(stats: dict, output_dir: Path):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots"); return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dataset Statistics", fontsize=14, fontweight="bold")

    # ── People count distribution ─────────────────────────────────────
    ax = axes[0, 0]
    from collections import Counter
    cnt = Counter(stats["people_counts"])
    ax.bar([str(k) for k in sorted(cnt)], [cnt[k] for k in sorted(cnt)],
           color="#4D96FF")
    ax.set_title("Pedestrian Count Distribution")
    ax.set_xlabel("num_people"); ax.set_ylabel("episodes")

    # ── Navigable area distribution ───────────────────────────────────
    ax = axes[0, 1]
    valid_nav = [v for v in stats["nav_areas"] if v > 0]
    if valid_nav:
        ax.hist(valid_nav, bins=30, color="#6BCB77", edgecolor="white")
    ax.set_title("Navigable Area Distribution")
    ax.set_xlabel("m²"); ax.set_ylabel("episodes")

    # ── Mode pie chart ────────────────────────────────────────────────
    ax = axes[1, 0]
    mode_cnt = Counter(stats["modes"])
    labels = list(mode_cnt.keys())
    sizes  = [mode_cnt[l] for l in labels]
    ax.pie(sizes, labels=labels, autopct="%1.1f%%",
           colors=["#FF6B6B", "#FFD93D"])
    ax.set_title("Generation Mode")

    # ── Elapsed time per episode ──────────────────────────────────────
    ax = axes[1, 1]
    valid_t = [t for t in stats["elapsed_list"] if t > 0]
    if valid_t:
        ax.hist(valid_t, bins=30, color="#FF6B6B", edgecolor="white")
    ax.set_title("Elapsed Time per Episode")
    ax.set_xlabel("seconds"); ax.set_ylabel("episodes")

    plt.tight_layout()
    plot_path = output_dir / "dataset_stats.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[PLOT] → {plot_path}")

    # ── Per-scene health bar chart ────────────────────────────────────
    ss = stats["scene_stats"]
    if len(ss) > 1:
        fig2, ax2 = plt.subplots(figsize=(max(10, len(ss)*0.6), 5))
        scene_ids  = sorted(ss.keys())
        ok_vals    = [ss[s]["placed_ok"]      for s in scene_ids]
        part_vals  = [ss[s]["placed_partial"]  for s in scene_ids]
        x = range(len(scene_ids))
        ax2.bar(x, ok_vals,   label="fully placed",   color="#6BCB77")
        ax2.bar(x, part_vals, bottom=ok_vals,
                label="partial placement", color="#FF6B6B")
        ax2.set_xticks(list(x)); ax2.set_xticklabels(scene_ids, rotation=45, ha="right")
        ax2.set_ylabel("episodes"); ax2.set_title("Per-Scene Placement Health")
        ax2.legend()
        plt.tight_layout()
        health_path = output_dir / "per_scene_health.png"
        plt.savefig(health_path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[PLOT] → {health_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Dataset quality statistics.")
    p.add_argument("--episodes_dir", type=Path, default=EPISODES_DIR)
    p.add_argument("--output_dir",   type=Path, default=OUTPUT_DIR)
    p.add_argument("--reports_dir",  type=Path, default=OUTPUT_DIR,
                   help="Directory to look for split_manifest_*.json")
    p.add_argument("--no_plot",      action="store_true",
                   help="Skip matplotlib plots.")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[STATS] Scanning {args.episodes_dir} ...")
    logs = load_all_logs(args.episodes_dir)
    print(f"[STATS] {len(logs)} log files found")

    if not logs:
        print("[STATS] Nothing to report."); return

    split_map = load_split_manifest(args.reports_dir)
    stats     = compute_stats(logs, split_map)

    print_scene_table(stats)
    save_csv(stats, args.output_dir / "scene_quality.csv")

    if not args.no_plot:
        make_plots(stats, args.output_dir)

    print("\n[STATS] Done.")


if __name__ == "__main__":
    main()