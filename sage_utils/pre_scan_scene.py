#!/usr/bin/env python3
"""Pre-scan v3_tracking_episodes for a scene and cache episode classification.

Usage:
  python3 pre_scan_scene.py 0003_839989

Writes to sage_utils/episode_caches/{scene_id}.json
"""
import json
import os
import sys

CACHE_DIR = os.path.join(os.path.dirname(__file__), "episode_caches")
V3_DIR = "/mnt/ssd1/zeyingg/SAGE-3D_Official/SAGE-3D_data/v3_tracking_episodes"


def scan(scene_id: str) -> dict:
    scene_dir = os.path.join(V3_DIR, scene_id)
    if not os.path.isdir(scene_dir):
        print(f"[ERROR] Scene directory not found: {scene_dir}", file=sys.stderr)
        sys.exit(1)

    stt, multi = [], []
    for fname in sorted(os.listdir(scene_dir)):
        if not fname.startswith("episode_") or not fname.endswith(".json"):
            continue
        fpath = os.path.join(scene_dir, fname)
        with open(fpath) as f:
            d = json.load(f)
        n = d["episode"]["characters"]["num_characters"]
        eid = d["episode"]["episode_id"]
        if n == 1:
            stt.append(eid)
        elif n >= 2:
            multi.append(eid)

    stt.sort()
    multi.sort()
    at = sorted([i for i in multi if i % 2 == 1])
    dt = sorted([i for i in multi if i % 2 == 0])

    cache = {
        "scene_id": scene_id,
        "total": len(stt) + len(multi),
        "stt_ids": stt,
        "multi_ids": multi,
        "at_ids": at,
        "dt_ids": dt,
    }
    return cache


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <scene_id> [scene_id ...]", file=sys.stderr)
        sys.exit(1)

    os.makedirs(CACHE_DIR, exist_ok=True)

    total_stt, total_at, total_dt = 0, 0, 0
    for scene_id in sys.argv[1:]:
        cache = scan(scene_id)
        path = os.path.join(CACHE_DIR, f"{scene_id}.json")
        with open(path, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"  {scene_id}: {cache['total']} eps → "
              f"STT={len(cache['stt_ids'])}  AT={len(cache['at_ids'])}  DT={len(cache['dt_ids'])}  "
              f"cached at {path}")
        total_stt += len(cache["stt_ids"])
        total_at += len(cache["at_ids"])
        total_dt += len(cache["dt_ids"])

    print(f"  Total across {len(sys.argv)-1} scene(s): STT={total_stt}  AT={total_at}  DT={total_dt}")


if __name__ == "__main__":
    main()
