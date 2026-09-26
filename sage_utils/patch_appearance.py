#!/usr/bin/env python3
"""
patch_appearance.py
────────────────────
给【已经生成好】的 tracking episode JSON 补上 appearance 块,
其他内容(robot / characters / commands / tracking)一个字节不动。

复用 clothing_appearance 的分配逻辑,并从每个 episode 自身的
episode_id + seed + 角色顺序派生外观,因此补丁结果与"当初带外观重新
生成"完全一致、可复现。

安全特性:
  • 默认跳过已有非空 appearance 的 episode(--force 可覆盖)。
  • 原子写入(临时文件 + os.replace),写坏不会损毁原文件。
  • --dry_run 只报告不写盘。

用法:
    # 单场景
    python patch_appearance.py \\
        --episode_dir /workspace/SAGE-3D_Official/SAGE-3D_data/v2_tracking_episodes \\
        --scene_ids 0001_839920 \\
        --profiles_json /workspace/FLUX/sage_utils/character_clothing_profiles.json

    # 全部场景(遍历 episode_dir 下所有子目录)
    python patch_appearance.py \\
        --episode_dir /workspace/SAGE-3D_Official/SAGE-3D_data/v2_tracking_episodes \\
        --profiles_json /workspace/FLUX/sage_utils/character_clothing_profiles.json

    # 先 dry-run 看会改哪些
    python patch_appearance.py --episode_dir ... --dry_run
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import tempfile

from clothing_appearance import ClothingProfiles, assign_episode_appearance


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episode_dir", required=True,
                   help="Root dir containing per-scene subdirs of episode_*.json")
    p.add_argument("--scene_ids", nargs="*", default=None,
                   help="Only patch these scene subdir names (default: all subdirs).")
    p.add_argument("--profiles_json",
                   default="/workspace/FLUX/sage_utils/character_clothing_profiles.json")
    p.add_argument("--force", action="store_true",
                   help="Overwrite even if appearance already exists.")
    p.add_argument("--dry_run", action="store_true",
                   help="Report what would change, write nothing.")
    return p.parse_args()


def extract_ep_id_from_name(path: str) -> int:
    """从文件名 episode_<N>.json 提取 N(仅作排序兜底用)。"""
    m = re.search(r"episode_(\d+)\.json$", os.path.basename(path))
    return int(m.group(1)) if m else 999999


def patch_one_file(path: str, profiles: ClothingProfiles,
                   force: bool, dry_run: bool) -> str:
    """
    返回状态字符串: 'patched' / 'skipped_existing' / 'error:<msg>' / 'would_patch'
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"error:read_failed({e})"

    ep = data.get("episode")
    if not isinstance(ep, dict):
        return "error:no_episode_block"

    # 幂等:已有非空 appearance 且未 force → 跳过
    if ep.get("appearance") not in (None, {}) and not force:
        return "skipped_existing"

    # 取 episode_id / seed —— 必须和生成端一致才能复现
    episode_id = ep.get("episode_id")
    seed = ep.get("seed")
    if episode_id is None or seed is None:
        return "error:missing_episode_id_or_seed"

    # 角色顺序:spawn_positions 的 key 顺序(第0个是 target "Character")
    try:
        spawn_positions = ep["characters"]["spawn_positions"]
    except (KeyError, TypeError):
        return "error:no_spawn_positions"
    char_names = list(spawn_positions.keys())
    if not char_names:
        return "error:empty_spawn_positions"

    # 生成 appearance(与新生成路径同一函数、同一派生)
    try:
        appearance_block = assign_episode_appearance(
            profiles, int(episode_id), int(seed), char_names)
    except Exception as e:
        return f"error:assign_failed({e})"

    if dry_run:
        return "would_patch"

    # 只加这一个键,其他不动
    ep["appearance"] = appearance_block

    # 原子写入:同目录临时文件 → os.replace
    try:
        dir_name = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=dir_name)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        os.replace(tmp_path, path)
    except Exception as e:
        # 清理可能残留的临时文件
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return f"error:write_failed({e})"

    return "patched"


def main():
    args = parse_args()

    if not os.path.isdir(args.episode_dir):
        print(f"[FATAL] episode_dir not found: {args.episode_dir}")
        return 1

    try:
        profiles = ClothingProfiles(args.profiles_json)
        print(f"[INFO] Loaded profiles: {profiles.num_assets} assets")
    except Exception as e:
        print(f"[FATAL] cannot load profiles_json: {e}")
        return 1

    # 决定要处理哪些 scene 子目录
    if args.scene_ids:
        scene_dirs = [os.path.join(args.episode_dir, s) for s in args.scene_ids]
    else:
        scene_dirs = [os.path.join(args.episode_dir, d)
                      for d in sorted(os.listdir(args.episode_dir))
                      if os.path.isdir(os.path.join(args.episode_dir, d))
                      and d != "logs"]

    if not scene_dirs:
        print("[FATAL] No scene subdirs to process.")
        return 1

    print(f"[INFO] {len(scene_dirs)} scene(s) to process"
          + (" [DRY RUN]" if args.dry_run else ""))

    totals = {
        "patched": 0, "skipped_existing": 0,
        "would_patch": 0, "error": 0,
    }
    distinct_counter = {"full": 0, "top_only": 0, "partial": 0}

    for scene_dir in scene_dirs:
        scene_id = os.path.basename(scene_dir)
        if not os.path.isdir(scene_dir):
            print(f"[WARN] not a dir, skip: {scene_dir}")
            continue
        files = sorted(glob.glob(os.path.join(scene_dir, "episode_*.json")),
                       key=extract_ep_id_from_name)
        if not files:
            print(f"[WARN] {scene_id}: no episode_*.json")
            continue

        scene_stat = {"patched": 0, "skipped_existing": 0,
                      "would_patch": 0, "error": 0}
        for fp in files:
            status = patch_one_file(fp, profiles, args.force, args.dry_run)

            if status.startswith("error:"):
                totals["error"] += 1
                scene_stat["error"] += 1
                print(f"  [ERR] {os.path.basename(fp)} -> {status}")
            else:
                totals[status] += 1
                scene_stat[status] += 1
                # 统计区分度(只在真正生成/将生成时读一次)
                if status in ("patched", "would_patch"):
                    try:
                        with open(fp, "r", encoding="utf-8") as f:
                            lvl = (json.load(f)["episode"].get("appearance") or {}
                                   ).get("distinct_level")
                        if lvl in distinct_counter:
                            distinct_counter[lvl] += 1
                    except Exception:
                        pass

        print(f"[{scene_id}] patched={scene_stat['patched']} "
              f"skipped={scene_stat['skipped_existing']} "
              f"would_patch={scene_stat['would_patch']} "
              f"err={scene_stat['error']}")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  patched          : {totals['patched']}")
    print(f"  would_patch (dry): {totals['would_patch']}")
    print(f"  skipped existing : {totals['skipped_existing']}")
    print(f"  errors           : {totals['error']}")
    if any(distinct_counter.values()):
        print(f"  distinct_level   : {distinct_counter}")
    if args.dry_run:
        print("\n[DRY RUN] No files were modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())