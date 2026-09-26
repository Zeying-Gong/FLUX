#!/usr/bin/env python3
"""
check_tracking_episodes.py
──────────────────────────
检查 tracking 数据集完整性：
  - 是否所有场景都有对应的 episode 文件夹？
  - 每个场景是否都包含 episode_0.json ~ episode_99.json？

Usage:
    python check_tracking_episodes.py
    (自动读取默认的语义地图和输出目录，也可通过参数指定)
"""
import os
import re
import glob
import argparse
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sem_dir", default="/workspace/SAGE-3D_Official/SAGE-3D_data/semantic_maps")
    p.add_argument("--episode_dir", default="/workspace/SAGE-3D_Official/SAGE-3D_data/v1_tracking_episodes")
    p.add_argument("--total_episodes", type=int, default=100)
    return p.parse_args()

def main():
    args = parse_args()

    # 1. 获取所有场景 ID（从语义地图文件名中提取）
    sem_files = glob.glob(os.path.join(args.sem_dir, "2D_Semantic_Map_*_Complete.json"))
    scene_ids = set()
    for f in sem_files:
        m = re.match(r"2D_Semantic_Map_\d+_(\d+)_Complete\.json", os.path.basename(f))
        if m:
            scene_ids.add(m.group(1))
    total_scenes = len(scene_ids)
    print(f"语义地图中共发现 {total_scenes} 个场景。")

    # 2. 检查每个场景的 episode 目录和文件
    missing_dir = []           # 完全没有文件夹的场景
    incomplete = []            # 文件夹存在但 episode 不全的场景
    episode_missing = defaultdict(list)  # 场景 -> 缺失的 episode_id 列表

    for sid in sorted(scene_ids):
        scene_dir = os.path.join(args.episode_dir, sid)
        if not os.path.isdir(scene_dir):
            missing_dir.append(sid)
            continue

        # 检查 episode_0.json ~ episode_99.json
        missing_eps = []
        for ep_id in range(args.total_episodes):
            ep_file = os.path.join(scene_dir, f"episode_{ep_id}.json")
            if not os.path.isfile(ep_file):
                missing_eps.append(ep_id)
        if missing_eps:
            incomplete.append(sid)
            episode_missing[sid] = missing_eps

    # 3. 输出报告
    print("\n========== 检查报告 ==========")
    print(f"总场景数: {total_scenes}")
    print(f"完全缺失文件夹的场景: {len(missing_dir)}")
    if missing_dir:
        print("  缺失场景ID:", ", ".join(missing_dir))
    print(f"文件夹存在但 episode 不全的场景: {len(incomplete)}")
    if incomplete:
        for sid in incomplete:
            missing = episode_missing[sid]
            print(f"  场景 {sid}: 缺失 {len(missing)} 个 episode, 例如前5个: {missing[:5]}")
    complete = total_scenes - len(missing_dir) - len(incomplete)
    print(f"完整的场景: {complete}")
    if complete == total_scenes:
        print("✅ 所有场景的 episode 均已完整生成！")
    else:
        print("⚠️  存在缺失，请检查上述列表。")

if __name__ == "__main__":
    main()