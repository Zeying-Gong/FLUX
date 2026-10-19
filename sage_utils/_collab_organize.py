#!/usr/bin/env python3
"""Move raw data into mode-based directory structure.

Raw layout:
  {run_dir}/raw/{scene_id}/{robot}_{camera}/
    episode_0000/
      frames/             (deleted)
      target_crops/       (deleted)
      task_description_*.txt (deleted)
      {ep_num}.json
      {ep_num}_info.json
      track_object.jpg
      rgb_video.mp4
      depth_video.mp4
      camera_info.json
      rgb/
      depth/

Target layout (via rename, zero copy):
  {run_dir}/stt/{scene_id}/{ep_num}/{robot}_{camera}/
    (same files, minus the deleted ones)

Afterward raw/ is removed.
"""
import argparse
import json
import os
import shutil

SKIP_DIRS = {"frames", "target_crops", "__pycache__"}


def _is_skip(name: str) -> bool:
    return name in SKIP_DIRS or name.startswith("task_description_ep")


def _move_episode(ep_src: str, target_dir: str):
    """Move episode contents into target_dir, skip unwanted files."""
    os.makedirs(target_dir, exist_ok=True)
    for name in os.listdir(ep_src):
        if _is_skip(name):
            path = os.path.join(ep_src, name)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
            continue
        shutil.move(os.path.join(ep_src, name), os.path.join(target_dir, name))
    # Remove the now-empty episode dir
    try:
        os.rmdir(ep_src)
    except OSError:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--run_name", required=True)
    args = p.parse_args()

    run_dir = args.run_dir
    episodes_dir = os.path.join(run_dir, "episodes")
    raw_dir = os.path.join(run_dir, "raw")

    if not os.path.isdir(raw_dir):
        print("  [Phase 3] No raw data found at", raw_dir)
        return

    for scene_id in sorted(os.listdir(raw_dir)):
        scene_raw = os.path.join(raw_dir, scene_id)
        if not os.path.isdir(scene_raw):
            continue

        for combo_name in sorted(os.listdir(scene_raw)):
            combo_raw = os.path.join(scene_raw, combo_name)
            if not os.path.isdir(combo_raw):
                continue

            for item in sorted(os.listdir(combo_raw)):
                ep_src = os.path.join(combo_raw, item)
                if not os.path.isdir(ep_src):
                    continue
                if not item.startswith("episode_"):
                    continue
                ep_idx = int(item.split("_")[1])

                # Determine mode from the episode JSON
                ep_json_path = os.path.join(episodes_dir, scene_id, f"episode_{ep_idx}.json")
                mode = "dt"
                ori_episode_id = ep_idx
                if os.path.isfile(ep_json_path):
                    with open(ep_json_path) as f:
                        ep_data = json.load(f)
                    mode = ep_data.get("episode", {}).get("mode", "dt")
                    ori_episode_id = ep_data.get("episode", {}).get("episode_id", ep_idx)

                # ── Target: {mode}/{scene_id}/{ep_idx}/{combo_name} ──
                target_dir = os.path.join(run_dir, mode, scene_id, str(ep_idx), combo_name)
                os.makedirs(target_dir, exist_ok=True)

                # ── Move episode contents ─────────────────────────────
                _move_episode(ep_src, target_dir)

                # ── Patch {ep_num}.json ──────────────────────────────
                summary_path = os.path.join(target_dir, f"{ep_idx}.json")
                if os.path.isfile(summary_path):
                    with open(summary_path) as f:
                        summary = json.load(f)
                    summary["episode_id"] = ep_idx
                    summary["ori_episode_id"] = ori_episode_id
                    summary["mode"] = mode
                    with open(summary_path, "w") as f:
                        json.dump(summary, f, indent=2)

                print(f"  [Phase 3] {mode}/{scene_id}/{ep_idx}/{combo_name}")

            # Remove empty combo dir
            try:
                os.rmdir(combo_raw)
            except OSError:
                pass

        # Remove empty scene dir
        try:
            os.rmdir(scene_raw)
        except OSError:
            pass

    # ── Remove raw directory ──────────────────────────────────────────
    try:
        os.rmdir(raw_dir)
    except OSError:
        pass
    print("  [Phase 3] Done.")


if __name__ == "__main__":
    main()
