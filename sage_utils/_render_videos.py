#!/usr/bin/env python3
"""Post-process: render MP4 videos from saved PNG frames.

Run after all Isaac Sim containers complete.  Scans {run_dir}/{mode}/{scene}/*/
for rgb/ and depth/ directories and generates rgb_video.mp4 + depth_video.mp4.

Usage:
  python3 _render_videos.py --run_dir <path>
"""
import argparse
import glob
import os
import sys


def _render_one(ep_dir: str, subdir: str, out_name: str, fps: float = 40.0,
                out_dir: str = None):
    frames = sorted(glob.glob(os.path.join(ep_dir, subdir, "*.png")))
    if not frames:
        return
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, out_name)
    else:
        out_path = os.path.join(ep_dir, out_name)
    if os.path.isfile(out_path):
        return  # already exists
    try:
        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio
        # probe ffmpeg plugin availability early with a clear message
        try:
            imageio.get_writer(out_path, fps=fps, codec="libx264")
        except Exception as e:
            if "imageio-ffmpeg" in str(e) or "ffmpeg" in str(e).lower():
                raise RuntimeError(
                    "imageio-ffmpeg plugin missing. Install it with: "
                    "pip install imageio-ffmpeg"
                )
            raise
        with imageio.get_writer(out_path, fps=fps, codec="libx264") as w:
            for fp in frames:
                w.append_data(imageio.imread(fp))
        print(f"  Video: {out_path} ({len(frames)} frames)")
        return True
    except ImportError as e:
        print(f"  WARN: imageio not available ({e}), skipping videos")
    except Exception as e:
        print(f"  WARN: {out_path} failed: {e}")
    return False


def _render_episode_dir(ep_dir: str, fps: float, out_dir: str = None):
    """Render rgb/depth videos for a single episode directory that may contain
    flat rgb/ + depth/ dirs, or combo subdirs each with their own rgb/depth."""
    found = 0
    items = os.listdir(ep_dir)
    if "rgb" in items:
        _out = out_dir
        if _out:
            _out = os.path.join(_out, os.path.basename(ep_dir.rstrip("/")))
        ok = _render_one(ep_dir, "rgb", "rgb_video.mp4", fps, _out)
        ok = _render_one(ep_dir, "depth", "depth_video.mp4", fps, _out) or ok
        if ok:
            found += 1
    else:
        for combo in items:
            combo_dir = os.path.join(ep_dir, combo)
            if not os.path.isdir(combo_dir):
                continue
            if "rgb" in os.listdir(combo_dir):
                _out = out_dir
                if _out:
                    _out = os.path.join(_out, os.path.basename(ep_dir.rstrip("/")), combo)
                ok = _render_one(combo_dir, "rgb", "rgb_video.mp4", fps, _out)
                ok = _render_one(combo_dir, "depth", "depth_video.mp4", fps, _out) or ok
                if ok:
                    found += 1
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default=None,
                   help="Scan this run dir for {mode}/{scene}/{ep}/{combo}/ "
                        "structures and render all episodes.")
    p.add_argument("--episode_dir", default=None,
                   help="Render a single episode directory only (flat rgb/+depth/ "
                        "or combo subdirs). Use for incremental rendering.")
    p.add_argument("--out_dir", default=None,
                   help="Where to write mp4 files. Defaults to next to the png "
                        "frames. Use this when the png dir is not writable "
                        "(e.g. created by docker as root).")
    p.add_argument("--fps", type=float, default=20.0)
    args = p.parse_args()

    if args.episode_dir:
        if not os.path.isdir(args.episode_dir):
            print(f"[VideoRender] episode_dir not found: {args.episode_dir}")
            return
        found = _render_episode_dir(args.episode_dir, args.fps, args.out_dir)
        if found:
            print(f"[VideoRender] Generated videos for {found} episode(s)")
        else:
            print("[VideoRender] No rgb/ directories found")
        return

    run_dir = args.run_dir
    if not run_dir or not os.path.isdir(run_dir):
        print(f"[VideoRender] run_dir not found: {run_dir}")
        return

    found = 0
    for mode in ("stt", "dt", "at"):
        mode_dir = os.path.join(run_dir, mode)
        if not os.path.isdir(mode_dir):
            continue
        for scene_id in sorted(os.listdir(mode_dir)):
            scene_dir = os.path.join(mode_dir, scene_id)
            if not os.path.isdir(scene_dir):
                continue
            for ep_str in sorted(os.listdir(scene_dir)):
                ep_dir = os.path.join(scene_dir, ep_str)
                if not os.path.isdir(ep_dir):
                    continue
                # ep_dir might contain combo subdirs or be a flat episode dir
                items = os.listdir(ep_dir)
                if "rgb" in items:
                    _out = args.out_dir
                    if _out:
                        _out = os.path.join(_out, mode, scene_id, ep_str)
                    ok = _render_one(ep_dir, "rgb", "rgb_video.mp4", args.fps, _out)
                    ok = _render_one(ep_dir, "depth", "depth_video.mp4", args.fps, _out) or ok
                    if ok:
                        found += 1
                else:
                    # combo subdirs
                    for combo in items:
                        combo_dir = os.path.join(ep_dir, combo)
                        if not os.path.isdir(combo_dir):
                            continue
                        if "rgb" in os.listdir(combo_dir):
                            _out = args.out_dir
                            if _out:
                                _out = os.path.join(_out, mode, scene_id, ep_str, combo)
                            ok = _render_one(combo_dir, "rgb", "rgb_video.mp4", args.fps, _out)
                            ok = _render_one(combo_dir, "depth", "depth_video.mp4", args.fps, _out) or ok
                            if ok:
                                found += 1

    if found:
        print(f"[VideoRender] Generated videos for {found} episodes")
    else:
        print("[VideoRender] No rgb/ directories found")


if __name__ == "__main__":
    main()
