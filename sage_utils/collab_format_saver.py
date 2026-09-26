"""
collab_format_saver.py
───────────────────────
Save collected episode data in the collaborator's dataset format.

Directory layout (per episode):
  {mode}_train_oracle/seed_{seed}/pass1/{episode_id}/
    episode.json                  # Episode summary  {finish, status, success, ... instruction}
    episode_info.json             # Full trajectory  [{step, source_fps, dt, ..., other_humans_pos}]
    track_object.jpg              # Target crop (first frame)
    {step_idx:04d}/
      {step_idx:04d}.png          # RGB frame
      {step_idx:04d}.json         # Same as episode.json (for per-keyframe self-containment)
      {step_idx:04d}_info.json    # Full trajectory (same as episode_info.json)
      track_object.jpg            # Target crop for this step
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _episode_id_str(eid: int) -> str:
    """Format episode ID. Uses collaborator's format: zero-padded 4-digit + random suffix,
    but for simplicity we use a fixed format."""
    return f"ep_{eid:06d}"


def make_episode_dir(root: str, mode: str, seed: int, episode_id: int) -> str:
    """Build the per-episode output directory path."""
    ep_name = _episode_id_str(episode_id)
    return os.path.join(root, f"{mode}_train_oracle", f"seed_{seed}", "pass1", ep_name)


def save_episode_json(
    ep_dir: str,
    instruction: str,
    finish: bool = True,
    status: str = "Normal",
    success: float = 1.0,
    following_rate: float = 0.0,
    following_step: int = 0,
    total_step: int = 0,
    collision: float = 0.0,
):
    """Save episode summary JSON (matches collaborator's {frame}.json)."""
    data = {
        "finish": finish,
        "status": status,
        "success": success,
        "following_rate": following_rate,
        "following_step": following_step,
        "total_step": total_step,
        "collision": collision,
        "instruction": instruction,
    }
    path = os.path.join(ep_dir, "episode.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_step_json(step_dir: str, summary: dict):
    """Save per-step JSON (same as episode.json, for alignment)."""
    path = os.path.join(step_dir, os.path.basename(step_dir) + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def save_info_json(
    ep_dir: str,
    ep_data: dict,
):
    """Save full trajectory as JSON array (matches collaborator's _info.json).
    Each element contains one step's data in the collaborator's schema.
    """
    n = len(ep_data.get("step", []))
    if n == 0:
        return

    # Pre-compute trajectories for robot_pre / other_humans_pos
    robot_traj = np.stack([
        np.asarray(ep_data.get("robot_pos_x", []), dtype=np.float64),
        np.asarray(ep_data.get("robot_pos_y", []), dtype=np.float64),
        np.asarray(ep_data.get("robot_pos_z", []), dtype=np.float64),
    ], axis=-1) if n > 0 else np.zeros((0, 3), dtype=np.float64)

    source_fps = 1.0 / 0.025  # 40 FPS like collaborator; caller can override
    dt = 0.025

    entries = []
    for i in range(n):
        step_num = int(ep_data["step"][i])
        # Robot position this step and previous
        rpos = robot_traj[i].tolist() if i < len(robot_traj) else [0.0, 0.0, 0.0]
        rpos_pre = robot_traj[i - 1].tolist() if i > 0 else rpos

        # Build other_humans_pos: sentinel -98 for inactive (same as collaborator)
        # We don't track other humans individually, so use sentinel defaults
        other = [[-98.0, -98.0, -98.0]] * 7

        entry = {
            "step": step_num,
            "source_fps": source_fps,
            "dt": dt,
            "dis_to_human": float(ep_data.get("target_dist", [0.0])[min(i, len(ep_data.get("target_dist", [])) - 1)]),
            "facing": 1.0 if float(ep_data.get("target_visible", [0])[min(i, len(ep_data.get("target_visible", [])) - 1)]) > 0 else 0.0,
            "base_velocity": [
                float(ep_data.get("action_forward", [0.0])[min(i, len(ep_data.get("action_forward", [])) - 1)]),
                0.0,
                0.0,
            ],
            "base_velocity_cmd": [
                float(ep_data.get("action_forward", [0.0])[min(i, len(ep_data.get("action_forward", [])) - 1)]),
                0.0,
                0.0,
            ],
            "slide": False,
            "navmesh_collision": False,
            "collision": bool(float(ep_data.get("contact_force", [0.0])[min(i, len(ep_data.get("contact_force", [])) - 1)]) >= 400.0),
            "robot_pos_pre": rpos_pre,
            "robot_yaw_pre": float(ep_data.get("robot_yaw", [0.0])[max(0, i - 1)]),
            "robot_pos": rpos,
            "robot_yaw": float(ep_data.get("robot_yaw", [0.0])[min(i, len(ep_data.get("robot_yaw", [])) - 1)]),
            "target_pos": [
                float(ep_data.get("target_pos_x", [0.0])[min(i, len(ep_data.get("target_pos_x", [])) - 1)]),
                float(ep_data.get("target_pos_y", [0.0])[min(i, len(ep_data.get("target_pos_y", [])) - 1)]),
                float(ep_data.get("target_pos_z", [0.0])[min(i, len(ep_data.get("target_pos_z", [])) - 1)]),
            ],
            "other_humans_pos": other,
        }
        entries.append(entry)

    path = os.path.join(ep_dir, "episode_info.json")
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)


def save_step_info_json(step_dir: str, step_entry: dict):
    """Save per-step info JSON containing the full trajectory array
    (same as episode_info.json, for structural alignment with collaborator)."""
    path = os.path.join(step_dir, os.path.basename(step_dir) + "_info.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([step_entry], f, indent=2)


def save_track_object_jpg(ep_dir: str, rgb: np.ndarray, bbox: Tuple[float, float, float, float],
                           step_dir: Optional[str] = None):
    """Crop target from RGB and save as track_object.jpg."""
    try:
        from PIL import Image
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(rgb.shape[1] - 1, int(round(x2)))
        y2 = min(rgb.shape[0] - 1, int(round(y2)))
        if x2 <= x1 or y2 <= y1:
            return
        crop = rgb[y1:y2, x1:x2]
        # Save episode-level
        ep_path = os.path.join(ep_dir, "track_object.jpg")
        os.makedirs(os.path.dirname(ep_path), exist_ok=True)
        Image.fromarray(crop).save(ep_path)
        # Save per-step
        if step_dir is not None:
            step_path = os.path.join(step_dir, "track_object.jpg")
            os.makedirs(os.path.dirname(step_path), exist_ok=True)
            Image.fromarray(crop).save(step_path)
    except Exception as e:
        print(f"[CollabFormat] track_object.jpg WARN: {e}")


def save_step_rgb(step_dir: str, step_idx: int, rgb: np.ndarray):
    """Save RGB frame as PNG in step directory."""
    from PIL import Image
    path = os.path.join(step_dir, f"{step_idx:04d}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(rgb).save(path)


def save_quality_result(
    ep_dir: str,
    ep_id: int,
    accepted: bool,
    metrics: dict,
    rejection_reasons: List[str],
    filter_config: dict,
):
    """Write quality.json and accepted/rejected marker (same as original)."""
    os.makedirs(ep_dir, exist_ok=True)
    status = "accepted" if accepted else "rejected"
    quality = {
        "schema_version": 1,
        "status": status,
        "episode_id": int(ep_id),
        "rejection_reasons": rejection_reasons,
        "metrics": metrics,
        "filter_config": filter_config,
    }
    with open(os.path.join(ep_dir, "quality.json"), "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)

    for marker in ("_ACCEPTED", "_REJECTED"):
        marker_path = os.path.join(ep_dir, marker)
        if os.path.exists(marker_path):
            os.remove(marker_path)
    with open(os.path.join(ep_dir, f"_{status.upper()}"), "w", encoding="utf-8") as f:
        f.write("\n")


def make_instruction(mode: str, appearance: dict, rng: Optional[random.Random] = None) -> str:
    """Generate instruction for the given oracle mode.

    STT: "Follow the person." / "Follow the man." / "Follow the woman."
    DT:  "Follow the {gender} wearing {color} top, {color} pants, ..."
    AT:  "Follow the first person you see." / "Pursue the first individual in your path."
    """
    import random as _random
    if rng is None:
        rng = _random.Random()

    by_char = appearance.get("by_character", {}) if appearance else {}
    target_info = by_char.get("Character", by_char.get(list(by_char.keys())[0]) if by_char else {})

    gender = "person"
    asset = target_info.get("asset", "")
    if asset.startswith("F_"):
        gender = "woman"
    elif asset.startswith("M_"):
        gender = "man"

    if mode == "stt":
        templates = [
            f"Follow the {gender}.",
            f"Follow the person.",
            f"Follow the {gender} in front of you.",
        ]
        return rng.choice(templates)

    elif mode == "at":
        templates = [
            "Follow the first person you see.",
            "Pursue the first individual in your path.",
            "Stay behind the first person you observe.",
            "Follow the person you see first.",
        ]
        people_count = len(by_char) if by_char else 0
        if people_count >= 3:
            templates.extend([
                "Follow the first person you see. Ignore everyone else.",
                "Track the first person you see. The others are distractions.",
            ])
        return rng.choice(templates)

    else:  # DT
        parts = target_info.get("parts", {})
        clothing_items = []
        for pname, pinfo in parts.items():
            cat = pinfo.get("category", "")
            color_word = pinfo.get("color_word", "")
            clothing_items.append((cat, color_word))
        top_items = [c for c in clothing_items if c[0] == "top"]
        bottom_items = [c for c in clothing_items if c[0] == "bottom"]
        shoe_items = [c for c in clothing_items if c[0] == "shoes"]
        hat_items = [c for c in clothing_items if c[0] == "hat"]
        desc_parts = []
        if hat_items:
            desc_parts.append(f"{hat_items[0][1]} hat")
        if top_items:
            desc_parts.append(f"{top_items[0][1]} top")
        if bottom_items:
            desc_parts.append(f"{bottom_items[0][1]} pants")
        if shoe_items:
            desc_parts.append(f"{shoe_items[0][1]} shoes")
        if desc_parts:
            return f"Pursue the {gender} wearing " + ", ".join(desc_parts) + "."
        else:
            templates = [
                f"Pursue the {gender} in front of you.",
                f"Follow the {gender}.",
            ]
            return rng.choice(templates)
