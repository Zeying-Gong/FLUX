"""
clothing_appearance_collab.py
─────────────────────────────
Extends clothing_appearance.py with STT/DT/AT oracle mode support.

  STT (Single-Target Tracking) — simple instructions, no appearance constraints.
  DT  (Distracted Tracking)    — fine-grained distinctive appearance (same as original).
  AT  (Ambiguity Tracking)     — target and distractors share identical appearance.
"""
from __future__ import annotations

import random
from typing import Optional

from clothing_appearance import (
    ClothingProfiles,
    COLOR_PALETTE,
    _evaluate_distinctness,
    assign_one_character,
)


def assign_episode_appearance(
    profiles: ClothingProfiles,
    episode_id: int,
    seed: int,
    character_names: list[str],
    mode: str = "dt",
) -> dict:
    """
    Like clothing_appearance.assign_episode_appearance but with a `mode` parameter.

    mode='stt': simple — all characters get random appearances with NO conflict
                avoidance (they may look alike or different, no constraint).
    mode='dt':  fine-grained — same as original: target and distractors all get
                distinct non-confusable colors (best for description-based tracking).
    mode='at':  ambiguity — ALL characters use the SAME asset and SAME recolor
                as the target, making them visually identical.
    """
    rng = random.Random(seed ^ 0xA11CE)
    target_asset = profiles.asset_for_index(episode_id)

    n = len(character_names)

    if mode == "at":
        # ── AT: everyone gets the same asset + same colors as target ──
        out: dict[str, dict] = {}
        used_top: list[str] = []
        used_bottom: list[str] = []
        for i, cname in enumerate(character_names):
            is_target = (i == 0)
            if is_target:
                out[cname] = assign_one_character(
                    rng, profiles, target_asset, is_target, used_top, used_bottom)
                target_appearance = out[cname]
            else:
                # Clone target appearance exactly (same asset, same parts, same colors)
                cloned = {
                    "asset": target_asset,
                    "is_target": False,
                    "keep_default": False,
                    "top_color": target_appearance["top_color"],
                    "bottom_color": target_appearance["bottom_color"],
                    "parts": {},
                }
                for pname, pinfo in target_appearance["parts"].items():
                    cloned["parts"][pname] = {
                        "category": pinfo["category"],
                        "color_word": pinfo["color_word"],
                        "rgb": pinfo["rgb"],
                        "source": "clone" if pinfo["rgb"] is not None else "default",
                    }
                out[cname] = cloned
        distinct_level = "full"  # intentionally identical
    elif mode == "stt":
        # ── STT: no conflict avoidance — each character independently random ──
        out = {}
        used_top = []
        used_bottom = []
        # Distractor assets: random selection (avoid target to add visual variety,
        # but no color conflict avoidance)
        distractor_assets = _pick_distractor_assets(profiles, episode_id, n, rng)
        for i, cname in enumerate(character_names):
            is_target = (i == 0)
            asset = target_asset if is_target else distractor_assets[i - 1]
            out[cname] = assign_one_character(
                rng, profiles, asset, is_target, used_top, used_bottom)
        distinct_level = _evaluate_distinctness(out, character_names)
    else:
        # ── DT: same as original — fine-grained distinctive colors ──
        from clothing_appearance import assign_episode_appearance as _orig
        return _orig(profiles, episode_id, seed, character_names)

    return {
        "by_character": out,
        "target_asset": target_asset,
        "distinct_level": distinct_level,
    }


def _pick_distractor_assets(
    profiles: ClothingProfiles,
    episode_id: int,
    n: int,
    rng: random.Random,
) -> list[str]:
    """Pick distractor assets cycling from asset_order after target."""
    target_asset = profiles.asset_for_index(episode_id)
    distractor_assets: list[str] = []
    if n > 1:
        start = (episode_id + 1) % profiles.num_assets
        pool = [profiles.asset_order[(start + k) % profiles.num_assets]
                for k in range(profiles.num_assets)]
        pool = [a for a in pool if a != target_asset]
        rng.shuffle(pool)
        for k in range(n - 1):
            if k < len(pool):
                distractor_assets.append(pool[k])
            else:
                distractor_assets.append(pool[k % len(pool)])
    return distractor_assets
