#!/usr/bin/env python3
"""
clothing_appearance.py
───────────────────────
纯 Python 模块(无 Isaac 依赖)。供 episode 生成端调用,
为一个 episode 内的所有角色分配【资产 + 逐部位颜色】,写入 appearance 字段。
采集端也 import 本模块的 PALETTE / 工具函数,保证两端一致。

设计依据(与用户确认):
  • 资产均匀覆盖:target 资产 = asset_order[episode_id % N];
    干扰行人从其余资产里按 seed 选,尽量不与已用资产重复。
  • 区分维度:同一 episode 内,行人之间 top + bottom 颜色词
    既不相同、也不属于同一混淆组。
  • 默认配色:仅【干扰行人】有 20% 概率保留默认色;target 永远随机染色
    (保证 caption 能用明确颜色词描述,对齐跟踪指令风格)。
  • 默认色为 None 的部位(original_* 旧资产,颜色在贴图里):
    保留默认时正常渲染,但不参与颜色混淆名额,靠资产语义区分。
  • 容量不足时降级:优先保证 top 区分,其次 bottom;再不够允许复用,
    并在 appearance 里标记 distinct_level。
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional


# ════════════════════════════════════════════════════════════════════
# 色板(CSS Level-1 16 色)+ 混淆组
# ════════════════════════════════════════════════════════════════════
COLOR_PALETTE: dict[str, tuple[float, float, float]] = {
    "black":   (0.000, 0.000, 0.000),
    "silver":  (0.753, 0.753, 0.753),
    "gray":    (0.502, 0.502, 0.502),
    "white":   (1.000, 1.000, 1.000),
    "maroon":  (0.502, 0.000, 0.000),
    "red":     (1.000, 0.000, 0.000),
    "purple":  (0.502, 0.000, 0.502),
    "fuchsia": (1.000, 0.000, 1.000),
    "green":   (0.000, 0.502, 0.000),
    "lime":    (0.000, 1.000, 0.000),
    "olive":   (0.502, 0.502, 0.000),
    "yellow":  (1.000, 1.000, 0.000),
    "navy":    (0.000, 0.000, 0.502),
    "blue":    (0.000, 0.000, 1.000),
    "teal":    (0.000, 0.502, 0.502),
    "aqua":    (0.000, 1.000, 1.000),
}

# 互相接近、视为"混淆"的颜色组(同组不可同时分给不同行人的同一维度)
CONFUSABLE_GROUPS: list[set[str]] = [
    {"black", "gray", "navy", "maroon"},
    {"white", "silver"},
    {"red", "maroon"},
    {"green", "lime", "olive"},
    {"blue", "navy"},
    {"purple", "fuchsia"},
    {"teal", "aqua"},
    {"yellow", "olive"},
]

# 主衣物部位(用于 caption / 区分),帽子鞋作为辅助维度也染但不参与区分约束
PRIMARY_CATEGORIES = ("top", "bottom")
SECONDARY_CATEGORIES = ("shoes", "hat", "vest")

DEFAULT_KEEP_PROB = 0.20   # 干扰行人保留默认配色的概率


# ── 颜色词关系工具 ──────────────────────────────────────────────────
def _word_to_group(word: str) -> frozenset[str]:
    """返回 word 所属的'冲突集合'(它自身 + 所有同混淆组成员)"""
    members = {word}
    for grp in CONFUSABLE_GROUPS:
        if word in grp:
            members |= grp
    return frozenset(members)


def colors_conflict(w1: Optional[str], w2: Optional[str]) -> bool:
    """两个颜色词是否相同或互相混淆。None 视为不冲突(未知色,不占名额)。"""
    if w1 is None or w2 is None:
        return False
    if w1 == w2:
        return True
    return w2 in _word_to_group(w1)


# ════════════════════════════════════════════════════════════════════
# profiles 载入
# ════════════════════════════════════════════════════════════════════
class ClothingProfiles:
    """封装 character_clothing_profiles.json"""

    def __init__(self, profiles_path: str):
        with open(profiles_path, encoding="utf-8") as f:
            obj = json.load(f)
        self.asset_order: list[str] = obj["_meta"]["asset_order"]
        self.num_assets: int = obj["_meta"]["num_assets"]
        self.profiles: dict = obj["profiles"]

    def asset_for_index(self, idx: int) -> str:
        return self.asset_order[idx % self.num_assets]

    def parts_of(self, asset: str) -> dict:
        return self.profiles.get(asset, {}).get("parts", {})

    def category_parts(self, asset: str, category: str) -> list[str]:
        """该资产属于某 category 的部位名列表(如 top -> ['shirt'])"""
        return [p for p, info in self.parts_of(asset).items()
                if info.get("category") == category]

    def usd_path_of(self, asset: str) -> str | None:
            return self.profiles.get(asset, {}).get("usd_path")

    def index_of(self, asset: str):
            """资产名 → 在 asset_order 中的下标(用于映射到 char_pool 同序列表)。"""
            try:
                return self.asset_order.index(asset)
            except ValueError:
                return None
# ════════════════════════════════════════════════════════════════════
# 单角色外观采样
# ════════════════════════════════════════════════════════════════════
def _sample_color_for_category(
    rng: random.Random,
    used_words_in_category: list[str],
) -> Optional[str]:
    """为某 category 选一个不与已用色冲突的颜色词;不够时返回最不冲突的。"""
    palette = list(COLOR_PALETTE.keys())
    rng.shuffle(palette)
    # 第一优先:完全不冲突
    for cand in palette:
        if not any(colors_conflict(cand, u) for u in used_words_in_category):
            return cand
    # 退化:允许冲突,返回随机一个(容量耗尽)
    return rng.choice(palette)


def assign_one_character(
    rng: random.Random,
    profiles: ClothingProfiles,
    asset: str,
    is_target: bool,
    used_top: list[str],
    used_bottom: list[str],
) -> dict:
    """
    为单个角色生成 appearance(逐部位颜色词 + rgb)。
    会就地更新 used_top / used_bottom(把本角色实际用的 top/bottom 颜色登记进去)。
    """
    parts_info = profiles.parts_of(asset)

    # 决定是否保留默认色(仅干扰行人,且按概率)
    keep_default = (not is_target) and (rng.random() < DEFAULT_KEEP_PROB)

    appearance_parts: dict[str, dict] = {}
    chosen_top: Optional[str] = None
    chosen_bottom: Optional[str] = None

    for part, info in parts_info.items():
        category = info.get("category", part)

        if keep_default:
            # 用默认色:rgb=None 表示采集端不染色(贴图原样)
            word = info.get("default_color_word")  # 可能是 None
            appearance_parts[part] = {
                "category": category,
                "color_word": word,
                "rgb": None,            # None → 采集端跳过染色
                "source": "default",
            }
            if category == "top":
                chosen_top = word
            elif category == "bottom":
                chosen_bottom = word
        else:
            # 随机染色:top/bottom 走防冲突采样,次要部位自由随机
            if category == "top":
                word = _sample_color_for_category(rng, used_top)
                chosen_top = word
            elif category == "bottom":
                word = _sample_color_for_category(rng, used_bottom)
                chosen_bottom = word
            else:
                # shoes/hat/vest:不参与区分约束,直接随机(避开本角色已用以求美观)
                self_used = [appearance_parts[p]["color_word"]
                             for p in appearance_parts]
                word = _sample_color_for_category(rng, self_used)

            appearance_parts[part] = {
                "category": category,
                "color_word": word,
                "rgb": list(COLOR_PALETTE[word]),
                "source": "random",
            }

    # 把本角色实际的 top/bottom 颜色登记到全局已用(供后续角色避让)
    if chosen_top is not None:
        used_top.append(chosen_top)
    if chosen_bottom is not None:
        used_bottom.append(chosen_bottom)

    return {
        "asset": asset,
        "is_target": is_target,
        "keep_default": keep_default,
        "top_color": chosen_top,
        "bottom_color": chosen_bottom,
        "parts": appearance_parts,
    }


# ════════════════════════════════════════════════════════════════════
# 整 episode 的外观分配(主入口)
# ════════════════════════════════════════════════════════════════════
def assign_episode_appearance(
    profiles: ClothingProfiles,
    episode_id: int,
    seed: int,
    character_names: list[str],   # 顺序:第0个是 target("Character"),其余干扰
) -> dict:
    """
    返回 {char_name: appearance_dict}。
    资产分配:target = asset_order[episode_id % N];
              干扰行人尽量用不同资产(从剩余里按 seed 取)。
    """
    rng = random.Random(seed ^ 0xA11CE)  # 与 episode 几何用的 seed 派生但不同步

    n = len(character_names)
    target_asset = profiles.asset_for_index(episode_id)

    # 为干扰行人分配资产:从 asset_order 里以 target 之后的位置滚动取,避开重复
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
                # 资产不够(>21 人,实际不会),允许复用
                distractor_assets.append(pool[k % len(pool)])

    used_top: list[str] = []
    used_bottom: list[str] = []
    out: dict[str, dict] = {}

    for i, cname in enumerate(character_names):
        is_target = (i == 0)
        asset = target_asset if is_target else distractor_assets[i - 1]
        out[cname] = assign_one_character(
            rng, profiles, asset, is_target, used_top, used_bottom)

    # 评估区分度:检查 top/bottom 是否有冲突对
    distinct_level = _evaluate_distinctness(out, character_names)

    return {
        "by_character": out,
        "target_asset": target_asset,
        "distinct_level": distinct_level,
    }


def _evaluate_distinctness(out: dict, names: list[str]) -> str:
    """返回 'full' / 'top_only' / 'partial' —— 描述区分质量,便于后续筛查。"""
    tops = [out[n]["top_color"] for n in names]
    bottoms = [out[n]["bottom_color"] for n in names]

    def has_conflict(words: list[str]) -> bool:
        for a in range(len(words)):
            for b in range(a + 1, len(words)):
                if colors_conflict(words[a], words[b]):
                    return True
        return False

    top_ok = not has_conflict(tops)
    bot_ok = not has_conflict(bottoms)
    if top_ok and bot_ok:
        return "full"
    if top_ok:
        return "top_only"
    return "partial"


# ── 自测 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--episode_id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_people", type=int, default=5)
    args = ap.parse_args()

    prof = ClothingProfiles(args.profiles)
    names = ["Character"] + [f"Character_{i:02d}" for i in range(1, args.num_people)]
    result = assign_episode_appearance(prof, args.episode_id, args.seed, names)

    print(f"target_asset = {result['target_asset']}")
    print(f"distinct_level = {result['distinct_level']}\n")
    for cname, ap_dict in result["by_character"].items():
        tgt = " [TARGET]" if ap_dict["is_target"] else ""
        dflt = " (default)" if ap_dict["keep_default"] else ""
        print(f"{cname}{tgt}{dflt}: asset={ap_dict['asset']}")
        print(f"    top={ap_dict['top_color']}  bottom={ap_dict['bottom_color']}")
        for part, info in ap_dict["parts"].items():
            print(f"      {part:14s} {info['color_word']}  src={info['source']}")
        print()