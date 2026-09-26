"""
debug_robot_vln.py
──────────────────
Standalone debug script for robot VLN landmark selection, path visualisation,
and LLM instruction generation.
No Isaac Sim required. Reads semantic map directly.

Architecture:
    2D semantic map
        → anchor map (objects, centroids)
        → endpoint pair sampling (random or from labels.json)
        → A* path planning
        → path analysis (length + turns)
        → landmark selection (segment-based, uniqueness-scored)
        → segmented ref path (A* through each landmark)
        → [optional] LLM instruction generation
        → VLN episode JSON + visualisation PNG

Usage:
    # Basic debug (no LLM):
    python debug_robot_vln.py \
        --semantic_map_json /path/to/2D_Semantic_Map_839920_Complete.json \
        --episode_ids 0 1 2 3 4 \
        --vis_dir ./debug_vln_vis

    # With LLM instruction generation:
    python debug_robot_vln.py \
        --semantic_map_json /path/to/2D_Semantic_Map_839920_Complete.json \
        --scene_text_file  /path/to/semantic_map_0001_839920.txt \
        --llm_base_url http://localhost:8000/v1 \
        --llm_model qwen3 \
        --episode_ids 0 1 2 3 4 \
        --vis_dir ./debug_vln_vis
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import random
import re
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt

# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def _parse():
    p = argparse.ArgumentParser()
    # Map input
    p.add_argument("--semantic_map_json", required=True)
    p.add_argument("--scene_text_file",   default=None,
                   help="Path to semantic_map_<scene_id>.txt for LLM context")
    # Episode config
    p.add_argument("--episode_ids",       nargs="+", type=int, default=[0])
    p.add_argument("--seed_offset",       type=int, default=0)
    p.add_argument("--robot_radius",      type=float, default=0.2,
                   help="Robot inflation radius in metres (default 0.2, matches trajectory_generator)")
    p.add_argument("--scale",             type=float, default=0.05)
    p.add_argument("--min_robot_dist",    type=float, default=2.0)
    p.add_argument("--max_robot_dist",    type=float, default=20.0)
    # Output
    p.add_argument("--vis_dir",           default="./debug_vln_vis")
    p.add_argument("--out_json",          default=None,
                   help="If given, save all episodes to this JSON file")
    # LLM (optional)
    p.add_argument("--llm_base_url",      default=None,
                   help="LLM server base URL, e.g. http://localhost:8000/v1")
    p.add_argument("--llm_model",         default="qwen3")
    p.add_argument("--llm_temperature",   type=float, default=0.7)
    p.add_argument("--llm_max_retries",   type=int, default=3)
    return p.parse_args()

ARGS = _parse()

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
_LANDMARK_SKIP = {
    "wall", "floor", "ceiling", "unable area", "door frame",
    "window", "curtain", "rug", "mat", "downlights", "chandelier",
}
MIN_REACHABLE_PX = 100

# Ref path colours
REF_COLORS = ["#4D96FF", "#FF6B6B"]
REF_LABELS = ["shortest", "socially-aware"]

# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# LLM prompts — two-layer: macro path + micro goal context
# ═══════════════════════════════════════════════════════════════════════════
_SYS_VLN = """You are generating a two-layer VLN navigation instruction for an indoor robot.
Output ONLY valid JSON. No markdown. No prose.

LANGUAGE RULES:
  FORBIDDEN: "near", "close to", "next to", "by the", "around the"
  REQUIRED directional vocabulary:
    "on your left/right", "straight ahead", "turn left/right at",
    "pass X on your left/right", "walk straight past", "stop in front of",
    "proceed toward", "facing you", "behind the", "to the left/right of"

Schema:
{
  "macro_instruction": "<20-35 words. Describe the route using objects the robot PASSES. Directional language only.>",
  "goal_description": "<20-35 words. Describe the GOAL object and 2-3 nearby objects to disambiguate. Specific enough that only ONE instance matches.>",
  "full_instruction": "<50-80 words. macro_instruction + goal_description combined into fluent first-person.>",
  "path_objects": [
    {"item_id": "...", "label": "...", "side": "left|right|ahead",
     "action": "pass|turn_left|turn_right", "desc": "..."}
  ],
  "goal_context": [
    {"item_id": "...", "label": "...", "relation": "to the left of|in front of|facing|behind",
     "desc": "..."}
  ]
}

CRITICAL RULES:
- path_objects: ONLY from PATH_OBJECTS list. 0-4 items. Empty OK for short paths.
- goal_context: 2-3 items from GOAL_CONTEXT list. Must disambiguate this specific instance.
- full_instruction must NOT contain "near". Use directional vocabulary throughout.
- goal_description must be specific enough to identify exactly ONE object instance."""

_USR_VLN = """SCENE: {scene_text}

ROBOT: from {start_label} → {goal_label}
PATH: {path_m:.1f}m, {n_turns} turn(s)

PATH_OBJECTS (objects robot passes en route, in order):
{path_objects_str}

GOAL_CONTEXT (objects around the goal for disambiguation):
{goal_context_str}

Generate the JSON now. NO "near"."""

# ═══════════════════════════════════════════════════════════════════════════
# Map helpers
# ═══════════════════════════════════════════════════════════════════════════
def load_map(sem_json, robot_r, scale):
    with open(sem_json, encoding="utf-8") as f:
        sem_data = json.load(f)
    all_y, all_x = [], []
    for inst in sem_data:
        for y, x in inst.get("mask_coords_m", []):
            try: all_y.append(float(y)); all_x.append(float(x))
            except: pass
    min_y, max_y = min(all_y), max(all_y)
    min_x, max_x = min(all_x), max(all_x)
    h = int(np.ceil((max_y - min_y) / scale)) + 1
    w = int(np.ceil((max_x - min_x) / scale)) + 1
    grid = np.zeros((h, w), dtype=np.uint8)
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
    if robot_r > 0:
        dist_m = distance_transform_edt(grid == 0, sampling=scale)
        grid = (dist_m <= robot_r).astype(np.uint8)
    esdf = distance_transform_edt(grid == 0, sampling=scale)
    return grid, esdf, min_x, min_y, max_x, max_y, scale, sem_data

def px_to_world(px, py, min_x, min_y, scale):
    return min_x + (px + 0.5) * scale, min_y + (py + 0.5) * scale

def world_to_px(x_m, y_m, min_x, min_y, scale):
    return (int(round((float(x_m) - min_x) / scale)),
            int(round((float(y_m) - min_y) / scale)))

def dist2(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

def path_length_m(pts_px, scale):
    """Geodesic path length from pixel list, in metres."""
    if not pts_px or len(pts_px) < 2: return 0.0
    return sum(math.hypot(pts_px[i][0]-pts_px[i-1][0],
                          pts_px[i][1]-pts_px[i-1][1])
               for i in range(1, len(pts_px))) * scale

# ═══════════════════════════════════════════════════════════════════════════
# A*
# ═══════════════════════════════════════════════════════════════════════════
def astar(grid, start_px, goal_px, esdf=None, esdf_w=0.0):
    H, W = grid.shape
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    open_set = [(0.0, start_px)]
    came_from = {}; g = {start_px: 0.0}
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal_px:
            path = [cur]
            while cur in came_from: cur = came_from[cur]; path.append(cur)
            return path[::-1]
        for d in dirs:
            nx, ny = cur[0]+d[0], cur[1]+d[1]
            if not (0<=nx<W and 0<=ny<H): continue
            if grid[ny,nx] == 1: continue
            nb = (nx, ny)
            step = math.hypot(d[0], d[1])
            if esdf is not None and esdf_w > 0:
                step += esdf_w / max(esdf[ny,nx], 1e-3)
            step = max(step, 1e-4)
            tg = g[cur] + step
            if nb not in g or tg < g[nb]:
                came_from[nb] = cur; g[nb] = tg
                heapq.heappush(open_set,
                    (tg + math.hypot(nx-goal_px[0], ny-goal_px[1]), nb))
    return None

def snap(grid, px_py):
    H, W = grid.shape
    px, py = px_py
    if 0<=px<W and 0<=py<H and grid[py,px]==0: return px_py
    visited = set(); q = deque([(px, py, 0)])
    while q:
        cx, cy, d = q.popleft()
        if d > 30: break
        if 0<=cx<W and 0<=cy<H and grid[cy,cx]==0: return (cx, cy)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (cx+dx, cy+dy)
            if nb not in visited: visited.add(nb); q.append((nb[0], nb[1], d+1))
    return None

def reachable_area_px(grid, start_px):
    H, W = grid.shape
    sx, sy = start_px
    if not (0<=sx<W and 0<=sy<H) or grid[sy,sx]!=0: return 0
    limit = MIN_REACHABLE_PX * 10
    visited = {start_px}; q = deque([start_px]); count = 0
    while q and count < limit:
        cx, cy = q.popleft(); count += 1
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (cx+dx, cy+dy); nx, ny = nb
            if nb not in visited and 0<=nx<W and 0<=ny<H and grid[ny,nx]==0:
                visited.add(nb); q.append(nb)
    return count

# ═══════════════════════════════════════════════════════════════════════════
# Anchor map  (same filtering as vln_trajectory_generator)
# ═══════════════════════════════════════════════════════════════════════════
def build_anchor_map(sem_data):
    """
    Build anchor_map: {item_id: {pos_m, label, mask_coords_m}}
    item_id format: label_N  (e.g. chair_3, sofa_1)
    Also returns itemid2inst for path-planning (boundary-pixel) access.
    """
    counter = {}; anchor_map = {}; itemid2inst = {}
    for inst in sem_data:
        label = str(inst.get("category_label", "")).lower().strip()
        if not label: continue
        coords = inst.get("mask_coords_m", [])
        if not coords: continue
        counter[label] = counter.get(label, 0) + 1
        item_id = f"{label.replace(' ','_')}_{counter[label]}"
        ys = [float(c[0]) for c in coords]
        xs = [float(c[1]) for c in coords]
        anchor_map[item_id] = {
            "pos_m":         (sum(xs)/len(xs), sum(ys)/len(ys)),
            "label":         label,
            "mask_coords_m": coords,
        }
        itemid2inst[item_id] = inst
        inst["_item_id"] = item_id   # back-reference
    return anchor_map, itemid2inst

def nearest_anchor(pos_world_xy, anchor_map, skip_labels=None):
    """Find item_id whose centroid is nearest to pos_world_xy = (x_m, y_m)."""
    best_id, best_d = None, float("inf")
    for item_id, info in anchor_map.items():
        if skip_labels and info["label"].lower() in skip_labels: continue
        cx, cy = info["pos_m"]
        d = math.hypot(cx - pos_world_xy[0], cy - pos_world_xy[1])
        if d < best_d: best_d = d; best_id = item_id
    return best_id

def instance_boundary_free(inst_mask_coords_m, grid, min_x, min_y, scale):
    """
    Find the nearest free pixel on the boundary of an instance.
    Matches vln_trajectory_generator.get_nearest_free_pixel_on_side() logic.
    """
    H, W = grid.shape
    # convert mask to pixel set
    pix_set = set()
    for y_m, x_m in inst_mask_coords_m:
        try:
            py = int(round((float(y_m) - min_y) / scale))
            px = int(round((float(x_m) - min_x) / scale))
            if 0 <= py < H and 0 <= px < W:
                pix_set.add((px, py))   # (col, row) convention
        except: pass
    if not pix_set: return None

    # boundary = pixels with at least one non-member 4-neighbour
    boundary = []
    for (px, py) in pix_set:
        if any((px+dx, py+dy) not in pix_set
               for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]):
            boundary.append((px, py))
    if not boundary: return None

    # BFS from boundary outward to find nearest free pixel
    visited = set(boundary); q = deque([(px, py, 0) for px, py in boundary])
    while q:
        cx, cy, d = q.popleft()
        if d > 50: break
        if 0<=cx<W and 0<=cy<H and grid[cy,cx]==0:
            return (cx, cy)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (cx+dx, cy+dy)
            if nb not in visited:
                visited.add(nb); q.append((nb[0], nb[1], d+1))
    return None

# ═══════════════════════════════════════════════════════════════════════════
# Landmark selection (identical to run_llm_pipeline logic)
# ═══════════════════════════════════════════════════════════════════════════
def detect_turns(path_px, angle_thresh_deg=35, min_gap=5):
    if not path_px or len(path_px) < 3: return []
    turns = []; last_turn = -min_gap
    for i in range(1, len(path_px)-1):
        dx1 = path_px[i][0]-path_px[i-1][0]; dy1 = path_px[i][1]-path_px[i-1][1]
        dx2 = path_px[i+1][0]-path_px[i][0];  dy2 = path_px[i+1][1]-path_px[i][1]
        l1 = math.hypot(dx1,dy1); l2 = math.hypot(dx2,dy2)
        if l1 < 1e-6 or l2 < 1e-6: continue
        cos_a = max(-1.0, min(1.0, (dx1*dx2+dy1*dy2)/(l1*l2)))
        angle = math.degrees(math.acos(cos_a))
        if angle > angle_thresh_deg and (i - last_turn) >= min_gap:
            turns.append(i); last_turn = i
    return turns


def select_landmarks(robot_path_px, start_world_xy, goal_world_xy,
                     anchor_map, scale, min_x, min_y):
    """
    Select key landmarks along the robot path.

    Parameters
    ----------
    robot_path_px   : list of (px, py) – pixel path from A*
    start_world_xy  : (x_m, y_m) of robot start
    goal_world_xy   : (x_m, y_m) of robot goal
    anchor_map      : {item_id: {pos_m, label, ...}}
    scale, min_x, min_y : grid coordinate frame

    Returns
    -------
    seg_lm    : list of landmark dicts (path order)
    debug_info: dict with path_m, n_turns, n_landmarks, landmarks
    """
    path_m  = path_length_m(robot_path_px, scale)
    turns   = detect_turns(robot_path_px)
    n_turns = len(turns)

    n_landmarks = max(2, int(path_m / 3.0))
    n_landmarks = max(n_landmarks, n_turns + 1)
    n_landmarks = min(n_landmarks, 5)

    # Category frequency counts (uniqueness score)
    cat_counts: dict[str, int] = {}
    for info in anchor_map.values():
        cat = info["label"].lower().strip()
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    max_cat = max(cat_counts.values(), default=1)

    # Scene diagonal for distance normalisation
    scene_diag = math.hypot(
        goal_world_xy[0] - start_world_xy[0],
        goal_world_xy[1] - start_world_xy[1]) * 2.0 + 1.0

    def anchor_score(item_id, probe_world_xy):
        """Lower = better landmark. 0.7*uniqueness + 0.3*proximity."""
        info = anchor_map.get(item_id)
        if not info: return None
        cat = info["label"].lower().strip()
        if cat in _LANDMARK_SKIP: return None
        cx, cy = info["pos_m"]
        d = math.hypot(cx - probe_world_xy[0], cy - probe_world_xy[1])
        return (0.7 * (cat_counts.get(cat, 1) / max_cat)
                + 0.3 * (d / (scene_diag + 1e-6)))

    def px_to_world_local(px, py):
        return px_to_world(px, py, min_x, min_y, scale)

    # Build sample points (turn positions + evenly-spaced)
    n = len(robot_path_px)
    turn_pts_px = [robot_path_px[ti] for ti in turns]
    step_pts_px = [robot_path_px[min(int(n * f / max(n_landmarks, 1)), n-1)]
                   for f in range(n_landmarks + 1)]
    all_sample_px = turn_pts_px + step_pts_px

    # Convert to world coords for distance calc
    turn_world = {px_to_world_local(p[0], p[1]) for p in turn_pts_px}
    sample_world = [px_to_world_local(p[0], p[1]) for p in all_sample_px]

    # Assign to segments
    seg_size = max(1, len(sample_world) // n_landmarks)
    segments: list[list] = [[] for _ in range(n_landmarks)]
    for k, pt in enumerate(sample_world):
        seg_idx = min(k // seg_size, n_landmarks - 1)
        segments[seg_idx].append(pt)

    seen_ids: set[str] = set()
    seg_lm = []

    for seg_idx, seg_pts in enumerate(segments):
        if not seg_pts: continue
        rep = (sum(p[0] for p in seg_pts) / len(seg_pts),
               sum(p[1] for p in seg_pts) / len(seg_pts))
        is_turn = any(tp in seg_pts for tp in turn_world)

        best_id, best_sc = None, float("inf")
        for item_id in anchor_map:
            if item_id in seen_ids: continue
            sc = anchor_score(item_id, rep)
            if sc is None: continue
            if is_turn: sc *= 0.5   # boost turn segments
            if sc < best_sc: best_sc = sc; best_id = item_id

        if best_id:
            seen_ids.add(best_id)
            info = anchor_map[best_id]
            cat  = info["label"].lower().strip()
            cnt  = cat_counts.get(cat, 1)
            cx, cy = info["pos_m"]
            lm_px  = world_to_px(cx, cy, min_x, min_y, scale)
            seg_lm.append({
                "seg_idx":  seg_idx,
                "is_turn":  is_turn,
                "item_id":  best_id,
                "label":    info["label"],
                "cnt":      cnt,
                "score":    round(best_sc, 4),
                "lm_px":    lm_px,
                "pos_m":    info["pos_m"],
            })

    debug_info = {
        "path_m":       round(path_m, 2),
        "n_turns":      n_turns,
        "n_landmarks":  n_landmarks,
        "landmarks":    [(e["item_id"], e["label"],
                          f"x{e['cnt']}", f"score={e['score']}",
                          "TURN" if e["is_turn"] else "")
                         for e in seg_lm],
    }
    return seg_lm, debug_info


# ═══════════════════════════════════════════════════════════════════════════
# Segmented reference path
# ═══════════════════════════════════════════════════════════════════════════
def plan_ref_path(grid, esdf, start_px, goal_px, key_wpts_px):
    """
    Segmented A*: start → kw[0] → kw[1] → ... → goal.
    Guarantees the path passes through (or very close to) each key waypoint.
    """
    waypoints = [start_px] + [snap(grid, kp) or kp for kp in key_wpts_px] + [goal_px]
    full = []
    for i in range(len(waypoints) - 1):
        seg = astar(grid, waypoints[i], waypoints[i+1], esdf=esdf, esdf_w=0.12)
        if seg is None: return None
        full.extend(seg if i == 0 else seg[1:])
    return full


# ═══════════════════════════════════════════════════════════════════════════
# Waypoint sequence for episode output
# ═══════════════════════════════════════════════════════════════════════════
def path_to_waypoints(path_px, min_x, min_y, scale, fixed_z=0.5, sample_step=1):
    """
    Convert pixel path to world-coord waypoint list (position + yaw quaternion).
    Matches vln_trajectory_generator.generate_trajectory_points() format.
    """
    world = [px_to_world(px, py, min_x, min_y, scale) for px, py in path_px]
    sampled = world[::sample_step]
    points = []
    for j, (wx, wy) in enumerate(sampled):
        if j < len(sampled) - 1:
            wx2, wy2 = sampled[j+1]
        else:
            wx2, wy2 = sampled[j]
        yaw = math.atan2(wy2 - wy, wx2 - wx)
        qz = math.sin(yaw / 2.0); qw = math.cos(yaw / 2.0)
        points.append({
            "point":    str(j),
            "position": [wx, wy, fixed_z],
            "rotation": [0.0, 0.0, qz, qw],
        })
    return points


# ═══════════════════════════════════════════════════════════════════════════
# LLM instruction generation
# ═══════════════════════════════════════════════════════════════════════════
def _llm_call(base_url, model, system, user, temperature, max_tokens=600):
    """Call OpenAI-compatible API."""
    import re as _re
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key="EMPTY")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        print(f"    [LLM] error: {e}")
        return None
    raw = resp.choices[0].message.content or ""
    raw = _re.sub(r"<think>.*?</think>\s*", "", raw, flags=_re.DOTALL).strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except Exception:
        # Try salvaging truncated JSON
        try:
            s = raw
            oc = s.count('{') - s.count('}')
            os_ = s.count('[') - s.count(']')
            if oc > 0 or os_ > 0:
                if s.count('"') % 2 == 1: s += '"'
                s += ']' * max(0, os_) + '}' * max(0, oc)
                return json.loads(s)
        except Exception:
            pass
        print(f"    [LLM] non-JSON: {raw[:200]}")
        return raw


def cat_cnt_local(info, anchor_map):
    """Count instances of same category in anchor_map."""
    cat = info["label"].lower().strip()
    return sum(1 for v in anchor_map.values() if v["label"].lower().strip() == cat)


def _objects_along_path(robot_path_px, anchor_map, min_x, min_y, scale,
                        max_dist_m=2.5, max_objects=4):
    """
    Objects whose centroid is within max_dist_m of any path point.
    Sorted by path position (where they are first encountered).
    """
    results = []
    seen = set()
    sampled = robot_path_px[::5]   # every 5th pixel for speed
    for item_id, info in anchor_map.items():
        if info["label"].lower() in _LANDMARK_SKIP: continue
        cx, cy = info["pos_m"]
        best_d, best_k = float("inf"), 0
        for k, (ppx, ppy) in enumerate(sampled):
            wx = min_x + (ppx + 0.5) * scale
            wy = min_y + (ppy + 0.5) * scale
            d = math.hypot(cx - wx, cy - wy)
            if d < best_d: best_d = d; best_k = k
        if best_d <= max_dist_m and item_id not in seen:
            results.append((item_id, info, best_d, best_k))
            seen.add(item_id)
    results.sort(key=lambda x: x[3])   # path order
    return results[:max_objects]


def _objects_around_goal(goal_world_xy, anchor_map, max_dist_m=3.0,
                         max_objects=3, exclude_id=None):
    """Objects near the goal for disambiguation, sorted by distance."""
    results = []
    for item_id, info in anchor_map.items():
        if item_id == exclude_id: continue
        if info["label"].lower() in _LANDMARK_SKIP: continue
        cx, cy = info["pos_m"]
        d = math.hypot(cx - goal_world_xy[0], cy - goal_world_xy[1])
        if d <= max_dist_m:
            results.append((item_id, info, d))
    results.sort(key=lambda x: x[2])
    return results[:max_objects]


def generate_vln_instruction(robot_path_px, goal_world_xy, goal_item_id,
                              start_label, goal_label,
                              anchor_map, min_x, min_y, scale,
                              scene_text, base_url, model, temperature,
                              max_retries, path_m=0.0, n_turns=0):
    """
    Two-layer VLN instruction:
      Macro: objects the robot passes → route description (disambiguation by path)
      Micro: objects around the goal  → precise final stop (disambiguation by context)
    """
    path_objs = _objects_along_path(robot_path_px, anchor_map, min_x, min_y, scale)
    goal_ctx  = _objects_around_goal(goal_world_xy, anchor_map,
                                     exclude_id=goal_item_id)

    # Format strings for prompt
    if path_objs:
        path_objects_str = "\n".join(
            f"- {iid} ({inf['label']}) [x{cat_cnt_local(inf, anchor_map)}] "
            f"at path pos ~{int(k*5/(max(len(robot_path_px),1))*100)}% dist={d:.1f}m"
            for iid, inf, d, k in path_objs)
    else:
        path_objects_str = "(none — short path or open corridor)"

    if goal_ctx:
        goal_context_str = "\n".join(
            f"- {iid} ({inf['label']}) [x{cat_cnt_local(inf, anchor_map)}] {d:.1f}m from goal"
            for iid, inf, d in goal_ctx)
    else:
        goal_context_str = "(no nearby objects — goal is in open area)"

    # No-LLM fallback
    if not base_url:
        path_part = ("; ".join(f"pass the {inf['label']}" for _, inf, _, _ in path_objs) + ". ") if path_objs else ""
        ctx_part  = (", ".join(f"to the side of the {inf['label']}" for _, inf, _ in goal_ctx[:2])) if goal_ctx else ""
        instr = (f"Walk from the {start_label} toward the {goal_label}. "
                 + path_part
                 + f"Stop at the {goal_label}"
                 + (f", {ctx_part}." if ctx_part else "."))
        return instr, {
            "macro_instruction": path_part.strip(),
            "goal_description":  f"Stop at the {goal_label}" + (f", {ctx_part}." if ctx_part else "."),
            "full_instruction":  instr,
            "path_objects":      [{"item_id": i, "label": inf["label"]} for i, inf, _, _ in path_objs],
            "goal_context":      [{"item_id": i, "label": inf["label"]} for i, inf, _ in goal_ctx],
        }

    # LLM call
    for attempt in range(max_retries):
        result = _llm_call(
            base_url, model, _SYS_VLN,
            _USR_VLN.format(
                scene_text=scene_text[:1500],
                start_label=start_label, goal_label=goal_label,
                path_m=path_m, n_turns=n_turns,
                path_objects_str=path_objects_str,
                goal_context_str=goal_context_str,
            ),
            temperature=temperature, max_tokens=700,
        )
        if isinstance(result, dict):
            instr = result.get("full_instruction", "")
            print(f"    [LLM] macro: {result.get('macro_instruction','')[:70]}")
            print(f"    [LLM] goal:  {result.get('goal_description','')[:70]}")
            print(f"    [LLM] full:  {instr[:80]}")
            return instr, result
        print(f"    [LLM] attempt {attempt+1} failed")

    instr = f"Walk from the {start_label} toward the {goal_label} and stop there."
    return instr, {}

def visualize(ep_id, grid, esdf, min_x, min_y, scale,
              start_px, goal_px, robot_path_px,
              ref_path_px, seg_lm, debug_info,
              start_label, goal_label, instruction, out_path):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("[VIS] matplotlib not available"); return

    H, W = grid.shape
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    ax_occ, ax_ref = axes

    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[grid==0] = [240,240,240]; vis[grid==1] = [50,50,50]

    def _draw_base(ax):
        ax.imshow(vis, origin="lower")
        if start_px:
            ax.plot(start_px[0], start_px[1], 's', color='cyan',
                    markersize=14, zorder=9)
            ax.annotate(f"START\n({start_label})", start_px,
                        fontsize=7, color='cyan', fontweight='bold',
                        xytext=(4, 4), textcoords='offset points')
        if goal_px:
            ax.plot(goal_px[0], goal_px[1], '*', color='cyan',
                    markersize=18, zorder=9)
            ax.annotate(f"GOAL\n({goal_label})", goal_px,
                        fontsize=7, color='cyan', fontweight='bold',
                        xytext=(4, 4), textcoords='offset points')

    def _draw_landmarks(ax):
        for ki, lm in enumerate(seg_lm):
            kpx = lm["lm_px"]
            circle = plt.Circle((kpx[0], kpx[1]), radius=6,
                                 color='yellow', fill=False, lw=2.5, zorder=10)
            ax.add_patch(circle)
            ax.plot(kpx[0], kpx[1], 'p', color='white', markersize=10,
                    markeredgecolor='yellow', markeredgewidth=2, zorder=11)
            turn_tag = " ⟳" if lm["is_turn"] else ""
            ax.annotate(f"KW{ki}: {lm['label']}\n({lm['item_id']}) x{lm['cnt']}{turn_tag}",
                        kpx, fontsize=7, color='yellow', fontweight='bold',
                        xytext=(8, 4), textcoords='offset points', zorder=12)

    # ax1: occupancy + A* path + landmarks
    _draw_base(ax_occ)
    ax_occ.set_title("① Occupancy + A* path + Landmarks", fontsize=12, fontweight='bold')
    if robot_path_px and len(robot_path_px) >= 2:
        ax_occ.plot([p[0] for p in robot_path_px], [p[1] for p in robot_path_px],
                    '--', color='#888888', lw=1.5, alpha=0.6, zorder=4, label="A* path")
    _draw_landmarks(ax_occ)
    ax_occ.legend(loc="upper right", fontsize=8)

    # ax2: segmented ref path
    _draw_base(ax_ref)
    ax_ref.set_title("② Segmented Ref Path (through key waypoints)",
                     fontsize=12, fontweight='bold')
    if ref_path_px and len(ref_path_px) >= 2:
        ax_ref.plot([p[0] for p in ref_path_px], [p[1] for p in ref_path_px],
                    '-', color=REF_COLORS[0], lw=2.5, alpha=0.9, zorder=8,
                    label=REF_LABELS[0])
    _draw_landmarks(ax_ref)
    legend_elems = [Line2D([0],[0], color=REF_COLORS[0], lw=2.5, label=REF_LABELS[0])]
    ax_ref.legend(handles=legend_elems, loc="upper right", fontsize=8)

    lm_names = " → ".join(f"{e['item_id']}(x{e['cnt']})" for e in seg_lm)
    instr_short = (instruction[:120] + "...") if len(instruction) > 120 else instruction
    plt.suptitle(
        f"Episode {ep_id}  |  path={debug_info['path_m']}m  "
        f"turns={debug_info['n_turns']}  n_landmarks={debug_info['n_landmarks']}\n"
        f"Path landmarks: {lm_names}\n"
        f"Instruction: {instr_short}",
        fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[VIS] → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(ARGS.vis_dir, exist_ok=True)

    print(f"[LOAD] {ARGS.semantic_map_json}")
    grid, esdf, min_x, min_y, max_x, max_y, scale, sem_data = load_map(
        ARGS.semantic_map_json, ARGS.robot_radius, ARGS.scale)
    H, W = grid.shape
    print(f"[LOAD] grid={grid.shape}  scale={scale}m/px  "
          f"free_px={(grid==0).sum()}")

    # Scene text for LLM
    scene_text = ""
    if ARGS.scene_text_file and Path(ARGS.scene_text_file).exists():
        scene_text = Path(ARGS.scene_text_file).read_text(encoding="utf-8").strip()
        print(f"[LOAD] scene_text: {len(scene_text)} chars")
    elif ARGS.llm_base_url:
        print("[WARN] --llm_base_url set but no --scene_text_file; LLM prompt will have empty scene.")

    anchor_map, itemid2inst = build_anchor_map(sem_data)

    # Category statistics
    cat_counts: dict[str, int] = {}
    for info in anchor_map.values():
        c = info["label"].lower().strip()
        cat_counts[c] = cat_counts.get(c, 0) + 1
    print(f"[ANCHOR] {len(anchor_map)} objects")
    unique_cats = sorted((k, v) for k, v in cat_counts.items() if v == 1)
    print(f"[ANCHOR] Unique categories (x1): "
          + ", ".join(f"{k}" for k, _ in unique_cats[:15]))

    # Navigable candidates: all anchors with a free pixel on their boundary
    candidates = {}  # item_id → free_px
    for item_id, info in anchor_map.items():
        fp = instance_boundary_free(info["mask_coords_m"], grid, min_x, min_y, scale)
        if fp is not None:
            candidates[item_id] = fp
    print(f"[CAND] {len(candidates)} navigable objects "
          f"(out of {len(anchor_map)} total)")

    # Free pixel list for random sampling
    free_ys, free_xs = np.where(grid == 0)
    free_list = list(zip(free_xs.tolist(), free_ys.tolist()))

    all_episodes = []

    for ep_id in ARGS.episode_ids:
        seed = ARGS.seed_offset + ep_id
        rng  = random.Random(seed)

        print(f"\n{'='*60}")
        print(f"[EP {ep_id}]  seed={seed}")

        # ── Sample a start and goal object pair ───────────────────────
        # Prefer object-to-object navigation (like vln_trajectory_generator)
        # but fall back to free-pixel sampling if candidates are sparse.
        start_px = goal_px = None
        start_world_xy = goal_world_xy = None
        start_label = goal_label = "area"
        robot_path_px = []
        _start_item_id = _goal_item_id = None

        cand_ids = list(candidates.keys())
        rng.shuffle(cand_ids)

        placed = False
        for si in range(min(len(cand_ids), 200)):
            s_id = cand_ids[si]
            s_fp = candidates[s_id]
            for gi in range(si+1, min(len(cand_ids), 200)):
                g_id = cand_ids[gi]
                g_fp = candidates[g_id]
                # Euclidean distance filter (metres)
                sx_m, sy_m = px_to_world(s_fp[0], s_fp[1], min_x, min_y, scale)
                gx_m, gy_m = px_to_world(g_fp[0], g_fp[1], min_x, min_y, scale)
                euclid = math.hypot(sx_m - gx_m, sy_m - gy_m)
                if euclid < ARGS.min_robot_dist or euclid > ARGS.max_robot_dist:
                    continue
                path = astar(grid, s_fp, g_fp, esdf=esdf, esdf_w=0.12)
                if path is None: continue
                path_m = path_length_m(path, scale)
                if path_m < ARGS.min_robot_dist or path_m > ARGS.max_robot_dist:
                    continue
                # Accept
                start_px = s_fp; goal_px = g_fp
                start_world_xy = (sx_m, sy_m)
                goal_world_xy  = (gx_m, gy_m)
                start_label = anchor_map[s_id]["label"]
                goal_label  = anchor_map[g_id]["label"]
                robot_path_px = path
                _start_item_id = s_id
                _goal_item_id  = g_id
                placed = True
                print(f"  [robot] {s_id} → {g_id}  geo={path_m:.1f}m  "
                      f"euclid={euclid:.1f}m")
                break
            if placed: break

        if not placed:
            print("  [WARN] object-pair sampling failed; trying free-pixel fallback")
            for attempt in range(500):
                spx, spy = free_list[rng.randint(0, len(free_list)-1)]
                gpx, gpy = free_list[rng.randint(0, len(free_list)-1)]
                if reachable_area_px(grid, (spx,spy)) < MIN_REACHABLE_PX: continue
                if reachable_area_px(grid, (gpx,gpy)) < MIN_REACHABLE_PX: continue
                path = astar(grid, (spx,spy), (gpx,gpy), esdf=esdf, esdf_w=0.12)
                if path is None: continue
                path_m = path_length_m(path, scale)
                if not (ARGS.min_robot_dist <= path_m <= ARGS.max_robot_dist): continue
                start_px = (spx,spy); goal_px = (gpx,gpy)
                start_world_xy = px_to_world(spx, spy, min_x, min_y, scale)
                goal_world_xy  = px_to_world(gpx, gpy, min_x, min_y, scale)
                robot_path_px  = path
                print(f"  [robot] fallback attempt={attempt+1} geo={path_m:.1f}m")
                placed = True; break

        if not placed:
            print("  [SKIP] Could not sample valid robot path"); continue

        # ── Landmark selection ─────────────────────────────────────────
        seg_lm, debug_info = select_landmarks(
            robot_path_px, start_world_xy, goal_world_xy,
            anchor_map, scale, min_x, min_y)

        print(f"  [LM] path={debug_info['path_m']}m  "
              f"turns={debug_info['n_turns']}  n_landmarks={debug_info['n_landmarks']}")
        for item in debug_info['landmarks']:
            print(f"       {item}")

        # ── Segmented reference path ───────────────────────────────────
        key_wpts_px = [lm["lm_px"] for lm in seg_lm]
        ref_path_px = plan_ref_path(grid, esdf, start_px, goal_px, key_wpts_px)
        if ref_path_px:
            print(f"  [REF] segmented path: {len(ref_path_px)} pixels")
        else:
            print("  [WARN] ref path planning failed, using A* shortest")
            ref_path_px = robot_path_px

        # ── LLM instruction generation ────────────────────────────────
        # ── LLM instruction generation (two-layer) ──────────────────────
        instruction, llm_raw = generate_vln_instruction(
            robot_path_px=robot_path_px,
            goal_world_xy=goal_world_xy,
            goal_item_id=_goal_item_id,
            start_label=start_label,
            goal_label=goal_label,
            anchor_map=anchor_map,
            min_x=min_x, min_y=min_y, scale=scale,
            scene_text=scene_text,
            base_url=ARGS.llm_base_url,
            model=ARGS.llm_model,
            temperature=ARGS.llm_temperature,
            max_retries=ARGS.llm_max_retries,
            path_m=debug_info["path_m"],
            n_turns=debug_info["n_turns"],
        )
        print(f"  [VLN] {instruction[:100]}")

        # ── Build episode dict ─────────────────────────────────────────
        points = path_to_waypoints(ref_path_px, min_x, min_y, scale)
        points = path_to_waypoints(robot_path_px, min_x, min_y, scale)
        episode = {
            "episode_id":       ep_id,
            "seed":             seed,
            "start_item_id":    _start_item_id,
            "goal_item_id":     _goal_item_id,
            "start_label":      start_label,
            "goal_label":       goal_label,
            "path_m":           debug_info["path_m"],
            "n_turns":          debug_info["n_turns"],
            "instruction":      instruction,
            "macro_instruction": llm_raw.get("macro_instruction", "") if isinstance(llm_raw, dict) else "",
            "goal_description":  llm_raw.get("goal_description", "") if isinstance(llm_raw, dict) else "",
            "path_objects":     llm_raw.get("path_objects", []) if isinstance(llm_raw, dict) else [],
            "goal_context":     llm_raw.get("goal_context", []) if isinstance(llm_raw, dict) else [],
            "points":           points,
        }
        all_episodes.append(episode)

        # ── Visualise ─────────────────────────────────────────────────
        out_vis = os.path.join(ARGS.vis_dir, f"debug_vln_ep{ep_id}.png")
        visualize(ep_id, grid, esdf, min_x, min_y, scale,
                  start_px, goal_px, robot_path_px,
                  ref_path_px, seg_lm, debug_info,
                  start_label, goal_label, instruction, out_vis)

    # ── Save all episodes ─────────────────────────────────────────────
    if ARGS.out_json:
        with open(ARGS.out_json, "w", encoding="utf-8") as f:
            json.dump(all_episodes, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVE] {len(all_episodes)} episodes → {ARGS.out_json}")

    print(f"\n[DONE] {len(all_episodes)}/{len(ARGS.episode_ids)} episodes generated")


if __name__ == "__main__":
    main()