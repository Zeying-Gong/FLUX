"""
generate_episode_json_interaction.py
─────────────────────────────────────
LLM-guided episode generator (Isaac Sim required).

Pipeline:
  Step 1 (code)  : sample valid robot start/goal + pedestrian spawns
                   with all distance / connectivity constraints satisfied
  Step 2 (LLM)   : given confirmed spawn anchors, decide each pedestrian's
                   action sequence + narrative (semantic only, no coords)
  Step 3 (code)  : resolve item_id → Isaac coords, run A*, insert aux actions

Falls back to pure random sampling when LLM fails.

Output:
    <output_dir>/<scene_id>/episode_<id>.json
    <output_dir>/<scene_id>/episode_<id>_vis.png
    <output_dir>/<scene_id>/episode_<id>_log.json   ← quality / resolution log

Usage
-----
    /isaac-sim/python.sh generate_episode_json_interaction.py \\
        --usda  .../usda/839873.usda \\
        --semantic_map_json .../2D_Semantic_Map_839873_Complete.json \\
        --scene_text_file   .../scene_text/semantic_map_0001_839920.txt \\
        --episode_id 0 --seed 0 --num_people 3 --use_llm

    # Random-only (no LLM):
    /isaac-sim/python.sh generate_episode_json_interaction.py \\
        --usda  .../usda/839873.usda \\
        --semantic_map_json .../2D_Semantic_Map_839873_Complete.json \\
        --episode_id 0 --seed 0 --num_people 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import datetime
import heapq
import numpy as np
from collections import defaultdict, deque
from pathlib import Path
from scipy.ndimage import distance_transform_edt

# ═══════════════════════════════════════════════════════════════════════════
# CLI  — parsed before SimulationApp so --help is fast
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--usda",               required=True)
    p.add_argument("--output_dir",         default=None)
    p.add_argument("--num_people",         type=int,   default=3)
    p.add_argument("--num_waypoints",      type=int,   default=3,
                   help="Waypoints per character in random mode.")
    p.add_argument("--episode_ids",        nargs="+",  type=int, default=[0],
                   help="List of episode ids to generate in one process.")
    p.add_argument("--seed_offset",        type=int,   default=0,
                   help="seed = seed_offset + episode_id")
    p.add_argument("--overwrite",          action="store_true")
    # 2D map
    p.add_argument("--semantic_map_json",  default=None, required=True)
    p.add_argument("--robot_radius_2d",    type=float, default=0.3)
    p.add_argument("--scale_m_per_px",     type=float, default=0.05)
    # NavMesh
    p.add_argument("--collision_root",     default="/World/scene_collision")
    p.add_argument("--volume_padding",     type=float, default=1.2)
    p.add_argument("--fallback_size",      type=float, default=100.0)
    p.add_argument("--warmup_frames",      type=int,   default=60)
    p.add_argument("--cache_dir",          default=None)
    p.add_argument("--force_rebake",       action="store_true")
    # LLM
    p.add_argument("--use_llm",            action="store_true")
    p.add_argument("--scene_text_file",    default=None,
                   help="Path to semantic_map_<scene_id>.txt")
    p.add_argument("--llm_base_url",       default="http://localhost:8000/v1")
    p.add_argument("--llm_model",          default=
                   "/workspace/SAGE-3D_Official/Qwen/Qwen/Qwen3-8B")
    p.add_argument("--llm_temperature",    type=float, default=0.8)
    p.add_argument("--llm_max_retries",    type=int,   default=3)
    # vis only
    p.add_argument("--vis_only",           action="store_true")
    return p.parse_args()

ARGS = parse_args()

# ═══════════════════════════════════════════════════════════════════════════
# SimulationApp
# ═══════════════════════════════════════════════════════════════════════════
from isaacsim import SimulationApp

_exp = os.environ.get("EXP_PATH")
if _exp is None:
    print("[FATAL] EXP_PATH not set"); sys.exit(1)

simulation_app = SimulationApp(
    launch_config={
        "renderer": "RayTracedLighting",
        "headless": True,
        "enable_cameras": False,
        "crash_reporter/enabled": False,
        "crash_reporter/skip_old_dump_upload": True,
    },
    experience=os.path.join(
        _exp, "isaacsim.exp.action_and_event_data_generation.base.kit"),
)

import carb
import omni.usd
from isaacsim.replicator.agent.core.stage_util import CharacterUtil
from navmesh_utils import cache_paths, bake_navmesh
sys.path.insert(0, os.path.dirname(__file__))

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
MIN_ROBOT_DIST              = 5.0
MAX_ROBOT_DIST              = 20.0
MIN_CHAR_DIST               = 5.0
MIN_CHAR_SPACING            = 5.0
MIN_CHAR_GOAL_TO_ROBOT_GOAL = 5.0
# waypoint 测地距离降级阈值
WAYPOINT_DIST_TIERS = [4.0, 2.5, 1.5]
# Island 面积阈值
MIN_ISLAND_AREA_PER_PERSON = 8.0   # 每人需要 8m²
MIN_ISLAND_AREA_THRESHOLD  = 6.0   # island 小于此面积直接忽略
# spawn 点最小可达像素数（过滤死角）
MIN_REACHABLE_PX = 100

# Transition matrix for auxiliary actions after GoTo
_TRANSITION = {
    "GoTo": {"Idle": 0.4, "LookAround": 0.4, None: 0.2},
}
_IDLE_RANGE       = (3.0, 10.0)
_LOOKAROUND_RANGE = (3.0,  8.0)

# Relation → direction offset in semantic-map coords (x right, y up)
_RELATION_DIR = {
    "near":        None,
    "in_front_of": ( 0.0,  1.0),
    "behind":      ( 0.0, -1.0),
    "left_of":     (-1.0,  0.0),
    "right_of":    ( 1.0,  0.0),
}

# Object categories to skip when building anchor map
_ANCHOR_SKIP = {
    "wall", "floor", "ceiling", "unable area", "door frame",
    "window", "curtain", "rug", "mat", "downlights", "chandelier",
}

_LABEL_RE = re.compile(r"^(.+?)_(\d+)$")

# ═══════════════════════════════════════════════════════════════════════════
# 2-D map helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_2d_map(sem_json: str, robot_r: float, scale: float):
    with open(sem_json, encoding="utf-8") as f:
        sem_data = json.load(f)
    all_y, all_x = [], []
    for inst in sem_data:
        for y, x in inst.get("mask_coords_m", []):
            try: all_y.append(float(y)); all_x.append(float(x))
            except: pass
    if not all_y:
        return None, None, None, None, None, None
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
    return grid, esdf, min_x, min_y, scale, sem_data


def _build_anchor_map(sem_data: list) -> tuple[str, dict]:
    """Extract anchor objects → {item_id: {pos_m, label}}."""
    counter = {}
    anchor_map = {}
    rows = []
    for inst in sem_data:
        label = str(inst.get("category_label", "")).lower().strip()
        if not label or label in _ANCHOR_SKIP:
            continue
        coords = inst.get("mask_coords_m", [])
        if not coords:
            continue
        ys = [float(c[0]) for c in coords]
        xs = [float(c[1]) for c in coords]
        counter[label] = counter.get(label, 0) + 1
        item_id = f"{label.replace(' ', '_')}_{counter[label]}"
        anchor_map[item_id] = {
            "pos_m": (sum(xs) / len(xs), sum(ys) / len(ys)),
            "label": label,
        }
        rows.append(item_id)
    groups = [rows[i:i+6] for i in range(0, len(rows), 6)]
    obj_list_str = "\n".join("- " + ", ".join(g) for g in groups)
    return obj_list_str, anchor_map


def _world_to_px(x_m, y_m, min_x, min_y, scale):
    return (int(round((float(x_m) - min_x) / scale)),
            int(round((float(y_m) - min_y) / scale)))

def _px_to_world(px, py, min_x, min_y, scale):
    return min_x + (px + 0.5) * scale, min_y + (py + 0.5) * scale

def _sm_to_isaac(x_m, y_m, map_min_x, map_max_x, map_min_y, map_max_y):
    px = (map_min_x + map_max_x) - float(x_m)
    py = (map_min_y + map_max_y) - float(y_m)
    return -px, -py

def _astar(grid, start_px, goal_px, esdf=None, min_corridor_m=0.0):
    """A* with optional ESDF-based corridor width constraint.
    
    min_corridor_m: pixels where esdf < this value are treated as blocked.
    Pass esdf array and min_corridor_m > 0 to avoid narrow passages.
    """
    H, W = grid.shape
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    open_set = [(0.0, start_px)]
    came_from = {}
    g = {start_px: 0.0}
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal_px:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]; path.append(cur)
            return path[::-1]
        for d in dirs:
            nx, ny = cur[0]+d[0], cur[1]+d[1]
            if not (0 <= nx < W and 0 <= ny < H): continue
            if grid[ny, nx] == 1: continue
            # ★ 走廊宽度约束：ESDF 值 < min_corridor_m 的像素视为不可通行
            if esdf is not None and min_corridor_m > 0:
                if esdf[ny, nx] < min_corridor_m:
                    continue
            nb = (nx, ny)
            tg = g[cur] + math.hypot(d[0], d[1])
            if nb not in g or tg < g[nb]:
                came_from[nb] = cur; g[nb] = tg
                heapq.heappush(open_set, (tg + math.hypot(nx-goal_px[0], ny-goal_px[1]), nb))
    return None

def _snap(grid, px_py):
    H, W = grid.shape
    px, py = px_py
    if 0 <= px < W and 0 <= py < H and grid[py, px] == 0:
        return px_py
    visited = set(); q = deque([(px, py, 0)])
    while q:
        cx, cy, d = q.popleft()
        if d > 30: break
        if 0 <= cx < W and 0 <= cy < H and grid[cy, cx] == 0:
            return (cx, cy)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (cx+dx, cy+dy)
            if nb not in visited: visited.add(nb); q.append((nb[0], nb[1], d+1))
    return None

def _reachable_area_px(grid, start_px) -> int:
    """BFS 估算从 start_px 出发可达的自由像素数（上限 MIN_REACHABLE_PX*10 提前退出）。"""
    H, W = grid.shape
    sx, sy = start_px
    if not (0 <= sx < W and 0 <= sy < H) or grid[sy, sx] != 0:
        return 0
    limit = MIN_REACHABLE_PX * 10
    visited = {start_px}
    q = deque([start_px])
    count = 0
    while q and count < limit:
        cx, cy = q.popleft()
        count += 1
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (cx+dx, cy+dy)
            nx, ny = nb
            if nb not in visited and 0 <= nx < W and 0 <= ny < H and grid[ny, nx] == 0:
                visited.add(nb)
                q.append(nb)
    return count

def _dist2(a, b):
    return math.hypot(float(a[0])-float(b[0]), float(a[1])-float(b[1]))

def _path_length(pts):
    if not pts or len(pts) < 2: return 0.0
    return sum(math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
               for i in range(1, len(pts)))

def _navmesh_connected(nm, a, b) -> bool:
    if nm is None: return True
    try:
        pa = carb.Float3(float(a[0]), float(a[1]), float(a[2]))
        pb = carb.Float3(float(b[0]), float(b[1]), float(b[2]))
        ra = nm.query_closest_point(pa, 1.0)
        rb = nm.query_closest_point(pb, 1.0)
        if ra is None or ra[0] is None or rb is None or rb[0] is None:
            return False
        sa, sb = ra[0], rb[0]
        if (math.hypot(float(sa[0])-float(pa[0]), float(sa[1])-float(pa[1])) > 1.0 or
            math.hypot(float(sb[0])-float(pb[0]), float(sb[1])-float(pb[1])) > 1.0):
            return False
        return nm.query_shortest_path(sa, sb, agent_radius=0.3) is not None
    except Exception:
        return False

# ═══════════════════════════════════════════════════════════════════════════
# Transition-matrix auxiliary actions
# ═══════════════════════════════════════════════════════════════════════════

def _sample_aux(rng: random.Random):
    row = _TRANSITION["GoTo"]
    k = rng.choices(list(row.keys()), weights=list(row.values()), k=1)[0]
    if k == "Idle":
        d = round(rng.uniform(*_IDLE_RANGE), 1)
        return {"cmd": "Idle", "params": [str(d)]}
    if k == "LookAround":
        d = round(rng.uniform(*_LOOKAROUND_RANGE), 1)
        return {"cmd": "LookAround", "params": [str(d)]}
    return None

# ═══════════════════════════════════════════════════════════════════════════
# Anchor resolution with relation offset
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_anchor(item_id, relation, anchor_map, to_isaac_fn,
                    grid, min_x, min_y, scale):
    info = anchor_map.get(item_id)
    if info is None: return None
    cx_m, cy_m = info["pos_m"]
    direction = _RELATION_DIR.get(relation)
    if direction is None:
        px = int(round((cx_m - min_x) / scale))
        py = int(round((cy_m - min_y) / scale))
    else:
        tx_m = cx_m + direction[0] * 1.0
        ty_m = cy_m + direction[1] * 1.0
        px = int(round((tx_m - min_x) / scale))
        py = int(round((ty_m - min_y) / scale))
    snapped = _snap(grid, (px, py))
    if snapped is None: return None
    x_m, y_m = _px_to_world(snapped[0], snapped[1], min_x, min_y, scale)
    ix, iy = to_isaac_fn(x_m, y_m)
    return (ix, iy, 0.0)

# ═══════════════════════════════════════════════════════════════════════════
# LLM helpers
# ═══════════════════════════════════════════════════════════════════════════

_SYS_STEP2 = """\
You are assigning a position and action sequence to a single pedestrian in an indoor scene.

YOU MUST output ONLY a single valid JSON object. No markdown fences. No prose. No explanation.
The FIRST character of your response must be '{' and the LAST must be '}'.

Relation options: "near" | "in_front_of" | "behind" | "left_of" | "right_of"

Output schema (copy exactly, fill in values):
{"ped_id": "<copy from input>", "actions": [{"type": "GoTo", "anchor": {"item_id": "<id>", "relation": "<rel>"}}, {"type": "Idle", "duration_s": 5}]}

Rules:
- 2 to 5 actions total.
- Only use item_ids from OBJECT_LIST. Never invent ids.
- Start with at least one GoTo."""

_USR_STEP2 = """\
SCENE_DESCRIPTION:
{scene_text}

PEDESTRIAN:
  ped_id: {ped_id}
  role_hint: {role_hint}
  spawn_near: {spawn_item}

OBJECT_LIST (use ONLY these exact item_ids):
{object_list}

ALREADY PLACED PEDESTRIANS:
{placed_summary}

Respond with JSON only:"""

_SYS_NARRATIVE = """You are writing a short scene narrative for a robot simulation dataset.
Given a scene description and pedestrian activities, write ONE sentence (max 30 words)
describing what the pedestrians are doing in third-person.
Output plain text only — no JSON, no markdown."""

_USR_NARRATIVE = """Scene: {scene_text}
Pedestrians: {ped_summary}
Write one sentence describing the pedestrian activity. /no_think"""

_SYS_VLN = """You are generating a Vision-and-Language Navigation (VLN) instruction for a robot.
The robot must navigate from a start location to a goal location in an indoor scene.

Write a natural navigation instruction in FIRST-PERSON from the robot's perspective.
Use directional landmarks: "turn left", "turn right", "go straight", "pass", "stop in front of",
"on your left", "on your right", etc.
Reference visible objects and furniture by their GENERIC TYPE only (e.g. "the chair", "the desk",
"the counter"). NEVER use internal identifiers like chair_24 or cup_74 — the robot has no
access to object IDs. Keep it under 40 words. Output plain text only."""

_USR_VLN = """Scene description: {scene_text}
Robot start: near {start_item}
Robot goal: near {goal_item}
Pedestrians present: {ped_summary}

Write the navigation instruction now. /no_think"""


# def _llm_call(client, model, system, user, temperature, max_tokens=512):
#     import re as _re
#     try:
#         resp = client.chat.completions.create(
#             model=model,
#             messages=[{"role": "system", "content": system},
#                       {"role": "user",   "content": user}],
#             temperature=temperature,
#             max_tokens=max_tokens,
#         )
#     except Exception as e:
#         import traceback
#         print(f"    [LLM] API error: {e}")
#         traceback.print_exc()
#         return None
#     raw = resp.choices[0].message.content or ""
#     raw = _re.sub(r"<think>.*?</think>\s*", "", raw, flags=_re.DOTALL).strip()
#     if raw.startswith("```"):
#         raw = raw.split("```", 2)[1]
#         if raw.startswith("json"): raw = raw[4:]
#         raw = raw.rsplit("```", 1)[0].strip()
#     try:
#         return json.loads(raw)
#     except Exception:
#         # Print for debugging — caller checks isinstance(result, dict)
#         print(f"    [LLM] non-JSON response (first 300 chars):\n{raw[:300]}")
#         return raw  # narrative step returns plain text

def _llm_call(client, model, system, user, temperature, max_tokens=512):
    import re as _re
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        import traceback
        print(f"    [LLM] API error: {e}")
        traceback.print_exc()
        return None
    raw = resp.choices[0].message.content or ""
    # ★ 打印完整原始响应（截断500字）
    print(f"    [LLM-RAW] ({len(raw)} chars): {repr(raw[:500])}")
    raw = _re.sub(r"<think>.*?</think>\s*", "", raw, flags=_re.DOTALL).strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except Exception:
        print(f"    [LLM] non-JSON response (first 300 chars):\n{raw[:300]}")
        return raw

def _nearest_anchor(pos_isaac, anchor_map, to_isaac_fn):
    """Find the item_id whose Isaac-coord centroid is closest to pos_isaac."""
    best_id, best_d = None, float("inf")
    for item_id, info in anchor_map.items():
        ix, iy = to_isaac_fn(*info["pos_m"])
        d = _dist2((ix, iy, 0), pos_isaac)
        if d < best_d:
            best_d = d; best_id = item_id
    return best_id

# ═══════════════════════════════════════════════════════════════════════════
# Core LLM pipeline  (Step 2 + narrative, after code-side Step 1)
# ═══════════════════════════════════════════════════════════════════════════

def run_llm_pipeline(
    spawn_positions: list,        # list of Isaac (x,y,z) — already validated
    robot_start, robot_goal,
    scene_text: str,
    object_list_str: str,
    anchor_map: dict,
    to_isaac_fn,
    grid, min_x, min_y, scale,
    nm,
    plan_path_fn,
    rng: random.Random,
    llm_client, llm_model: str, temperature: float, max_retries: int,
) -> tuple[list[dict], str, str, list[str]]:
    """
    Returns:
        ped_results  : list of {spawn, commands, ped_id, role_hint}
        narrative    : third-person pedestrian scene description
        vln_instr    : first-person robot navigation instruction
        llm_log      : list of log strings for resolution_log
    """
    from openai import OpenAI as _OAI
    llm_log = []

    ROLE_HINTS = ["resting", "transit", "working", "waiting", "browsing"]
    placed_summary = "none"
    ped_results = []
    # GoTo target 距 robot_goal 的最小距离，随重试次数降级
    _GOAL_DIST_TIERS = [MIN_CHAR_GOAL_TO_ROBOT_GOAL, 2.0, 0.5]

    print(f"    [LLM-pipeline] {len(spawn_positions)} spawns, {len(anchor_map)} anchors")
    for i, spawn_pos in enumerate(spawn_positions):
        ped_id    = f"p{i+1}"
        role_hint = rng.choice(ROLE_HINTS)
        spawn_item = _nearest_anchor(spawn_pos, anchor_map, to_isaac_fn) or "unknown"
        print(f"    [LLM-pipeline] {ped_id} role={role_hint} spawn_near={spawn_item}")

        placed_ok = False
        for attempt in range(max_retries):
            goal_dist_min = _GOAL_DIST_TIERS[min(attempt, len(_GOAL_DIST_TIERS)-1)]
            print(f"    [LLM-pipeline] {ped_id} attempt {attempt+1}/{max_retries} calling LLM...")  # ★
            result = _llm_call(
                llm_client, llm_model,
                _SYS_STEP2,
                _USR_STEP2.format(
                    scene_text   = scene_text,
                    ped_id       = ped_id,
                    role_hint    = role_hint,
                    spawn_item   = spawn_item,
                    object_list  = object_list_str,
                    placed_summary = placed_summary,
                ),
                temperature=temperature,
            )
            print(f"    [LLM-pipeline] {ped_id} result type={type(result).__name__}")  # ★

            if not isinstance(result, dict):
                llm_log.append(f"{ped_id} attempt {attempt+1}: non-JSON response")
                continue

            valid_ids = set(anchor_map.keys())
            commands = []
            prev = spawn_pos
            actions_ok = True

            for action in result.get("actions", []):
                atype = action.get("type")
                if atype == "GoTo":
                    tgt_id  = action.get("anchor", {}).get("item_id", "")
                    tgt_rel = action.get("anchor", {}).get("relation", "near")
                    if tgt_id not in valid_ids:
                        llm_log.append(f"{ped_id} attempt {attempt+1}: invalid GoTo id '{tgt_id}'")
                        actions_ok = False; break
                    target = _resolve_anchor(tgt_id, tgt_rel, anchor_map,
                                             to_isaac_fn, grid, min_x, min_y, scale)
                    if target is None:
                        llm_log.append(f"{ped_id}: resolve failed for '{tgt_id}'")
                        continue
                    if _dist2(target, robot_goal) < goal_dist_min:
                        llm_log.append(f"{ped_id}: GoTo target too close to robot_goal (dist={_dist2(target,robot_goal):.1f}<{goal_dist_min}), skipping")
                        continue
                    if not _navmesh_connected(nm, target, robot_start):
                        llm_log.append(f"{ped_id}: GoTo target NavMesh disconnected")
                        continue
                    path_pts = plan_path_fn(prev, target)
                    if path_pts is None:
                        llm_log.append(f"{ped_id}: no path to '{tgt_id}'")
                        continue
                    x, y, z = target
                    commands.append({
                        "cmd": "GoTo",
                        "params": [f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", "_"],
                        "path": path_pts,
                    })
                    aux = _sample_aux(rng)
                    if aux: commands.append(aux)
                    prev = target
                elif atype == "Idle":
                    commands.append({"cmd": "Idle",
                                     "params": [str(float(action.get("duration_s", 5)))]})
                elif atype == "LookAround":
                    commands.append({"cmd": "LookAround",
                                     "params": [str(float(action.get("duration_s", 5)))]})
            if not actions_ok:
                continue

            # ★ GoTo-swap：如果某个 GoTo target 太靠近 robot_goal，
            #   尝试在序列内找一个距离够远的 GoTo 与之互换
            goto_indices = [j for j, c in enumerate(commands) if c.get("cmd") == "GoTo"]
            if len(goto_indices) >= 2:
                for j in goto_indices:
                    cmd_j = commands[j]
                    params_j = cmd_j.get("params", [])
                    if len(params_j) < 3:
                        continue
                    pos_j = (float(params_j[0]), float(params_j[1]), float(params_j[2]))
                    if _dist2(pos_j, robot_goal) >= goal_dist_min:
                        continue   # 这个 GoTo 本身没问题，不需要换
                    # pos_j 太靠近 robot_goal，找一个可以替换的候选
                    for k in goto_indices:
                        if k == j:
                            continue
                        cmd_k = commands[k]
                        params_k = cmd_k.get("params", [])
                        if len(params_k) < 3:
                            continue
                        pos_k = (float(params_k[0]), float(params_k[1]), float(params_k[2]))
                        if _dist2(pos_k, robot_goal) < goal_dist_min:
                            continue   # 候选也太近，换了没用
                        # ★ 互换 params 和 path（坐标信息），保留各自在序列中的位置
                        commands[j]["params"], commands[k]["params"] = \
                            commands[k]["params"], commands[j]["params"]
                        commands[j]["path"], commands[k]["path"] = \
                            commands[k]["path"], commands[j]["path"]
                        llm_log.append(
                            f"{ped_id}: swapped GoTo[{j}]↔GoTo[{k}] "
                            f"(dist {_dist2(pos_j,robot_goal):.1f}m < {goal_dist_min:.1f}m)"
                        )
                        break   # 每个"太近"的 GoTo 只换一次

            # Strip orphan Idle/LookAround before first GoTo
            first_goto = next((j for j, c in enumerate(commands) if c.get("cmd") == "GoTo"), None)

            ped_results.append({
                "ped_id":    ped_id,
                "role_hint": role_hint,
                "spawn":     spawn_pos,
                "commands":  commands,
            })
            placed_summary = "; ".join(
                f"{p['ped_id']}(near {_nearest_anchor(p['spawn'], anchor_map, to_isaac_fn)})"
                for p in ped_results
            )
            llm_log.append(f"{ped_id} OK ({len(commands)} cmds, attempt {attempt+1})")
            placed_ok = True
            break

        if not placed_ok:
            llm_log.append(f"{ped_id} FAILED all {max_retries} attempts → fallback random")

    ped_summary = "; ".join(f"{p['ped_id']}:{p['role_hint']}" for p in ped_results)

    # ★ 没有成功放置任何行人，不生成 narrative/vln，直接返回
    if not ped_results:
        llm_log.append("no pedestrians placed — skipping narrative and vln")
        print(f"    [LLM] no pedestrians placed, skipping narrative/vln")
        return [], "", "", llm_log

    # Pedestrian narrative (third-person)
    narrative_raw = _llm_call(
        llm_client, llm_model, _SYS_NARRATIVE,
        _USR_NARRATIVE.format(scene_text=scene_text, ped_summary=ped_summary),
        temperature=0.7, max_tokens=60,
    )
    narrative = narrative_raw if isinstance(narrative_raw, str) else ""
    llm_log.append(f"narrative: {narrative[:80]}")

    # Robot VLN instruction (first-person, directional)
    def _id_to_label(item_id: str) -> str:
        """'cup_74' → 'cup',  'dining_table_3' → 'dining table'"""
        if item_id and item_id in anchor_map:
            return anchor_map[item_id]["label"].replace("_", " ")
        # fallback: strip trailing _<number>
        import re
        return re.sub(r"_\d+$", "", item_id).replace("_", " ") if item_id else "area"

    start_item = _id_to_label(_nearest_anchor(robot_start, anchor_map, to_isaac_fn))
    goal_item  = _id_to_label(_nearest_anchor(robot_goal,  anchor_map, to_isaac_fn))
    vln_raw = _llm_call(
        llm_client, llm_model, _SYS_VLN,
        _USR_VLN.format(
            scene_text=scene_text, start_item=start_item,
            goal_item=goal_item,
            ped_summary=ped_summary if ped_summary else "none",
        ),
        temperature=0.7, max_tokens=80,
    )
    vln_instr = vln_raw if isinstance(vln_raw, str) else ""
    llm_log.append(f"vln_instr: {vln_instr[:80]}")
    print(f"    [LLM] narrative: {narrative[:60]}")
    print(f"    [LLM] vln_instr: {vln_instr[:60]}")
    return ped_results, narrative, vln_instr, llm_log

# ═══════════════════════════════════════════════════════════════════════════
# Helpers (stage / output path)
# ═══════════════════════════════════════════════════════════════════════════

def _update(n=1):
    for _ in range(n): simulation_app.update()

def open_stage(usda_path):
    if not os.path.exists(usda_path):
        print(f"[FATAL] USDA not found: {usda_path}"); return False
    omni.usd.get_context().open_stage(usda_path)
    _update(30)
    waited = 0
    while omni.usd.get_context().get_stage_loading_status()[1] > 0:
        simulation_app.update(); waited += 1
        if waited > 3000: print("[WARN] Stage still loading after 30s."); break
    return omni.usd.get_context().get_stage() is not None

def _scene_id():
    return os.path.splitext(os.path.basename(ARGS.usda))[0]

def _output_dir():
    sid = _scene_id()
    if ARGS.output_dir:
        d = os.path.join(ARGS.output_dir, sid)
    else:
        usda_dir = os.path.dirname(os.path.abspath(ARGS.usda))
        d = os.path.join(os.path.dirname(usda_dir), "episodes", sid)
    os.makedirs(d, exist_ok=True)
    return d

def _sample_free_point(grid, esdf, rng, min_clearance,
                       _free_cache: dict | None = None):
    """Sample a free pixel. Uses precomputed pixel list if available (much faster)."""
    # _free_cache is a mutable dict passed in to persist the list across calls
    if _free_cache is not None:
        key = (id(grid), min_clearance)
        if key not in _free_cache:
            ys, xs = np.where((grid == 0) & (esdf >= min_clearance))
            if len(ys) == 0:
                ys, xs = np.where(grid == 0)
            _free_cache[key] = list(zip(xs.tolist(), ys.tolist()))
        candidates = _free_cache[key]
        if not candidates:
            return None
        px, py = candidates[rng.randint(0, len(candidates) - 1)]
        return (px, py)
    # Fallback: original random scan
    H, W = grid.shape
    for _ in range(500):
        py = rng.randint(0, H-1); px = rng.randint(0, W-1)
        if grid[py, px] == 0 and esdf[py, px] >= min_clearance:
            return (px, py)
    for _ in range(500):
        py = rng.randint(0, H-1); px = rng.randint(0, W-1)
        if grid[py, px] == 0:
            return (px, py)
    return None

# ═══════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════

def _save_vis(out_path, grid, esdf, min_x, min_y, scale,
              map_min_x, map_max_x, map_min_y, map_max_y,
              robot_start, robot_goal, spawn_dict, commands_dict,
              narrative="", episode_id=0):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, matplotlib.patches as mpatches
    except ImportError:
        return
    def from_isaac(ix, iy):
        x_m, y_m = -float(ix), -float(iy)
        return (map_min_x+map_max_x)-x_m, (map_min_y+map_max_y)-y_m
    def i2px(ix, iy):
        x2, y2 = from_isaac(ix, iy)
        return int(round((x2-min_x)/scale)), int(round((y2-min_y)/scale))
    H, W = grid.shape
    colors = ['#FF6B6B','#FFD93D','#6BCB77','#4D96FF']
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    for ax_idx, ax in enumerate(axes):
        if ax_idx == 0:
            vis = np.zeros((H, W, 3), dtype=np.uint8)
            vis[grid==0]=[240,240,240]; vis[grid==1]=[50,50,50]
            ax.imshow(vis, origin="lower"); ax.set_title("Occupancy Map")
        else:
            ax.imshow(np.clip(esdf,0,2), origin="lower", cmap="viridis")
            ax.set_title("ESDF (m, clipped 2m)")
        if robot_start:
            rs = i2px(*robot_start[:2]); ax.plot(rs[0],rs[1],'s',color='cyan',markersize=12,zorder=5)
        if robot_goal:
            rg = i2px(*robot_goal[:2]); ax.plot(rg[0],rg[1],'*',color='cyan',markersize=16,zorder=5)
        for ci, (cn, sd) in enumerate(spawn_dict.items()):
            color = colors[ci % len(colors)]
            sp = i2px(*sd["pos"][:2])
            ax.plot(sp[0],sp[1],'o',color=color,markersize=10,zorder=6)
            goto_cmds = [cmd for cmd in commands_dict.get(cn, []) if cmd.get("cmd") == "GoTo"]
            for wi, cmd in enumerate(goto_cmds):
                # 路径线
                pts = [i2px(p[0], p[1]) for p in cmd.get("path", [])]
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            '-', color=color, lw=1.5, alpha=0.7, zorder=4)
                # waypoint marker
                params = cmd.get("params", [])
                if len(params) >= 2:
                    wp = i2px(float(params[0]), float(params[1]))
                    is_last = (wi == len(goto_cmds) - 1)
                    ax.plot(wp[0], wp[1],
                            '*' if is_last else 'D',
                            color=color,
                            markersize=14 if is_last else 9,
                            markeredgecolor='black',
                            markeredgewidth=1.2, zorder=7)
                    ax.text(wp[0] + 3, wp[1] + 3, str(wi + 1),
                            fontsize=8, color=color, fontweight='bold', zorder=8)
    if narrative:
        fig.text(0.5, 0.01, narrative, ha='center', fontsize=9, style='italic')
    plt.suptitle(f"Episode {episode_id} — {_scene_id()}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    vis_path = out_path.replace(".json", "_vis.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"[VIS] → {vis_path}")

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def _generate_one_episode(
    episode_id: int, seed: int,
    out_dir: str,
    grid, esdf, min_x, min_y, scale,
    map_min_x, map_max_x, map_min_y, map_max_y,
    sem_data, object_list_str, anchor_map,
    nm, to_isaac, sample_point_fn, plan_path_fn,
    nav_area_m2: float, actual_num: int,
    valid_islands: list, px_to_island: dict,   # ← 新增
    get_isaac_island,                           # ← 新增
) -> dict:
    """Generate one episode given pre-loaded scene resources. Returns log dict."""
    t0 = time.time()
    random.seed(seed); np.random.seed(seed)
    rng = random.Random(seed)

    out_path = os.path.join(out_dir, f"episode_{episode_id}.json")
    log_path = out_path.replace(".json", "_log.json")

    if os.path.exists(out_path) and not ARGS.overwrite:
        print(f"  [SKIP] episode_{episode_id}"); return {"skipped": True}

    # Need a fresh sample_point that uses this episode's rng
    # Precompute free-pixel list once — O(H*W) vs O(500) per call
    _free_cache: dict = {}

    def sample_point():
        for _attempt in range(10):
            px_py = _sample_free_point(grid, esdf, rng, ARGS.robot_radius_2d * 0.5,
                                       _free_cache=_free_cache)
            if px_py is None:
                return None
            if _reachable_area_px(grid, px_py) < MIN_REACHABLE_PX:
                continue  # 死角，换一个
            x_2d, y_2d = _px_to_world(px_py[0], px_py[1], min_x, min_y, scale)
            ix, iy = to_isaac(x_2d, y_2d)
            return (ix, iy, 0.0)
        return None

    # ── STEP 1: sample robot + pedestrian positions ──────────────────
    robot_start = robot_goal = None
    robot_island_idx = None   # ← 在最顶部初始化，保证后续不会 NameError

    for _ in range(200):
        s = sample_point(); g = sample_point()
        if s is None or g is None: continue
        if _dist2(s, g) < 2.0: continue
        path_pts = plan_path_fn(s, g)
        if path_pts is None: continue
        geo = _path_length(path_pts)
        if not (MIN_ROBOT_DIST <= geo <= MAX_ROBOT_DIST): continue
        if not _navmesh_connected(nm, s, g): continue
        # ★ robot 所在 island 必须有足够容量放至少 1 个行人
        s_island = get_isaac_island(s)
        if s_island is None: continue
        island_capacity = max(1, int(valid_islands[s_island]["area_m2"] / MIN_ISLAND_AREA_PER_PERSON))
        if island_capacity < 1: continue   # 理论上不会触发，但保险
        robot_start, robot_goal = s, g
        robot_island_idx = s_island
        print(f"  [ep{episode_id}] robot geodesic={geo:.1f}m  island={robot_island_idx}  "
              f"island_area={valid_islands[robot_island_idx]['area_m2']:.1f}m²  "
              f"capacity={island_capacity}")
        break

    if robot_start is None:
        print(f"  [ep{episode_id}] WARN: robot pose sampling failed 200 attempts, using relaxed fallback")
        for _ in range(500):
            s = sample_point()
            g = sample_point()
            if s is None or g is None: continue
            if _dist2(s, g) < MIN_ROBOT_DIST: continue
            path_pts = plan_path_fn(s, g)
            if path_pts is None: continue
            # ★ 同样要求 island 有容量
            s_island = get_isaac_island(s)
            if s_island is None: continue
            if int(valid_islands[s_island]["area_m2"] / MIN_ISLAND_AREA_PER_PERSON) < 1: continue
            robot_start, robot_goal = s, g
            robot_island_idx = s_island
            print(f"  [ep{episode_id}] fallback robot geodesic={_path_length(path_pts):.1f}m  island={robot_island_idx}")
            break

    if robot_start is None:
        # 最后手段：只要求 island 合法，放弃距离约束
        print(f"  [ep{episode_id}] WARN: using largest valid island fallback")
        # 直接在最大的合法 island 里采样
        largest_island_idx = 0   # valid_islands 已按面积降序排列
        for _ in range(500):
            s = sample_point()
            g = sample_point()
            if s is None or g is None: continue
            if get_isaac_island(s) != largest_island_idx: continue
            if get_isaac_island(g) != largest_island_idx: continue
            if plan_path_fn(s, g) is None: continue
            robot_start, robot_goal = s, g
            robot_island_idx = largest_island_idx
            print(f"  [ep{episode_id}] WARN: fallback to largest island, dist={_dist2(s,g):.1f}m")
            break

    robot_orientation = rng.uniform(0, 2*math.pi)

    NUM_CHAIN_WPT = 2

    # ── 计算每个 island 应放几个行人 ──────────────────────────────────
    # 保证至少 1 人在 robot_island，其余按面积比例分配
    def _plan_island_allocation(actual_num, robot_island_idx, valid_islands):
        """返回 list，长度 actual_num，每个元素是该行人目标 island idx（None=任意）。"""
        if actual_num <= 0:
            return []
        if actual_num == 1:
            return [robot_island_idx]   # 唯一一个人必须同岛

        # 按面积算每个 island 的权重
        total_area = sum(isl["area_m2"] for isl in valid_islands)
        weights = [isl["area_m2"] / total_area for isl in range(len(valid_islands))]

        # 先给 robot_island 保留 1 个名额
        remaining = actual_num - 1
        allocation = [robot_island_idx]   # 第一个人：robot 同岛

        if remaining > 0:
            # 用 largest-remainder 方法按比例分配剩余名额
            raw = [w * remaining for w in weights]
            floored = [int(r) for r in raw]
            remainders = sorted(enumerate(raw), key=lambda x: -(x[1] - int(x[1])))
            extra = remaining - sum(floored)
            for idx, _ in remainders[:extra]:
                floored[idx] += 1
            for isl_idx, count in enumerate(floored):
                allocation.extend([isl_idx] * count)

        return allocation[:actual_num]

    island_plan = _plan_island_allocation(actual_num, robot_island_idx, valid_islands)
    spawn_positions = []
    ped_chains      = []

    for i in range(actual_num):
        target_island = island_plan[i] if i < len(island_plan) else None
        placed = False

        for _ in range(200):
            p0 = sample_point()
            if p0 is None: continue
            # ★ 只要求离 robot_start 有安全距离，不管 goal
            if _dist2(p0, robot_start) < MIN_CHAR_DIST: continue
            if any(_dist2(p0, sp) < MIN_CHAR_SPACING for sp in spawn_positions): continue

            # ★ island 约束：按分配计划
            p0_island = get_isaac_island(p0)
            if target_island is not None and p0_island != target_island:
                continue
            elif target_island is None and p0_island is None:
                continue   # 不在任何合法 island 里，跳过

            if plan_path_fn(p0, robot_start) is None: continue
            if not _navmesh_connected(nm, p0, robot_start): continue

            # Step B: waypoint chain
            chain = [p0]; prev = p0; chain_ok = True
            for _ in range(NUM_CHAIN_WPT):
                wpt_ok = False
                for _ in range(200):
                    w = sample_point()
                    if w is None: continue
                    if _dist2(w, prev) < 2.0: continue
                    pts = plan_path_fn(prev, w)
                    if pts is None: continue
                    if _path_length(pts) < MIN_CHAR_DIST: continue
                    # ★ waypoint 不限制离 robot_goal 的距离
                    if not _navmesh_connected(nm, w, robot_start): continue
                    chain.append(w); prev = w; wpt_ok = True; break
                if not wpt_ok:
                    chain_ok = False; break
            if not chain_ok:
                continue
            spawn_positions.append(p0)
            ped_chains.append(chain)
            placed = True
            print(f"  [ep{episode_id}] ped{i} placed: island={p0_island} "
                f"spawn + {len(chain)-1} waypoints")
            break

        if not placed:
            # Fallback：放宽距离到 2m，允许任意合法 island
            # 但如果这是第一个人（必须同岛），仍然坚持 robot_island
            must_same_island = (i == 0)
            for _ in range(200):
                p0 = sample_point()
                if p0 is None: continue
                if _dist2(p0, robot_start) < 2.0: continue
                p0_island = get_isaac_island(p0)
                if p0_island is None: continue
                if must_same_island and p0_island != robot_island_idx: continue
                if plan_path_fn(p0, robot_start) is None: continue
                chain = [p0]; prev = p0; chain_ok = True
                for _ in range(NUM_CHAIN_WPT):
                    wpt_ok = False
                    for _ in range(200):
                        w = sample_point()
                        if w is None: continue
                        if _dist2(w, prev) < 2.0: continue
                        pts = plan_path_fn(prev, w)
                        if pts is None: continue
                        if _path_length(pts) < MIN_CHAR_DIST: continue
                        chain.append(w); prev = w; wpt_ok = True; break
                    if not wpt_ok: chain_ok = False; break
                if not chain_ok: continue
                spawn_positions.append(p0)
                ped_chains.append(chain)
                placed = True
                print(f"  [ep{episode_id}] ped{i} placed (relaxed fallback): "
                    f"island={p0_island}  {len(chain)} pts")
                break

        if not placed:
            print(f"  [ep{episode_id}] WARN: could not place pedestrian {i}")

    # ── STEP 2+3: LLM or random waypoints ───────────────────────────
    spawn_positions_dict = {}
    commands_dict        = {}
    narrative            = ""
    vln_instr            = ""
    llm_log              = []
    mode_used            = "random"

    if ARGS.use_llm and ARGS.scene_text_file:
        scene_text = Path(ARGS.scene_text_file).read_text(encoding="utf-8").strip()
        from openai import OpenAI as _OAI
        llm_client = _OAI(base_url=ARGS.llm_base_url, api_key="EMPTY")
        # Quick connectivity check
        try:
            models = llm_client.models.list()
            print(f"  [ep{episode_id}] LLM server OK: {[m.id for m in models.data]}")
        except Exception as _e:
            print(f"  [ep{episode_id}] WARN: LLM server unreachable ({_e}) — falling back to random")
            ARGS.use_llm = False
        print(f"  [ep{episode_id}] LLM action pipeline...")
        print(f"  [ep{episode_id}] spawn_positions count: {len(spawn_positions)}")
        print(f"  [ep{episode_id}] anchor_map size: {len(anchor_map)}")
        print(f"  [ep{episode_id}] scene_text length: {len(scene_text)} chars")
        # If no spawn positions were placed, sample from anchor map directly
        if not spawn_positions and anchor_map:
            print(f"  [ep{episode_id}] WARN: no spawn positions — filtering anchors by island")
            anchor_coords = []
            for k in list(anchor_map.keys()):
                coord = _resolve_anchor(k, "near", anchor_map, to_isaac,
                                        grid, min_x, min_y, scale)
                if coord is None: continue
                # ★ 只用和 robot 同 island 的 anchor
                if get_isaac_island(coord) != robot_island_idx: continue
                anchor_coords.append(coord)
                if len(anchor_coords) >= actual_num:
                    break
            spawn_positions = anchor_coords
            print(f"  [ep{episode_id}] anchor fallback: {len(spawn_positions)} valid spawns in island {robot_island_idx}")
        ped_results, narrative, vln_instr, llm_log = run_llm_pipeline(
            spawn_positions=spawn_positions,
            robot_start=robot_start, robot_goal=robot_goal,
            scene_text=scene_text,
            object_list_str=object_list_str,
            anchor_map=anchor_map,
            to_isaac_fn=to_isaac,
            grid=grid, min_x=min_x, min_y=min_y, scale=scale,
            nm=nm, plan_path_fn=plan_path_fn,
            rng=rng,
            llm_client=llm_client, llm_model=ARGS.llm_model,
            temperature=ARGS.llm_temperature, max_retries=ARGS.llm_max_retries,
        )
        if ped_results:
            mode_used = "llm"
            for p in ped_results:
                ci = len(spawn_positions_dict)
                cn = CharacterUtil.get_character_name_by_index(ci)
                spawn_positions_dict[cn] = {
                    "pos": [float(p["spawn"][0]), float(p["spawn"][1]), 0.0],
                    "rot": 0.0,
                }
                commands_dict[cn] = p["commands"]
        else:
            llm_log.append("LLM returned empty → random fallback")

    if not spawn_positions_dict:
        mode_used = "random"
        for i, chain in enumerate(ped_chains):
            cn = CharacterUtil.get_character_name_by_index(i)
            spawn_positions_dict[cn] = {
                "pos": [float(chain[0][0]), float(chain[0][1]), 0.0], "rot": 0.0,
            }
            commands = []; prev = chain[0]
            for wpt in chain[1:]:
                placed_wp = False
                for min_dist in WAYPOINT_DIST_TIERS:
                    if _dist2(wpt, prev) < min_dist * 0.4:
                        continue
                    path_pts = plan_path_fn(prev, wpt)
                    if path_pts is None:
                        continue
                    if _path_length(path_pts) < min_dist:
                        continue
                    x, y, z = wpt
                    commands.append({
                        "cmd": "GoTo",
                        "params": [f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", "_"],
                        "path": path_pts,
                    })
                    aux = _sample_aux(rng)
                    if aux: commands.append(aux)
                    prev = wpt
                    placed_wp = True
                    break
                if not placed_wp:
                    print(f"    [ep{episode_id}] {cn} wpt failed all tiers, skipping")
            commands_dict[cn] = commands
    # ★ 如果最终没有放置任何行人，清空 narrative 和 vln_instr
        if not spawn_positions_dict:
            narrative  = ""
            vln_instr  = ""
            mode_used  = "empty"
            print(f"  [ep{episode_id}] WARN: no pedestrians placed, narrative/vln cleared")
    # ── Write JSON ───────────────────────────────────────────────────
    episode_data = {
        "episode": {
            "episode_id":  episode_id,
            "seed":        seed,
            "mode":        mode_used,
            "narrative":   narrative,
            "vln_instruction": vln_instr,
            "robot": {
                "start_pos":         list(robot_start),
                "goal_pos":          list(robot_goal),
                "start_orientation": robot_orientation,
            },
            "characters": {
                "num_characters":  len(spawn_positions_dict),
                "spawn_positions": spawn_positions_dict,
                "commands":        commands_dict,
            },
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(episode_data, f, indent=4)

    elapsed = round(time.time() - t0, 2)
    log_data = {
        "scene_id":       _scene_id(),
        "episode_id":     episode_id,
        "seed":           seed,
        "mode":           mode_used,
        "num_requested":  actual_num,
        "num_placed":     len(spawn_positions_dict),
        "navigable_m2":   round(nav_area_m2, 1),
        "narrative":      narrative,
        "vln_instruction": vln_instr,
        "llm_log":        llm_log,
        "elapsed_s":      elapsed,
        "generated_at":   datetime.datetime.now().isoformat(),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    print(f"  [ep{episode_id}] ✓ mode={mode_used} placed={len(spawn_positions_dict)} "
          f"elapsed={elapsed}s → {out_path}")

    _save_vis(
        out_path, grid, esdf, min_x, min_y, scale,
        map_min_x, map_max_x, map_min_y, map_max_y,
        robot_start, robot_goal,
        spawn_positions_dict, commands_dict, narrative,
        episode_id=episode_id,
    )
    return log_data


def main() -> int:
    out_dir = _output_dir()
    print(f"[GEN] Output dir: {out_dir}")

    # ── Load 2D map (once per process) ──────────────────────────────
    if not ARGS.semantic_map_json or not os.path.exists(ARGS.semantic_map_json):
        print("[FATAL] --semantic_map_json required"); simulation_app.close(); return 3

    grid, esdf, min_x, min_y, scale, sem_data = _load_2d_map(
        ARGS.semantic_map_json, ARGS.robot_radius_2d, ARGS.scale_m_per_px)
    if grid is None:
        print("[FATAL] 2D map load failed"); simulation_app.close(); return 3

    all_y, all_x = [], []
    for inst in sem_data:
        for y, x in inst.get("mask_coords_m", []):
            try: all_y.append(float(y)); all_x.append(float(x))
            except: pass
    map_min_x, map_max_x = min(all_x), max(all_x)
    map_min_y, map_max_y = min(all_y), max(all_y)

    def to_isaac(x_m, y_m):
        return _sm_to_isaac(x_m, y_m, map_min_x, map_max_x, map_min_y, map_max_y)

    def plan_path(start_isaac, goal_isaac):
        def from_isaac(ix, iy):
            x_m, y_m = -ix, -iy
            return (map_min_x+map_max_x)-x_m, (map_min_y+map_max_y)-y_m
        sx_2d, sy_2d = from_isaac(start_isaac[0], start_isaac[1])
        gx_2d, gy_2d = from_isaac(goal_isaac[0], goal_isaac[1])
        spx = _world_to_px(sx_2d, sy_2d, min_x, min_y, scale)
        gpx = _world_to_px(gx_2d, gy_2d, min_x, min_y, scale)
        spx = _snap(grid, spx); gpx = _snap(grid, gpx)
        if spx is None or gpx is None: return None
        path_px = _astar(grid, spx, gpx,
                         esdf=esdf,
                         min_corridor_m=ARGS.robot_radius_2d * 0.5)
        if path_px is None: return None
        result = []
        for px, py in path_px:
            x_2d, y_2d = _px_to_world(px, py, min_x, min_y, scale)
            ix, iy = to_isaac(x_2d, y_2d)
            result.append([ix, iy, 0.0])
        return result

    # ── 连通域分析 ────────────────────────────────────────────────────
    def _find_islands(grid, scale):
        """BFS 找出所有连通域，返回按面积降序排列的列表。
        每个 island: {"pixels": set of (px,py), "area_m2": float}
        """
        H, W = grid.shape
        visited = np.zeros((H, W), dtype=bool)
        islands = []
        for start_y in range(H):
            for start_x in range(W):
                if grid[start_y, start_x] != 0 or visited[start_y, start_x]:
                    continue
                # BFS
                pixels = set()
                q = deque([(start_x, start_y)])
                visited[start_y, start_x] = True
                while q:
                    cx, cy = q.popleft()
                    pixels.add((cx, cy))
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nx, ny = cx+dx, cy+dy
                        if 0 <= nx < W and 0 <= ny < H and not visited[ny,nx] and grid[ny,nx] == 0:
                            visited[ny, nx] = True
                            q.append((nx, ny))
                islands.append({
                    "pixels":   pixels,
                    "area_m2":  len(pixels) * scale ** 2,
                })
        islands.sort(key=lambda x: x["area_m2"], reverse=True)
        return islands

    all_islands = _find_islands(grid, scale)
    valid_islands = [isl for isl in all_islands if isl["area_m2"] >= MIN_ISLAND_AREA_THRESHOLD]

    print(f"[GEN] Total islands: {len(all_islands)}, valid (≥{MIN_ISLAND_AREA_THRESHOLD}m²): {len(valid_islands)}")
    for idx, isl in enumerate(valid_islands):
        capacity = max(1, int(isl["area_m2"] / MIN_ISLAND_AREA_PER_PERSON))
        print(f"  island[{idx}] area={isl['area_m2']:.1f}m²  capacity={capacity}")

    if not valid_islands:
        print("[WARN] No valid island found, using full grid as single island")
        all_px = set(zip(*np.where(grid == 0)[::-1]))  # (px, py)
        valid_islands = [{"pixels": all_px, "area_m2": len(all_px) * scale**2}]

    # 总容量上限
    total_capacity = sum(
        max(1, int(isl["area_m2"] / MIN_ISLAND_AREA_PER_PERSON))
        for isl in valid_islands
    )
    actual_num = max(1, min(ARGS.num_people, total_capacity))
    nav_area_m2 = sum(isl["area_m2"] for isl in valid_islands)
    print(f"[GEN] nav_area={nav_area_m2:.0f}m²  total_capacity={total_capacity}  actual_num={actual_num}")

    # 为每个 island 建一个快速查找集合（pixel → island index）
    px_to_island: dict[tuple, int] = {}
    for idx, isl in enumerate(valid_islands):
        for px_py in isl["pixels"]:
            px_to_island[px_py] = idx

    def _get_isaac_island(isaac_pos) -> int | None:
        """给定 Isaac 坐标，返回它所在的 island index，不在任何合法 island 里则返回 None。"""
        def from_isaac(ix, iy):
            x_m, y_m = -float(ix), -float(iy)
            return (map_min_x + map_max_x) - x_m, (map_min_y + map_max_y) - y_m
        x_2d, y_2d = from_isaac(isaac_pos[0], isaac_pos[1])
        px_py = _world_to_px(x_2d, y_2d, min_x, min_y, scale)
        return px_to_island.get(px_py)

    object_list_str, anchor_map = _build_anchor_map(sem_data)

    # ── Open stage & bake NavMesh (once per process) ─────────────────
    if not open_stage(ARGS.usda):
        simulation_app.close(); return 1

    navvols_usda = cache_paths(ARGS.usda, ARGS.cache_dir)
    nav_ok, inav = bake_navmesh(
        app=simulation_app, navvols_usda=navvols_usda,
        force_rebake=ARGS.force_rebake, collision_root=ARGS.collision_root,
        volume_padding=ARGS.volume_padding, fallback_size=ARGS.fallback_size,
        warmup_frames=ARGS.warmup_frames, semantic_map_json=None,
    )
    nm = inav.get_navmesh() if nav_ok else None
    if nm is None: print("[WARN] NavMesh bake failed")

    # ── Loop over episode ids (Isaac Sim stays alive) ─────────────────
    print(f"[GEN] Generating episodes: {ARGS.episode_ids}")
    for ep_id in ARGS.episode_ids:
        seed = ARGS.seed_offset + ep_id
        print(f"\n[GEN] ── episode_{ep_id}  seed={seed} ──────────────────")
        _generate_one_episode(
            episode_id=ep_id, seed=seed,
            out_dir=out_dir,
            grid=grid, esdf=esdf, min_x=min_x, min_y=min_y, scale=scale,
            map_min_x=map_min_x, map_max_x=map_max_x,
            map_min_y=map_min_y, map_max_y=map_max_y,
            sem_data=sem_data, object_list_str=object_list_str, anchor_map=anchor_map,
            nm=nm, to_isaac=to_isaac,
            sample_point_fn=None,   # each episode builds its own via rng
            plan_path_fn=plan_path,
            nav_area_m2=nav_area_m2, actual_num=actual_num,
            valid_islands=valid_islands,        # ← 新增
            px_to_island=px_to_island,          # ← 新增
            get_isaac_island=_get_isaac_island, # ← 新增
        )

    simulation_app.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())