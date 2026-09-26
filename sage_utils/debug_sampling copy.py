"""
debug_sampling.py
─────────────────────────────────────────────────────────────────────────────
独立调试脚本：从 2D 语义地图采样机器人 + 行人 spawn/waypoint，
不依赖 Isaac Sim / NavMesh，NavMesh 连通性检查用 stub（返回 True）。

用法：
    python debug_sampling.py \
        --semantic_map_json /path/to/2D_Semantic_Map_839920_Complete.json \
        --episode_ids 0 1 2 \
        --num_people 3 \
        --vis                    # 生成可视化 PNG

调试完成后把 _generate_spawn() 的逻辑 paste 回 pipeline 即可。
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--semantic_map_json", required=True)
    p.add_argument("--num_people",        type=int,   default=3)
    p.add_argument("--episode_ids",       nargs="+",  type=int, default=[0])
    p.add_argument("--seed_offset",       type=int,   default=0)
    p.add_argument("--robot_radius_2d",   type=float, default=0.3)
    p.add_argument("--scale_m_per_px",    type=float, default=0.05)
    p.add_argument("--output_dir",        default="./debug_episodes")
    p.add_argument("--vis",               action="store_true")
    return p.parse_args()

ARGS = _parse()

# ══════════════════════════════════════════════════════════════════════
# Constants  （和 pipeline 保持一致）
# ══════════════════════════════════════════════════════════════════════
MIN_ROBOT_DIST              = 5.0
MAX_ROBOT_DIST              = 20.0
MIN_CHAR_DIST               = 3.0   # spawn 离 robot_start 的最小距离
MIN_CHAR_SPACING            = 3.0   # 行人之间最小间距
MIN_CHAR_GOAL_TO_ROBOT_GOAL = 3.0   # 行人最后一个 waypoint 离 robot_goal 的最小距离
MIN_ISLAND_AREA_THRESHOLD   = 6.0   # island 小于此面积直接忽略（过滤犄角旮旯）
MIN_REACHABLE_PX            = 100
NUM_CHAIN_WPT               = 2     # 每个行人 spawn 后的 waypoint 数

# ── 路径规划 cost 权重 ────────────────────────────────────────────────
# 行人路径：ESDF 排墙权重（越大路径越靠走廊中心，但会绕远）
PED_ESDF_WEIGHT            = 0.08
# 机器人路径：ESDF 排墙权重
ROBOT_ESDF_WEIGHT          = 0.12
# 机器人路径：行人排斥场参数
ROBOT_PED_REPULSION_SIGMA  = 1.5   # 高斯半径（m）
ROBOT_PED_REPULSION_WEIGHT = 2.0   # 峰值 cost
# actual_num = floor(nav_area / AREA_PER_PERSON_M2)，上限 num_people
# 调小 → 人更多；调大 → 人更少
AREA_PER_PERSON_M2          = 5.0

# island allocation：每 island 每人最少占用面积（只影响 allocation，不影响总人数）
MIN_ISLAND_AREA_PER_PERSON  = 4.0

# ══════════════════════════════════════════════════════════════════════
# 2-D map helpers
# ══════════════════════════════════════════════════════════════════════
def load_2d_map(sem_json: str, robot_r: float, scale: float):
    with open(sem_json, encoding="utf-8") as f:
        sem_data = json.load(f)
    all_y, all_x = [], []
    for inst in sem_data:
        for y, x in inst.get("mask_coords_m", []):
            try:
                all_y.append(float(y)); all_x.append(float(x))
            except Exception:
                pass
    if not all_y:
        raise RuntimeError("No coords in semantic map")
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
                except Exception:
                    pass
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

def sm_to_isaac(x_m, y_m, map_min_x, map_max_x, map_min_y, map_max_y):
    px = (map_min_x + map_max_x) - float(x_m)
    py = (map_min_y + map_max_y) - float(y_m)
    return -px, -py

def from_isaac(ix, iy, map_min_x, map_max_x, map_min_y, map_max_y):
    x_m, y_m = -float(ix), -float(iy)
    return (map_min_x + map_max_x) - x_m, (map_min_y + map_max_y) - y_m

# ══════════════════════════════════════════════════════════════════════
# A* + helpers
# ══════════════════════════════════════════════════════════════════════
def astar(grid, start_px, goal_px, esdf=None, min_corridor_m=0.0,
          esdf_weight=0.0, extra_cost=None):
    """
    esdf_weight > 0：把 ESDF 距离倒数叠加到移动 cost，路径主动远离墙壁。
    extra_cost：shape=(H,W) 的额外 cost map（如行人排斥场）。
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
            if esdf is not None and min_corridor_m > 0:
                if esdf[ny, nx] < min_corridor_m: continue
            nb = (nx, ny)
            step = math.hypot(d[0], d[1])
            if esdf is not None and esdf_weight > 0:
                clearance = max(esdf[ny, nx], 1e-3)
                step += esdf_weight / clearance
            if extra_cost is not None:
                step += float(extra_cost[ny, nx])
            tg = g[cur] + step
            if nb not in g or tg < g[nb]:
                came_from[nb] = cur; g[nb] = tg
                heapq.heappush(open_set,
                    (tg + math.hypot(nx-goal_px[0], ny-goal_px[1]), nb))
    return None

def snap(grid, px_py):
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
            if nb not in visited:
                visited.add(nb); q.append((nb[0], nb[1], d+1))
    return None

def reachable_area_px(grid, start_px) -> int:
    H, W = grid.shape
    sx, sy = start_px
    if not (0 <= sx < W and 0 <= sy < H) or grid[sy, sx] != 0:
        return 0
    limit = MIN_REACHABLE_PX * 10
    visited = {start_px}; q = deque([start_px]); count = 0
    while q and count < limit:
        cx, cy = q.popleft(); count += 1
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (cx+dx, cy+dy); nx, ny = nb
            if nb not in visited and 0<=nx<W and 0<=ny<H and grid[ny,nx]==0:
                visited.add(nb); q.append(nb)
    return count

def dist2(a, b):
    return math.hypot(float(a[0])-float(b[0]), float(a[1])-float(b[1]))

def path_length(pts):
    if not pts or len(pts) < 2: return 0.0
    return sum(math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
               for i in range(1, len(pts)))

# ══════════════════════════════════════════════════════════════════════
# Island 分析
# ══════════════════════════════════════════════════════════════════════
def find_islands(grid, scale):
    H, W = grid.shape
    visited = np.zeros((H, W), dtype=bool)
    islands = []
    for start_y in range(H):
        for start_x in range(W):
            if grid[start_y, start_x] != 0 or visited[start_y, start_x]:
                continue
            pixels = set(); q = deque([(start_x, start_y)])
            visited[start_y, start_x] = True
            while q:
                cx, cy = q.popleft(); pixels.add((cx, cy))
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = cx+dx, cy+dy
                    if 0<=nx<W and 0<=ny<H and not visited[ny,nx] and grid[ny,nx]==0:
                        visited[ny,nx] = True; q.append((nx,ny))
            islands.append({"pixels": pixels, "area_m2": len(pixels)*scale**2})
    islands.sort(key=lambda x: x["area_m2"], reverse=True)
    return islands

def plan_island_allocation(actual_num, robot_island_idx, valid_islands):
    """
    Island 分配策略：
      - 第一个人固定分到 robot_island（保证至少有人和机器人同岛）
      - 其余 (actual_num-1) 个人按各 island 面积比例分配（largest-remainder 取整）
      - 每个 island 的 cap = max(1, floor(area / MIN_ISLAND_AREA_PER_PERSON))
      - 如果按比例分配后总量仍不足 actual_num，循环把剩余名额按面积降序补给各 island
      - 返回长度恰好为 actual_num 的列表
    """
    if actual_num <= 0:
        return []
    if actual_num == 1:
        return [robot_island_idx]

    n_isl = len(valid_islands)
    caps  = [max(1, int(isl["area_m2"] / MIN_ISLAND_AREA_PER_PERSON))
             for isl in valid_islands]
    total_area = sum(isl["area_m2"] for isl in valid_islands)

    # 按面积比例分配全部 actual_num 个名额（含 robot_island 那 1 个）
    weights = [isl["area_m2"] / total_area for isl in valid_islands]
    raw     = [w * actual_num for w in weights]
    counts  = [int(r) for r in raw]

    # largest-remainder 补足到 actual_num
    remainders = sorted(enumerate(raw), key=lambda x: -(x[1] - int(x[1])))
    extra = actual_num - sum(counts)
    for idx, _ in remainders[:extra]:
        counts[idx] += 1

    # 应用 caps（超出的额度重新分配给未满的 island，按面积降序）
    overflow = sum(max(0, c - caps[idx]) for idx, c in enumerate(counts))
    counts   = [min(c, caps[idx]) for idx, c in enumerate(counts)]
    if overflow > 0:
        order = sorted(range(n_isl), key=lambda k: -valid_islands[k]["area_m2"])
        for idx in order:
            if overflow <= 0: break
            room = caps[idx] - counts[idx]
            add  = min(room, overflow)
            counts[idx] += add
            overflow    -= add

    # 保证 robot_island 至少有 1 人（可能在比例分配后为 0）
    if counts[robot_island_idx] == 0:
        # 从分配最多的 island 里借 1 个
        donor = max((k for k in range(n_isl) if k != robot_island_idx and counts[k] > 0),
                    key=lambda k: counts[k], default=None)
        if donor is not None:
            counts[donor]          -= 1
            counts[robot_island_idx] += 1

    # 展开为列表：robot_island 的第一个名额排最前
    allocation = [robot_island_idx]
    remaining_counts = counts[:]
    remaining_counts[robot_island_idx] -= 1   # 已放一个
    # 按面积降序展开其余（大岛先排，让优先级高的先采样）
    order = sorted(range(n_isl), key=lambda k: -valid_islands[k]["area_m2"])
    for idx in order:
        allocation.extend([idx] * remaining_counts[idx])

    assert len(allocation) == actual_num, \
        f"allocation len={len(allocation)} != actual_num={actual_num}"

    return allocation

# ══════════════════════════════════════════════════════════════════════
# Stub：NavMesh 连通性（不依赖 Isaac Sim，直接用 A* 代替）
# ══════════════════════════════════════════════════════════════════════
def navmesh_connected_stub(plan_fn, a, b) -> bool:
    """用 A* 路径规划代替 NavMesh 连通性检查。"""
    return plan_fn(a, b) is not None

# ══════════════════════════════════════════════════════════════════════
# 核心采样函数
# ══════════════════════════════════════════════════════════════════════
def sample_free_point(grid, esdf, rng, min_clearance, _free_cache=None):
    if _free_cache is not None:
        key = (id(grid), min_clearance)
        if key not in _free_cache:
            ys, xs = np.where((grid == 0) & (esdf >= min_clearance))
            if len(ys) == 0:
                ys, xs = np.where(grid == 0)
            _free_cache[key] = list(zip(xs.tolist(), ys.tolist()))
        candidates = _free_cache[key]
        if not candidates: return None
        px, py = candidates[rng.randint(0, len(candidates)-1)]
        return (px, py)
    H, W = grid.shape
    for _ in range(1000):
        py = rng.randint(0, H-1); px = rng.randint(0, W-1)
        if grid[py,px]==0 and esdf[py,px]>=min_clearance:
            return (px, py)
    return None


def generate_spawn(
    episode_id: int,
    seed: int,
    grid, esdf, min_x, min_y, scale,
    map_min_x, map_max_x, map_min_y, map_max_y,
    valid_islands: list,
    px_to_island: dict,
    actual_num: int,
) -> dict:
    """
    核心采样函数（纯 2D，无 Isaac Sim 依赖）。
    返回 dict 包含：
        robot_start, robot_goal, robot_island_idx,
        pedestrians: list of {spawn, chain}
        log: list of str
    """
    t0 = time.time()
    random.seed(seed); np.random.seed(seed)
    rng = random.Random(seed)
    log = []

    _free_cache: dict = {}

    def to_isaac(x_m, y_m):
        return sm_to_isaac(x_m, y_m, map_min_x, map_max_x, map_min_y, map_max_y)

    def from_isaac_local(ix, iy):
        return from_isaac(ix, iy, map_min_x, map_max_x, map_min_y, map_max_y)

    def sample_point():
        for _ in range(10):
            px_py = sample_free_point(grid, esdf, rng,
                                      ARGS.robot_radius_2d * 0.5, _free_cache)
            if px_py is None: return None
            if reachable_area_px(grid, px_py) < MIN_REACHABLE_PX: continue
            x_2d, y_2d = px_to_world(px_py[0], px_py[1], min_x, min_y, scale)
            ix, iy = to_isaac(x_2d, y_2d)
            return (ix, iy, 0.0)
        return None

    def plan_path(a, b, corridor_m=None, esdf_weight=0.0, extra_cost=None):
        """
        corridor_m=None  → 机器人默认用 robot_radius*0.5
        esdf_weight      → ESDF 排墙 cost 强度
        extra_cost       → 叠加额外 cost map（如行人排斥场），shape=(H,W)
        """
        if corridor_m is None:
            corridor_m = ARGS.robot_radius_2d * 0.5
        x_2d_a, y_2d_a = from_isaac_local(a[0], a[1])
        x_2d_b, y_2d_b = from_isaac_local(b[0], b[1])
        spx = snap(grid, world_to_px(x_2d_a, y_2d_a, min_x, min_y, scale))
        gpx = snap(grid, world_to_px(x_2d_b, y_2d_b, min_x, min_y, scale))
        if spx is None or gpx is None:
            return None
        path_px = astar(grid, spx, gpx, esdf=esdf,
                        min_corridor_m=corridor_m,
                        esdf_weight=esdf_weight,
                        extra_cost=extra_cost)
        if path_px is None: return None
        pts = []
        for px, py in path_px:
            x_2d, y_2d = px_to_world(px, py, min_x, min_y, scale)
            ix, iy = to_isaac(x_2d, y_2d)
            pts.append([ix, iy, 0.0])
        return pts

    def plan_path_ped(a, b):
        """行人路径：无 corridor 约束，ESDF cost 让路径远离墙壁。"""
        return plan_path(a, b, corridor_m=0.0, esdf_weight=PED_ESDF_WEIGHT)

    def build_ped_repulsion(spawn_list, sigma_m=2.0, weight=3.0):
        """
        在 spawn_list 的像素位置周围叠加高斯排斥场。
        sigma_m：影响半径（m）；weight：峰值 cost。
        返回 shape=(H,W) 的 float32 cost map。
        """
        H, W = grid.shape
        cost = np.zeros((H, W), dtype=np.float32)
        sigma_px = sigma_m / scale
        for pos in spawn_list:
            x_2d, y_2d = from_isaac_local(pos[0], pos[1])
            cx, cy = world_to_px(x_2d, y_2d, min_x, min_y, scale)
            # 只在有限窗口内计算，避免全图扫描
            r = int(sigma_px * 3)
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < W and 0 <= ny < H and grid[ny, nx] == 0:
                        d2 = (dx**2 + dy**2) / (sigma_px**2)
                        cost[ny, nx] += weight * math.exp(-0.5 * d2)
        return cost

    def get_island(isaac_pos):
        x_2d, y_2d = from_isaac_local(isaac_pos[0], isaac_pos[1])
        pxpy = world_to_px(x_2d, y_2d, min_x, min_y, scale)
        return px_to_island.get(pxpy)

    # ── 机器人采样（三级降级）────────────────────────────────────────
    robot_start = robot_goal = None
    robot_island_idx = None

    # Level 1: 严格约束
    log.append("=== ROBOT SAMPLING: Level 1 (strict) ===")
    for attempt in range(200):
        s = sample_point(); g = sample_point()
        if s is None or g is None: continue
        d_sg = dist2(s, g)
        if d_sg < 2.0:
            continue
        path_pts = plan_path(s, g)
        if path_pts is None:
            continue
        geo = path_length(path_pts)
        if not (MIN_ROBOT_DIST <= geo <= MAX_ROBOT_DIST):
            continue
        s_island = get_island(s)
        if s_island is None:
            continue
        island_cap = max(1, int(valid_islands[s_island]["area_m2"] / MIN_ISLAND_AREA_PER_PERSON))
        if island_cap < 1: continue
        robot_start, robot_goal = s, g
        robot_island_idx = s_island
        msg = (f"  [L1 OK attempt={attempt+1}] "
               f"geo={geo:.1f}m dist_euclid={d_sg:.1f}m "
               f"island={s_island} area={valid_islands[s_island]['area_m2']:.1f}m² "
               f"capacity={island_cap}")
        print(msg); log.append(msg)
        break
    else:
        log.append("  [L1 FAIL] 200 attempts exhausted")

    # Level 2: 放宽 MAX_ROBOT_DIST
    if robot_start is None:
        log.append("=== ROBOT SAMPLING: Level 2 (relaxed dist) ===")
        for attempt in range(500):
            s = sample_point(); g = sample_point()
            if s is None or g is None: continue
            if dist2(s, g) < MIN_ROBOT_DIST: continue
            path_pts = plan_path(s, g)
            if path_pts is None: continue
            s_island = get_island(s)
            if s_island is None: continue
            if int(valid_islands[s_island]["area_m2"] / MIN_ISLAND_AREA_PER_PERSON) < 1: continue
            robot_start, robot_goal = s, g
            robot_island_idx = s_island
            msg = (f"  [L2 OK attempt={attempt+1}] "
                   f"geo={path_length(path_pts):.1f}m island={s_island}")
            print(msg); log.append(msg)
            break
        else:
            log.append("  [L2 FAIL] 500 attempts exhausted")

    # Level 3: 最大 island，放弃距离
    if robot_start is None:
        log.append("=== ROBOT SAMPLING: Level 3 (largest island only) ===")
        for attempt in range(500):
            s = sample_point(); g = sample_point()
            if s is None or g is None: continue
            if get_island(s) != 0: continue
            if get_island(g) != 0: continue
            if plan_path(s, g) is None: continue
            robot_start, robot_goal = s, g
            robot_island_idx = 0
            msg = f"  [L3 OK attempt={attempt+1}] dist={dist2(s,g):.1f}m"
            print(msg); log.append(msg)
            break
        else:
            log.append("  [L3 FAIL] giving up on robot sampling")

    if robot_start is None:
        log.append("FATAL: robot sampling completely failed")
        return {"success": False, "log": log}

    # ── 行人采样 ─────────────────────────────────────────────────────
    island_plan = plan_island_allocation(actual_num, robot_island_idx, valid_islands)
    log.append(f"=== PEDESTRIAN SAMPLING: {actual_num} peds, island_plan={island_plan} ===")

    spawn_positions = []
    ped_chains      = []

    for i in range(actual_num):
        target_island = island_plan[i] if i < len(island_plan) else None
        same_island   = (target_island == robot_island_idx)  # 是否需要连通性检查
        placed = False
        log.append(f"--- ped{i}: target_island={target_island} same_island={same_island} ---")

        # 主循环
        fail_reasons = {}
        for attempt in range(200):
            p0 = sample_point()
            if p0 is None:
                fail_reasons["sample_none"] = fail_reasons.get("sample_none",0)+1; continue

            # same_island 行人需要离 robot_start 够远；非 same_island 无此要求
            if same_island and dist2(p0, robot_start) < MIN_CHAR_DIST:
                fail_reasons["too_close_start"] = fail_reasons.get("too_close_start",0)+1; continue

            if any(dist2(p0, sp) < MIN_CHAR_SPACING for sp in spawn_positions):
                fail_reasons["too_close_other_ped"] = fail_reasons.get("too_close_other_ped",0)+1; continue

            p0_island = get_island(p0)
            if target_island is not None and p0_island != target_island:
                fail_reasons["wrong_island"] = fail_reasons.get("wrong_island",0)+1; continue
            if target_island is None and p0_island is None:
                fail_reasons["no_island"] = fail_reasons.get("no_island",0)+1; continue

            # ★ 只有 same_island 行人才检查到 robot_start 的连通性
            if same_island and plan_path_ped(p0, robot_start) is None:
                fail_reasons["no_path_to_robot"] = fail_reasons.get("no_path_to_robot",0)+1; continue

            # waypoint chain：chain 存 (waypoint, path_to_waypoint) 对
            chain = [(p0, [])]   # spawn 点没有入射路径
            prev = p0; chain_ok = True
            wpt_fail_reasons = {}
            for wpt_i in range(NUM_CHAIN_WPT):
                wpt_ok = False
                for _ in range(200):
                    w = sample_point()
                    if w is None:
                        wpt_fail_reasons["sample_none"] = wpt_fail_reasons.get("sample_none",0)+1; continue
                    if dist2(w, prev) < 2.0:
                        wpt_fail_reasons["too_close_prev"] = wpt_fail_reasons.get("too_close_prev",0)+1; continue
                    # ★ 非 same_island 行人：waypoint 必须留在自己 island 内
                    if not same_island and get_island(w) != target_island:
                        wpt_fail_reasons["wrong_island"] = wpt_fail_reasons.get("wrong_island",0)+1; continue
                    pts = plan_path_ped(prev, w)
                    if pts is None:
                        wpt_fail_reasons["no_path"] = wpt_fail_reasons.get("no_path",0)+1; continue
                    if path_length(pts) < MIN_CHAR_DIST:
                        wpt_fail_reasons["path_too_short"] = wpt_fail_reasons.get("path_too_short",0)+1; continue
                    chain.append((w, pts)); prev = w; wpt_ok = True; break
                if not wpt_ok:
                    chain_ok = False
                    log.append(f"  ped{i} wpt{wpt_i} failed: {wpt_fail_reasons}")
                    break

            if not chain_ok:
                fail_reasons["chain_failed"] = fail_reasons.get("chain_failed",0)+1; continue

            # ★ 终点调换：若最后一个 waypoint 离 robot_goal 太近，
            #   找所有 waypoint 里离 robot_goal 最远的换到最后，并重新规划受影响路径段。
            # chain[0]=spawn(固定), chain[1:]=waypoints
            wpts = chain[1:]   # list of (pos, path_from_prev)
            if len(wpts) >= 2 and dist2(wpts[-1][0], robot_goal) < MIN_CHAR_GOAL_TO_ROBOT_GOAL:
                # 找所有候选中离 robot_goal 最远的（排除最后一个自己）
                best_k, best_d = None, -1.0
                for k in range(len(wpts) - 1):
                    d = dist2(wpts[k][0], robot_goal)
                    if d >= MIN_CHAR_GOAL_TO_ROBOT_GOAL and d > best_d:
                        best_d = d; best_k = k
                if best_k is not None:
                    old_last = dist2(wpts[-1][0], robot_goal)
                    wpts[best_k], wpts[-1] = wpts[-1], wpts[best_k]
                    prev_of_bestk = chain[best_k][0]
                    new_path_bestk = plan_path_ped(prev_of_bestk, wpts[best_k][0])
                    if new_path_bestk is not None:
                        wpts[best_k] = (wpts[best_k][0], new_path_bestk)
                    prev_of_last = wpts[-2][0]
                    new_path_last = plan_path_ped(prev_of_last, wpts[-1][0])
                    if new_path_last is not None:
                        wpts[-1] = (wpts[-1][0], new_path_last)
                    log.append(f"  ped{i} swapped wpt[{best_k}]↔wpt[-1]: "
                               f"last dist {old_last:.1f}m→{best_d:.1f}m from robot_goal")
                else:
                    log.append(f"  ped{i} last wpt {dist2(wpts[-1][0], robot_goal):.1f}m "
                               f"from robot_goal, no better swap candidate")
            chain = [chain[0]] + wpts

            spawn_positions.append(chain[0][0])
            ped_chains.append(chain)
            placed = True
            d_info = f"dist_to_robot={dist2(p0,robot_start):.1f}m" if same_island else "bg_ped"
            msg = (f"  [ped{i} OK attempt={attempt+1}] "
                   f"island={p0_island} {d_info} chain_len={len(chain)}")
            print(msg); log.append(msg)
            break

        if not placed:
            log.append(f"  ped{i} main loop FAIL reasons: {fail_reasons}")
            # Fallback：放宽距离，must_same 仅对 ped0（必须和机器人同岛）
            must_same = (i == 0) or same_island
            log.append(f"  ped{i} fallback: must_same_island={must_same}")
            fb_fail = {}
            for attempt in range(200):
                p0 = sample_point()
                if p0 is None:
                    fb_fail["sample_none"] = fb_fail.get("sample_none",0)+1; continue
                if same_island and dist2(p0, robot_start) < 2.0:
                    fb_fail["too_close_start"] = fb_fail.get("too_close_start",0)+1; continue
                p0_island = get_island(p0)
                if p0_island is None:
                    fb_fail["no_island"] = fb_fail.get("no_island",0)+1; continue
                if must_same and p0_island != robot_island_idx:
                    fb_fail["wrong_island"] = fb_fail.get("wrong_island",0)+1; continue
                if not must_same and target_island is not None and p0_island != target_island:
                    fb_fail["wrong_island"] = fb_fail.get("wrong_island",0)+1; continue
                # ★ same_island 才检查连通性
                if same_island and plan_path_ped(p0, robot_start) is None:
                    fb_fail["no_path"] = fb_fail.get("no_path",0)+1; continue
                chain = [(p0, [])]; prev = p0; chain_ok = True
                for _ in range(NUM_CHAIN_WPT):
                    wpt_ok = False
                    for _ in range(200):
                        w = sample_point()
                        if w is None: continue
                        if dist2(w, prev) < 2.0: continue
                        if not same_island and get_island(w) != p0_island: continue
                        pts = plan_path_ped(prev, w)
                        if pts is None: continue
                        if path_length(pts) < MIN_CHAR_DIST: continue
                        chain.append((w, pts)); prev = w; wpt_ok = True; break
                    if not wpt_ok: chain_ok = False; break
                if not chain_ok:
                    fb_fail["chain_failed"] = fb_fail.get("chain_failed",0)+1; continue
                # ★ 同样做终点调换
                wpts = chain[1:]
                if len(wpts) >= 2 and dist2(wpts[-1][0], robot_goal) < MIN_CHAR_GOAL_TO_ROBOT_GOAL:
                    best_k, best_d = None, -1.0
                    for k in range(len(wpts) - 1):
                        d = dist2(wpts[k][0], robot_goal)
                        if d >= MIN_CHAR_GOAL_TO_ROBOT_GOAL and d > best_d:
                            best_d = d; best_k = k
                    if best_k is not None:
                        wpts[best_k], wpts[-1] = wpts[-1], wpts[best_k]
                        prev_of_bestk = chain[best_k][0]
                        new_path_bestk = plan_path_ped(prev_of_bestk, wpts[best_k][0])
                        if new_path_bestk is not None:
                            wpts[best_k] = (wpts[best_k][0], new_path_bestk)
                        prev_of_last = wpts[-2][0]
                        new_path_last = plan_path_ped(prev_of_last, wpts[-1][0])
                        if new_path_last is not None:
                            wpts[-1] = (wpts[-1][0], new_path_last)
                        log.append(f"  ped{i} fallback swapped wpt[{best_k}]↔wpt[-1] "
                                   f"new_last_dist={best_d:.1f}m")
                chain = [chain[0]] + wpts
                spawn_positions.append(chain[0][0])
                ped_chains.append(chain)
                placed = True
                msg = (f"  [ped{i} FALLBACK OK attempt={attempt+1}] "
                       f"island={p0_island} chain_len={len(chain)}")
                print(msg); log.append(msg)
                break

            if not placed:
                log.append(f"  ped{i} fallback FAIL: {fb_fail}")
                print(f"  [ped{i}] WARN: completely failed to place")

    elapsed = round(time.time()-t0, 2)
    log.append(f"=== DONE: {len(spawn_positions)}/{actual_num} peds placed, {elapsed}s ===")
    print(f"  [ep{episode_id}] spawn done: {len(spawn_positions)}/{actual_num} peds  ({elapsed}s)")

    # ── 机器人路径规划（行人采样完后，叠加行人排斥场）──────────────
    ped_repulsion = build_ped_repulsion(
        spawn_positions,
        sigma_m=ROBOT_PED_REPULSION_SIGMA,
        weight=ROBOT_PED_REPULSION_WEIGHT,
    )
    robot_path = plan_path(
        robot_start, robot_goal,
        corridor_m=ARGS.robot_radius_2d * 0.5,
        esdf_weight=ROBOT_ESDF_WEIGHT,
        extra_cost=ped_repulsion,
    )
    if robot_path is None:
        # 降级：去掉行人排斥，只用 ESDF cost
        robot_path = plan_path(robot_start, robot_goal,
                               esdf_weight=ROBOT_ESDF_WEIGHT)
        log.append("  [robot_path] repulsion fallback: using ESDF-only path")
    if robot_path is None:
        robot_path = plan_path(robot_start, robot_goal)
        log.append("  [robot_path] final fallback: plain A*")
    log.append(f"  [robot_path] len={len(robot_path) if robot_path else 0} pts")
    print(f"  [ep{episode_id}] robot_path: {len(robot_path) if robot_path else 0} pts")

    return {
        "success": True,
        "episode_id": episode_id,
        "seed": seed,
        "robot_start": robot_start,
        "robot_goal": robot_goal,
        "robot_path": robot_path,
        "robot_island_idx": robot_island_idx,
        "pedestrians": [
            {
                "spawn": list(ped_chains[i][0][0]),
                "chain": [list(pt) for pt, _ in ped_chains[i]],
                "paths": [path for _, path in ped_chains[i]],
            }
            for i in range(len(ped_chains))
        ],
        "num_placed": len(spawn_positions),
        "log": log,
    }

# ══════════════════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════════════════
def visualize(result, grid, esdf, min_x, min_y, scale,
              map_min_x, map_max_x, map_min_y, map_max_y, out_path):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[VIS] matplotlib not available, skipping"); return

    def i2px(ix, iy):
        x_2d = (map_min_x+map_max_x) - (-float(ix))
        y_2d = (map_min_y+map_max_y) - (-float(iy))
        return world_to_px(x_2d, y_2d, min_x, min_y, scale)

    H, W = grid.shape
    colors = ['#FF6B6B','#FFD93D','#6BCB77','#4D96FF','#FF9F45','#C77DFF']

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    for ax_idx, ax in enumerate(axes):
        if ax_idx == 0:
            vis = np.zeros((H, W, 3), dtype=np.uint8)
            vis[grid==0] = [240,240,240]; vis[grid==1] = [50,50,50]
            ax.imshow(vis, origin="lower")
            ax.set_title(f"Occupancy + Spawn  (ep{result['episode_id']})")
        else:
            ax.imshow(np.clip(esdf, 0, 2), origin="lower", cmap="viridis")
            ax.set_title("ESDF (m, clipped 2m)")

        # 机器人
        if result.get("robot_start"):
            rs = i2px(*result["robot_start"][:2])
            ax.plot(rs[0], rs[1], 's', color='cyan', markersize=14,
                    zorder=5, label="robot start")
            ax.annotate("R_start", rs, fontsize=8, color='cyan',
                        xytext=(5,5), textcoords='offset points')
        if result.get("robot_goal"):
            rg = i2px(*result["robot_goal"][:2])
            ax.plot(rg[0], rg[1], '*', color='cyan', markersize=18,
                    zorder=5, label="robot goal")
            ax.annotate("R_goal", rg, fontsize=8, color='cyan',
                        xytext=(5,5), textcoords='offset points')
        # 机器人路径（虚线，行人排斥后的规划结果）
        robot_path = result.get("robot_path")
        if robot_path and len(robot_path) >= 2:
            rp = [i2px(p[0], p[1]) for p in robot_path]
            ax.plot([p[0] for p in rp], [p[1] for p in rp],
                    '--', color='cyan', lw=2.0, alpha=0.8, zorder=4,
                    label="robot path")

        # 行人
        for pi, ped in enumerate(result.get("pedestrians", [])):
            color = colors[pi % len(colors)]
            chain = ped["chain"]
            paths = ped.get("paths", [])

            # spawn 点
            sp = i2px(*chain[0][:2])
            ax.plot(sp[0], sp[1], 'o', color=color, markersize=11,
                    zorder=6, markeredgecolor='black', markeredgewidth=1)
            ax.annotate(f"P{pi}", sp, fontsize=9, color=color,
                        fontweight='bold', xytext=(5,5), textcoords='offset points')

            # waypoints + 真实 A* 路径
            for wi in range(1, len(chain)):
                wp = i2px(*chain[wi][:2])
                is_last = (wi == len(chain)-1)
                ax.plot(wp[0], wp[1],
                        '*' if is_last else 'D',
                        color=color,
                        markersize=14 if is_last else 9,
                        markeredgecolor='black', markeredgewidth=1.2,
                        zorder=7)
                ax.annotate(f"P{pi}W{wi}", wp, fontsize=7, color=color,
                            xytext=(5,-10), textcoords='offset points')
                # 画 A* 路径（而非直线）
                if wi < len(paths) and paths[wi]:
                    path_px_list = [i2px(p[0], p[1]) for p in paths[wi]]
                    xs = [p[0] for p in path_px_list]
                    ys = [p[1] for p in path_px_list]
                    ax.plot(xs, ys, '-', color=color, lw=1.8, alpha=0.7, zorder=4)
                else:
                    # 没有路径数据时退化为直线（带虚线以示区别）
                    prev_px = i2px(*chain[wi-1][:2])
                    ax.plot([prev_px[0], wp[0]], [prev_px[1], wp[1]],
                            '--', color=color, lw=1.2, alpha=0.5, zorder=4)

    plt.suptitle(
        f"ep{result['episode_id']}  seed={result['seed']}  "
        f"placed={result['num_placed']}  island={result.get('robot_island_idx')}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[VIS] → {out_path}")

# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(ARGS.output_dir, exist_ok=True)

    print(f"[LOAD] {ARGS.semantic_map_json}")
    grid, esdf, min_x, min_y, max_x, max_y, scale, sem_data = load_2d_map(
        ARGS.semantic_map_json, ARGS.robot_radius_2d, ARGS.scale_m_per_px)
    print(f"[LOAD] grid={grid.shape}  free_px={int((grid==0).sum())}  "
          f"scale={scale}m/px")

    map_min_x, map_max_x = min_x, max_x
    map_min_y, map_max_y = min_y, max_y

    # Island 分析
    all_islands = find_islands(grid, scale)
    valid_islands = [isl for isl in all_islands
                     if isl["area_m2"] >= MIN_ISLAND_AREA_THRESHOLD]
    print(f"[ISLANDS] total={len(all_islands)}  valid(≥{MIN_ISLAND_AREA_THRESHOLD}m²)={len(valid_islands)}")
    for idx, isl in enumerate(valid_islands):
        cap = max(1, int(isl["area_m2"]/MIN_ISLAND_AREA_PER_PERSON))
        print(f"  island[{idx}] area={isl['area_m2']:.1f}m²  capacity={cap}")

    if not valid_islands:
        print("[WARN] no valid island, using full grid")
        all_px = set(zip(*np.where(grid==0)[::-1]))
        valid_islands = [{"pixels": all_px, "area_m2": len(all_px)*scale**2}]

    px_to_island: dict = {}
    for idx, isl in enumerate(valid_islands):
        for pxpy in isl["pixels"]:
            px_to_island[pxpy] = idx

    total_cap = sum(max(1, int(isl["area_m2"]/MIN_ISLAND_AREA_PER_PERSON))
                    for isl in valid_islands)
    nav_area  = sum(isl["area_m2"] for isl in valid_islands)
    # actual_num 由场景面积 / 每人面积决定，再和 num_people 取 min
    area_based_num = max(1, int(nav_area / AREA_PER_PERSON_M2))
    actual_num = min(ARGS.num_people, area_based_num)
    print(f"[GEN] nav_area={nav_area:.1f}m²  AREA_PER_PERSON={AREA_PER_PERSON_M2}m²  "
          f"area_based={area_based_num}  requested={ARGS.num_people}  actual_num={actual_num}")
    print(f"[GEN] island_capacity_total={total_cap}  "
          f"(used only for per-island allocation, not capping total)")

    all_results = []
    for ep_id in ARGS.episode_ids:
        seed = ARGS.seed_offset + ep_id
        print(f"\n{'='*60}")
        print(f"[EP {ep_id}]  seed={seed}")
        print(f"{'='*60}")

        result = generate_spawn(
            episode_id=ep_id, seed=seed,
            grid=grid, esdf=esdf,
            min_x=min_x, min_y=min_y, scale=scale,
            map_min_x=map_min_x, map_max_x=map_max_x,
            map_min_y=map_min_y, map_max_y=map_max_y,
            valid_islands=valid_islands,
            px_to_island=px_to_island,
            actual_num=actual_num,
        )

        # 打印详细 log
        print(f"\n[LOG ep{ep_id}]")
        for line in result.get("log", []):
            print(f"  {line}")

        # 保存 JSON
        out_json = os.path.join(ARGS.output_dir, f"debug_ep{ep_id}.json")
        # 去掉不可序列化的 pixels
        save_result = {k: v for k, v in result.items() if k != "log"}
        with open(out_json, "w") as f:
            json.dump(save_result, f, indent=2)
        print(f"[SAVE] → {out_json}")

        # 保存 log
        out_log = os.path.join(ARGS.output_dir, f"debug_ep{ep_id}_log.txt")
        with open(out_log, "w") as f:
            f.write("\n".join(result.get("log", [])))
        print(f"[SAVE] → {out_log}")

        # 可视化
        if ARGS.vis and result.get("success"):
            out_vis = os.path.join(ARGS.output_dir, f"debug_ep{ep_id}_vis.png")
            visualize(result, grid, esdf, min_x, min_y, scale,
                      map_min_x, map_max_x, map_min_y, map_max_y, out_vis)

        all_results.append(result)

    # 汇总
    print(f"\n{'='*60}")
    print(f"[SUMMARY] {len(ARGS.episode_ids)} episodes")
    for r in all_results:
        status = "OK" if r.get("success") else "FAIL"
        placed = r.get("num_placed", 0)
        print(f"  ep{r.get('episode_id','?')}  {status}  "
              f"peds={placed}/{actual_num}  "
              f"robot_island={r.get('robot_island_idx')}")

if __name__ == "__main__":
    main()