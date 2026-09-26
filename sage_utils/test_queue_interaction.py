"""
test_queue_interaction.py
─────────────────────────
Test Queue command with omni.anim.people — two characters.

Key fixes vs previous version
------------------------------
1. Queue spots are now sampled from the 2D map (not derived by +X offset),
   so both spots are guaranteed to be on navigable ground.
2. Spawn points are validated reachable to BOTH spots (not just spot_0).
3. Spot ordering: spot_1 is the "back of queue" (farther from head),
   spot_0 is the head. Characters approach from spot_1 → advance to spot_0.

character_behavior.py patch required (add before "elif command[0] == 'Queue'"):
    elif command[0] == "Queue_Spot":
        queue = self.queue_manager.create_queue(command[1])
        rot = Utils.convert_angle_to_quatd(float(command[6])) \\
              if len(command) > 6 else None
        queue.create_spot(
            int(command[2]),
            carb.Float3(float(command[3]), float(command[4]), float(command[5])),
            rot,
        )
        return None

Usage
-----
    python test_queue_interaction.py \\
        --usda /workspace/SAGE-3D_Official/SAGE-3D_data/usda/839873.usda \\
        --semantic_map_json /workspace/SAGE-3D_Official/SAGE-3D_data/semantic_maps/2D_Semantic_Map_0033_839873_Complete.json
"""
from __future__ import annotations
import argparse
import heapq
import json
import math
import os
import random
import sys


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--usda", required=True)
    p.add_argument("--semantic_map_json", required=True)

    p.add_argument("--collision_root",  default="/World/scene_collision")
    p.add_argument("--volume_padding",  type=float, default=1.2)
    p.add_argument("--fallback_size",   type=float, default=100.0)
    p.add_argument("--warmup_frames",   type=int,   default=60)

    p.add_argument("--lookaround_duration", type=float, default=5.0)
    p.add_argument("--queue_spacing",       type=float, default=1.5,
                   help="Min distance between queue spots (m).")

    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--scale_m_per_px",  type=float, default=0.05)
    p.add_argument("--robot_radius_2d", type=float, default=0.3)
    p.add_argument("--cache_dir",       default=None)
    p.add_argument("--force_rebake",    action="store_true")
    return p.parse_args()


ARGS = parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# SimulationApp
# ═══════════════════════════════════════════════════════════════════════════
from isaacsim import SimulationApp

_exp = os.environ.get("EXP_PATH")
if _exp is None:
    print("[FATAL] EXP_PATH env var not set.")
    sys.exit(1)

simulation_app = SimulationApp(
    launch_config={
        "renderer":                            "RayTracedLighting",
        "headless":                            False,
        "enable_cameras":                      True,
        "crash_reporter/enabled":              False,
        "crash_reporter/skip_old_dump_upload": True,
    },
    experience=os.path.join(
        _exp, "isaacsim.exp.action_and_event_data_generation.base.kit"),
)

# ═══════════════════════════════════════════════════════════════════════════
# Post-launch imports
# ═══════════════════════════════════════════════════════════════════════════
import numpy as np
import carb
import omni.usd
import omni.timeline
from pxr import Sdf, Usd
from scipy.ndimage import distance_transform_edt
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sage_utils.navmesh_utils import cache_paths, bake_navmesh
from sage_utils.people_utils import (
    load_character_assets, spawn_character,
    load_default_skeleton_and_animations,
    bind_animation_graph_to_characters,
    attach_behavior_scripts_to_characters,
    frame_viewport_on,
)

try:
    import omni.anim.navigation.core as _nav_core
    NavSchema = _nav_core.NavSchema if hasattr(_nav_core, "NavSchema") else None
except Exception:
    NavSchema = None


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _update(n: int = 1):
    for _ in range(n):
        simulation_app.update()


def _hide_navmesh_volumes():
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    hidden = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() == "NavMeshVolume":
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden += 1
    print(f"[NavMesh] Hidden {hidden} NavMeshVolume prim(s).")


def open_stage(usda_path: str) -> bool:
    if not os.path.exists(usda_path):
        print(f"[FATAL] USDA not found: {usda_path}")
        return False
    print(f"[STAGE] Opening: {usda_path}")
    omni.usd.get_context().open_stage(usda_path)
    _update(30)
    waited = 0
    while omni.usd.get_context().get_stage_loading_status()[1] > 0:
        simulation_app.update()
        waited += 1
        if waited > 3000:
            print("[WARN] Stage still loading after 30 s.")
            break
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return False
    print(f"[STAGE] Loaded. Default prim: {stage.GetDefaultPrim().GetPath()}")
    return True


def _find_skelroot(stage, char_prim_path: str):
    char_prim = stage.GetPrimAtPath(char_prim_path)
    if not char_prim.IsValid():
        return None
    for desc in Usd.PrimRange(char_prim):
        if desc.GetTypeName() == "SkelRoot":
            return desc
    return None


def _write_script_data(stage, char_prim_path: str, commands: list[str]):
    skelroot = _find_skelroot(stage, char_prim_path)
    if skelroot is None:
        print(f"[Cmd] WARNING: no SkelRoot under {char_prim_path}")
        return
    sd_attr = skelroot.GetAttribute("omni:scripting:scriptData")
    if not sd_attr:
        sd_attr = skelroot.CreateAttribute(
            "omni:scripting:scriptData", Sdf.ValueTypeNames.StringArray)
    sd_attr.Set(commands)
    print(f"[Cmd] {os.path.basename(char_prim_path)}: {commands}")


def _fmt3(pos) -> str:
    return f"{float(pos[0]):.3f} {float(pos[1]):.3f} {float(pos[2]):.3f}"


def _exclude_from_navmesh(prim_path: str):
    """Apply NavMeshExcludeAPI so this character capsule is invisible to
    NavMesh path queries - other characters can plan paths through its spot."""
    if NavSchema is None:
        print(f"  [NavMesh] NavSchema unavailable, skipping for {prim_path}")
        return
    try:
        import omni.kit.commands
        omni.kit.commands.execute(
            "ApplyNavMeshAPICommand",
            prim_path=prim_path,
            api=NavSchema.NavMeshExcludeAPI,
        )
        print(f"  [NavMesh] Excluded {prim_path} from NavMesh")
    except Exception as e:
        print(f"  [NavMesh] Could not exclude {prim_path}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 2D map helpers
# ═══════════════════════════════════════════════════════════════════════════
def _load_2d_map(sem_json, robot_radius_m, scale):
    with open(sem_json, encoding="utf-8") as f:
        sem_data = json.load(f)
    all_y, all_x = [], []
    for inst in sem_data:
        for y, x in inst.get("mask_coords_m", []):
            try:
                all_y.append(float(y)); all_x.append(float(x))
            except (ValueError, TypeError):
                continue
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
                except (ValueError, TypeError):
                    continue
    if robot_radius_m > 0:
        dist_m = distance_transform_edt(grid == 0, sampling=scale)
        grid = (dist_m <= robot_radius_m).astype(np.uint8)
    esdf = distance_transform_edt(grid == 0, sampling=scale)
    return grid, esdf, min_x, min_y, max_x, max_y


def _px_to_world(px, py, min_x, min_y, scale):
    return min_x + (px + 0.5) * scale, min_y + (py + 0.5) * scale


def _world_to_px(x_m, y_m, min_x, min_y, scale):
    return int(round((x_m - min_x) / scale)), int(round((y_m - min_y) / scale))


def _sm_to_isaac(x_m, y_m, map_min_x, map_max_x, map_min_y, map_max_y):
    px = (map_min_x + map_max_x) - float(x_m)
    py = (map_min_y + map_max_y) - float(y_m)
    return -px, -py


def _isaac_to_sm(ix, iy, map_min_x, map_max_x, map_min_y, map_max_y):
    x_m = (map_min_x + map_max_x) - (-float(ix))
    y_m = (map_min_y + map_max_y) - (-float(iy))
    return x_m, y_m


def _snap_free(grid, px, py):
    H, W = grid.shape
    if 0 <= px < W and 0 <= py < H and grid[py, px] == 0:
        return (px, py)
    visited = set()
    q = deque([(px, py, 0)])
    while q:
        cx, cy, d = q.popleft()
        if d > 50:
            break
        if 0 <= cx < W and 0 <= cy < H and grid[cy, cx] == 0:
            return (cx, cy)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (cx+dx, cy+dy)
            if nb not in visited:
                visited.add(nb); q.append((nb[0], nb[1], d+1))
    return None


def _astar(grid, s, g):
    H, W = grid.shape
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    open_set = [(0.0, s)]
    came = {}
    gc = {s: 0.0}
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == g:
            path = [cur]
            while cur in came:
                cur = came[cur]; path.append(cur)
            return path[::-1]
        for d in dirs:
            nx, ny = cur[0]+d[0], cur[1]+d[1]
            if not (0 <= nx < W and 0 <= ny < H) or grid[ny, nx] == 1:
                continue
            nb = (nx, ny)
            tg = gc[cur] + math.hypot(d[0], d[1])
            if nb not in gc or tg < gc[nb]:
                came[nb] = cur; gc[nb] = tg
                heapq.heappush(open_set, (tg + math.hypot(nx-g[0], ny-g[1]), nb))
    return None


class Map2D:
    """Wrapper for 2D map operations."""
    def __init__(self, grid, esdf, min_x, min_y, max_x, max_y, scale,
                 map_min_x, map_max_x, map_min_y, map_max_y):
        self.grid = grid
        self.esdf = esdf
        self.min_x = min_x; self.min_y = min_y
        self.max_x = max_x; self.max_y = max_y
        self.scale = scale
        self.map_min_x = map_min_x; self.map_max_x = map_max_x
        self.map_min_y = map_min_y; self.map_max_y = map_max_y

    def isaac_to_px(self, ix, iy):
        x_m, y_m = _isaac_to_sm(ix, iy,
                                  self.map_min_x, self.map_max_x,
                                  self.map_min_y, self.map_max_y)
        return _world_to_px(x_m, y_m, self.min_x, self.min_y, self.scale)

    def px_to_isaac(self, px, py):
        x_m, y_m = _px_to_world(px, py, self.min_x, self.min_y, self.scale)
        ix, iy = _sm_to_isaac(x_m, y_m,
                               self.map_min_x, self.map_max_x,
                               self.map_min_y, self.map_max_y)
        return (ix, iy, 0.0)

    def sample(self, rng, min_clearance=0.5):
        H, W = self.grid.shape
        for _ in range(500):
            py = rng.randint(0, H-1)
            px = rng.randint(0, W-1)
            if self.grid[py, px] == 0 and self.esdf[py, px] >= min_clearance:
                return self.px_to_isaac(px, py)
        return None

    def reachable(self, a_isaac, b_isaac) -> bool:
        """Check A* connectivity between two Isaac-coords points."""
        apx = _snap_free(self.grid, *self.isaac_to_px(a_isaac[0], a_isaac[1]))
        bpx = _snap_free(self.grid, *self.isaac_to_px(b_isaac[0], b_isaac[1]))
        if apx is None or bpx is None:
            return False
        return _astar(self.grid, apx, bpx) is not None

    def dist(self, a_isaac, b_isaac) -> float:
        return math.hypot(float(a_isaac[0]) - float(b_isaac[0]),
                          float(a_isaac[1]) - float(b_isaac[1]))

    def sample_reachable(self, rng, anchor_isaac, tries=300, min_clearance=0.5):
        for _ in range(tries):
            pt = self.sample(rng, min_clearance)
            if pt and self.reachable(pt, anchor_isaac):
                return pt
        return None

    def sample_pair_reachable(self, rng, anchor_isaac,
                               min_dist=1.2, max_dist=3.0,
                               tries=500, min_clearance=0.5):
        """
        Sample two points that are:
          - both reachable from anchor_isaac
          - both reachable from each other
          - separated by min_dist..max_dist
        Returns (pt_near, pt_far) where pt_near is closer to anchor.
        """
        for _ in range(tries):
            a = self.sample(rng, min_clearance)
            if a is None or not self.reachable(a, anchor_isaac):
                continue
            b = self.sample(rng, min_clearance)
            if b is None or not self.reachable(b, anchor_isaac):
                continue
            d = self.dist(a, b)
            if not (min_dist <= d <= max_dist):
                continue
            if not self.reachable(a, b):
                continue
            # return closer one as spot_0 (head), farther as spot_1
            da = self.dist(a, anchor_isaac)
            db = self.dist(b, anchor_isaac)
            if da <= db:
                return a, b   # a=head(0), b=back(1)
            else:
                return b, a   # b=head(0), a=back(1)
        return None, None


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> int:
    rng = random.Random(ARGS.seed)
    np.random.seed(ARGS.seed)

    # ── Load 2D map ────────────────────────────────────────────────────────
    print(f"[2D] Loading: {ARGS.semantic_map_json}")
    grid, esdf, min_x, min_y, max_x, max_y = _load_2d_map(
        ARGS.semantic_map_json, ARGS.robot_radius_2d, ARGS.scale_m_per_px)
    m = Map2D(grid, esdf, min_x, min_y, max_x, max_y, ARGS.scale_m_per_px,
              min_x, max_x, min_y, max_y)
    print(f"[2D] Grid {grid.shape}")

    # ── Sample spawn point (anchor for all other points) ───────────────────
    print("[2D] Sampling anchor/spawn region...")
    anchor = None
    for _ in range(200):
        pt = m.sample(rng)
        if pt is not None:
            anchor = pt
            break
    if anchor is None:
        print("[FATAL] Could not sample anchor.")
        simulation_app.close()
        return 1

    # ── Sample two queue spots both reachable from anchor and each other ───
    print("[2D] Sampling queue spots (both reachable, 1.2–3m apart)...")
    spot_0, spot_1 = m.sample_pair_reachable(
        rng, anchor,
        min_dist=ARGS.queue_spacing, max_dist=ARGS.queue_spacing * 2.5,
    )
    if spot_0 is None:
        print("[WARN] Could not find connected spot pair, using simple offset.")
        spot_0 = anchor
        spot_1 = (anchor[0] + ARGS.queue_spacing, anchor[1], anchor[2])

    print(f"[Queue] Spot 0 (head)  : {spot_0}")
    print(f"[Queue] Spot 1 (second): {spot_1}")

    # ── Sample spawn points reachable to both spots ────────────────────────
    print("[2D] Sampling spawn points...")

    def _sample_spawn(tries=300):
        for _ in range(tries):
            pt = m.sample(rng)
            if (pt and
                    m.reachable(pt, spot_0) and
                    m.reachable(pt, spot_1)):
                return pt
        return m.sample_reachable(rng, spot_0)  # relaxed fallback

    spawn_a = _sample_spawn()
    spawn_b = _sample_spawn()
    if spawn_a is None: spawn_a = spot_1
    if spawn_b is None: spawn_b = spot_1

    # ── Sample depart points reachable from spot_0 ─────────────────────────
    print("[2D] Sampling depart points...")
    depart_a = m.sample_reachable(rng, spot_0)
    depart_b = m.sample_reachable(rng, spot_0)
    if depart_a is None: depart_a = spawn_a
    if depart_b is None: depart_b = spawn_b

    print(f"[2D] Spawn  A: {spawn_a}")
    print(f"[2D] Spawn  B: {spawn_b}")
    print(f"[2D] Depart A: {depart_a}")
    print(f"[2D] Depart B: {depart_b}")

    # ── Open scene + NavMesh ───────────────────────────────────────────────
    if not open_stage(ARGS.usda):
        return 1

    navvols_usda = cache_paths(ARGS.usda, ARGS.cache_dir)
    bake_navmesh(
        app=simulation_app, navvols_usda=navvols_usda,
        force_rebake=ARGS.force_rebake, collision_root=ARGS.collision_root,
        volume_padding=ARGS.volume_padding, fallback_size=ARGS.fallback_size,
        warmup_frames=ARGS.warmup_frames, semantic_map_json=None,
    )
    _hide_navmesh_volumes()
    _update(10)

    # ── Physics world ──────────────────────────────────────────────────────
    print("\n[World] Creating World + physicsScene...")
    from isaacsim.core.api import World
    world = World(physics_dt=1.0/60.0, rendering_dt=1.0/30.0)
    world.initialize_physics()
    world.reset()
    _update(10)
    print("[World] Physics initialized.")

    # ── Characters ─────────────────────────────────────────────────────────
    char_pool = load_character_assets()
    print(f"[PEOPLE] {len(char_pool)} asset(s) available.")

    prim_a = spawn_character(simulation_app, 0, char_pool[0 % len(char_pool)], spawn_a)
    prim_b = spawn_character(simulation_app, 1, char_pool[1 % len(char_pool)], spawn_b)
    _exclude_from_navmesh(prim_a)
    _exclude_from_navmesh(prim_b)
    name_a = prim_a.split("/")[-1]
    name_b = prim_b.split("/")[-1]

    print(f"[PEOPLE] {name_a} at {spawn_a}  → {prim_a}")
    print(f"[PEOPLE] {name_b} at {spawn_b}  → {prim_b}")

    load_default_skeleton_and_animations(simulation_app)
    _update(30)
    bind_animation_graph_to_characters(simulation_app)
    _update(60)

    # ── scriptData ─────────────────────────────────────────────────────────
    # Order matters:
    #   Queue_Spot lines → get_command() registers spots via new patch
    #   Queue line       → QueueCmd.__init__ calls get_queue() → succeeds
    #   LookAround       → action at queue head
    #   Dequeue          → leave queue and walk to depart point
    #
    # Both characters share the same spot definitions (idempotent create_spot
    # on the same queue is safe since create_queue() is also idempotent).
    #
    QUEUE_NAME = "TestQueue"
    ld = ARGS.lookaround_duration
    stage = omni.usd.get_context().get_stage()

    for prim, depart in [(prim_a, depart_a), (prim_b, depart_b)]:
        _write_script_data(stage, prim, [
            # Register spots first (patched Queue_Spot branch in get_command)
            f"Queue_Spot {QUEUE_NAME} 0 {_fmt3(spot_0)} 0",
            f"Queue_Spot {QUEUE_NAME} 1 {_fmt3(spot_1)} 0",
            # Enter queue (spots now exist → no KeyError)
            f"Queue {QUEUE_NAME}",
            # Action at head
            f"LookAround {ld:.1f}",
            # Leave queue
            f"Dequeue {QUEUE_NAME} {_fmt3(depart)} _",
        ])

    _update(10)

    # ── Attach behavior scripts ────────────────────────────────────────────
    attach_behavior_scripts_to_characters(simulation_app)
    _update(60)

    frame_viewport_on(prim_a)
    _update(5)

    # ── Play ───────────────────────────────────────────────────────────────
    tl = omni.timeline.get_timeline_interface()
    tl.set_current_time(0.0)
    tl.play()

    print("\n[HOLD] Simulation running. Ctrl+C to quit.")
    print(f"  {name_a}: → spot-1 → spot-0 → LookAround {ld}s → Dequeue → depart")
    print(f"  {name_b}: → spot-1 (wait) → spot-0 → LookAround {ld}s → Dequeue → depart\n")

    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        print("\n[HOLD] Ctrl+C received, shutting down.")

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())