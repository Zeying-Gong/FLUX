#!/usr/bin/env python3
"""
test_dataset_pipeline_and_create.py  (with visualization, NO simulation)
──────────────────────────────────────────────────────────────────────────
NavMesh‑based episode generator for scenes with broken semantic maps.
Generates episode JSONs + a lightweight visualization PNG for each episode.
Now WITHOUT running the actual character simulation (fast generation).

Usage
-----
    # Batch generate 100 episodes (headless)
    CUDA_VISIBLE_DEVICES=0 /isaac-sim/python.sh \
        test_dataset_pipeline_and_create.py \
        --usda /path/to/scene.usda \
        --output_dir /path/to/v1_tracking_episodes_hard \
        --num_episodes 100

    # With distance constraint 2-10 m
    CUDA_VISIBLE_DEVICES=0 /isaac-sim/python.sh \
        test_dataset_pipeline_and_create.py \
        --usda /path/to/scene.usda \
        --output_dir /path/to/v1_tracking_episodes_hard \
        --num_episodes 100 \
        --min_robot_char_dist 2.0 --max_robot_char_dist 10.0

    # Or use --dist_range for convenience
    CUDA_VISIBLE_DEVICES=0 /isaac-sim/python.sh \
        test_dataset_pipeline_and_create.py \
        --usda /path/to/scene.usda \
        --output_dir /path/to/v1_tracking_episodes_hard \
        --num_episodes 100 \
        --dist_range 2,10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback

# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--usda", required=True)
    p.add_argument("--output_dir", required=True,
                   help="Root output directory (scene_id subfolder created automatically).")
    p.add_argument("--num_episodes", type=int, default=100)
    p.add_argument("--num_waypoints", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_pair_attempts", type=int, default=1000,
                   help="Max random tries to find a robot+char pair with LOS and distance constraint.")
    p.add_argument("--skip_los", action="store_true",
                   help="Disable line‑of‑sight check.")
    p.add_argument("--vis", action="store_true",
                   help="Enable GUI for visual inspection (only stage opening, no simulation).")
    p.add_argument("--no_vis_img", action="store_true",
                   help="Do not save the lightweight episode visualization PNG.")
    # New distance constraints
    p.add_argument("--min_robot_char_dist", type=float, default=3.0,
                   help="Minimum Euclidean distance between robot and character (meters).")
    p.add_argument("--max_robot_char_dist", type=float, default=6.0,
                   help="Maximum Euclidean distance between robot and character (meters).")
    p.add_argument("--dist_range", type=str, default=None,
                   help="Convenience: '2,10' or '3,6' (overrides min/max).")
    # NavMesh / Isaac Sim
    p.add_argument("--collision_root", default="/World/scene_collision")
    p.add_argument("--volume_padding", type=float, default=1.2)
    p.add_argument("--fallback_size", type=float, default=100.0)
    p.add_argument("--warmup_frames", type=int, default=60)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--force_rebake", action="store_true")
    p.add_argument("--semantic_map_json", default=None,
                help="Optional semantic map JSON to create NavMesh exclusion volumes "
                    "(e.g., 2D_Semantic_Map_xxx_Complete.json).")
    # Disable simulation (always true now, kept for compatibility)
    p.add_argument("--no_sim", action="store_true", default=True,
                   help="Do not run actual character simulation (fast generation).")
    return p.parse_args()


ARGS = parse_args()

# Override distance range if provided
if ARGS.dist_range:
    parts = ARGS.dist_range.split(',')
    if len(parts) == 2:
        ARGS.min_robot_char_dist = float(parts[0])
        ARGS.max_robot_char_dist = float(parts[1])
        print(f"[CLI] Using distance range: {ARGS.min_robot_char_dist} - {ARGS.max_robot_char_dist} m")
    else:
        print(f"[WARN] Invalid --dist_range '{ARGS.dist_range}', using defaults")

# ═══════════════════════════════════════════════════════════════════════════
# SimulationApp (headless or GUI)
# ═══════════════════════════════════════════════════════════════════════════
from isaacsim import SimulationApp

_exp = os.environ.get("EXP_PATH")
if _exp is None:
    print("[FATAL] EXP_PATH env var not set.")
    sys.exit(1)

CUSTOM_APP_PATH = os.path.join(
    _exp, "isaacsim.exp.action_and_event_data_generation.base.kit")

simulation_app = SimulationApp(
    launch_config={
        "renderer": "RayTracedLighting",
        "headless": not ARGS.vis,
        "enable_cameras": ARGS.vis,
        "crash_reporter/enabled": False,
        "crash_reporter/skip_old_dump_upload": True,
    },
    experience=CUSTOM_APP_PATH,
)

# ═══════════════════════════════════════════════════════════════════════════
# Post‑launch imports
# ═══════════════════════════════════════════════════════════════════════════
import numpy as np
import carb
import omni.usd
import omni.timeline
from pxr import Sdf, Usd, Gf

# Make sage_utils importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage_utils.navmesh_utils import (
    cache_paths, bake_navmesh,
    safe_query_random_point,
    add_exclude_volumes_from_semantic_map,
)
from sage_utils.people_utils import (
    load_character_assets, spawn_character,
    load_default_skeleton_and_animations,
    bind_animation_graph_to_characters,
    attach_behavior_scripts_to_characters,
    write_commands_to_scriptdata,
    frame_viewport_on,
    spawn_on_navmesh,
)
from isaacsim.replicator.agent.core.settings import PrimPaths
from isaacsim.replicator.agent.core.stage_util import CharacterUtil


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _update(n=1):
    for _ in range(n):
        simulation_app.update()

def open_stage(usda_path: str) -> bool:
    if not os.path.exists(usda_path):
        print("[FATAL] USDA not found:", usda_path)
        return False
    print("[STAGE] Opening:", usda_path)
    omni.usd.get_context().open_stage(usda_path)
    _update(30)
    waited = 0
    while omni.usd.get_context().get_stage_loading_status()[1] > 0:
        simulation_app.update()
        waited += 1
        if waited > 3000:
            print("[WARN] Stage still loading after 30s.")
            break
    return omni.usd.get_context().get_stage() is not None

def _hide_navmesh_volumes():
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    hidden = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() == "NavMeshVolume":
            imageable = UsdGeom.Imageable(prim)
            imageable.MakeInvisible()
            hidden += 1
    print(f"[NavMesh] Hidden {hidden} NavMeshVolume prim(s).")

def check_line_of_sight(start: tuple, end: tuple) -> bool:
    from omni.physx import get_physx_scene_query_interface
    scene_query = get_physx_scene_query_interface()
    origin = carb.Float3(*start)
    direction = carb.Float3(end[0]-start[0], end[1]-start[1], end[2]-start[2])
    dist = math.sqrt(direction.x**2 + direction.y**2 + direction.z**2)
    if dist < 1e-4:
        return True
    direction = carb.Float3(direction.x/dist, direction.y/dist, direction.z/dist)
    hit_info = scene_query.raycast_closest(origin, direction, dist)
    return not hit_info['hit']


# ═══════════════════════════════════════════════════════════════════════════
# Lightweight visualization (robot + character paths)
# ═══════════════════════════════════════════════════════════════════════════
def save_episode_vis(episode_dict: dict, out_path: str):
    if ARGS.no_vis_img:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[VIS] matplotlib not available, skip episode visualization.")
        return

    robot = episode_dict["robot"]
    robot_start = robot["start_pos"]
    robot_ori = robot["start_orientation"]
    char_data = episode_dict["characters"]
    spawn = char_data["spawn_positions"]["Character"]["pos"]
    commands = char_data["commands"]["Character"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    # robot start
    ax.plot(robot_start[0], robot_start[1], "s", color="cyan", markersize=10, label="Robot start")
    # robot orientation arrow
    arrow_len = 1.0
    ax.arrow(robot_start[0], robot_start[1],
             arrow_len * math.cos(robot_ori),
             arrow_len * math.sin(robot_ori),
             head_width=0.3, head_length=0.3, fc="cyan", ec="cyan")

    # character spawn
    ax.plot(spawn[0], spawn[1], "o", color="red", markersize=8, label="Character spawn")

    # plot all goto paths
    prev_pt = spawn[:2]
    for idx, cmd in enumerate(commands):
        if cmd["cmd"] != "GoTo":
            continue
        path = cmd["path"]
        if not path:
            continue
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, "-", color="blue", linewidth=1.2, alpha=0.7)

        # waypoint marker
        wp = path[-1]
        is_last = (idx == len([c for c in commands if c["cmd"]=="GoTo"]) - 1)
        marker = "*" if is_last else "D"
        size = 12 if is_last else 8
        ax.plot(wp[0], wp[1], marker, color="blue", markersize=size, markeredgecolor="k")
        prev_pt = wp[:2]

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Episode {episode_dict['episode_id']} – Robot & Character Paths")
    ax.legend()
    plt.tight_layout()
    vis_path = out_path.replace(".json", "_vis.png")
    plt.savefig(vis_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIS] → {vis_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Generate ONE episode (NO simulation, only data)
# ═══════════════════════════════════════════════════════════════════════════
def generate_one_episode(
    episode_id: int,
    nm,
    char_pool: list[str],
) -> dict:
    rng = random.Random(ARGS.seed + episode_id)
    print(f"\n[Gen] Episode {episode_id} (distance constraint: {ARGS.min_robot_char_dist}–{ARGS.max_robot_char_dist} m)")

    # Sample robot + character with LOS and distance constraint
    robot_start = None
    char_spawn = None
    for attempt in range(1, ARGS.max_pair_attempts + 1):
        # Sample candidate character position
        pt_char = safe_query_random_point(nm)
        if not pt_char:
            continue
        cand_char = (pt_char.x, pt_char.y, pt_char.z)

        # Sample candidate robot position
        pt_robot = safe_query_random_point(nm)
        if not pt_robot:
            continue
        cand_robot = (pt_robot.x, pt_robot.y, pt_robot.z)

        # --- NEW: Euclidean distance constraint ---
        dist = math.hypot(cand_robot[0] - cand_char[0],
                          cand_robot[1] - cand_char[1])
        if not (ARGS.min_robot_char_dist <= dist <= ARGS.max_robot_char_dist):
            continue

        # Check NavMesh connectivity
        path = nm.query_shortest_path(
            carb.Float3(*cand_robot), carb.Float3(*cand_char), agent_radius=0.3)
        if not path:
            continue

        # Optional line-of-sight
        if not ARGS.skip_los and not check_line_of_sight(cand_robot, cand_char):
            continue

        robot_start = cand_robot
        char_spawn = cand_char
        print(f"[Gen] Pair found after {attempt} attempts (distance={dist:.2f}m).")
        break
    else:
        raise RuntimeError(
            f"Could not find valid robot+char pair after {ARGS.max_pair_attempts} attempts. "
            "Try --skip_los, adjust distance range, or increase --max_pair_attempts."
        )

    # Robot orientation: face the character
    dx = char_spawn[0] - robot_start[0]
    dy = char_spawn[1] - robot_start[1]
    robot_orientation = math.atan2(dy, dx)

    # Character waypoints (same as original)
    waypoints = [char_spawn]
    goto_commands = []
    for wpt_idx in range(ARGS.num_waypoints):
        for _ in range(50):
            pt = safe_query_random_point(nm)
            if pt:
                next_wp = (pt.x, pt.y, pt.z)
                path = nm.query_shortest_path(
                    carb.Float3(*waypoints[-1]), carb.Float3(*next_wp), agent_radius=0.3)
                if path and len(path.get_points()) >= 2:
                    path_points = [(p[0], p[1], p[2]) for p in path.get_points()]
                    goto_commands.append({
                        "cmd": "GoTo",
                        "params": [f"{next_wp[0]:.4f}", f"{next_wp[1]:.4f}", f"{next_wp[2]:.4f}", "_"],
                        "path": path_points,
                    })
                    waypoints.append(next_wp)
                    break
        else:
            goto_commands.append({
                "cmd": "GoTo",
                "params": [f"{waypoints[-1][0]:.4f}", f"{waypoints[-1][1]:.4f}", f"{waypoints[-1][2]:.4f}", "_"],
                "path": [waypoints[-1]],
            })
            print("[WARN] Waypoint", wpt_idx+1, "fallback to last point.")

    # Build episode dict (no simulation performed)
    episode_dict = {
        "episode_id": episode_id,
        "seed": ARGS.seed + episode_id,
        "robot": {
            "start_pos": list(robot_start),
            "goal_pos": list(char_spawn),
            "start_orientation": robot_orientation,
        },
        "characters": {
            "num_characters": 1,
            "spawn_positions": {
                "Character": {
                    "pos": list(char_spawn),
                    "rot": 0.0,
                }
            },
            "commands": {
                "Character": goto_commands,
            },
        },
        "tracking": {
            "target_character": "Character",
        },
    }

    # No simulation: just return the episode data
    print(f"[Episode {episode_id}] Data generated (no simulation).")
    return episode_dict


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> int:
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)

    if not open_stage(ARGS.usda):
        simulation_app.close()
        return 1

    # ── Add exclusion volumes from semantic map (if provided) ──
    if ARGS.semantic_map_json and os.path.exists(ARGS.semantic_map_json):
        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            try:
                count = add_exclude_volumes_from_semantic_map(
                    simulation_app, stage, ARGS.semantic_map_json,
                    flip_x=True, flip_y=True, negate_xy=True
                )
                print(f"[SemanticMap] Added {count} exclusion volumes.")
            except Exception as e:
                print(f"[WARN] Failed to add exclusion volumes: {e}")
                
    # NavMesh
    navvols_usda = cache_paths(ARGS.usda, ARGS.cache_dir)
    print("[Cache] NavVols USDA:", navvols_usda)
    # 删除已存在的 navvols 文件，避免 SdfLayer.CreateNew 冲突
    if os.path.exists(navvols_usda):
        os.remove(navvols_usda)
        print("[Cache] Removed existing navvols file to avoid SdfLayer conflict")

    nav_ok, inav = bake_navmesh(
        app=simulation_app,
        navvols_usda=navvols_usda,
        force_rebake=ARGS.force_rebake,
        collision_root=ARGS.collision_root,
        volume_padding=ARGS.volume_padding,
        fallback_size=ARGS.fallback_size,
        warmup_frames=ARGS.warmup_frames,
        semantic_map_json=None,
    )
    if not nav_ok:
        print("[FATAL] NavMesh bake failed.")
        simulation_app.close()
        return 1

    nm = inav.get_navmesh()
    _hide_navmesh_volumes()
    _update(10)

    char_pool = load_character_assets()
    print("[PEOPLE]", len(char_pool), "asset(s) available.")

    scene_id = os.path.splitext(os.path.basename(ARGS.usda))[0]
    out_dir = os.path.join(ARGS.output_dir, scene_id)
    os.makedirs(out_dir, exist_ok=True)
    print("[Output] Saving episodes to:", out_dir)

    success = 0
    for ep_id in range(ARGS.num_episodes):
        try:
            # Generate episode data (no simulation)
            ep_dict = generate_one_episode(ep_id, nm, char_pool)
            out_path = os.path.join(out_dir, f"episode_{ep_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"episode": ep_dict}, f, indent=2)
            print("[SAVE]", out_path)

            # Save visualization
            save_episode_vis(ep_dict, out_path)

            success += 1
        except Exception as e:
            print("[ERROR] Episode", ep_id, "failed:")
            traceback.print_exc()

    print(f"\n[DONE] Generated {success}/{ARGS.num_episodes} episodes.")
    simulation_app.close()
    return 0 if success == ARGS.num_episodes else 1


if __name__ == "__main__":
    sys.exit(main())