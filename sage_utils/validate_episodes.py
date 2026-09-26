#!/usr/bin/env python3
"""
validate_episodes.py  — 行人执行验证（稳定版）
────────────────────────────────────────────
Usage:
    export DISPLAY=:0
    CUDA_VISIBLE_DEVICES=0 /isaac-sim/python.sh validate_episodes.py \
        --usda_dir /workspace/.../usda \
        --episode_dir /workspace/.../v1_episodes \
        --scene_ids 839873
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# 1. 参数解析
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--usda_dir", required=True)
    p.add_argument("--episode_dir", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--scene_ids", nargs="*", default=None)
    p.add_argument("--worker_id", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--ped_speed", type=float, default=1.2)
    p.add_argument("--arrive_tolerance", type=float, default=0.5)
    return p.parse_args()

ARGS = parse_args()

# 确保 DISPLAY
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"

# ═══════════════════════════════════════════════════════════════════
# 2. 启动 SimulationApp
# ═══════════════════════════════════════════════════════════════════
from isaacsim import SimulationApp

sim_app = SimulationApp(
    launch_config={
        "renderer": "RayTracedLighting",
        "headless": True,
        "enable_cameras": False,
    },
    experience="/isaac-sim/apps/isaacsim.exp.action_and_event_data_generation.base.kit",
)

import omni.usd
from pxr import Usd, UsdGeom

# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════
def update(n=1):
    for _ in range(n):
        sim_app.update()

def open_stage(usda_path):
    if not os.path.exists(usda_path):
        print(f"[ERROR] USDA not found: {usda_path}")
        return False
    omni.usd.get_context().open_stage(usda_path)
    update(30)
    waited = 0
    while omni.usd.get_context().get_stage_loading_status()[1] > 0:
        sim_app.update()
        waited += 1
        if waited > 3000:
            print("[WARN] Stage still loading after 30s.")
            break
    return omni.usd.get_context().get_stage() is not None

def get_world_pos(prim_path):
    """获取 prim 在世界空间中的位置 (x,y,z)"""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return None
    xform_api = UsdGeom.XformCommonAPI(prim)
    # 直接获取世界变换矩阵的平移部分（XformCommonAPI 不直接给世界矩阵，但 ComputeLocalToWorldTransform 可用）
    xformable = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default()
    mat = xformable.ComputeLocalToWorldTransform(time)
    trans = mat.ExtractTranslation()
    return (trans[0], trans[1], trans[2])

def set_world_pos(prim_path, pos):
    """设置 prim 的世界位置"""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        print(f"[ERROR] set_world_pos: prim {prim_path} not found")
        return False
    xform_api = UsdGeom.XformCommonAPI(prim)
    xform_api.SetTranslate(pos)
    return True

def create_character(name):
    """创建一个简单的 Xform prim 作为行人代理"""
    stage = omni.usd.get_context().get_stage()
    path = f"/World/Characters/{name}"
    prim = stage.DefinePrim(path, "Xform")
    # 确保它有一个 XformCommonAPI 可用
    return str(prim.GetPath())

def move_agent_along_path(prim_path, path_points, speed, tolerance, max_steps, agent_name=""):
    """移动 agent 依次经过路径点，返回 (success, final_distance, steps_used)"""
    if not path_points:
        return True, 0.0, 0

    total_steps = 0
    for i, target in enumerate(path_points):
        reached = False
        for step in range(max_steps):
            total_steps += 1
            cur = get_world_pos(prim_path)
            if cur is None:
                return False, 1e6, total_steps
            dx, dy, dz = target[0]-cur[0], target[1]-cur[1], target[2]-cur[2]
            dist = math.hypot(dx, dy, dz)
            if dist <= tolerance:
                reached = True
                break
            step_size = min(speed * 0.01, dist)
            d = (dx/dist, dy/dist, dz/dist)
            new_pos = (cur[0]+d[0]*step_size, cur[1]+d[1]*step_size, cur[2]+d[2]*step_size)
            set_world_pos(prim_path, new_pos)
            update(1)
        if not reached:
            cur = get_world_pos(prim_path)
            if cur is None:
                return False, 1e6, total_steps
            final_dist = math.hypot(target[0]-cur[0], target[1]-cur[1], target[2]-cur[2])
            print(f"  [MOVE] {agent_name} failed to reach waypoint {i}, dist left={final_dist:.2f}m")
            return False, final_dist, total_steps
    cur = get_world_pos(prim_path)
    if cur is None:
        return False, 1e6, total_steps
    final = math.hypot(target[0]-cur[0], target[1]-cur[1], target[2]-cur[2])
    return final <= tolerance, final, total_steps

# ═══════════════════════════════════════════════════════════════════
# 场景验证（仅行人）
# ═══════════════════════════════════════════════════════════════════
def validate_scene(usda_path, scene_id):
    out_dir = ARGS.output_dir or os.path.join(ARGS.episode_dir, "validation")
    log_dir = os.path.join(out_dir, "logs", scene_id)
    os.makedirs(log_dir, exist_ok=True)

    if not open_stage(usda_path):
        return {"scene_id": scene_id, "error": "stage open failed"}

    ep_dir = os.path.join(ARGS.episode_dir, scene_id)
    if not os.path.isdir(ep_dir):
        return {"scene_id": scene_id, "error": f"no episode dir {ep_dir}"}

    ep_files = sorted(
        (f for f in os.listdir(ep_dir) if f.startswith("episode_") and f.endswith(".json")),
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    results = []
    for ep_file in ep_files:
        ep_path = os.path.join(ep_dir, ep_file)
        try:
            with open(ep_path, "r") as f:
                data = json.load(f)
            ep = data["episode"]
        except Exception as e:
            msg = f"Failed to load {ep_file}: {e}"
            print(f"  [ERROR] {msg}")
            results.append({"episode": ep_file, "result": "error", "reason": msg})
            continue

        ep_id = ep["episode_id"]
        log_path = os.path.join(log_dir, f"ep_{ep_id}.log")
        lines = [f"Scene: {scene_id} Episode: {ep_id}"]

        spawns = ep["characters"]["spawn_positions"]
        commands = ep["characters"]["commands"]

        # 按命名规则排序
        char_names = []
        for i in range(len(spawns)):
            if i == 0:
                name = "Character"
            elif i < 10:
                name = f"Character_0{i}"
            else:
                name = f"Character_{i}"
            if name in spawns:
                char_names.append(name)
        if not char_names:
            char_names = list(spawns.keys())

        ped_fail = False
        for ci, cn in enumerate(char_names):
            if cn not in spawns:
                msg = f"Ped {ci} ({cn}) missing spawn"
                print(f"  [FAIL] ep{ep_id}: {msg}")
                lines.append(f"FAIL: {msg}")
                ped_fail = True
                break

            sp = spawns[cn]["pos"]
            prim_path = create_character(cn)
            set_world_pos(prim_path, sp)
            update(5)

            cmds = commands.get(cn, [])
            for cmd_idx, cmd in enumerate(cmds):
                if cmd.get("cmd") == "GoTo":
                    params = cmd.get("params", [])
                    if len(params) < 2:
                        msg = f"Ped {cn} GoTo missing params at cmd {cmd_idx}"
                        print(f"  [FAIL] ep{ep_id}: {msg}")
                        lines.append(f"FAIL: {msg}")
                        ped_fail = True
                        break
                    target = (float(params[0]), float(params[1]), float(params[2]) if len(params)>2 else 0.0)
                    if "path" in cmd and cmd["path"]:
                        path = [(p[0], p[1], p[2]) for p in cmd["path"]]
                    else:
                        path = [target]
                    ok, d, steps = move_agent_along_path(
                        prim_path, path, ARGS.ped_speed, ARGS.arrive_tolerance, ARGS.max_steps//2, f"{cn}")
                    if not ok:
                        msg = f"Ped {cn} GoTo blocked at cmd {cmd_idx} (dist={d:.2f}m)"
                        print(f"  [FAIL] ep{ep_id}: {msg}")
                        lines.append(f"FAIL: {msg}")
                        ped_fail = True
                        break
                    lines.append(f" {cn} GoTo ok")
                elif cmd.get("cmd") in ("Idle", "LookAround"):
                    update(50)
            if ped_fail:
                break

        if ped_fail:
            results.append({"episode": ep_id, "result": "fail", "reason": lines[-1] if lines else "ped fail"})
        else:
            lines.append("PASS")
            results.append({"episode": ep_id, "result": "pass"})
            print(f"  [PASS] ep{ep_id}")

        with open(log_path, "w") as f:
            f.write("\n".join(lines))

    return {"scene_id": scene_id, "results": results}

# ═══════════════════════════════════════════════════════════════════
def main():
    usda_dir = Path(ARGS.usda_dir)
    if ARGS.scene_ids:
        usda_files = {sid: str(usda_dir / f"{sid}.usda") for sid in ARGS.scene_ids
                      if (usda_dir / f"{sid}.usda").exists()}
    else:
        usda_files = {fp.stem: str(fp) for fp in usda_dir.glob("*.usda")}
    if not usda_files:
        print("[FATAL] No USDAs found")
        sim_app.close()
        return 1

    all_res = {}
    pass_cnt = fail_cnt = 0
    for sid, path in sorted(usda_files.items()):
        print(f"\n[VALIDATE] Scene {sid}")
        res = validate_scene(path, sid)
        all_res[sid] = res
        for ep in res.get("results", []):
            if ep.get("result") == "pass":
                pass_cnt += 1
            elif ep.get("result") == "fail":
                fail_cnt += 1

    out_dir = ARGS.output_dir or os.path.join(ARGS.episode_dir, "validation")
    os.makedirs(out_dir, exist_ok=True)
    summary_name = f"summary_worker_{ARGS.worker_id}.json" if ARGS.worker_id > 0 else "summary.json"
    summary_path = os.path.join(out_dir, summary_name)
    with open(summary_path, "w") as f:
        json.dump({"total_pass": pass_cnt, "total_fail": fail_cnt, "scenes": all_res}, f, indent=2)

    print(f"\n[DONE] Pass={pass_cnt} Fail={fail_cnt}")
    sim_app.close()
    return 0 if fail_cnt == 0 else 1

if __name__ == "__main__":
    sys.exit(main())