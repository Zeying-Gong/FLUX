#!/usr/bin/env python3
"""Load a SAGE scene without creating robots, people, physics, or sensors."""

from __future__ import annotations

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Scene-only visual diagnostic")
    parser.add_argument(
        "--usda",
        default="/workspace/SAGE-3D_Official/SAGE-3D_data/usda/839888.usda",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=0,
        help="Kit updates before exit; 0 keeps the viewer open until interrupted",
    )
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


ARGS = parse_args()

from isaacsim import SimulationApp


exp_path = os.environ.get("EXP_PATH")
if not exp_path:
    print("[FATAL] EXP_PATH is not set", file=sys.stderr)
    sys.exit(1)

experience = os.path.join(
    exp_path, "isaacsim.exp.action_and_event_data_generation.base.kit"
)
app = SimulationApp(
    launch_config={
        "renderer": "RayTracedLighting",
        "headless": ARGS.headless,
        "enable_cameras": True,
        "crash_reporter/enabled": False,
    },
    experience=experience,
)

import omni.usd
from pxr import UsdGeom


def update(count=1):
    for _ in range(count):
        app.update()


def main():
    if not os.path.isfile(ARGS.usda):
        print(f"[FATAL] Scene does not exist: {ARGS.usda}", file=sys.stderr)
        return 1

    print(f"[SceneOnly] Opening {ARGS.usda}")
    omni.usd.get_context().open_stage(ARGS.usda)
    update(30)

    waited = 0
    while omni.usd.get_context().get_stage_loading_status()[1] > 0:
        update()
        waited += 1
        if waited >= 3000:
            print("[SceneOnly] WARN: stage loading timeout")
            break

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[FATAL] Stage failed to open", file=sys.stderr)
        return 1

    collision_root = stage.GetPrimAtPath("/World/scene_collision")
    if collision_root and collision_root.IsValid():
        UsdGeom.Imageable(collision_root).MakeInvisible()

    for prim in stage.Traverse():
        if prim.GetTypeName() == "NavMeshVolume":
            UsdGeom.Imageable(prim).MakeInvisible()

    update(10)
    print("[SceneOnly] Top-level prims already present in the scene:")
    pseudo_root = stage.GetPseudoRoot()
    for prim in pseudo_root.GetChildren():
        print(f"  {prim.GetPath()} [{prim.GetTypeName()}]")
        for child in prim.GetChildren():
            print(f"    {child.GetPath()} [{child.GetTypeName()}]")

    if stage.GetPrimAtPath("/World/Robot").IsValid():
        print("[SceneOnly] WARNING: /World/Robot is authored by the scene itself")
    else:
        print("[SceneOnly] Confirmed: this process did not create /World/Robot")

    if ARGS.updates > 0:
        update(ARGS.updates)
        return 0

    print("[SceneOnly] Viewer ready. Use the viewport freely; press Ctrl+C to exit.")
    try:
        while app.is_running():
            update()
    except KeyboardInterrupt:
        pass
    return 0


try:
    exit_code = main()
finally:
    app.close()

sys.exit(exit_code)
