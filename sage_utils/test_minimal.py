#!/usr/bin/env python3
import os, sys
os.environ["EXP_PATH"] = "/isaac-sim/apps"
from isaacsim import SimulationApp

sim_app = SimulationApp(
    launch_config={
        "renderer": "RayTracedLighting",
        "headless": True,
        "enable_cameras": False,
    },
)

import omni.usd
print("[TEST] SimulationApp started OK")
omni.usd.get_context().open_stage("/workspace/SAGE-3D_Official/SAGE-3D_data/usda/839873.usda")
for _ in range(30): sim_app.update()
print("[TEST] Stage opened OK")
sim_app.close()
print("[TEST] Done")