# ── 1. Launch IsaacSim ──────────────────────────────────────────────────────
from isaacsim import SimulationApp
import os
CUSTOM_APP_PATH = os.path.join(
    os.environ["EXP_PATH"],
    "isaacsim.exp.action_and_event_data_generation.base.kit"
)

simulation_app = SimulationApp(
    launch_config={
        "renderer": "RayTracedLighting",
        "headless": True,
        "enable_cameras": True,
    },
    experience=CUSTOM_APP_PATH,
)
# cell 1: 打开 USD
from pxr import Usd, UsdGeom, Sdf

USD_PATH = "/workspace/FLUX/assets/robots/dingo.usd"

stage = Usd.Stage.Open(USD_PATH)
print(f"Stage default prim: {stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None}")
print(f"Root layer: {stage.GetRootLayer().identifier}")
# cell 2: 确认 GroundPlane 存在 + 查看它的位置
# 因为这是编辑 dingo.usd 本身(不是场景 reference),路径应该是 /dingo/GroundPlane
# 或 /GroundPlane,取决于 USD 里 default prim 名字。先列出来看看。

for prim in stage.Traverse():
    path = str(prim.GetPath())
    if "GroundPlane" in path or "ground" in path.lower():
        print(f"  Found: {path}  type={prim.GetTypeName()}")
# cell 3: 删除 GroundPlane
# 根据上一个 cell 的打印结果,填入正确的完整路径

# 例子:如果上一步打印的是 "/dingo/GroundPlane",用这个:
GROUND_PATH = "/dingo/GroundPlane"   # ← 根据 cell 2 实际结果修改

# 验证路径有效
prim = stage.GetPrimAtPath(GROUND_PATH)
if not prim.IsValid():
    print(f"ERROR: {GROUND_PATH} not valid, check cell 2 output")
else:
    print(f"Removing: {GROUND_PATH}")
    stage.RemovePrim(GROUND_PATH)
    print("Removed.")
# cell 4: 保存(就地覆盖 dingo.usd)
stage.GetRootLayer().Save()
print(f"Saved to {USD_PATH}")