"""
preview_character_clothing.py
──────────────────────────────
逐个加载 People 角色,自动定时给衣物部位随机换色,
让你直观确认每个角色都能正常染色。

可选 GUI 模式(默认)或 headless 模式 + 截图。

用法:
    # GUI 模式:打开 Isaac Sim 窗口,逐个角色循环换装
    /isaac-sim/python.sh preview_character_clothing.py \\
        --root /workspace/FLUX/assets/isaacsim_assets/Assets/Isaac/4.5/Isaac/People/Characters

    # 截图模式:不开窗口,每个角色保存几张图到本地
    /isaac-sim/python.sh preview_character_clothing.py \\
        --root /workspace/FLUX/assets/isaacsim_assets/Assets/Isaac/4.5/Isaac/People/Characters \\
        --headless --screenshot_dir /workspace/FLUX/sage_utils/clothing_previews

    # 只看一个角色
    /isaac-sim/python.sh preview_character_clothing.py \\
        --root /workspace/FLUX/assets/isaacsim_assets/Assets/Isaac/4.5/Isaac/People/Characters \\
        --only male_adult_construction_03 \\
        --num_changes 10 \\
        --seconds_per_change 3
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

# 1) 必须在 pxr / omni 导入前启动 SimulationApp
from isaacsim import SimulationApp

_parser_for_headless = argparse.ArgumentParser(add_help=False)
_parser_for_headless.add_argument("--headless", action="store_true")
_pre_args, _ = _parser_for_headless.parse_known_args()

_app = SimulationApp({
    "headless": _pre_args.headless,
    "renderer": "RayTracedLighting",
    "width": 1280,
    "height": 720,
})

# 2) 启动后才能 import 这些
import omni.usd
import omni.kit.app
import carb
from pxr import Usd, UsdShade, UsdGeom, Sdf, Gf, UsdLux


# ── 配置 ─────────────────────────────────────────────────────────────────

# 哪些部位算"衣服"(可以染色),按重要性从高到低
CLOTHING_PARTS = (
    # 主衣
    "shirt", "tshirt", "scrubsshirt", "policeshirt", "labcoat",
    "jeans", "bluejeans", "workpants", "policepants", "scrubspants",
    # 鞋
    "sneakers", "workboots", "tennisshoes", "policeshoes",
    # 帽子
    "hardhat", "policehat", "policecap",
    # 工装外套
    "vest", "safetyvest",
    # 配饰(可选,通常不染)
    # "gloves", "idbadge", "stethoscope", "policeradio", "earprotection",
)

# 永远跳过的部位(主要是皮肤)
SKIP_PARTS_CONTAINS = ("skin",)
SKIP_MATERIAL_NAME_CONTAINS = ("retro__reflective",)

# 候选染色 input 名,按优先级顺序尝试
COLOR_INPUTS = ("diffuse_tint", "diffuse_color_constant")

EXCLUDED_FOLDERS = {"biped_demo"}


# ── helpers ──────────────────────────────────────────────────────────────

def _update(n: int = 1) -> None:
    for _ in range(n):
        _app.update()


def find_character_usds(root: Path) -> list[tuple[str, Path]]:
    chars: list[tuple[str, Path]] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name.startswith(".") or folder.name in EXCLUDED_FOLDERS:
            continue
        usd_file = next(
            (f for f in folder.iterdir() if f.suffix in (".usd", ".usda")),
            None,
        )
        if usd_file is not None:
            chars.append((folder.name, usd_file))
    return chars


def setup_blank_stage_with_light() -> Usd.Stage:
    """新建一个空 stage,加 dome light + ground plane,方便观察"""
    omni.usd.get_context().new_stage()
    _update(2)
    stage = omni.usd.get_context().get_stage()

    # World xform
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # Dome light (整体环境光)
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1000.0)

    # Distant light(打方向)
    dist = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    dist.CreateIntensityAttr(2500.0)
    UsdGeom.Xformable(dist).AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    # Ground plane
    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    ground.CreatePointsAttr([
        (-5, -5, 0), (5, -5, 0), (5, 5, 0), (-5, 5, 0),
    ])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

    _update(2)
    return stage


def add_character_reference(stage: Usd.Stage, char_name: str,
                             usd_path: Path) -> str:
    """通过 reference 把角色 USD 加到 stage 里;返回角色根 prim path"""
    char_root_path = f"/World/{char_name}"
    char_root = UsdGeom.Xform.Define(stage, char_root_path).GetPrim()
    char_root.GetReferences().AddReference(str(usd_path))
    _update(5)  # 等加载
    return char_root_path


def find_clothing_materials(stage: Usd.Stage,
                             char_root_path: str) -> dict[str, Usd.Prim]:
    """在角色 prim 下找到所有衣物 material,返回 {part: material_prim}"""
    char_root = stage.GetPrimAtPath(char_root_path)
    if not char_root.IsValid():
        return {}

    found: dict[str, Usd.Prim] = {}
    for prim in Usd.PrimRange(char_root):
        if prim.GetTypeName() != "Material":
            continue
        name = prim.GetName()
        # 跳过 skin
        if any(s in name for s in SKIP_PARTS_CONTAINS):
            continue
        if any(s in name for s in SKIP_MATERIAL_NAME_CONTAINS):   # ← 新增
            continue
        tokens = name.split("__")
        if len(tokens) < 3:
            continue
        part = tokens[-1]
        if part in CLOTHING_PARTS and part not in found:
            found[part] = prim
    return found


def set_material_color(mat_prim: Usd.Prim,
                        rgb: tuple[float, float, float]) -> str | None:
    """
    尝试给 material 染色。
    返回成功使用的 input 名;失败返回 None。
    """
    mat = UsdShade.Material(mat_prim)
    if not mat:
        return None

    # 找 surface shader
    surface = mat.GetSurfaceOutput()
    shader_prim = None
    if surface:
        src = surface.GetConnectedSource()
        if src:
            shader_prim = src[0].GetPrim()
    if shader_prim is None:
        # 兜底:取第一个 Shader child
        for c in mat_prim.GetChildren():
            if c.GetTypeName() == "Shader":
                shader_prim = c
                break
    if shader_prim is None:
        return None

    shader = UsdShade.Shader(shader_prim)
    color_vec = Gf.Vec3f(*rgb)

    for input_name in COLOR_INPUTS:
        inp = shader.GetInput(input_name)
        if inp:
            try:
                inp.Set(color_vec)
                return input_name
            except Exception:
                continue

    # 如果 input 不存在,主动创建一个(MDL 通常已经存在,但保险)
    for input_name in COLOR_INPUTS:
        try:
            inp = shader.CreateInput(input_name, Sdf.ValueTypeNames.Color3f)
            inp.Set(color_vec)
            return input_name + "(created)"
        except Exception:
            continue

    return None


def random_rgb(rng: random.Random) -> tuple[float, float, float]:
    """生成饱和度较高的随机颜色,避免出现灰扑扑的"""
    h = rng.random()
    s = 0.55 + rng.random() * 0.45  # 0.55 ~ 1.0
    v = 0.55 + rng.random() * 0.45
    # HSV -> RGB
    import colorsys
    return colorsys.hsv_to_rgb(h, s, v)


def position_camera_to_view_character(char_root_path: str) -> None:
    """用 viewport camera API 设置默认相机位置(允许鼠标导航且不被控制器覆盖)"""
    eye    = Gf.Vec3d(0.0, -4.5, 1.6)   # 角色正前方 (+X);若看到背面改成 -4.5
    target = Gf.Vec3d(0.0, 0.0, 1.0)

    try:
        from omni.kit.viewport.utility import get_active_viewport
        from omni.kit.viewport.utility.camera_state import ViewportCameraState

        vp = get_active_viewport()
        cam_state = ViewportCameraState(vp.camera_path, vp)
        cam_state.set_position_world(eye, True)
        cam_state.set_target_world(target, True)
    except Exception as e:
        print(f"[WARN] camera positioning failed: {e}")

    # 焦距单独设(广角看全身)
    stage = omni.usd.get_context().get_stage()
    try:
        from omni.kit.viewport.utility import get_active_viewport
        cam_prim = stage.GetPrimAtPath(get_active_viewport().camera_path)
        fl = cam_prim.GetAttribute("focalLength")
        if fl:
            fl.Set(20.0)
    except Exception:
        pass


def take_screenshot(out_path: Path) -> bool:
    """保存当前视口截图"""
    try:
        import omni.kit.viewport.utility as vp_util
        vp = vp_util.get_active_viewport()
        if vp is None:
            return False
        # 等几帧让渲染稳定
        _update(20)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        vp_util.capture_viewport_to_file(vp, str(out_path))
        _update(5)
        return True
    except Exception as e:
        print(f"[WARN] screenshot failed: {e}")
        return False


# ── 主流程 ───────────────────────────────────────────────────────────────

def preview_one_character(
    char_name: str,
    usd_path: Path,
    num_changes: int,
    seconds_per_change: float,
    rng: random.Random,
    screenshot_dir: Path | None,
) -> None:
    print(f"\n{'='*60}\n[CHARACTER] {char_name}\n{'='*60}")

    stage = setup_blank_stage_with_light()
    char_root_path = add_character_reference(stage, char_name, usd_path)
    position_camera_to_view_character(char_root_path)
    _update(20)

    materials = find_clothing_materials(stage, char_root_path)
    if not materials:
        print(f"[{char_name}] ⚠  no clothing materials found, skipping")
        return

    print(f"[{char_name}] clothing parts: {sorted(materials.keys())}")

    for i in range(num_changes):
        log_parts: list[str] = []
        for part, mat_prim in materials.items():
            rgb = random_rgb(rng)
            used = set_material_color(mat_prim, rgb)
            ok = "✓" if used else "✗"
            log_parts.append(
                f"{part}({ok} {used or '-'} "
                f"R={rgb[0]:.2f} G={rgb[1]:.2f} B={rgb[2]:.2f})"
            )
        print(f"[{char_name}] change {i+1}/{num_changes}: " + ", ".join(log_parts))

        if screenshot_dir is not None:
            shot = screenshot_dir / char_name / f"change_{i+1:02d}.png"
            take_screenshot(shot)

        # 等若干秒,让用户/渲染看一下
        t_end = time.time() + seconds_per_change
        while time.time() < t_end:
            _update(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--only", default=None,
                    help="只测试这个角色名(默认全部)")
    ap.add_argument("--num_changes", type=int, default=5,
                    help="每个角色随机换装的次数")
    ap.add_argument("--seconds_per_change", type=float, default=2.5,
                    help="每次换装后停留秒数(GUI 模式下看图用)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true",
                    help="不开窗口(配合 --screenshot_dir 使用)")
    ap.add_argument("--screenshot_dir", type=Path, default=None,
                    help="每次换装保存截图到这个目录")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    chars = find_character_usds(args.root)
    if args.only:
        chars = [(n, p) for n, p in chars if n == args.only]
        if not chars:
            print(f"[ERROR] character '{args.only}' not found under {args.root}")
            _app.close()
            return

    print(f"[INFO] Will preview {len(chars)} character(s)")
    for name, usd_path in chars:
        preview_one_character(
            char_name=name,
            usd_path=usd_path,
            num_changes=args.num_changes,
            seconds_per_change=args.seconds_per_change,
            rng=rng,
            screenshot_dir=args.screenshot_dir,
        )

    print("\n[OK] done. Closing app...")
    _app.close()


if __name__ == "__main__":
    main()