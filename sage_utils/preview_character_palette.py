"""
preview_character_palette.py
──────────────────────────────
用【离散色板】给衣物部位采样上色,返回 {part: color_word},
并解析角色的性别 / 类型(供后续喂 LLM 生成 caption)。

支持截图模式逐次保存,方便离线检查。

用法:
    # 截图模式(推荐检查):一个角色,采样若干次,每次存图
    /isaac-sim/python.sh preview_character_palette.py \\
        --root /workspace/FLUX/assets/isaacsim_assets/Assets/Isaac/4.5/Isaac/People/Characters \\
        --only male_adult_construction_03 \\
        --num_changes 6 \\
        --headless \\
        --screenshot_dir /workspace/FLUX/sage_utils/palette_previews

    # GUI 模式
    /isaac-sim/python.sh preview_character_palette.py \\
        --root /workspace/FLUX/assets/isaacsim_assets/Assets/Isaac/4.5/Isaac/People/Characters \\
        --only male_adult_construction_03 --num_changes 6
"""
from __future__ import annotations

import argparse
import json
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
import carb
from pxr import Usd, UsdShade, UsdGeom, Sdf, Gf, UsdLux

# ── 离散色板 ────────────────────────────────────────────────────────────
# 基于标准 CSS/X11 命名颜色,经 CIELAB ΔE(CIE76)验证两两感知距离。
# 精简掉了与已有色过近的 cyan/magenta/tan(分别并入 teal/purple/brown 体系)。
# 当前最小 ΔE ≈ 21.9 (white vs beige),其余多在 35 以上。

COLOR_PALETTE: dict[str, tuple[float, float, float]] = {
    "black":   (0.000, 0.000, 0.000),  # #000000
    "silver":  (0.753, 0.753, 0.753),  # #C0C0C0
    "gray":    (0.502, 0.502, 0.502),  # #808080
    "white":   (1.000, 1.000, 1.000),  # #FFFFFF
    "maroon":  (0.502, 0.000, 0.000),  # #800000
    "red":     (1.000, 0.000, 0.000),  # #FF0000
    "purple":  (0.502, 0.000, 0.502),  # #800080
    "fuchsia": (1.000, 0.000, 1.000),  # #FF00FF
    "green":   (0.000, 0.502, 0.000),  # #008000
    "lime":    (0.000, 1.000, 0.000),  # #00FF00
    "olive":   (0.502, 0.502, 0.000),  # #808000
    "yellow":  (1.000, 1.000, 0.000),  # #FFFF00
    "navy":    (0.000, 0.000, 0.502),  # #000080
    "blue":    (0.000, 0.000, 1.000),  # #0000FF
    "teal":    (0.000, 0.502, 0.502),  # #008080
    "aqua":    (0.000, 1.000, 1.000),  # #00FFFF
}

# 经 ΔE 验证、感知上确实接近(ΔE < ~35)的组;
# 采样时同一角色身上尽量避免同组颜色并存。
CONFUSABLE_GROUPS: list[set[str]] = [
    {"black", "gray", "navy", "maroon"},   # 深色系
    {"white", "silver"},                    # 浅中性
    {"red", "maroon"},
    {"green", "lime", "olive"},
    {"blue", "navy"},
    {"purple", "fuchsia"},
    {"teal", "aqua"},
    {"yellow", "olive"},
]


# ── 衣物部位配置 ────────────────────────────────────────────────────────

# 可染色的部位(完整列表,帽子/鞋也在内,作为辅助区分维度)
CLOTHING_PARTS = (
    # 主上衣
    "shirt", "tshirt", "scrubsshirt", "policeshirt", "labcoat",
    # 主下裤
    "jeans", "bluejeans", "workpants", "policepants", "scrubspants",
    # 鞋
    "sneakers", "workboots", "tennisshoes", "policeshoes",
    # 帽子
    "hardhat", "policehat", "policecap",
    # 外套 / 背心
    "vest", "safetyvest",
)

# 把细分部位名归一到"语义大类",供 caption 用("上衣/裤子/鞋/帽子")
PART_TO_CATEGORY: dict[str, str] = {
    "shirt": "top", "tshirt": "top", "scrubsshirt": "top",
    "policeshirt": "top", "labcoat": "top",
    "jeans": "bottom", "bluejeans": "bottom", "workpants": "bottom",
    "policepants": "bottom", "scrubspants": "bottom",
    "sneakers": "shoes", "workboots": "shoes", "tennisshoes": "shoes",
    "policeshoes": "shoes",
    "hardhat": "hat", "policehat": "hat", "policecap": "hat",
    "vest": "vest", "safetyvest": "vest",
}

SKIP_PARTS_CONTAINS = ("skin",)
SKIP_MATERIAL_NAME_CONTAINS = ("retro__reflective",)
COLOR_INPUTS = ("diffuse_tint", "diffuse_color_constant")
EXCLUDED_FOLDERS = {"biped_demo"}


# ── 角色元信息解析 ──────────────────────────────────────────────────────

def parse_character_meta(char_name: str) -> dict[str, str]:
    """从角色文件夹名解析 gender / role(供 LLM 生成 caption)"""
    n = char_name.lower()

    if "female" in n or n.startswith("f_"):
        gender = "female"
    elif "male" in n or n.startswith("m_"):
        gender = "male"
    else:
        gender = "unknown"

    if "police" in n:
        role = "police officer"
    elif "construction" in n:
        role = "construction worker"
    elif "medical" in n:
        role = "medical staff"
    elif "business" in n:
        role = "business person"
    else:
        role = "person"

    age = "adult"  # 当前资产全是成人;留字段以备扩展
    return {"gender": gender, "role": role, "age": age}


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
    omni.usd.get_context().new_stage()
    _update(2)
    stage = omni.usd.get_context().get_stage()

    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1000.0)

    dist = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    dist.CreateIntensityAttr(2500.0)
    UsdGeom.Xformable(dist).AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    ground.CreatePointsAttr([(-5, -5, 0), (5, -5, 0), (5, 5, 0), (-5, 5, 0)])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

    _update(2)
    return stage


def add_character_reference(stage: Usd.Stage, char_name: str,
                             usd_path: Path) -> str:
    char_root_path = f"/World/{char_name}"
    char_root = UsdGeom.Xform.Define(stage, char_root_path).GetPrim()
    char_root.GetReferences().AddReference(str(usd_path))
    _update(60)   # 等贴图/材质异步加载完成(原来 5 太短,会截到灰皮肤)
    return char_root_path


def _collect_bound_material_paths(char_root: Usd.Prim) -> set[str]:
    """遍历角色下所有可见 Mesh / GeomSubset,收集它们实际绑定的 material 路径。

    用 UsdShade.MaterialBindingAPI 解析 direct binding。
    返回 {material_prim_path, ...}。
    """
    bound: set[str] = set()
    for prim in Usd.PrimRange(char_root):
        type_name = prim.GetTypeName()
        if type_name not in ("Mesh", "GeomSubset"):
            continue

        # 跳过不可见几何(visibility = invisible)
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            vis_attr = imageable.GetVisibilityAttr()
            if vis_attr and vis_attr.Get() == UsdGeom.Tokens.invisible:
                continue

        binding_api = UsdShade.MaterialBindingAPI(prim)
        # 计算最终绑定的材质(含继承)
        mat, _ = binding_api.ComputeBoundMaterial()
        if mat:
            mat_prim = mat.GetPrim()
            if mat_prim and mat_prim.IsValid():
                bound.add(str(mat_prim.GetPath()))
    return bound


def find_clothing_materials(stage: Usd.Stage,
                            char_root_path: str) -> dict[str, Usd.Prim]:
    """返回 {part: material_prim},part 为细分部位名(shirt/jeans/...)

    只保留【确实绑定到可见 Mesh/GeomSubset】的材质,
    避免染了一个场景里没有对应几何的材质(如某角色定义了 hardhat
    material 却没戴帽子),从而导致 caption 描述不存在的衣物。
    """
    char_root = stage.GetPrimAtPath(char_root_path)
    if not char_root.IsValid():
        return {}

    bound_mat_paths = _collect_bound_material_paths(char_root)

    found: dict[str, Usd.Prim] = {}
    for prim in Usd.PrimRange(char_root):
        if prim.GetTypeName() != "Material":
            continue
        name = prim.GetName()
        if any(s in name for s in SKIP_PARTS_CONTAINS):
            continue
        if any(s in name for s in SKIP_MATERIAL_NAME_CONTAINS):
            continue
        # ★ 过滤:material 必须真的被某个可见几何绑定
        if str(prim.GetPath()) not in bound_mat_paths:
            continue
        tokens = name.split("__")
        if len(tokens) < 3:
            continue
        part = tokens[-1]
        if part in CLOTHING_PARTS and part not in found:
            found[part] = prim
    return found


def _get_shader(mat_prim: Usd.Prim) -> UsdShade.Shader | None:
    mat = UsdShade.Material(mat_prim)
    if not mat:
        return None
    surface = mat.GetSurfaceOutput()
    if surface:
        src = surface.GetConnectedSource()
        if src:
            return UsdShade.Shader(src[0].GetPrim())
    for c in mat_prim.GetChildren():
        if c.GetTypeName() == "Shader":
            return UsdShade.Shader(c)
    return None


def read_default_color(mat_prim: Usd.Prim) -> tuple[float, float, float] | None:
    """读取材质默认 tint(若有);用于'保留原色'选项或记录"""
    shader = _get_shader(mat_prim)
    if shader is None:
        return None
    for input_name in COLOR_INPUTS:
        inp = shader.GetInput(input_name)
        if inp:
            val = inp.Get()
            if val is not None:
                try:
                    return tuple(float(x) for x in val)
                except Exception:
                    pass
    return None


def set_material_color(mat_prim: Usd.Prim,
                       rgb: tuple[float, float, float]) -> str | None:
    shader = _get_shader(mat_prim)
    if shader is None:
        return None
    color_vec = Gf.Vec3f(*rgb)
    for input_name in COLOR_INPUTS:
        inp = shader.GetInput(input_name)
        if inp:
            try:
                inp.Set(color_vec)
                return input_name
            except Exception:
                continue
    for input_name in COLOR_INPUTS:
        try:
            inp = shader.CreateInput(input_name, Sdf.ValueTypeNames.Color3f)
            inp.Set(color_vec)
            return input_name + "(created)"
        except Exception:
            continue
    return None


# ── 色板采样 ────────────────────────────────────────────────────────────

def _too_confusable(c1: str, c2: str) -> bool:
    for grp in CONFUSABLE_GROUPS:
        if c1 in grp and c2 in grp:
            return True
    return False


def sample_palette_for_parts(
    parts: list[str],
    rng: random.Random,
) -> dict[str, str]:
    """
    给一组部位采样颜色词。
    约束:同一角色身上,语义相邻的部位(top/bottom)尽量不撞色、不互相混淆。
    返回 {part: color_word}
    """
    assigned: dict[str, str] = {}
    used_words: list[str] = []
    palette_keys = list(COLOR_PALETTE.keys())

    for part in parts:
        candidates = palette_keys[:]
        rng.shuffle(candidates)
        chosen = None
        for cand in candidates:
            # 不和已用颜色完全相同
            if cand in used_words:
                continue
            # 不和已用颜色互相混淆
            if any(_too_confusable(cand, u) for u in used_words):
                continue
            chosen = cand
            break
        if chosen is None:  # 实在没空位就允许重复
            chosen = rng.choice(palette_keys)
        assigned[part] = chosen
        used_words.append(chosen)
    return assigned


# ── 相机 / 截图 ─────────────────────────────────────────────────────────

def position_camera_to_view_character(char_root_path: str) -> None:
    eye    = Gf.Vec3d(0.0, -4.5, 1.6)
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
    try:
        import omni.kit.viewport.utility as vp_util
        vp = vp_util.get_active_viewport()
        if vp is None:
            return False
        _update(120)
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
    meta = parse_character_meta(char_name)
    print(f"\n{'='*60}")
    print(f"[CHARACTER] {char_name}")
    print(f"  meta: gender={meta['gender']}  role={meta['role']}  age={meta['age']}")
    print('='*60)

    stage = setup_blank_stage_with_light()
    char_root_path = add_character_reference(stage, char_name, usd_path)
    position_camera_to_view_character(char_root_path)
    _update(20)

    materials = find_clothing_materials(stage, char_root_path)
    if not materials:
        print(f"[{char_name}] ⚠  no clothing materials found, skipping")
        return

    parts = list(materials.keys())
    print(f"[{char_name}] clothing parts (bound & visible): {sorted(parts)}")

    # 额外:列出"材质存在但未绑定到可见几何"而被过滤掉的部位,供核对
    char_root_prim = stage.GetPrimAtPath(char_root_path)
    bound_paths = _collect_bound_material_paths(char_root_prim)
    skipped = []
    for prim in Usd.PrimRange(char_root_prim):
        if prim.GetTypeName() != "Material":
            continue
        nm = prim.GetName()
        if any(s in nm for s in SKIP_PARTS_CONTAINS):
            continue
        if any(s in nm for s in SKIP_MATERIAL_NAME_CONTAINS):
            continue
        tk = nm.split("__")
        if len(tk) < 3 or tk[-1] not in CLOTHING_PARTS:
            continue
        if str(prim.GetPath()) not in bound_paths:
            skipped.append(tk[-1])
    if skipped:
        print(f"[{char_name}] filtered (material exists but no visible geom): "
              f"{sorted(set(skipped))}")

    # 记录默认配色(只读一次)
    defaults = {p: read_default_color(materials[p]) for p in parts}
    print(f"[{char_name}] default tints: " +
          ", ".join(f"{p}={tuple(round(x,2) for x in v) if v else None}"
                    for p, v in defaults.items()))

    for i in range(num_changes):
        # 1) 色板采样,拿到 {part: color_word}
        part_to_word = sample_palette_for_parts(parts, rng)

        # 2) 上色 + 构造 caption 用的结构
        appearance = {
            "asset": char_name,
            "gender": meta["gender"],
            "role": meta["role"],
            "parts": {},   # {part: {category, color_word, rgb}}
        }
        log_parts = []
        for part in parts:
            word = part_to_word[part]
            rgb = COLOR_PALETTE[word]
            used = set_material_color(materials[part], rgb)
            ok = "✓" if used else "✗"
            cat = PART_TO_CATEGORY.get(part, part)
            appearance["parts"][part] = {
                "category": cat,
                "color_word": word,
                "rgb": [round(x, 3) for x in rgb],
            }
            log_parts.append(f"{cat}:{part}={word}({ok})")

        print(f"[{char_name}] change {i+1}/{num_changes}: " +
              ", ".join(log_parts))

        # 3) 截图 + 落地一份 appearance JSON,方便核对图文是否一致
        if screenshot_dir is not None:
            base = screenshot_dir / char_name / f"change_{i+1:02d}"
            ok_shot = take_screenshot(base.with_suffix(".png"))
            with base.with_suffix(".json").open("w") as f:
                json.dump(appearance, f, indent=2, ensure_ascii=False)

        t_end = time.time() + seconds_per_change
        while time.time() < t_end:
            _update(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--only", default=None)
    ap.add_argument("--num_changes", type=int, default=5)
    ap.add_argument("--seconds_per_change", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--screenshot_dir", type=Path, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    chars = find_character_usds(args.root)
    if args.only:
        chars = [(n, p) for n, p in chars if n == args.only]
        if not chars:
            print(f"[ERROR] character '{args.only}' not found")
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