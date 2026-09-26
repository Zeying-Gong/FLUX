#!/usr/bin/env python3
"""
build_character_clothing_profiles.py
─────────────────────────────────────
离线(需启动 Isaac Sim)预计算每个 People 角色的服装 profile:
  - 实际可见且可染色的部位列表(过滤掉 material 存在但无几何的部位)
  - 每个部位的默认 tint 颜色(RGB)
  - 每个部位默认色映射到色板的最近邻 color_word(ΔE CIE76)

产出一个静态 JSON,供【纯 Python 的生成端】直接读取,
无需在生成 episode 时再启动 Isaac Sim。

用法:
    /isaac-sim/python.sh build_character_clothing_profiles.py \\
        --root /workspace/FLUX/assets/isaacsim_assets/Assets/Isaac/4.5/Isaac/People/Characters \\
        --out  /workspace/FLUX/sage_utils/character_clothing_profiles.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# 启动 SimulationApp(headless)以获得 pxr
from isaacsim import SimulationApp
_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

import omni.usd
from pxr import Usd, UsdShade, UsdGeom


# ════════════════════════════════════════════════════════════════════
# 共享配置  —— 生成端 / 采集端会 import 同一份(见 clothing_palette.py)
# 为避免循环依赖,这里内联一份;最终请统一到 clothing_palette.py。
# ════════════════════════════════════════════════════════════════════

# CSS Level-1 16 色(sRGB 0-1)
COLOR_PALETTE: dict[str, tuple[float, float, float]] = {
    "black":   (0.000, 0.000, 0.000),
    "silver":  (0.753, 0.753, 0.753),
    "gray":    (0.502, 0.502, 0.502),
    "white":   (1.000, 1.000, 1.000),
    "maroon":  (0.502, 0.000, 0.000),
    "red":     (1.000, 0.000, 0.000),
    "purple":  (0.502, 0.000, 0.502),
    "fuchsia": (1.000, 0.000, 1.000),
    "green":   (0.000, 0.502, 0.000),
    "lime":    (0.000, 1.000, 0.000),
    "olive":   (0.502, 0.502, 0.000),
    "yellow":  (1.000, 1.000, 0.000),
    "navy":    (0.000, 0.000, 0.502),
    "blue":    (0.000, 0.000, 1.000),
    "teal":    (0.000, 0.502, 0.502),
    "aqua":    (0.000, 1.000, 1.000),
}

CLOTHING_PARTS = (
    "shirt", "tshirt", "scrubsshirt", "policeshirt", "labcoat",
    "jeans", "bluejeans", "workpants", "policepants", "scrubspants",
    "sneakers", "workboots", "tennisshoes", "policeshoes",
    "hardhat", "policehat", "policecap",
    "vest", "safetyvest",
)

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


# ── ΔE 最近邻:RGB(0-1) → 最接近的色板 color_word ───────────────────
def _srgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(max(0.0, min(1.0, v))) for v in rgb)
    # linear sRGB -> XYZ (D65)
    x = 0.4124 * r + 0.3576 * g + 0.1805 * b
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = 0.0193 * r + 0.1192 * g + 0.9505 * b
    xr, yr, zr = x / 0.95047, y / 1.0, z / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx, fy, fz = f(xr), f(yr), f(zr)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    bb = 200 * (fy - fz)
    return (L, a, bb)


_PALETTE_LAB = {name: _srgb_to_lab(rgb) for name, rgb in COLOR_PALETTE.items()}


def nearest_palette_word(rgb: tuple[float, float, float]) -> tuple[str, float]:
    """返回 (最近色板词, ΔE 距离)"""
    lab = _srgb_to_lab(rgb)
    best_word, best_d = None, float("inf")
    for word, plab in _PALETTE_LAB.items():
        d = ((lab[0] - plab[0]) ** 2 +
             (lab[1] - plab[1]) ** 2 +
             (lab[2] - plab[2]) ** 2) ** 0.5
        if d < best_d:
            best_d, best_word = d, word
    return best_word, best_d


# ── USD helpers ─────────────────────────────────────────────────────
def find_character_usds(root: Path) -> list[tuple[str, Path]]:
    chars = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name.startswith(".") or folder.name in EXCLUDED_FOLDERS:
            continue
        usd_file = next((f for f in folder.iterdir()
                         if f.suffix in (".usd", ".usda")), None)
        if usd_file is not None:
            chars.append((folder.name, usd_file))
    return chars


def _get_shader(mat_prim: Usd.Prim):
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


def read_default_color(mat_prim: Usd.Prim):
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


def collect_bound_material_paths(char_root: Usd.Prim) -> set[str]:
    bound: set[str] = set()
    for prim in Usd.PrimRange(char_root):
        if prim.GetTypeName() not in ("Mesh", "GeomSubset"):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            vis = imageable.GetVisibilityAttr()
            if vis and vis.Get() == UsdGeom.Tokens.invisible:
                continue
        binding_api = UsdShade.MaterialBindingAPI(prim)
        mat, _ = binding_api.ComputeBoundMaterial()
        if mat:
            mp = mat.GetPrim()
            if mp and mp.IsValid():
                bound.add(str(mp.GetPath()))
    return bound


def build_profile(char_name: str, usd_path: Path) -> dict:
    stage = omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    root_path = f"/World/{char_name}"
    UsdGeom.Xform.Define(stage, "/World")
    char_root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    char_root.GetReferences().AddReference(str(usd_path))
    for _ in range(60):
        _app.update()

    char_root = stage.GetPrimAtPath(root_path)
    bound = collect_bound_material_paths(char_root)

    parts: dict[str, dict] = {}
    for prim in Usd.PrimRange(char_root):
        if prim.GetTypeName() != "Material":
            continue
        name = prim.GetName()
        if any(s in name for s in SKIP_PARTS_CONTAINS):
            continue
        if any(s in name for s in SKIP_MATERIAL_NAME_CONTAINS):
            continue
        if str(prim.GetPath()) not in bound:   # 必须绑定到可见几何
            continue
        tokens = name.split("__")
        if len(tokens) < 3:
            continue
        part = tokens[-1]
        if part not in CLOTHING_PARTS or part in parts:
            continue

        default_rgb = read_default_color(prim)
        if default_rgb is not None:
            word, de = nearest_palette_word(default_rgb)
        else:
            word, de = None, None

        parts[part] = {
            "category": PART_TO_CATEGORY.get(part, part),
            "material_name": name,
            "default_rgb": [round(x, 4) for x in default_rgb] if default_rgb else None,
            "default_color_word": word,
            "default_deltaE": round(de, 1) if de is not None else None,
        }

    return {
        "asset": char_name,
        "usd_path": str(usd_path),
        "parts": parts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    chars = find_character_usds(args.root)
    print(f"[INFO] {len(chars)} characters found")

    profiles = {}
    for i, (name, usd_path) in enumerate(chars):
        print(f"[{i+1:3d}/{len(chars)}] {name}")
        prof = build_profile(name, usd_path)
        profiles[name] = prof
        cats = sorted({p["category"] for p in prof["parts"].values()})
        print(f"    parts={sorted(prof['parts'].keys())}  categories={cats}")
        for part, info in prof["parts"].items():
            print(f"      {part:14s} default={info['default_color_word']} "
                  f"(ΔE={info['default_deltaE']})")

    # 固定资产顺序(供 episode_id % N 轮询用),写进文件
    asset_order = [name for name, _ in chars]

    out_obj = {
        "_meta": {
            "num_assets": len(chars),
            "asset_order": asset_order,
            "palette": COLOR_PALETTE,
            "part_to_category": PART_TO_CATEGORY,
        },
        "profiles": profiles,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] wrote {args.out}")

    _app.close()


if __name__ == "__main__":
    main()