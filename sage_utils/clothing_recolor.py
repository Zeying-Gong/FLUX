#!/usr/bin/env python3
"""
clothing_recolor.py
────────────────────
采集端(Isaac Sim 内)使用的角色染色工具。
按 episode 的 appearance.parts 给 spawn 出来的角色逐部位染色。

依赖 pxr(必须在 SimulationApp 启动后才能 import 本模块的函数,
但模块本身可安全 import —— pxr 在函数内延迟使用)。
"""
from __future__ import annotations

from typing import Optional


SKIP_MATERIAL_NAME_CONTAINS = ("retro__reflective",)
SKIP_PARTS_CONTAINS = ("skin",)
COLOR_INPUTS = ("diffuse_tint", "diffuse_color_constant")


def _get_shader(mat_prim):
    from pxr import UsdShade
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


def _set_material_color(mat_prim, rgb) -> Optional[str]:
    """给单个 material 染色,返回成功使用的 input 名或 None。"""
    from pxr import Gf, Sdf, UsdShade
    shader = _get_shader(mat_prim)
    if shader is None:
        return None
    color_vec = Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))
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


def _find_material_by_part(stage, char_prim_path: str) -> dict:
    """
    在角色 prim 子树下,找到每个细分部位名对应的 material prim。
    返回 {part_name: material_prim}。只收集绑定到可见几何的材质。
    """
    from pxr import Usd, UsdGeom, UsdShade

    char_root = stage.GetPrimAtPath(char_prim_path)
    if not char_root or not char_root.IsValid():
        return {}

    # 收集绑定到可见几何的 material 路径
    bound: set[str] = set()
    for prim in Usd.PrimRange(char_root):
        if prim.GetTypeName() not in ("Mesh", "GeomSubset"):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            vis = imageable.GetVisibilityAttr()
            if vis and vis.Get() == UsdGeom.Tokens.invisible:
                continue
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if mat:
            mp = mat.GetPrim()
            if mp and mp.IsValid():
                bound.add(str(mp.GetPath()))

    found: dict = {}
    for prim in Usd.PrimRange(char_root):
        if prim.GetTypeName() != "Material":
            continue
        name = prim.GetName()
        if any(s in name for s in SKIP_PARTS_CONTAINS):
            continue
        if any(s in name for s in SKIP_MATERIAL_NAME_CONTAINS):
            continue
        if str(prim.GetPath()) not in bound:
            continue
        tokens = name.split("__")
        if len(tokens) < 3:
            continue
        part = tokens[-1]
        if part not in found:
            found[part] = prim
    return found


def recolor_character(stage, char_prim_path: str,
                      parts_appearance: dict) -> dict:
    """
    按 appearance 的 parts 给一个角色染色。

    parts_appearance: {part_name: {"color_word":..., "rgb":[r,g,b] or None, ...}}
      rgb 为 None 的部位 → 保留默认色(不染),用于 keep_default 角色。

    返回 {part_name: status}  status ∈ {"recolored:<input>", "skip_default",
                                         "no_material", "failed"}。
    """
    part_to_mat = _find_material_by_part(stage, char_prim_path)
    result: dict = {}

    for part, info in parts_appearance.items():
        rgb = info.get("rgb")
        if rgb is None:
            result[part] = "skip_default"
            continue
        mat_prim = part_to_mat.get(part)
        if mat_prim is None:
            result[part] = "no_material"
            continue
        used = _set_material_color(mat_prim, rgb)
        result[part] = f"recolored:{used}" if used else "failed"

    return result