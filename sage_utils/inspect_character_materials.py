"""
inspect_character_materials.py
───────────────────────────────
扫描 Isaac Sim People 资产目录下的所有角色 USD,
列出每个角色 /Looks/ 下的 material prims,
检查 material 命名规约 (opacity__type__part) 是否统一,
以及 shader 输入(用于 Color Tint 的 diffuse_tint 等)是否可用。

不启动 Isaac Sim,纯 USD 读取。

用法:
    python inspect_character_materials.py \\
        --root /workspace/FLUX/assets/isaacsim_assets/Assets/Isaac/4.5/Isaac/People/Characters \\
        --out /workspace/FLUX/sage_utils/character_material_report.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# 必须在任何 pxr/omni 导入之前启动 SimulationApp
from isaacsim import SimulationApp
_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

from pxr import Usd, UsdShade

EXCLUDED_FOLDERS = {"biped_demo"}


def find_character_usds(root: Path) -> list[tuple[str, Path]]:
    """返回 [(character_name, usd_path), ...]"""
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


def find_looks_scopes(stage: Usd.Stage) -> list[Usd.Prim]:
    """找出 stage 里所有名为 Looks 的 prim(Scope 或 Xform)"""
    looks = []
    for prim in stage.Traverse():
        if prim.GetName() == "Looks":
            looks.append(prim)
    return looks


def inspect_material(mat_prim: Usd.Prim) -> dict[str, Any]:
    """提取一个 material prim 的关键信息"""
    info: dict[str, Any] = {
        "name": mat_prim.GetName(),
        "path": str(mat_prim.GetPath()),
        "shaders": [],
    }
    mat = UsdShade.Material(mat_prim)
    if not mat:
        return info

    # 收集 surface output 连到的 shader
    surface_output = mat.GetSurfaceOutput()
    shader_prims: list[Usd.Prim] = []
    if surface_output:
        src, _, _ = surface_output.GetConnectedSource() or (None, None, None)
        if src is not None:
            shader_prims.append(src.GetPrim())

    # 兜底:把 Material 内所有 Shader 类型 prim 也加进来
    for child in mat_prim.GetChildren():
        if child.GetTypeName() == "Shader" and child not in shader_prims:
            shader_prims.append(child)

    for shader_prim in shader_prims:
        shader = UsdShade.Shader(shader_prim)
        if not shader:
            continue
        shader_info: dict[str, Any] = {
            "shader_prim": str(shader_prim.GetPath()),
            "shader_id": None,
            "mdl_source_asset": None,
            "color_inputs": [],
        }
        # 1) UsdPreviewSurface 类型用 info:id
        id_attr = shader_prim.GetAttribute("info:id")
        if id_attr and id_attr.HasAuthoredValue():
            shader_info["shader_id"] = id_attr.Get()

        # 2) MDL 类型用 info:mdl:sourceAsset
        mdl_attr = shader_prim.GetAttribute("info:mdl:sourceAsset")
        if mdl_attr and mdl_attr.HasAuthoredValue():
            try:
                shader_info["mdl_source_asset"] = str(mdl_attr.Get())
            except Exception:
                pass

        # 3) 找出能用于颜色调整的输入
        color_candidates = (
            "diffuse_tint",
            "diffuseColor",
            "diffuse_color_constant",
            "tint",
            "base_color",
            "albedo_tint",
        )
        for inp in shader.GetInputs():
            inp_name = inp.GetBaseName()
            if inp_name in color_candidates:
                val = None
                try:
                    val = inp.Get()
                    if val is not None:
                        val = tuple(val) if hasattr(val, "__iter__") else val
                except Exception:
                    pass
                shader_info["color_inputs"].append({
                    "name": inp_name,
                    "type": str(inp.GetTypeName()),
                    "value": val,
                })

        info["shaders"].append(shader_info)

    return info


def inspect_character(name: str, usd_path: Path) -> dict[str, Any]:
    """打开一个角色 USD,提取 Looks 下所有 material 信息"""
    result: dict[str, Any] = {
        "character": name,
        "usd_path": str(usd_path),
        "looks_scopes": [],
        "materials": [],
        "parts": [],          # 解析出的部位名(opacity__type__part 的 part)
        "naming_ok": True,    # 全部命名都符合 opacity__type__part
        "error": None,
    }
    try:
        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            result["error"] = "Usd.Stage.Open returned None"
            return result

        looks_prims = find_looks_scopes(stage)
        result["looks_scopes"] = [str(p.GetPath()) for p in looks_prims]

        seen_paths = set()
        for looks in looks_prims:
            for child in looks.GetChildren():
                if child.GetTypeName() != "Material":
                    continue
                if str(child.GetPath()) in seen_paths:
                    continue
                seen_paths.add(str(child.GetPath()))

                mat_info = inspect_material(child)
                result["materials"].append(mat_info)

                # 解析命名 opacity__type__part
                tokens = mat_info["name"].split("__")
                if len(tokens) >= 3:
                    result["parts"].append(tokens[-1])
                else:
                    result["naming_ok"] = False
                    result["parts"].append(f"<RAW:{mat_info['name']}>")
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def summarize(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """跨角色汇总:哪些 part 名最常见,哪些角色异常"""
    part_counter: Counter = Counter()
    part_to_chars: dict[str, list[str]] = defaultdict(list)
    shader_id_counter: Counter = Counter()
    color_input_counter: Counter = Counter()
    naming_bad: list[str] = []
    errored: list[str] = []
    no_materials: list[str] = []

    for r in reports:
        if r["error"]:
            errored.append(r["character"])
            continue
        if not r["materials"]:
            no_materials.append(r["character"])
        if not r["naming_ok"]:
            naming_bad.append(r["character"])
        for p in r["parts"]:
            part_counter[p] += 1
            part_to_chars[p].append(r["character"])
        for m in r["materials"]:
            for s in m["shaders"]:
                if s["shader_id"]:
                    shader_id_counter[s["shader_id"]] += 1
                if s["mdl_source_asset"]:
                    shader_id_counter[f"MDL:{s['mdl_source_asset']}"] += 1
                for ci in s["color_inputs"]:
                    color_input_counter[ci["name"]] += 1

    return {
        "num_characters": len(reports),
        "errored": errored,
        "no_materials": no_materials,
        "naming_inconsistent": naming_bad,
        "part_frequency": dict(part_counter.most_common()),
        "shader_id_frequency": dict(shader_id_counter.most_common()),
        "color_input_frequency": dict(color_input_counter.most_common()),
        "parts_per_character_unique": {
            p: sorted(set(chars)) for p, chars in part_to_chars.items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", required=True, type=Path,
        help="Characters folder, e.g. .../Isaac/4.5/Isaac/People/Characters",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("character_material_report.json"),
        help="Output JSON path",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of characters (0 = all)",
    )
    args = ap.parse_args()

    chars = find_character_usds(args.root)
    if args.limit > 0:
        chars = chars[: args.limit]
    print(f"[INFO] Found {len(chars)} character folders under {args.root}")

    reports: list[dict[str, Any]] = []
    for i, (name, usd_path) in enumerate(chars, 1):
        print(f"[{i:3d}/{len(chars)}] {name}")
        reports.append(inspect_character(name, usd_path))

    summary = summarize(reports)

    # 打印精简摘要
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total characters         : {summary['num_characters']}")
    print(f"Errored                  : {len(summary['errored'])}")
    print(f"No materials found       : {len(summary['no_materials'])}")
    print(f"Naming inconsistent      : {len(summary['naming_inconsistent'])}")
    print("\nTop part names (across all characters):")
    for p, n in list(summary["part_frequency"].items())[:30]:
        print(f"  {n:4d}  {p}")
    print("\nShader id / MDL frequency:")
    for s, n in list(summary["shader_id_frequency"].items())[:20]:
        print(f"  {n:4d}  {s}")
    print("\nColor input frequency:")
    for c, n in summary["color_input_frequency"].items():
        print(f"  {n:4d}  {c}")
    if summary["errored"]:
        print(f"\nErrored characters: {summary['errored'][:10]}")
    if summary["naming_inconsistent"]:
        print(f"Naming inconsistent: {summary['naming_inconsistent'][:10]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({"summary": summary, "per_character": reports}, f, indent=2)
    print(f"\n[OK] Full report written to {args.out}")


if __name__ == "__main__":
    main()
    _app.close()