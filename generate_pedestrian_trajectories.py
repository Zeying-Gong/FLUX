# generate_pedestrian_trajectories.py
"""
离线生成行人轨迹 JSON，不启动完整仿真，只需 NavMesh bake。
用法:
    python generate_pedestrian_trajectories.py \
        --usda /workspace/SAGE-3D_Official/SAGE-3D_data/usda/839873.usda \
        --num_people 3 \
        --num_waypoints 5 \
        --output /tmp/trajectories_839873.json
"""
from __future__ import annotations
import argparse, json, os, sys, random
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--usda",          type=str, required=True)
    p.add_argument("--num_people",    type=int, default=1)
    p.add_argument("--num_waypoints", type=int, default=3)
    p.add_argument("--output",        type=str, default="/tmp/pedestrian_trajectories.json")
    p.add_argument("--collision_root",type=str, default="/World/scene_collision")
    p.add_argument("--seed",          type=int, default=0)
    p.add_argument("--volume_padding",type=float, default=1.2)
    p.add_argument("--fallback_size", type=float, default=100.0)
    p.add_argument("--warmup_frames", type=int, default=60)
    p.add_argument("--semantic_map_json", type=str, default=None)
    return p.parse_args()

ARGS = parse_args()

from isaacsim import SimulationApp
_exp = os.environ.get("EXP_PATH")
if _exp is None:
    print("[FATAL] EXP_PATH not set"); sys.exit(1)

CUSTOM_APP_PATH = os.path.join(
    _exp, "isaacsim.exp.action_and_event_data_generation.base.kit")

simulation_app = SimulationApp(
    launch_config={"renderer": "RayTracedLighting",
                   "headless": True,   # 离线生成，headless 即可
                   "enable_cameras": False,
                   "crash_reporter/enabled": False},
    experience=CUSTOM_APP_PATH,
)

import carb, omni, omni.usd, omni.kit.app, omni.kit.commands
import omni.client
import omni.timeline
from pxr import Sdf, Usd, UsdGeom, Gf
from isaacsim.replicator.agent.core.settings import AssetPaths, PrimPaths
from omni.anim.people.settings import PeopleSettings
from isaacsim.replicator.agent.core.stage_util import CharacterUtil

def _update(n=1):
    for _ in range(n): simulation_app.update()

# ── 复用 test_navmesh_usda.py 里已有的工具函数 ──────────────────────────
# （直接粘贴或 import，此处省略重复代码，假设你把它们放到 navmesh_utils.py）
# from test_navmesh_usda import (
#     open_stage,
#     try_bake_navmesh,
#     safe_query_random_point,
#     find_reachable_point,
#     _path_is_valid,
# )


def open_stage(usda_path: str) -> bool:
    if not os.path.exists(usda_path):
        print(f"[FATAL] USDA not found: {usda_path}")
        return False
    print(f"[STAGE] Opening: {usda_path}")
    omni.usd.get_context().open_stage(usda_path)
    _update(30)
    waited = 0
    while omni.usd.get_context().get_stage_loading_status()[1] > 0:
        simulation_app.update()
        waited += 1
        if waited > 3000:
            print("[WARN] Stage still loading after 30s, continuing anyway.")
            break
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[FATAL] Stage failed to open.")
        return False
    print(f"[STAGE] Loaded. Default prim: {stage.GetDefaultPrim().GetPath()}")
    return True

def set_subtree_visibility(stage, root_path: str, visible: bool) -> str | None:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        print(f"[Vis] {root_path} not found.")
        return None
    imageable = UsdGeom.Imageable(root)
    if not imageable:
        print(f"[Vis] {root_path} is not Imageable.")
        return None
    attr = imageable.GetVisibilityAttr()
    prev = attr.Get() if attr and attr.HasAuthoredValue() else "inherited"
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()
    print(f"[Vis] {root_path}: {prev} → "
          f"{'inherited' if visible else 'invisible'}")
    return prev


def compute_scene_bbox(stage):
    try:
        root = stage.GetDefaultPrim() if stage.HasDefaultPrim() \
               else stage.GetPseudoRoot()
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        bbox = cache.ComputeWorldBound(root)
        rng  = bbox.ComputeAlignedRange()
        if rng.IsEmpty():
            return None
        return rng.GetMin(), rng.GetMax()
    except Exception as e:
        print(f"[BBox] failed: {e}")
        return None

def size_navmesh_volume(stage, volume_path: Sdf.Path,
                        padding: float, fallback_size: float):
    prim = stage.GetPrimAtPath(volume_path)
    if not prim.IsValid():
        print(f"[NavMesh] Volume prim not found at {volume_path}")
        return
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    half_extent_stage_units = 0.5 / mpu
    bbox = compute_scene_bbox(stage)
    if bbox is None:
        print(f"[NavMesh] Using fallback {fallback_size}m cube.")
        center   = Gf.Vec3d(0.0, 0.0, fallback_size * 0.25)
        half_ext = Gf.Vec3d(fallback_size*0.5, fallback_size*0.5, fallback_size*0.5)
    else:
        bmin, bmax = bbox
        center   = Gf.Vec3d((bmin[0]+bmax[0])*0.5, (bmin[1]+bmax[1])*0.5,
                            (bmin[2]+bmax[2])*0.5)
        half_ext = Gf.Vec3d((bmax[0]-bmin[0])*0.5*padding,
                            (bmax[1]-bmin[1])*0.5*padding,
                            (bmax[2]-bmin[2])*0.5*padding)
        print(f"[NavMesh] Scene bbox min={tuple(bmin)} max={tuple(bmax)}")
        print(f"[NavMesh] Volume center={tuple(center)} "
              f"half_extent(padded)={tuple(half_ext)}")
    scale = Gf.Vec3d(half_ext[0]/half_extent_stage_units,
                     half_ext[1]/half_extent_stage_units,
                     half_ext[2]/half_extent_stage_units)
    mat = Gf.Matrix4d(1.0)
    mat.SetScale(scale)
    mat = mat * Gf.Matrix4d(1.0).SetTranslate(center)
    omni.kit.commands.execute("TransformPrim",
                              path=volume_path, new_transform_matrix=mat)
    print(f"[NavMesh] NavMeshVolume resized. scale={tuple(scale)}")

def add_exclude_volumes_from_semantic_map(stage, semantic_map_json,
                                          flip_x=True, flip_y=True,
                                          negate_xy=True) -> int:
    import json as _json
    with open(semantic_map_json) as f:
        items = _json.load(f)

    exclude_label_set = ["table", "chair", "sofa", "bed", "wardrobe",
                            "car", "desk", "counter", "cabinet"]

    # ★ 用 mask_coords_m 算全局边界，和 trajectory_2d_to_3d.py 完全一致
    all_y, all_x = [], []
    for inst in items:
        for y, x in inst.get("mask_coords_m", []):
            try:
                all_y.append(float(y))
                all_x.append(float(x))
            except (ValueError, TypeError):
                continue
    
    if not all_x:
        print("[Exclude] ERROR: no mask_coords_m found")
        return 0

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    print(f"[Exclude] map bounds from mask_coords_m: "
          f"x∈[{min_x:.3f},{max_x:.3f}] y∈[{min_y:.3f},{max_y:.3f}]")

    def sm_to_isaac(sm_x, sm_y):
        """与 trajectory_2d_to_3d.flip_position 完全一致（先flip后negate）"""
        px, py = sm_x, sm_y
        if flip_x:
            px = (min_x + max_x) - px
        if flip_y:
            py = (min_y + max_y) - py
        if negate_xy:
            px = -px
            py = -py
        return px, py

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    half_extent_stage_units = 0.5 / mpu
    bbox = compute_scene_bbox(stage)
    created = 0
    skipped = 0

    for item in items:
        label = str(item.get("category_label", "")).lower()
        if not any(keyword in label for keyword in exclude_label_set):
            continue
        x_l, y_b, x_r, y_t = [float(v) for v in item["bbox_m"]]
        z_min = float(item["min_z_m"])
        z_max = float(item["max_z_m"])

        corners_sm = [(x_l, y_b), (x_l, y_t), (x_r, y_b), (x_r, y_t)]
        corners_isaac = [sm_to_isaac(cx, cy) for cx, cy in corners_sm]
        ix_vals = [c[0] for c in corners_isaac]
        iy_vals = [c[1] for c in corners_isaac]
        ix_l, ix_r = min(ix_vals), max(ix_vals)
        iy_b, iy_t = min(iy_vals), max(iy_vals)
        cx = 0.5*(ix_l + ix_r)
        cy = 0.5*(iy_b + iy_t)

        PAD = 0.10
        hx = 0.5*(ix_r - ix_l) + PAD
        hy = 0.5*(iy_t - iy_b) + PAD

        if bbox:
            bmin, bmax = bbox
            margin = 1.0
            if not (bmin[0]-margin <= cx <= bmax[0]+margin and
                    bmin[1]-margin <= cy <= bmax[1]+margin):
                skipped += 1
                continue

        bottom = max(z_min - 0.05, -0.01)
        top    = z_max + 0.05
        hz     = 0.5 * (top - bottom)
        cz     = 0.5 * (top + bottom)

        existing = {p.GetPath() for p in stage.Traverse()
                    if p.GetName().startswith("NavMeshVolume")}
        omni.kit.commands.execute("CreateNavMeshVolumeCommand",
                                  parent_prim_path=Sdf.Path.emptyPath,
                                  volume_type=1, usd_context_name="", layer=None)
        _update(1)
        new_prims = [p.GetPath() for p in stage.Traverse()
                     if p.GetName().startswith("NavMeshVolume")
                     and p.GetPath() not in existing]
        if not new_prims:
            continue
        vol_path = new_prims[0]

        scale  = Gf.Vec3d(hx / half_extent_stage_units,
                          hy / half_extent_stage_units,
                          hz / half_extent_stage_units)
        mat = Gf.Matrix4d(1.0)
        mat.SetScale(scale)
        mat = mat * Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(cx, cy, cz))
        omni.kit.commands.execute("TransformPrim", path=vol_path,
                                  new_transform_matrix=mat)
        created += 1

    _update(10)
    print(f"[Exclude] Created {created}, skipped {skipped}.")
    return created

def try_bake_navmesh():
    """Returns (success, nav_interface). Toggles collision_root visibility."""
    import omni.anim.navigation.core as nav

    settings = carb.settings.get_settings()
    # agentMinRadius: 默认 0.5m，室内场景建议 0.25m
    settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/config/agentMinRadius", 0.5)
    # agentMaxStepHeight: 允许小台阶（默认 0.3）
    # settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/config/agentMaxStepHeight", 0.25)
    # agentMinHeight: 默认 2.0，可适当降低
    settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/config/agentMinHeight", 1.8)
    # agentMinIslandRadius: 孤立小岛低于这个尺寸会被丢弃，降低以保留更多可达区域
    settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/config/agentMinIslandRadius", 0.5)
    # 关闭自动重烘焙，避免场景更新时意外触发
    settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/config/autoRebakeOnChanges", False)
    # 告诉 CharacterBehavior 使用 NavMesh 规划路径（默认 None = False）
    settings.set(PeopleSettings.NAVMESH_ENABLED, True)
    settings.set(PeopleSettings.DYNAMIC_AVOIDANCE_ENABLED, True)
    # loop 次数设为 0 表示不循环（我们用 runtime GoTo 替代）
    settings.set(PeopleSettings.NUMBER_OF_LOOP, "0")
    _update(5)

    stage = omni.usd.get_context().get_stage()

    prev_vis = set_subtree_visibility(stage, ARGS.collision_root, visible=True)
    _update(5)

    existing = {p.GetPath() for p in stage.Traverse()
                if p.GetName().startswith("NavMeshVolume")}
    try:
        omni.kit.commands.execute(
            "CreateNavMeshVolumeCommand",
            parent_prim_path=Sdf.Path.emptyPath,
            volume_type=0,
            usd_context_name="",
            layer=None,
        )
    except Exception as e:
        print(f"[NavMesh] CreateNavMeshVolumeCommand failed: {e}")
        return False, None
    _update(5)

    new_volumes = [p.GetPath() for p in stage.Traverse()
                   if p.GetName().startswith("NavMeshVolume")
                   and p.GetPath() not in existing]
    if not new_volumes:
        print("[NavMesh] Could not find new NavMeshVolume.")
        return False, None
    volume_path = new_volumes[0]
    print(f"[NavMesh] Created {volume_path}")

    size_navmesh_volume(stage, volume_path,
                        padding=ARGS.volume_padding,
                        fallback_size=ARGS.fallback_size)

    # Carve out furniture tops so characters don't climb onto tables/sofas
    if ARGS.semantic_map_json and os.path.exists(ARGS.semantic_map_json):
        add_exclude_volumes_from_semantic_map(
            stage, ARGS.semantic_map_json,
        )
    elif ARGS.semantic_map_json:
        print(f"[Exclude] semantic_map_json not found: {ARGS.semantic_map_json}")

    print(f"[NavMesh] Warming up for {ARGS.warmup_frames} frames before bake...")
    _update(ARGS.warmup_frames)

    inav = nav.acquire_interface()
    print("[NavMesh] Baking...")
    inav.start_navmesh_baking_and_wait()
    nm = inav.get_navmesh()

    if ARGS.semantic_map_json and os.path.exists(ARGS.semantic_map_json):
        stage = omni.usd.get_context().get_stage()
        
    rp = safe_query_random_point(nm)
    print(f"[NavMesh] OK. Sample random point: {tuple(rp)}")

    success = True

    # if prev_vis == "invisible":
    set_subtree_visibility(stage, ARGS.collision_root, visible=False)
    _update(2)

    return success, inav


def _path_is_valid(path) -> bool:
    """兼容 INavMeshPath 对象和列表两种返回形式"""
    if path is None:
        return False
    # INavMeshPath 对象：用 get_points() 或 points 属性
    if hasattr(path, 'get_points'):
        pts = path.get_points()
        return pts is not None and len(pts) > 0
    if hasattr(path, 'points'):
        return path.points is not None and len(path.points) > 0
    # 兼容直接返回列表的情况
    try:
        return len(path) > 0
    except TypeError:
        # 对象存在但无 len，保守认为有效（说明路径找到了）
        return True

def find_reachable_point(nm, from_xyz, max_attempts=50):
    """从 from_xyz 出发，找一个 NavMesh 上可达的随机目标点"""
    from_pt = carb.Float3(float(from_xyz[0]), float(from_xyz[1]), float(from_xyz[2]))
    for i in range(max_attempts):
        candidate = safe_query_random_point(nm)
        if candidate is None:
            continue
        to_pt = carb.Float3(float(candidate[0]), float(candidate[1]), float(candidate[2]))
        try:
            path = nm.query_shortest_path(from_pt, to_pt, agent_radius=0.5)
            if _path_is_valid(path):
                return candidate
        except Exception:
            continue
    print(f"[NavMesh] WARNING: No reachable point found from {from_xyz[:2]} "
          f"after {max_attempts} attempts — NavMesh may be disconnected")
    return safe_query_random_point(nm)  # fallback

def safe_query_random_point(nm, max_z=0.3, retries=30):
    """安全的 NavMesh 随机点采样，过滤家具面高度点，防递归崩溃"""
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(500)  # 提前截断，避免真正崩溃
    try:
        best = None
        for _ in range(retries):
            try:
                p = nm.query_random_point()
                z = float(p[2])
                if z < max_z:
                    return p
                if best is None:
                    best = p  # 备用：至少返回个点
            except RecursionError:
                print("[WARN] query_random_point recursion, NavMesh may be damaged")
                break
            except Exception as e:
                print(f"[WARN] query_random_point failed: {e}")
                break
        return best
    finally:
        sys.setrecursionlimit(old_limit)

def build_trajectories(nm, num_people, num_waypoints, seed):
    random.seed(seed); np.random.seed(seed)
    
    spawn_positions = {}
    commands = {}
    
    for i in range(num_people):
        # char_name = f"Character" if i == 0 else f"Character_{i:02d}"
        char_name = CharacterUtil.get_character_name_by_index(i)
        spawn = safe_query_random_point(nm)
        if spawn is None:
            continue
        
        spawn_positions[char_name] = {
            "pos": [float(spawn[0]), float(spawn[1]), float(spawn[2])],
            "rot": 0.0
        }
        
        # 链式生成 waypoints → GoTo 命令
        char_commands = []
        prev = spawn
        for _ in range(num_waypoints):
            pt = find_reachable_point(nm, prev, max_attempts=50)
            if pt is None:
                break
            char_commands.append({
                "cmd": "GoTo",
                "params": [f"{float(pt[0]):.4f}", f"{float(pt[1]):.4f}", f"{float(pt[2]):.4f}", "_"]
            })
            prev = pt
        
        commands[char_name] = char_commands
        print(f"[GEN] {char_name}: spawn={tuple(spawn)}, {len(char_commands)} waypoints")
    
    return {
        "episode": {
            "episode_id": 0,
            "robot": {
                "start_pos": [0.0, 0.0, 0.0],
                "goal_pos":  [0.0, 0.0, 0.0],
                "start_orientation": 0.0
            },
            "characters": {
                "num_characters": len(spawn_positions),
                "spawn_positions": spawn_positions,
                "commands": commands
            }
        }
    }



# def main():
#     if not open_stage(ARGS.usda):
#         sys.exit(1)

#     nav_ok, inav = try_bake_navmesh()
#     if not nav_ok:
#         print("[FATAL] NavMesh bake failed"); sys.exit(2)

#     nm = inav.get_navmesh()
#     trajectories = build_trajectories(
#         nm, ARGS.num_people, ARGS.num_waypoints, ARGS.seed)

#     os.makedirs(os.path.dirname(os.path.abspath(ARGS.output)), exist_ok=True)
#     with open(ARGS.output, "w") as f:
#         json.dump({"usda": ARGS.usda, "trajectories": trajectories}, f, indent=2)
#     print(f"[GEN] Saved {len(trajectories)} trajectories → {ARGS.output}")

#     simulation_app.close()

def main():
    if not open_stage(ARGS.usda):
        sys.exit(1)

    nav_ok, inav = try_bake_navmesh()
    if not nav_ok:
        print("[FATAL] NavMesh bake failed"); sys.exit(2)

    nm = inav.get_navmesh()
    episode = build_trajectories(nm, ARGS.num_people, ARGS.num_waypoints, ARGS.seed)

    os.makedirs(os.path.dirname(os.path.abspath(ARGS.output)), exist_ok=True)
    with open(ARGS.output, "w") as f:
        json.dump(episode, f, indent=2)
    print(f"[GEN] Saved → {ARGS.output}")

    simulation_app.close()
if __name__ == "__main__":
    main()