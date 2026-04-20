"""People simulation for dynamic pedestrians in the scene - Modified to load from episode JSON."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import carb
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Sdf, UsdGeom, Gf
import NavSchema

from isaaclab.utils.configclass import configclass
import numpy as np
# Isaac Sim paths
ISAAC_SIM_PATH = "/isaac-sim"
sys.path.append(f"{ISAAC_SIM_PATH}/extscache/isaacsim.replicator.agent.core-0.5.14+106.5.0")

from isaacsim.replicator.agent.core.stage_util import CharacterUtil, StageUtil
from isaacsim.core.utils import prims
from omni.anim.people.scripts.custom_command.populate_anim_graph import populate_anim_graph
from isaacsim.replicator.agent.core.settings import AssetPaths, PrimPaths, BehaviorScriptPaths, Settings
from omni.metropolis.utils.semantics_util import SemanticsUtils
# 角色资源路径
# CHARACTER_ASSET_PATH = "/workspace/NavDP/assets/isaacsim_assets/Assets/Isaac/4.5/Isaac/People/Characters"

class PeopleSimulation:
    """Manages dynamic pedestrian simulation by loading from episode JSON.
    
    This class loads character positions and commands from an episode JSON file,
    similar to people_dataset_test.py but integrated into the IsaacLab workflow.
    """
    
    def __init__(self, episode_json_path: str, enable_dynamic_target: bool = False):
        """Initialize people simulation from episode JSON.
        
        Args:
            episode_json_path: Path to the episode JSON file containing spawn positions and commands.
        """
        self._episode_json_path = episode_json_path
        self._episode_data = None
        self._setup_complete = False
        self._setup_failed = False
        self._available_character_list = None
        
        # Timeline control
        self._timeline_started = False

        # Dyn PointNav
        self._enable_dynamic_target = enable_dynamic_target
        
        print(f"[INFO] PeopleSimulation: Initialized with episode JSON: {episode_json_path}")
    
    def _cleanup_existing_characters(self):
        from isaacsim.replicator.agent.core.settings import PrimPaths
        from omni.anim.people.scripts.global_character_position_manager import GlobalCharacterPositionManager
        
        # 清空 position manager 的旧数据
        try:
            char_manager = GlobalCharacterPositionManager.get_instance()
            char_manager._character_positions = {}
            char_manager._character_future_positions = {}
            char_manager._character_radius = {}
        except Exception as e:
            carb.log_warn(f"Failed to clear GlobalCharacterPositionManager: {e}")
        
        # 删除 character prims
        stage = omni.usd.get_context().get_stage()
        parent_path = PrimPaths.characters_parent_path()
        parent_prim = stage.GetPrimAtPath(parent_path)
        # 删 character prims —— 但保留 Biped_Setup,因为它是 AG 模板的承载者
        # 删了 biped 会导致下个 episode 的 character 拿不到 AG → init_character 失败 → 不动
        from isaacsim.replicator.agent.core.settings import PrimPaths as _PrimPaths
        biped_path_str = str(_PrimPaths.biped_prim_path())
        
        # 删 parent_path 下面除了 biped 之外的所有 child
        if parent_prim.IsValid():
            children_to_delete = []
            for child in parent_prim.GetChildren():
                child_path_str = str(child.GetPath())
                if child_path_str != biped_path_str:
                    children_to_delete.append(child_path_str)
            
            if children_to_delete:
                print(f"[PeopleSim cleanup] Deleting {len(children_to_delete)} character prims (preserving biped)")
                omni.kit.commands.execute("DeletePrimsCommand", paths=children_to_delete)
                
    async def setup_async(self):
        """Asynchronously setup people simulation.
        
        This must be called after the scene and NavMesh are ready.
        """
        # 停止 timeline，让新 episode 从 t=0 开始
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        timeline.set_current_time(0.0)

        self._cleanup_existing_characters()
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()
        try:
            # Step 1: Enable required extensions
            self._enable_extensions()
            await omni.kit.app.get_app().next_update_async()
            
            # Step 2: Configure settings
            self._configure_settings()
            await omni.kit.app.get_app().next_update_async()
            
            # Step 3: Load episode JSON
            print("[INFO] PeopleSimulation: Loading episode JSON...")
            if not self._load_episode_json():
                raise RuntimeError("Failed to load episode JSON")
            await omni.kit.app.get_app().next_update_async()
            
            # Step 4: Setup characters
            print("[INFO] PeopleSimulation: Setting up characters...")
            await self._setup_characters()

            self._setup_complete = True
            print("[INFO] PeopleSimulation: Setup complete ✓")
            
        except Exception as e:
            carb.log_error(f"PeopleSimulation setup failed: {e}")
            import traceback
            carb.log_error(traceback.format_exc())
            self._setup_failed = True
    
    def _enable_extensions(self):
        """Enable required Isaac Sim extensions."""
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        
        required_extensions = [
            "omni.anim.timeline",
            "omni.anim.graph.core",
            "omni.anim.retarget.core",
            "omni.anim.navigation.core",
            # "omni.anim.navigation.recast",
            "omni.anim.navigation.schema",
            "omni.anim.people",
            "isaacsim.replicator.agent.core",
        ]
        
        for ext in required_extensions:
            if not ext_manager.is_extension_enabled(ext):
                ext_manager.set_extension_enabled_immediate(ext, True)
        
        carb.log_info("PeopleSimulation: Extensions enabled")
    
    def _configure_settings(self):
        """Configure global settings for people simulation."""
        settings = carb.settings.get_settings()
        
        # Navigation and animation settings
        settings.set("/app/scripting/ignoreWarningDialog", True)
        settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
        settings.set("/app/omni.graph.scriptnode/enable_opt_in", False)
        settings.set("/rtx/raytracing/fractionalCutoutOpacity", True)
        
        carb.log_info("PeopleSimulation: Settings configured")
    
    def _load_episode_json(self) -> bool:
        """Load episode JSON file.
        
        Returns:
            True if successful, False otherwise.
        """
        if not os.path.exists(self._episode_json_path):
            carb.log_error(f"Episode JSON not found: {self._episode_json_path}")
            return False
        
        try:
            with open(self._episode_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._episode_data = data.get("episode")
            if not self._episode_data:
                carb.log_error("Invalid episode JSON format")
                return False
            
            carb.log_info(f"Episode loaded: ID={self._episode_data.get('episode_id')}, "
                         f"Characters={self._episode_data['characters'].get('num_characters')}")
            
            return True
            
        except Exception as e:
            carb.log_error(f"Failed to load episode JSON: {e}")
            return False
    
    async def _setup_characters(self):
        """Setup all characters from episode data."""
        try:
            # Load characters
            self._load_characters_from_episode()
            
            # Wait for characters to be created
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()
            
            # Setup all character components
            self._setup_all_characters()
            
            # Wait for setup to complete
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()
            
        except Exception as e:
            carb.log_error(f"Failed to setup characters: {e}")
            import traceback
            carb.log_error(traceback.format_exc())
            raise
    
    def _load_characters_from_episode(self):
        """Load characters based on episode data."""
        character_count = self._episode_data['characters'].get('num_characters')
        if character_count is None:
            carb.log_error("Invalid character count in episode data")
            return

        stage = omni.usd.get_context().get_stage()
        parent_path = PrimPaths.characters_parent_path()
        spawn_positions = self._episode_data['characters'].get("spawn_positions", {})
        if not spawn_positions:  # 提前判空，避免后续无效逻辑
            carb.log_error("[DynPointGoal] No characters found in episode")
            return

        # ===== 动态目标筛选：单次遍历+提前终止，无冗余 =====
        if self._enable_dynamic_target:
            robot_data = self._episode_data.get("robot")
            if not robot_data:
                carb.log_error("No robot data in episode")
                return
            
            robot_start = np.array(robot_data.get("start_pos", [0, 0, 0]))[:2]
            MIN_DISTANCE, MAX_DISTANCE = 5.0, 30.0
            target_character_name = None
            target_distance = 0.0
            target_spawn_data = None
            char_list = list(spawn_positions.items())  # 转列表方便索引/取值

            # 单次遍历：找第一个有效行人，找不到则取第一个行人（一次循环完成，无二次遍历）
            for idx, (char_name, spawn_data) in enumerate(char_list):
                char_pos = np.array(spawn_data.get('pos', [0, 0, 0]))[:2]
                distance = np.linalg.norm(char_pos - robot_start)
                # 找到第一个有效行人，立即记录并终止遍历（提前终止，提升性能）
                if MIN_DISTANCE <= distance <= MAX_DISTANCE:
                    target_character_name = char_name
                    target_distance = distance
                    target_spawn_data = spawn_data
                    print(f"[DynPointGoal] Selected target: {target_character_name} (distance: {target_distance:.2f}m)")
                    break
                # 若为第一个行人（无论是否有效），先记录为兜底值
                elif idx == 0:
                    target_character_name = char_name
                    target_distance = distance
                    target_spawn_data = spawn_data

            # 无有效行人时，输出兜底警告（复用首次遍历的兜底值，无二次计算）
            if not (MIN_DISTANCE <= target_distance <= MAX_DISTANCE):
                carb.log_warn(f"[DynPointGoal] No character in valid range, using: {target_character_name} (distance: {target_distance:.2f}m)")

            # ===== 生成目标行人 =====
            character_path = f"{parent_path}/{target_character_name}"
            character_prim = stage.GetPrimAtPath(character_path)
            if not character_prim.IsValid():
                spawn_pos = target_spawn_data.get('pos', [0, 0, 0])
                spawn_rot = target_spawn_data.get('rot', 0.0)
                print(f"Spawning target {target_character_name} at {spawn_pos}")
                self._spawn_character_by_idx(spawn_pos, spawn_rot, idx)

                print(f"\n{'='*80}")
                print(f"DEBUG: Character Selection Details")
                print(f"{'='*80}")
                print(f"Robot start from JSON: {robot_start}")
                print(f"Selected character: {target_character_name}")
                print(f"Character position from JSON: {spawn_pos}")
                print(f"{'='*80}\n")

            return  # 仅生成目标行人,直接返回
            
        for i in range(character_count):
            character_name = CharacterUtil.get_character_name_by_index(i)
            character_path = f"{parent_path}/{character_name}"
            character_prim = stage.GetPrimAtPath(character_path)
            
            if not character_prim.IsValid():
                spawn_data = spawn_positions.get(character_name)
                if not spawn_data:
                    carb.log_warn(f"No spawn data for {character_name}, skipping")
                    continue
                
                spawn_pos = spawn_data.get('pos', [0, 0, 0])
                spawn_rot = spawn_data.get('rot', 0.0)
                
                carb.log_info(f"[DEBUG] Spawning {character_name}:")
                carb.log_info(f"  JSON pos: {spawn_pos}")
                carb.log_info(f"  JSON rot: {spawn_rot}")
                
                # Spawn character
                character_prim = self._spawn_character_by_idx(spawn_pos, spawn_rot, i)
                
                # ===== 新增：验证spawn后的实际世界坐标 =====
                if character_prim:
                    from pxr import UsdGeom
                    xformable = UsdGeom.Xformable(character_prim)
                    transform_matrix = xformable.ComputeLocalToWorldTransform(0)
                    actual_world_pos = transform_matrix.ExtractTranslation()
                    carb.log_info(f"  Actual world pos: {actual_world_pos}")
                    
                    # 检查坐标是否匹配
                    diff = [
                        abs(actual_world_pos[0] - spawn_pos[0]),
                        abs(actual_world_pos[1] - spawn_pos[1]),
                        abs(actual_world_pos[2] - spawn_pos[2])
                    ]
                    if max(diff) > 0.1:
                        carb.log_error(f"  ⚠️ Position mismatch! Diff: {diff}")
    
    # def _spawn_character_by_name(self, spawn_location, spawn_rotation, idx, char_name):
    #         """Spawn a character with an explicit name (instead of CharacterUtil default name).
            
    #         Args:
    #             spawn_location: [x, y, z] position
    #             spawn_rotation: Rotation in degrees (yaw)
    #             idx: Character index (used for picking asset from _available_character_list)
    #             char_name: Explicit prim name (e.g. "Character_ep2") — must be unique per episode
                
    #         Returns:
    #             The spawned character prim, or None if failed.
    #         """
    #         spawn_location = carb.Float3(*spawn_location)
            
    #         if self._available_character_list is None:
    #             self._available_character_list = self._load_character_assets()
            
    #         list_len = len(self._available_character_list)
    #         if list_len == 0:
    #             carb.log_error("No character assets found")
    #             return None
            
    #         asset_idx = idx % list_len
    #         character_usd_path = self._available_character_list[asset_idx]
            
    #         return CharacterUtil.load_character_usd_to_stage(
    #             character_usd_path, spawn_location, spawn_rotation, char_name
    #         )
        
    def _spawn_character_by_idx(self, spawn_location, spawn_rotation, idx):
        """Spawn a character at the specified location.
        
        Args:
            spawn_location: [x, y, z] position
            spawn_rotation: Rotation in degrees (yaw)
            idx: Character index
            
        Returns:
            The spawned character prim, or None if failed.
        """
        spawn_location = carb.Float3(*spawn_location)
        
        # Load available character assets if not already loaded
        if self._available_character_list is None:
            self._available_character_list = self._load_character_assets()
        
        char_name = CharacterUtil.get_character_name_by_index(idx)
        list_len = len(self._available_character_list)
        
        if list_len == 0:
            carb.log_error("No character assets found")
            return None
        
        # Cycle through available characters
        asset_idx = idx % list_len
        character_usd_path = self._available_character_list[asset_idx]
        
        # Spawn character
        return CharacterUtil.load_character_usd_to_stage(
            character_usd_path, spawn_location, spawn_rotation, char_name
        )
    
    def _setup_all_characters(self):
        """Setup animation graphs, scripts, and semantics for all characters."""
        # Load default skeleton and animations
        self._load_default_skeleton_and_animations()
        
        # Get all SkelRoot prims
        skelroot_prim_list = CharacterUtil.get_characters_in_stage()
        
        # Setup animation graph
        self._setup_animation_graph_to_character(skelroot_prim_list)
        
        # Setup behavior scripts with commands from episode data
        self._setup_python_scripts_to_skelroot(skelroot_prim_list)
        
        # Setup semantics
        character_skelroot_list = CharacterUtil.get_characters_skelroot_list()
        SemanticsUtils.add_update_prim_metrosim_semantics(
            character_skelroot_list, type_value="class", name="character"
        )
    
    def _setup_python_scripts_to_skelroot(self, skelroot_prim_list):
        """Add behavior scripts to SkelRoot prims and set commands from episode data."""
        commands_dict = self._episode_data['characters'].get("commands", {})
            
        for skelroot_prim in skelroot_prim_list:
            skelroot_path = str(skelroot_prim.GetPrimPath())
            
            # Extract character name from path
            parent_path = PrimPaths.characters_parent_path()
            path_parts = skelroot_path.split('/')
            parent_parts = parent_path.split('/')
            
            if len(path_parts) > len(parent_parts):
                char_name = path_parts[len(parent_parts)]
            else:
                carb.log_warn(f"Cannot extract character name from {skelroot_path}")
                continue
            
            carb.log_info(f"\n{'='*60}")
            carb.log_info(f"Setting up commands for {char_name}")
            carb.log_info(f"{'='*60}")
            
            # Apply scripting API
            if not skelroot_prim.HasAttribute("omni:scripting:scripts"):
                try:
                    omni.kit.commands.execute(
                        "ApplyScriptingAPICommand", 
                        paths=[Sdf.Path(skelroot_path)]
                    )
                except Exception as e:
                    carb.log_warn(f"Failed to apply ScriptingAPI to {skelroot_path}: {e}")
                    continue
            
            # Set script path
            attr = skelroot_prim.GetAttribute("omni:scripting:scripts")
            script_path = BehaviorScriptPaths.behavior_script_path()
            attr.Set([r"{}".format(script_path)])
            
            # Get commands for this character
            char_commands = commands_dict.get(char_name, [])
            
            if not char_commands:
                carb.log_warn(f"No commands for {char_name}")
                continue
            
            carb.log_info(f"Processing {len(char_commands)} commands for {char_name}")
            
            # ===== 关键修改：为每个角色独立处理命令和路径 =====
            command_strings = []
            path_data = {}  # 存储 {goto_index: path}
            
            goto_index = 0  # ← 每个角色从 0 开始计数
            
            for cmd_idx, cmd in enumerate(char_commands):
                cmd_name = cmd.get("cmd")
                params = cmd.get("params", [])
                
                # 构建命令字符串
                cmd_str = f"{cmd_name} " + " ".join(str(p) for p in params)
                command_strings.append(cmd_str)
                
                carb.log_info(f"  Command {cmd_idx}: {cmd_str[:80]}...")
                
                # ===== 如果是 GoTo 命令，处理预计算路径 =====
                if cmd_name == "GoTo":
                    if "path" in cmd and len(cmd["path"]) > 0:
                        path_data[goto_index] = cmd["path"]
                        carb.log_info(f"    → Stored path for GoTo#{goto_index} ({len(cmd['path'])} points)")
                    else:
                        carb.log_warn(f"    → No precomputed path for GoTo#{goto_index}")
                    
                    goto_index += 1  # ← GoTo 计数器递增
            
            carb.log_info(f"Total: {len(command_strings)} commands, {len(path_data)} GoTo paths")
            
            # ===== 设置命令数据到 scriptData =====
            script_data_attr = skelroot_prim.GetAttribute("omni:scripting:scriptData")
            if script_data_attr:
                script_data_attr.Set(command_strings)
            else:
                script_data_attr = skelroot_prim.CreateAttribute(
                    "omni:scripting:scriptData", 
                    Sdf.ValueTypeNames.StringArray
                )
                script_data_attr.Set(command_strings)
            
            # ===== 设置路径数据到 pathData =====
            if path_data:
                import json
                path_json = json.dumps(path_data)
                
                path_attr = skelroot_prim.GetAttribute("omni:scripting:pathData")
                if path_attr:
                    path_attr.Set(path_json)
                else:
                    path_attr = skelroot_prim.CreateAttribute(
                        "omni:scripting:pathData",
                        Sdf.ValueTypeNames.String
                    )
                    path_attr.Set(path_json)
                
                print(f"✓ Saved {len(path_data)} precomputed paths for {char_name}")
            else:
                carb.log_info(f"✓ No precomputed paths for {char_name}")
            
            carb.log_info(f"{'='*60}\n")
    
    def _setup_animation_graph_to_character(self, character_list: list):
        """Add animation graph to all characters."""
        stage = omni.usd.get_context().get_stage()
        
        # Get animation graph from default biped
        default_biped_prim = PrimPaths.biped_prim_path()
        anim_graph_prim = CharacterUtil.get_anim_graph_from_character(
            stage.GetPrimAtPath(default_biped_prim)
        )
        
        if anim_graph_prim is None:
            carb.log_error("Unable to find animation graph on stage")
            return
        
        # Apply animation graph to each character
        for prim in character_list:
            # Remove existing animation graph if present
            try:
                if prim.GetTypeName() == "SkelRoot":
                    omni.kit.commands.execute(
                        "RemoveAnimationGraphAPICommand", 
                        paths=[Sdf.Path(prim.GetPrimPath())]
                    )
            except Exception:
                pass
            
            # Apply animation graph
            omni.kit.commands.execute(
                "ApplyAnimationGraphAPICommand",
                paths=[Sdf.Path(prim.GetPrimPath())],
                animation_graph_path=Sdf.Path(anim_graph_prim.GetPrimPath()),
            )
    
    def _load_default_skeleton_and_animations(self):
        """Load default biped skeleton and animations."""
        stage = omni.usd.get_context().get_stage()
        parent_path = PrimPaths.characters_parent_path()
        
        if not stage.GetPrimAtPath(parent_path):
            prims.create_prim(parent_path, "Xform")
        
        if Settings.skip_biped_setup():
            return
        
        biped_prim_path = PrimPaths.biped_prim_path()
        biped_prim = stage.GetPrimAtPath(biped_prim_path)
        
        # 关键修改:不仅检查 prim 是否存在,还检查 prim 是否 valid 且未被销毁
        # _cleanup_existing_characters 删了整个 parent path,biped 也跟着没了,
        # 这时 GetPrimAtPath 可能返回一个"已销毁"的 prim 但 valid 检查不严
        biped_needs_recreate = not biped_prim or not biped_prim.IsValid()
        
        if biped_needs_recreate:
            print(f"[PeopleSim] Recreating biped at {biped_prim_path}")
            prim = prims.create_prim(
                biped_prim_path,
                "Xform",
                usd_path=AssetPaths.default_biped_asset_path(),
            )
            prim.GetAttribute("visibility").Set("invisible")
            # 关键:重新创建 biped 后必须重跑 populate_anim_graph
            populate_anim_graph()
            print(f"[PeopleSim] populate_anim_graph re-run after biped recreate")
        else:
            # biped 还在,正常走原逻辑
            populate_anim_graph()
    
    def _load_character_assets(self):
        """Load available character asset list.
        
        Returns:
            List of character USD file paths.
        """
        import omni.client
        
        assets_root_path = AssetPaths.default_character_path() # CHARACTER_ASSET_PATH, local path
        
        # List all folders
        result, folder_list = omni.client.list(f"{assets_root_path}/")
        if result != omni.client.Result.OK:
            return []
        
        # 定义需要排除的文件夹名称（可扩展）
        EXCLUDED_FOLDERS = {"biped_demo"}  # 使用集合提高查找效率
        
        # Filter folders containing .usd files
        character_assets = []
        for folder in folder_list:
            # 跳过非文件夹项
            if not (folder.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN):
                continue
            # 跳过隐藏文件夹
            if folder.relative_path.startswith("."):
                continue
            # 排除指定的biped_demo文件夹
            if folder.relative_path in EXCLUDED_FOLDERS:
                continue
            
            folder_path = f"{assets_root_path}/{folder.relative_path}"
            
            # Check if folder contains .usd files
            result, file_list = omni.client.list(f"{folder_path}/")
            if result == omni.client.Result.OK:
                for file in file_list:
                    if file.relative_path.endswith((".usd", ".usda")):
                        character_assets.append(f"{folder_path}/{file.relative_path}")
                        break
        
        return character_assets
    
    def update(self, dt: float):
        """Update people simulation.
        
        Args:
            dt: Time step in seconds.
        """
        # Characters are animated by timeline and animation graph
        # No additional update needed
        pass
    
    def reset(self):
        """Reset people simulation."""
        # For now, we don't need to reset anything
        # Commands are loaded from JSON and repeat via timeline looping
        pass
    
    def _get_num_people(self) -> int:
        """Get number of people in simulation."""
        if self._episode_data:
            return self._episode_data['characters'].get('num_characters', 0)
        return 0
    
    @property
    def is_ready(self) -> bool:
        """Whether people simulation is ready."""
        return self._setup_complete and not self._setup_failed
    
    @property
    def num_people(self) -> int:
        """Number of people in simulation."""
        return self._get_num_people()