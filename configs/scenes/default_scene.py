from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.scene import InteractiveSceneCfg, DynamicSceneCfg
from isaaclab.assets import ArticulationCfg,AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg, CameraCfg, RayCasterCfg
from dataclasses import MISSING
from isaaclab.sim.spawners import materials
import isaaclab.sim as sim_utils

GOAL_CFG = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Goal",\
    spawn = sim_utils.SphereCfg(visual_material=materials.PreviewSurfaceCfg(diffuse_color=(1.0,0.0,0.0)),visible=False,radius=0.25),
)

BENCH_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/Scene",
    terrain_type="usd",
    usd_path=f"",
)

@configclass
class ExplorationSceneCfg(InteractiveSceneCfg):
    terrain: TerrainImporterCfg = MISSING
    robot: ArticulationCfg = MISSING
    contact_sensor: ContactSensorCfg = MISSING
    camera_sensor: CameraCfg = MISSING
    metric_sensor: CameraCfg = MISSING

@configclass
class PointNavSceneCfg(InteractiveSceneCfg):
    terrain: TerrainImporterCfg = MISSING
    robot: ArticulationCfg = MISSING
    contact_sensor: ContactSensorCfg = MISSING
    camera_sensor: CameraCfg = MISSING
    goal: AssetBaseCfg = MISSING
    
@configclass
class ImageNavSceneCfg(InteractiveSceneCfg):
    terrain: TerrainImporterCfg = MISSING
    robot: ArticulationCfg = MISSING
    contact_sensor: ContactSensorCfg = MISSING
    camera_sensor: CameraCfg = MISSING
    goal_camera: CameraCfg = MISSING
    goal_marker: AssetBaseCfg = MISSING

@configclass
class PixelNavSceneCfg(InteractiveSceneCfg):
    terrain: TerrainImporterCfg = MISSING
    robot: ArticulationCfg = MISSING
    contact_sensor: ContactSensorCfg = MISSING
    camera_sensor: CameraCfg = MISSING
    goal_marker: AssetBaseCfg = MISSING
    
@configclass
class QuadrupedPointNavSceneCfg(PointNavSceneCfg):
    height_sensor: RayCasterCfg = MISSING

@configclass
class QuadrupedImageNavSceneCfg(PointNavSceneCfg):
    height_sensor: RayCasterCfg = MISSING
    
@configclass
class QuadrupedExplorationSceneCfg(ExplorationSceneCfg):
    height_sensor: RayCasterCfg = MISSING

@configclass
class HumanoidPointNavSceneCfg(PointNavSceneCfg):
    height_sensor: RayCasterCfg = MISSING

@configclass
class HumanoidImageNavSceneCfg(PointNavSceneCfg):
    height_sensor: RayCasterCfg = MISSING
    
@configclass
class HumanoidExplorationSceneCfg(ExplorationSceneCfg):
    height_sensor: RayCasterCfg = MISSING
    
@configclass
class SocialNavSceneCfg(DynamicSceneCfg):
    """Scene configuration for social navigation with dynamic pedestrians.
    
    This scene includes:
    - Terrain (USD scene)
    - Robot (wheeled/quadruped/humanoid)
    - Contact sensors
    - Camera sensors
    - Goal marker
    - Dynamic pedestrians from episode JSON
    """
    terrain: TerrainImporterCfg = MISSING
    robot: ArticulationCfg = MISSING
    contact_sensor: ContactSensorCfg = MISSING
    camera_sensor: CameraCfg = MISSING
    goal: AssetBaseCfg = MISSING
    
    def __post_init__(self):
        """Post initialization to set default values for people simulation."""
        # Enable people simulation by default for social nav
        if not hasattr(self, 'people_simulation'):
            self.people_simulation = True
    
@configclass
class DynPointGoalSceneCfg(DynamicSceneCfg):
    """动态点导航场景 - 目标是单个移动行人
    
    This scene includes:
    - Terrain (USD scene)
    - Robot (wheeled/quadruped/humanoid)
    - Contact sensors
    - Camera sensors
    - Goal marker (for visualization)
    - Dynamic pedestrians from episode JSON
    - Target character is selected by episode_id % 15
    """
    terrain: TerrainImporterCfg = MISSING
    robot: ArticulationCfg = MISSING
    contact_sensor: ContactSensorCfg = MISSING
    camera_sensor: CameraCfg = MISSING
    goal: AssetBaseCfg = MISSING

    def __post_init__(self):
        """Post initialization to set default values for people simulation."""
        if not hasattr(self, 'people_simulation'):
            self.people_simulation = True
        self.enable_dynamic_target = True

@configclass
class DynExploreSceneCfg(DynamicSceneCfg):
    """动态探索场景 - 在有人环境下自由探索
    
    This scene includes:
    - Terrain (USD scene)
    - Robot (wheeled/quadruped/humanoid)
    - Contact sensors
    - Camera sensors (standard + metric for occupancy mapping)
    - Dynamic pedestrians from episode JSON
    - No goal marker (exploration task)
    """
    terrain: TerrainImporterCfg = MISSING
    robot: ArticulationCfg = MISSING
    contact_sensor: ContactSensorCfg = MISSING
    camera_sensor: CameraCfg = MISSING
    metric_sensor: CameraCfg = MISSING  # For occupancy mapping

    def __post_init__(self):
        """Post initialization to set default values for people simulation."""
        if not hasattr(self, 'people_simulation'):
            self.people_simulation = True
        # self.enable_exploration = True  # ← 启用探索模式


    


        
    
    
    
    
    
    
    