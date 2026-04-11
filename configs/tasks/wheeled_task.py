import math
import random
import torch
import trimesh
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import MISSING
from typing import Literal
from isaaclab.envs import ManagerBasedRLEnvCfg,ManagerBasedEnvCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners import materials
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns, CameraCfg
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import CameraCfg
from isaacsim.core.prims import XFormPrim
import isaacsim.core.utils.numpy.rotations as rot_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation, RigidObject
from configs.robots import *
from .usd_utils import *

import os
import json
import glob
from configs.scenes import SocialNavSceneCfg, DynPointGoalSceneCfg, DynExploreSceneCfg
from scipy.spatial.transform import Rotation as R

reset_counter = 0
def camera_rgb_data(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("camera")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.output['rgb']
def camera_depth_data(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("camera")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.output['distance_to_image_plane']
def oracle_imu_pose_data(env: ManagerBasedEnv, 
                         robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot_asset = env.scene[robot_asset_cfg.name]
    robot_rot = math_utils.matrix_from_quat(robot_asset.data.root_quat_w)
    robot_pos = robot_asset.data.root_pos_w
    goal_primview = XFormPrim(prim_paths_expr="/World/envs/env_.*/Goal", name="xform_view") # XFormPrimView
    goal_pos = goal_primview.get_world_poses()[0]
    rel_pos = torch.zeros((goal_pos.shape[0], 3))
    for i in range(rel_pos.shape[0]):
        rel_pos[i] = torch.matmul(torch.inverse(robot_rot[i]),(goal_pos[i] - robot_pos[i]).T)
    return rel_pos

def pixel_projection_data(env: ManagerBasedEnv,
                          robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("camera")):
    camera_asset = env.scene[robot_asset_cfg.name]
    camera_w_pos = camera_asset._data.pos_w 
    camera_w_rot = math_utils.matrix_from_quat(camera_asset._data.quat_w_world)
    camera_intrinsic = camera_asset._data.intrinsic_matrices
    goal_primview = XFormPrim(prim_paths_expr="/World/envs/env_.*/Goal", name="xform_view") # XFormPrimView
    goal_pos = goal_primview.get_world_poses()[0]
    pixel_coords = torch.zeros((goal_pos.shape[0], 2))
    for i in range(camera_intrinsic.shape[0]):
        frame_coord = torch.matmul(torch.inverse(camera_w_rot[i]),(goal_pos[i] - camera_w_pos[i]).T)
        pixel_coord_x =  -frame_coord[1] * camera_intrinsic[i,0,0] / frame_coord[0] + camera_intrinsic[i,0,2]
        pixel_coord_y =  -frame_coord[2] * camera_intrinsic[i,1,1] / frame_coord[0] + camera_intrinsic[i,1,2]
        pixel_coords[i] = torch.as_tensor([pixel_coord_x,pixel_coord_y],dtype=torch.float32,device=camera_w_pos.device)
    return pixel_coords
    
def stuck_terminal_check(env: ManagerBasedEnv,
                         robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                         window_size: int = 10,
                         threshold: float = 0.05):
    if not hasattr(env, '_recent_positions'):
        env._recent_positions = deque(maxlen=window_size)
    robot_asset = env.scene[robot_asset_cfg.name]
    pos = robot_asset.data.root_pos_w[0, :2].cpu().numpy()  # 只看x, y
    env._recent_positions.append(pos)
    if len(env._recent_positions) < window_size:
        return False 
    current = env._recent_positions[-1]
    max_dist = max(np.linalg.norm(current - np.array(p)) for p in list(env._recent_positions)[:-1])
    return bool(max_dist < threshold)

def arrival_terminal_check(env: ManagerBasedEnv,
                           robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot_asset = env.scene[robot_asset_cfg.name]
    robot_pos = robot_asset.data.root_pos_w
    goal_primview = XFormPrim(prim_paths_expr="/World/envs/env_.*/Goal", name="xform_view") # XFormPrimView
    goal_pos = goal_primview.get_world_poses()[0]
    robot_vel = robot_asset.data.root_lin_vel_w
    distance = torch.square(robot_pos[:,0:2] - goal_pos[:,0:2]).sum(axis=1).sqrt()
    velocity = torch.abs(robot_vel).sum(axis=1)
    return (distance < 1.0) & (velocity < 0.5)

def exploration_reset(env: ManagerBasedEnv, 
                      env_ids: torch.Tensor, 
                      init_point_path:str,
                      height_offset:float,
                      robot_visible:bool,
                      light_enabled:bool,
                      robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    global reset_counter
    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    np.random.seed(1234)
    sample_points = np.load(init_point_path)
    
    if not robot_visible:
        for i in range(env_ids.shape[0]):
            hide_entity(f"/World/envs/env_{env_ids[i]}/Robot")
    if light_enabled:
        if reset_counter == 0:
            for light_idx,pts in enumerate(sample_points[:,0]):
                pts = pts + np.array([0.0, 0.0, 1.5])
                add_point_light(torch.as_tensor(pts, dtype=torch.float32, device=robot_asset.data.root_pos_w.device),
                                prim_path= f"/World/envs/env_{env_ids[0]}/point_light_{light_idx}")
                
    random_robot_points = []
    random_init_orientions = []
    for i in range(env_ids.shape[0]):
        idx = int((i + reset_counter) % sample_points.shape[0])
        start_goal_pair = sample_points[idx]
        start_points = np.array([start_goal_pair[0], start_goal_pair[1], 0])
        init_orientions = start_goal_pair[4]
        random_robot_points.append(start_points)
        random_init_orientions.append(init_orientions)
        
    random_robot_points = np.array(random_robot_points)
    tensor_robot_points = torch.tensor(random_robot_points, dtype=torch.float32, device=robot_asset.data.root_pos_w.device) + env.scene.env_origins[env_ids]
    tensor_robot_points[:, 2] = tensor_robot_points[:, 2] + height_offset
    random_init_orientions = np.array(random_init_orientions)
    random_init_orientions = torch.tensor(random_init_orientions, dtype=torch.float32, device=robot_asset.data.root_pos_w.device)
    if len(tensor_robot_points.shape) == 1:
        tensor_robot_points = tensor_robot_points.unsqueeze(0)
        
    angle = random_init_orientions
    angle = angle.unsqueeze(-1).cpu().numpy()
    batch_init_rotation = torch.tensor(rot_utils.euler_angles_to_quats(np.concatenate((angle*0.0, angle*0.0, angle), axis=-1))).to(robot_asset.data.root_pos_w.device)
    robot_asset.write_root_pose_to_sim(torch.concat((tensor_robot_points, batch_init_rotation.to(torch.float32)),dim=-1),env_ids)
    reset_counter += env_ids.shape[0]

    if hasattr(env, '_recent_positions'):
        env._recent_positions.clear()

def pointnav_reset(env: ManagerBasedEnv, 
                   env_ids: torch.Tensor, 
                   init_point_path:str,
                   height_offset:float,
                   robot_visible:bool,
                   light_enabled:bool,
                   robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    global reset_counter
    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    np.random.seed(1234)
    sample_points = np.load(init_point_path)
    
    if not robot_visible:
        for i in range(env_ids.shape[0]):
            hide_entity(f"/World/envs/env_{env_ids[i]}/Robot")
    if light_enabled:
        if reset_counter == 0:
            for light_idx,pts in enumerate(sample_points[:,0]):
                pts = pts + np.array([0.0, 0.0, 1.5])
                add_point_light(torch.as_tensor(pts, dtype=torch.float32, device=robot_asset.data.root_pos_w.device),
                                prim_path= f"/World/envs/env_{env_ids[0]}/point_light_{light_idx}")
    
    random_robot_points = []
    random_goal_points = []
    random_init_orientions = []
    for i in range(env_ids.shape[0]):
        idx = int((i + reset_counter) % sample_points.shape[0])
        start_goal_pair = sample_points[idx]
        start_points = np.array([start_goal_pair[0], start_goal_pair[1], 0])
        goal_points = np.array([start_goal_pair[2], start_goal_pair[3], 0])
        init_orientions = start_goal_pair[4]
        random_robot_points.append(start_points)
        random_goal_points.append(goal_points)
        random_init_orientions.append(init_orientions)
        
    random_robot_points = np.array(random_robot_points)
    random_goal_points = np.array(random_goal_points)
    random_init_orientions = np.array(random_init_orientions)
    random_init_orientions = torch.tensor(random_init_orientions, dtype=torch.float32, device=robot_asset.data.root_pos_w.device)
    tensor_robot_points = torch.tensor(random_robot_points, dtype=torch.float32, device=robot_asset.data.root_pos_w.device) + env.scene.env_origins[env_ids]
    tensor_robot_points[:, 2] = tensor_robot_points[:, 2] + height_offset
    if len(tensor_robot_points.shape) == 1:
        tensor_robot_points = tensor_robot_points.unsqueeze(0)
    tensor_goal_points = torch.tensor(random_goal_points, dtype=torch.float32, device=robot_asset.data.root_pos_w.device) + env.scene.env_origins[env_ids]
    tensor_goal_points[:, 2] = tensor_goal_points[:, 2] + 1.5
    
    angle = random_init_orientions
    angle = angle.unsqueeze(-1).cpu().numpy()
    batch_init_rotation = torch.tensor(rot_utils.euler_angles_to_quats(np.concatenate((angle*0.0, angle*0.0, angle), axis=-1))).to(robot_asset.data.root_pos_w.device)
    robot_asset.write_root_pose_to_sim(torch.concat((tensor_robot_points, batch_init_rotation.to(torch.float32)),dim=-1),env_ids)
    for i, env_id in enumerate(env_ids):
        goal_primview = XFormPrim(prim_paths_expr=f"/World/envs/env_{env_id}/Goal", name="xform_view") # XFormPrimView
        goal_primview.set_world_poses(tensor_goal_points[i].unsqueeze(0),batch_init_rotation[i].unsqueeze(0))
    reset_counter += env_ids.shape[0]

    if hasattr(env, '_recent_positions'):
        env._recent_positions.clear()

def imagenav_reset(env: ManagerBasedEnv, 
                   env_ids: torch.Tensor, 
                   init_point_path:str,
                   height_offset:float,
                   camera_offset:float,
                   robot_visible:bool,
                   light_enabled:bool,
                   robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    global reset_counter
    np.random.seed(1234)
    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    sample_points = np.load(init_point_path)
    
    if not robot_visible:
        for i in range(env_ids.shape[0]):
            hide_entity(f"/World/envs/env_{env_ids[i]}/Robot")
    if light_enabled:
        if reset_counter == 0:
            for light_idx,pts in enumerate(sample_points[:,0]):
                pts = pts + np.array([0.0, 0.0, 1.5])
                add_point_light(torch.as_tensor(pts, dtype=torch.float32, device=robot_asset.data.root_pos_w.device),
                                prim_path= f"/World/envs/env_{env_ids[0]}/point_light_{light_idx}")
    
    random_robot_points = []
    random_goal_points = []
    random_init_orientions = []
    for i in range(env_ids.shape[0]):
        idx = int((i + reset_counter) % sample_points.shape[0])
        start_goal_pair = sample_points[idx]
        start_points = np.array([start_goal_pair[0], start_goal_pair[1], 0])
        goal_points = np.array([start_goal_pair[2], start_goal_pair[3], 0])
        init_orientions = start_goal_pair[4]
        random_robot_points.append(start_points)
        random_goal_points.append(goal_points)
        random_init_orientions.append(init_orientions)
        
    random_robot_points = np.array(random_robot_points)
    random_goal_points = np.array(random_goal_points)
    random_init_orientions = np.array(random_init_orientions)
    random_init_orientions = torch.tensor(random_init_orientions, dtype=torch.float32, device=robot_asset.data.root_pos_w.device)
    tensor_robot_points = torch.tensor(random_robot_points, dtype=torch.float32, device=robot_asset.data.root_pos_w.device) + env.scene.env_origins[env_ids]
    tensor_robot_points[:, 2] = tensor_robot_points[:, 2] + height_offset
    if len(tensor_robot_points.shape) == 1:
        tensor_robot_points = tensor_robot_points.unsqueeze(0)
    tensor_goal_points = torch.tensor(random_goal_points, dtype=torch.float32, device=robot_asset.data.root_pos_w.device) + env.scene.env_origins[env_ids]
    tensor_goal_points[:, 2] = tensor_goal_points[:, 2] + 1.5
    
    angle = random_init_orientions
    angle = angle.unsqueeze(-1).cpu().numpy()
    batch_init_rotation = torch.tensor(rot_utils.euler_angles_to_quats(np.concatenate((angle*0.0, angle*0.0, angle), axis=-1))).to(robot_asset.data.root_pos_w.device)
    robot_asset.write_root_pose_to_sim(torch.concat((tensor_robot_points, batch_init_rotation.to(torch.float32)),dim=-1),env_ids)
    
    for i, env_id in enumerate(env_ids):
        goal_primview = XFormPrim(prim_paths_expr=f"/World/envs/env_{env_id}/goal_cam", name="xform_view") # XFormPrimView
        goal_image_point = tensor_goal_points[i]
        goal_image_point[2] = robot_asset.data.root_pos_w[i,2] + camera_offset
        goal_image_rot = torch.tensor(rot_utils.euler_angles_to_quats(np.concatenate((angle*0.0 + np.pi/2, angle*0.0, angle - np.pi/2), axis=-1))).to(robot_asset.data.root_pos_w.device)
        goal_primview.set_world_poses(goal_image_point.unsqueeze(0),goal_image_rot)
        
    for i, env_id in enumerate(env_ids):
        goal_primview = XFormPrim(prim_paths_expr=f"/World/envs/env_{env_id}/Goal", name="xform_view") # XFormPrimView
        goal_primview.set_world_poses(tensor_goal_points[i].unsqueeze(0),batch_init_rotation[i].unsqueeze(0))
    reset_counter += env_ids.shape[0]
 
    if hasattr(env, '_recent_positions'):
        env._recent_positions.clear()

def pixelnav_reset(env: ManagerBasedEnv, 
                   env_ids: torch.Tensor, 
                   init_point_path:str,
                   height_offset:float,
                   robot_visible:bool,
                   light_enabled:bool,
                   robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    global reset_counter
    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    np.random.seed(1234)
    sample_points = np.load(init_point_path)
    
    if not robot_visible:
        for i in range(env_ids.shape[0]):
            hide_entity(f"/World/envs/env_{env_ids[i]}/Robot")
    if light_enabled:
        if reset_counter == 0:
            for light_idx,pts in enumerate(sample_points[:,0]):
                pts = pts + np.array([0.0, 0.0, 1.5])
                add_point_light(torch.as_tensor(pts, dtype=torch.float32, device=robot_asset.data.root_pos_w.device),
                                prim_path= f"/World/envs/env_{env_ids[0]}/point_light_{light_idx}")
    
    random_robot_points = []
    random_goal_points = []
    random_init_orientions = []
    for i in range(env_ids.shape[0]):
        idx = int((i + reset_counter) % sample_points.shape[0])
        start_goal_pair = sample_points[idx]
        start_points = np.array([start_goal_pair[0], start_goal_pair[1], 0])
        goal_points = np.array([start_goal_pair[2], start_goal_pair[3], 0])
        init_orientions = start_goal_pair[4]
        random_robot_points.append(start_points)
        random_goal_points.append(goal_points)
        random_init_orientions.append(init_orientions)
        
    random_robot_points = np.array(random_robot_points)
    random_goal_points = np.array(random_goal_points)
    random_init_orientions = np.array(random_init_orientions)
    random_init_orientions = torch.tensor(random_init_orientions, dtype=torch.float32, device=robot_asset.data.root_pos_w.device)
    tensor_robot_points = torch.tensor(random_robot_points, dtype=torch.float32, device=robot_asset.data.root_pos_w.device) + env.scene.env_origins[env_ids]
    tensor_robot_points[:, 2] = tensor_robot_points[:, 2] + height_offset
    if len(tensor_robot_points.shape) == 1:
        tensor_robot_points = tensor_robot_points.unsqueeze(0)
    tensor_goal_points = torch.tensor(random_goal_points, dtype=torch.float32, device=robot_asset.data.root_pos_w.device) + env.scene.env_origins[env_ids]
    tensor_goal_points[:, 2] = tensor_goal_points[:, 2] + 1.5
    
    angle = random_init_orientions
    angle = angle.unsqueeze(-1).cpu().numpy()
    batch_init_rotation = torch.tensor(rot_utils.euler_angles_to_quats(np.concatenate((angle*0.0, angle*0.0, angle), axis=-1))).to(robot_asset.data.root_pos_w.device)
    robot_asset.write_root_pose_to_sim(torch.concat((tensor_robot_points, batch_init_rotation.to(torch.float32)),dim=-1),env_ids)
    for i, env_id in enumerate(env_ids):
        goal_primview = XFormPrim(prim_paths_expr=f"/World/envs/env_{env_id}/Goal", name="xform_view") # XFormPrimView
        goal_primview.set_world_poses(tensor_goal_points[i].unsqueeze(0),batch_init_rotation[i].unsqueeze(0))
    reset_counter += env_ids.shape[0]

    if hasattr(env, '_recent_positions'):
        env._recent_positions.clear()

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

@configclass
class ExploreObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_pos = ObsTerm(func=mdp.root_pos_w)
        base_rot = ObsTerm(func=mdp.root_quat_w)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    @configclass
    class RGBImageCfg(ObsGroup):
        rgb_measurement = ObsTerm(
            func = camera_rgb_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    @configclass
    class DepthImageCfg(ObsGroup):
        depth_measurement = ObsTerm(
            func = camera_depth_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    @configclass
    class MetricRGBImageCfg(ObsGroup):
        depth_measurement = ObsTerm(
            func = camera_rgb_data,
            params = {'asset_cfg':SceneEntityCfg("metric_sensor")}
        )
    @configclass
    class MetricDepthImageCfg(ObsGroup):
        depth_measurement = ObsTerm(
            func = camera_depth_data,
            params = {'asset_cfg':SceneEntityCfg("metric_sensor")}
        )
    policy: PolicyCfg = PolicyCfg()
    rgb: RGBImageCfg = RGBImageCfg()
    depth: DepthImageCfg = DepthImageCfg()
    metric_rgb: MetricRGBImageCfg=MetricRGBImageCfg()
    metric_depth: MetricDepthImageCfg=MetricDepthImageCfg()

@configclass
class PointNavObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_pos = ObsTerm(func=mdp.root_pos_w)
        base_rot = ObsTerm(func=mdp.root_quat_w)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class RGBImageCfg(ObsGroup):
        rgb_measurement = ObsTerm(
            func = camera_rgb_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    @configclass
    class DepthImageCfg(ObsGroup):
        depth_measurement = ObsTerm(
            func = camera_depth_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    @configclass
    class GoalPoseCfg(ObsGroup):
        pose_measurement = ObsTerm(
            func = oracle_imu_pose_data,
            params = {'robot_asset_cfg':SceneEntityCfg("robot")}
        )
    policy: PolicyCfg = PolicyCfg()
    rgb: RGBImageCfg = RGBImageCfg()
    depth: DepthImageCfg = DepthImageCfg()
    goal_pose: GoalPoseCfg = GoalPoseCfg()

@configclass
class ImageNavObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_pos = ObsTerm(func=mdp.root_pos_w)
        base_rot = ObsTerm(func=mdp.root_quat_w)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class RGBImageCfg(ObsGroup):
        rgb_measurement = ObsTerm(
            func = camera_rgb_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    @configclass
    class DepthImageCfg(ObsGroup):
        depth_measurement = ObsTerm(
            func = camera_depth_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    @configclass
    class GoalImageCfg(ObsGroup):
        pose_measurement = ObsTerm(
            func = camera_rgb_data,
            params = {'asset_cfg':SceneEntityCfg("goal_camera")}
        )
    
    @configclass
    class GoalPoseCfg(ObsGroup):
        pose_measurement = ObsTerm(
            func = oracle_imu_pose_data,
            params = {'robot_asset_cfg':SceneEntityCfg("robot")}
        )
    policy: PolicyCfg = PolicyCfg()
    rgb: RGBImageCfg = RGBImageCfg()
    depth: DepthImageCfg = DepthImageCfg()
    goal_image: GoalImageCfg = GoalImageCfg()
    goal_pose: GoalPoseCfg = GoalPoseCfg()

@configclass
class PixelNavObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_pos = ObsTerm(func=mdp.root_pos_w)
        base_rot = ObsTerm(func=mdp.root_quat_w)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class RGBImageCfg(ObsGroup):
        rgb_measurement = ObsTerm(
            func = camera_rgb_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    @configclass
    class DepthImageCfg(ObsGroup):
        depth_measurement = ObsTerm(
            func = camera_depth_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    @configclass
    class GoalPoseCfg(ObsGroup):
        pose_measurement = ObsTerm(
            func = oracle_imu_pose_data,
            params = {'robot_asset_cfg':SceneEntityCfg("robot")}
        )
    
    @configclass
    class GoalPixelCfg(ObsGroup):
        pixel_measurement = ObsTerm(
            func = pixel_projection_data,
            params = {'robot_asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    policy: PolicyCfg = PolicyCfg()
    rgb: RGBImageCfg = RGBImageCfg()
    depth: DepthImageCfg = DepthImageCfg()
    goal_pose: GoalPoseCfg = GoalPoseCfg()
    goal_pixel: GoalPixelCfg = GoalPixelCfg()

@configclass
class ExploreEventCfg:
    """Configuration for events.""" 
    reset_pose = EventTerm(func=exploration_reset,
                           mode='reset',
                           params={})

@configclass
class PointNavEventCfg:
    """Configuration for events.""" 
    reset_pose = EventTerm(func=pointnav_reset,
                           mode='reset',
                           params={})

@configclass
class ImageNavEventCfg:
    """Configuration for events.""" 
    reset_pose = EventTerm(func=imagenav_reset,
                           mode='reset',
                           params={})

@configclass
class PixelNavEventCfg:
    """Configuration for events.""" 
    reset_pose = EventTerm(func=pixelnav_reset,
                           mode='reset',
                           params={})
    
@configclass
class DingoActionsCfg:
    joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=DINGO_WHEEL_JOINTS, scale=1.0, use_default_offset=True, debug_vis=True)
     
@configclass
class DingoExploreTerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=DINGO_BASE_LINK), "threshold": DINGO_THRESHOLD},
    # )
    struck = DoneTerm(func=stuck_terminal_check,
                      params={"robot_asset_cfg": SceneEntityCfg("robot"), 
                              "window_size": 30, 
                              "threshold": 0.1})
    
@configclass
class PointNavTerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    arrive_goal = DoneTerm(func=arrival_terminal_check,
                           params={"robot_asset_cfg":SceneEntityCfg("robot")})
    stuck = DoneTerm(func=stuck_terminal_check,
                      params={"robot_asset_cfg": SceneEntityCfg("robot"), 
                              "window_size": 30, 
                              "threshold": 0.1})
    
@configclass
class ImageNavTerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    arrive_goal = DoneTerm(func=arrival_terminal_check,
                           params={"robot_asset_cfg":SceneEntityCfg("robot")})
    stuck = DoneTerm(func=stuck_terminal_check,
                      params={"robot_asset_cfg": SceneEntityCfg("robot"), 
                              "window_size": 30, 
                              "threshold": 0.1})

@configclass
class PixelNavTerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    arrive_goal = DoneTerm(func=arrival_terminal_check,
                           params={"robot_asset_cfg":SceneEntityCfg("robot")})
    stuck = DoneTerm(func=stuck_terminal_check,
                      params={"robot_asset_cfg": SceneEntityCfg("robot"), 
                              "window_size": 30, 
                              "threshold": 0.1})
    
@configclass
class DingoPointNavCfg(ManagerBasedRLEnvCfg):
    scene: InteractiveSceneCfg = MISSING
    observations = PointNavObservationsCfg()
    actions = DingoActionsCfg()
    terminations = PointNavTerminationsCfg()
    events = PointNavEventCfg()
    rewards = RewardsCfg()
    def __post_init__(self):
        self.sim.render_interval = 15
        self.decimation = 15
        self.episode_length_s = 120.0
        self.sim.dt = 0.01
        self.sim.disable_contact_processing = True
        
@configclass
class DingoImageNavCfg(ManagerBasedRLEnvCfg):
    scene: InteractiveSceneCfg = MISSING
    observations = ImageNavObservationsCfg()
    actions = DingoActionsCfg()
    terminations = ImageNavTerminationsCfg()
    events = ImageNavEventCfg()
    rewards = RewardsCfg()
    def __post_init__(self):
        self.sim.render_interval = 15
        self.decimation = 15
        self.episode_length_s = 120.0
        self.sim.dt = 0.01
        self.sim.disable_contact_processing = True

@configclass
class DingoPixelNavCfg(ManagerBasedRLEnvCfg):
    scene: InteractiveSceneCfg = MISSING
    observations = PixelNavObservationsCfg()
    actions = DingoActionsCfg()
    terminations = PixelNavTerminationsCfg()
    events = PixelNavEventCfg()
    rewards = RewardsCfg()
    def __post_init__(self):
        self.sim.render_interval = 15
        self.decimation = 15
        self.episode_length_s = 120.0
        self.sim.dt = 0.01
        self.sim.disable_contact_processing = True
        
@configclass
class DingoExplorationCfg(ManagerBasedRLEnvCfg):
    scene: InteractiveSceneCfg = MISSING
    observations = ExploreObservationsCfg()
    actions = DingoActionsCfg()
    terminations = DingoExploreTerminationsCfg()
    events = ExploreEventCfg()
    rewards = RewardsCfg()
    def __post_init__(self):
        self.sim.render_interval = 15
        self.decimation = 15
        self.episode_length_s = 120.0
        self.sim.dt = 0.01
        self.sim.disable_contact_processing = True

### Dynamic

def socialnav_reset(env: ManagerBasedEnv, 
                    env_ids: torch.Tensor,
                    episode_json_dir: str,
                    num_episodes: int,
                    height_offset: float,
                    robot_visible: bool,
                    light_enabled: bool,
                    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Reset function for social navigation that reads from episode JSON files.
    
    Args:
        env: The environment instance
        env_ids: Environment IDs to reset
        episode_json_dir: Directory containing episode JSON files
        num_episodes: Total number of episodes available
        height_offset: Height offset for robot spawn
        robot_visible: Whether robot should be visible
        light_enabled: Whether to enable point lights
        robot_asset_cfg: Robot asset configuration
    """
    global reset_counter
    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    
    # Validate episode directory and files
    if not os.path.exists(episode_json_dir):
        raise RuntimeError(f"Episode directory not found: {episode_json_dir}")
    
    # Check for required episode files
    available_episodes = []
    for episode_id in range(num_episodes):
        episode_path = os.path.join(episode_json_dir, f"episode_{episode_id}.json")
        if not os.path.exists(episode_path):
            raise RuntimeError(f"Missing episode file: {episode_path}")
        available_episodes.append(episode_path)
    
    if not robot_visible:
        for i in range(env_ids.shape[0]):
            hide_entity(f"/World/envs/env_{env_ids[i]}/Robot")
    
    random_robot_points = []
    random_goal_points = []
    random_init_orientations = []
    
    for i in range(env_ids.shape[0]):
        # Cycle through available episodes
        # episode_idx = int((i + reset_counter) % num_episodes)
        # episode_path = available_episodes[episode_idx]
        
        episode_idx = reset_counter % num_episodes
        episode_path = os.path.join(episode_json_dir, f"episode_{episode_idx}.json")

        # Load episode JSON
        try:
            with open(episode_path, 'r', encoding='utf-8') as f:
                episode_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load episode file {episode_path}: {e}")
        
        # Extract robot data
        episode = episode_data.get("episode")
        if not episode:
            raise RuntimeError(f"Invalid episode format in {episode_path}")
        
        robot_data = episode.get("robot")
        if not robot_data:
            raise RuntimeError(f"No robot data in episode {episode_path}")
        
        # Get start position, goal position, and orientation
        start_pos = robot_data.get("start_pos", [0, 0, 0])
        goal_pos = robot_data.get("goal_pos", [0, 0, 0])
        start_orientation = robot_data.get("start_orientation", 0.0)
        
        # Convert to numpy arrays
        start_points = np.array([start_pos[0], start_pos[1], 0])
        goal_points = np.array([goal_pos[0], goal_pos[1], 0])
        
        random_robot_points.append(start_points)
        random_goal_points.append(goal_points)
        random_init_orientations.append(start_orientation)
    
    # Convert to tensors
    random_robot_points = np.array(random_robot_points)
    random_goal_points = np.array(random_goal_points)
    random_init_orientations = np.array(random_init_orientations)
    random_init_orientations = torch.tensor(
        random_init_orientations, 
        dtype=torch.float32, 
        device=robot_asset.data.root_pos_w.device
    )
    
    tensor_robot_points = torch.tensor(
        random_robot_points, 
        dtype=torch.float32, 
        device=robot_asset.data.root_pos_w.device
    ) + env.scene.env_origins[env_ids]
    tensor_robot_points[:, 2] = tensor_robot_points[:, 2] + height_offset
    
    if len(tensor_robot_points.shape) == 1:
        tensor_robot_points = tensor_robot_points.unsqueeze(0)
    
    tensor_goal_points = torch.tensor(
        random_goal_points, 
        dtype=torch.float32, 
        device=robot_asset.data.root_pos_w.device
    ) + env.scene.env_origins[env_ids]
    tensor_goal_points[:, 2] = tensor_goal_points[:, 2] + 1.5
    
    # Setup lighting (only once)
    if light_enabled and reset_counter == 0:
        for light_idx, start_point in enumerate(random_robot_points):
            light_pos = start_point + np.array([0.0, 0.0, 1.5])
            add_point_light(
                torch.as_tensor(light_pos, dtype=torch.float32, device=robot_asset.data.root_pos_w.device),
                prim_path=f"/World/envs/env_{env_ids[0]}/point_light_{light_idx}"
            )
    
    # Set robot orientation
    angle = random_init_orientations
    angle = angle.unsqueeze(-1).cpu().numpy()
    batch_init_rotation = torch.tensor(
        rot_utils.euler_angles_to_quats(
            np.concatenate((angle*0.0, angle*0.0, angle), axis=-1)
        )
    ).to(robot_asset.data.root_pos_w.device)
    
    # Write robot pose
    robot_asset.write_root_pose_to_sim(
        torch.concat((tensor_robot_points, batch_init_rotation.to(torch.float32)), dim=-1),
        env_ids
    )
    
    # Set goal positions
    for i, env_id in enumerate(env_ids):
        goal_primview = XFormPrim(
            prim_paths_expr=f"/World/envs/env_{env_id}/Goal", 
            name="xform_view"
        )
        goal_primview.set_world_poses(
            tensor_goal_points[i].unsqueeze(0),
            batch_init_rotation[i].unsqueeze(0)
        )
    
    reset_counter += env_ids.shape[0]
    if hasattr(env, '_recent_positions'):
        env._recent_positions.clear()

@configclass
class SocialNavObservationsCfg:
    """Observation specifications for Social Navigation MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_pos = ObsTerm(func=mdp.root_pos_w)
        base_rot = ObsTerm(func=mdp.root_quat_w)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class RGBImageCfg(ObsGroup):
        rgb_measurement = ObsTerm(
            func=camera_rgb_data,
            params={'asset_cfg': SceneEntityCfg("camera_sensor")}
        )
    
    @configclass
    class DepthImageCfg(ObsGroup):
        depth_measurement = ObsTerm(
            func=camera_depth_data,
            params={'asset_cfg': SceneEntityCfg("camera_sensor")}
        )
    
    @configclass
    class GoalPoseCfg(ObsGroup):
        pose_measurement = ObsTerm(
            func=oracle_imu_pose_data,
            params={'robot_asset_cfg': SceneEntityCfg("robot")}
        )
    
    policy: PolicyCfg = PolicyCfg()
    rgb: RGBImageCfg = RGBImageCfg()
    depth: DepthImageCfg = DepthImageCfg()
    goal_pose: GoalPoseCfg = GoalPoseCfg()


@configclass
class SocialNavTerminationsCfg:
    """Termination terms for Social Navigation MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    arrive_goal = DoneTerm(
        func=arrival_terminal_check,
        params={"robot_asset_cfg": SceneEntityCfg("robot")}
    )
    stuck = DoneTerm(
        func=stuck_terminal_check,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"), 
            "window_size": 30, 
            "threshold": 0.1
        }
    )


@configclass
class SocialNavEventCfg:
    """Configuration for social navigation events."""
    reset_pose = EventTerm(
        func=socialnav_reset,
        mode='reset',
        params={}
    )        

@configclass
class DingoSocialNavCfg(ManagerBasedRLEnvCfg):
    """Configuration for Dingo Social Navigation task."""
    scene: SocialNavSceneCfg = MISSING
    observations = SocialNavObservationsCfg()
    actions = DingoActionsCfg()
    terminations = SocialNavTerminationsCfg()
    events = SocialNavEventCfg()
    rewards = RewardsCfg()
    
    def __post_init__(self):
        self.sim.render_interval = 15
        self.decimation = 15
        self.episode_length_s = 120.0
        self.sim.dt = 0.01
        self.sim.disable_contact_processing = True
        self.people_simulation = True

## Dynamic PointNav

def dynamic_target_pose_data(env: ManagerBasedEnv, 
                              robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """获取动态目标（行人）的相对位置（实时）
    
    这个函数从people manager获取目标行人的位置，并转换为相对于机器人的坐标
    """
    from omni.anim.people.scripts.global_character_position_manager import GlobalCharacterPositionManager

    robot_asset = env.scene[robot_asset_cfg.name]
    robot_rot = math_utils.matrix_from_quat(robot_asset.data.root_quat_w)
    robot_pos = robot_asset.data.root_pos_w
    
    char_manager = GlobalCharacterPositionManager.get_instance()
    all_chars = char_manager.get_all_managed_characters()
    
    if len(all_chars) == 0:
        return torch.zeros(robot_pos.shape[0], dtype=torch.bool, device=robot_pos.device)
    
    # 只有一个行人，直接取第一个
    target_path = list(all_chars)[0]
    
    pos = char_manager.get_character_current_pos(target_path)
    target_pos = torch.tensor([float(pos[0]), float(pos[1]), float(pos[2])], 
                                dtype=torch.float32, device=robot_pos.device)
    
    # 计算相对位置
    rel_pos = torch.zeros((robot_pos.shape[0], 3), device=robot_pos.device)
    for i in range(rel_pos.shape[0]):
        rel_pos[i] = torch.matmul(torch.inverse(robot_rot[i]), 
                                    (target_pos - robot_pos[i]).T)
    return rel_pos

def arrival_dynamic_target_check(env: ManagerBasedEnv,
                                  robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                  threshold: float = 1.0):
    """检查是否到达动态目标（行人）附近
    
    成功条件：距离 < 1.0m
    """
    from omni.anim.people.scripts.global_character_position_manager import (
        GlobalCharacterPositionManager,
    )
    robot_asset = env.scene[robot_asset_cfg.name]
    robot_pos = robot_asset.data.root_pos_w
    robot_vel = robot_asset.data.root_lin_vel_w
    
    char_manager = GlobalCharacterPositionManager.get_instance()
    all_chars = char_manager.get_all_managed_characters()
    
    # ===== 简化：直接获取唯一的行人位置 =====
    if len(all_chars) == 0:
        return torch.zeros(robot_pos.shape[0], dtype=torch.bool, device=robot_pos.device)
    
    # 只有一个行人，直接取第一个
    target_path = list(all_chars)[0]
    
    pos = char_manager.get_character_current_pos(target_path)
    target_pos = torch.tensor([float(pos[0]), float(pos[1]), float(pos[2])], 
                                dtype=torch.float32, device=robot_pos.device)
    
    # 计算距离（去掉速度限制）
    distance = torch.square(robot_pos[:, 0:2] - target_pos[:2]).sum(axis=1).sqrt()
    
    return (distance < threshold)

def dynpointgoal_reset(env: ManagerBasedEnv, 
                       env_ids: torch.Tensor,
                       episode_json_dir: str,
                       num_episodes: int,
                       height_offset: float,
                       robot_visible: bool,
                       light_enabled: bool,
                       robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """动态点导航的reset函数
    
    从episode JSON读取机器人起点，目标行人由episode_id % 15确定
    不设置goal位置（因为目标是移动的）
    """
    global reset_counter
    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    
    # Validate episode directory
    if not os.path.exists(episode_json_dir):
        raise RuntimeError(f"Episode directory not found: {episode_json_dir}")
    
    if not robot_visible:
        for i in range(env_ids.shape[0]):
            hide_entity(f"/World/envs/env_{env_ids[i]}/Robot")
    
    random_robot_points = []
    random_init_orientations = []
    
    for i in range(env_ids.shape[0]):

        episode_idx = reset_counter % num_episodes
        episode_path = os.path.join(episode_json_dir, f"episode_{episode_idx}.json")
        
        # Load episode JSON
        try:
            with open(episode_path, 'r', encoding='utf-8') as f:
                episode_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load episode file {episode_path}: {e}")
        
        episode = episode_data.get("episode")
        if not episode:
            raise RuntimeError(f"Invalid episode format in {episode_path}")
        
        robot_data = episode.get("robot")
        if not robot_data:
            raise RuntimeError(f"No robot data in episode {episode_path}")
        
        # Get start position and orientation
        start_pos = robot_data.get("start_pos", [0, 0, 0])
        start_orientation = robot_data.get("start_orientation", 0.0)
        
        start_points = np.array([start_pos[0], start_pos[1], 0])
        
        random_robot_points.append(start_points)
        random_init_orientations.append(start_orientation)
    
    # Convert to tensors
    random_robot_points = np.array(random_robot_points)
    random_init_orientations = np.array(random_init_orientations)
    random_init_orientations = torch.tensor(
        random_init_orientations, 
        dtype=torch.float32, 
        device=robot_asset.data.root_pos_w.device
    )
    
    tensor_robot_points = torch.tensor(
        random_robot_points, 
        dtype=torch.float32, 
        device=robot_asset.data.root_pos_w.device
    ) + env.scene.env_origins[env_ids]
    tensor_robot_points[:, 2] = tensor_robot_points[:, 2] + height_offset
    
    if len(tensor_robot_points.shape) == 1:
        tensor_robot_points = tensor_robot_points.unsqueeze(0)
    
    # Setup lighting (only once)
    if light_enabled and reset_counter == 0:
        for light_idx, start_point in enumerate(random_robot_points):
            light_pos = start_point + np.array([0.0, 0.0, 1.5])
            add_point_light(
                torch.as_tensor(light_pos, dtype=torch.float32, device=robot_asset.data.root_pos_w.device),
                prim_path=f"/World/envs/env_{env_ids[0]}/point_light_{light_idx}"
            )
    
    # Set robot orientation
    angle = random_init_orientations
    angle = angle.unsqueeze(-1).cpu().numpy()
    batch_init_rotation = torch.tensor(
        rot_utils.euler_angles_to_quats(
            np.concatenate((angle*0.0, angle*0.0, angle), axis=-1)
        )
    ).to(robot_asset.data.root_pos_w.device)
    
    # Write robot pose
    robot_asset.write_root_pose_to_sim(
        torch.concat((tensor_robot_points, batch_init_rotation.to(torch.float32)), dim=-1),
        env_ids
    )
    
    # 注意：不设置Goal位置，因为目标是动态的
    
    reset_counter += env_ids.shape[0]
    if hasattr(env, '_recent_positions'):
        env._recent_positions.clear()

@configclass
class DynPointGoalObservationsCfg:
    """动态点导航的观测配置"""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_pos = ObsTerm(func=mdp.root_pos_w)
        base_rot = ObsTerm(func=mdp.root_quat_w)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class RGBImageCfg(ObsGroup):
        rgb_measurement = ObsTerm(
            func=camera_rgb_data,
            params={'asset_cfg': SceneEntityCfg("camera_sensor")}
        )
    
    @configclass
    class DepthImageCfg(ObsGroup):
        depth_measurement = ObsTerm(
            func=camera_depth_data,
            params={'asset_cfg': SceneEntityCfg("camera_sensor")}
        )
    
    @configclass
    class GoalPoseCfg(ObsGroup):
        """动态目标（行人）的相对位置"""
        pose_measurement = ObsTerm(
            func=dynamic_target_pose_data,
            params={'robot_asset_cfg': SceneEntityCfg("robot")}
        )
    
    policy: PolicyCfg = PolicyCfg()
    rgb: RGBImageCfg = RGBImageCfg()
    depth: DepthImageCfg = DepthImageCfg()
    goal_pose: GoalPoseCfg = GoalPoseCfg()


@configclass
class DynPointGoalTerminationsCfg:
    """动态点导航的终止条件"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    arrive_goal = DoneTerm(
        func=arrival_dynamic_target_check,
        params={"robot_asset_cfg": SceneEntityCfg("robot"), "threshold": 1.0}
    )
    stuck = DoneTerm(
        func=stuck_terminal_check,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"), 
            "window_size": 30, 
            "threshold": 0.1
        }
    )


@configclass
class DynPointGoalEventCfg:
    """动态点导航的事件配置"""
    reset_pose = EventTerm(
        func=dynpointgoal_reset,
        mode='reset',
        params={}
    )


@configclass
class DingoDynPointGoalCfg(ManagerBasedRLEnvCfg):
    """Dingo动态点导航任务配置"""
    scene: DynPointGoalSceneCfg = MISSING
    observations = DynPointGoalObservationsCfg()
    actions = DingoActionsCfg()
    terminations = DynPointGoalTerminationsCfg()
    events = DynPointGoalEventCfg()
    rewards = RewardsCfg()
    
    def __post_init__(self):
        self.sim.render_interval = 15
        self.decimation = 15
        self.episode_length_s = 120.0
        self.sim.dt = 0.01
        self.sim.disable_contact_processing = True
        self.people_simulation = True

## Dynamic NoGoal

def dynexplore_reset(env: ManagerBasedEnv, 
                     env_ids: torch.Tensor,
                     episode_json_dir: str,
                     num_episodes: int,
                     height_offset: float,
                     robot_visible: bool,
                     light_enabled: bool,
                     robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """动态探索的reset函数 - 在有人环境下探索
    
    从episode JSON读取机器人起点
    没有目标点，机器人自由探索直到stuck或碰撞
    """
    global reset_counter
    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    
    # Validate episode directory
    if not os.path.exists(episode_json_dir):
        raise RuntimeError(f"Episode directory not found: {episode_json_dir}")
    
    if not robot_visible:
        for i in range(env_ids.shape[0]):
            hide_entity(f"/World/envs/env_{env_ids[i]}/Robot")
    
    random_robot_points = []
    random_init_orientations = []
    
    for i in range(env_ids.shape[0]):
        episode_idx = reset_counter % num_episodes
        episode_path = os.path.join(episode_json_dir, f"episode_{episode_idx}.json")
        
        # Load episode JSON
        try:
            with open(episode_path, 'r', encoding='utf-8') as f:
                episode_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load episode file {episode_path}: {e}")
        
        episode = episode_data.get("episode")
        if not episode:
            raise RuntimeError(f"Invalid episode format in {episode_path}")
        
        robot_data = episode.get("robot")
        if not robot_data:
            raise RuntimeError(f"No robot data in episode {episode_path}")
        
        # Get start position and orientation
        start_pos = robot_data.get("start_pos", [0, 0, 0])
        start_orientation = robot_data.get("start_orientation", 0.0)
        
        start_points = np.array([start_pos[0], start_pos[1], 0])
        
        random_robot_points.append(start_points)
        random_init_orientations.append(start_orientation)
    
    # Convert to tensors
    random_robot_points = np.array(random_robot_points)
    random_init_orientations = np.array(random_init_orientations)
    random_init_orientations = torch.tensor(
        random_init_orientations, 
        dtype=torch.float32, 
        device=robot_asset.data.root_pos_w.device
    )
    
    tensor_robot_points = torch.tensor(
        random_robot_points, 
        dtype=torch.float32, 
        device=robot_asset.data.root_pos_w.device
    ) + env.scene.env_origins[env_ids]
    tensor_robot_points[:, 2] = tensor_robot_points[:, 2] + height_offset
    
    if len(tensor_robot_points.shape) == 1:
        tensor_robot_points = tensor_robot_points.unsqueeze(0)
    
    # Setup lighting (only once)
    if light_enabled and reset_counter == 0:
        for light_idx, start_point in enumerate(random_robot_points):
            light_pos = start_point + np.array([0.0, 0.0, 1.5])
            add_point_light(
                torch.as_tensor(light_pos, dtype=torch.float32, device=robot_asset.data.root_pos_w.device),
                prim_path=f"/World/envs/env_{env_ids[0]}/point_light_{light_idx}"
            )
    
    # Set robot orientation
    angle = random_init_orientations
    angle = angle.unsqueeze(-1).cpu().numpy()
    batch_init_rotation = torch.tensor(
        rot_utils.euler_angles_to_quats(
            np.concatenate((angle*0.0, angle*0.0, angle), axis=-1)
        )
    ).to(robot_asset.data.root_pos_w.device)
    
    # Write robot pose
    robot_asset.write_root_pose_to_sim(
        torch.concat((tensor_robot_points, batch_init_rotation.to(torch.float32)), dim=-1),
        env_ids
    )
    
    reset_counter += env_ids.shape[0]
    if hasattr(env, '_recent_positions'):
        env._recent_positions.clear()


@configclass
class DynExploreObservationsCfg:
    """动态探索的观测配置 - 无目标点"""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_pos = ObsTerm(func=mdp.root_pos_w)
        base_rot = ObsTerm(func=mdp.root_quat_w)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class RGBImageCfg(ObsGroup):
        rgb_measurement = ObsTerm(
            func=camera_rgb_data,
            params={'asset_cfg': SceneEntityCfg("camera_sensor")}
        )
    
    @configclass
    class DepthImageCfg(ObsGroup):
        depth_measurement = ObsTerm(
            func=camera_depth_data,
            params={'asset_cfg': SceneEntityCfg("camera_sensor")}
        )
    
    @configclass
    class MetricRGBImageCfg(ObsGroup):
        rgb_measurement = ObsTerm(
            func=camera_rgb_data,
            params={'asset_cfg': SceneEntityCfg("metric_sensor")}
        )
    
    @configclass
    class MetricDepthImageCfg(ObsGroup):
        depth_measurement = ObsTerm(
            func=camera_depth_data,
            params={'asset_cfg': SceneEntityCfg("metric_sensor")}
        )
    
    policy: PolicyCfg = PolicyCfg()
    rgb: RGBImageCfg = RGBImageCfg()
    depth: DepthImageCfg = DepthImageCfg()
    metric_rgb: MetricRGBImageCfg = MetricRGBImageCfg()
    metric_depth: MetricDepthImageCfg = MetricDepthImageCfg()

def collision_with_people_check(env: ManagerBasedEnv,
                                robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """检查是否与行人碰撞（复用 social_metrics 的统计）
    
    从环境的 social_metrics_trackers 中读取碰撞状态，避免重复计算
    
    注意：需要在主循环中设置 env.social_metrics_trackers
    
    Returns:
        torch.Tensor: 布尔张量，True表示发生碰撞
    """
    camera_asset = env.scene[robot_asset_cfg.name]
    # 获取环境中的 social_metrics_trackers
    if not hasattr(env, 'social_metrics_trackers'):
        # 如果没有 tracker，返回 False（不终止）
        num_envs = env.num_envs if hasattr(env, 'num_envs') else 1
        return torch.zeros(num_envs, dtype=torch.bool, device=camera_asset._data.device)
    
    # 从 tracker 读取碰撞状态
    collision_flags = []
    for tracker in env.social_metrics_trackers:
        # 检查是否当前处于 INTIMATE_SPACE（碰撞状态）
        collision_flags.append(tracker.in_intimate_space)
    
    return torch.tensor(collision_flags, dtype=torch.bool, device=camera_asset._data.device)

@configclass
class DynExploreTerminationsCfg:
    """动态探索的终止条件 - stuck或碰撞行人"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    stuck = DoneTerm(
        func=stuck_terminal_check,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"), 
            "window_size": 30, 
            "threshold": 0.1
        }
    )
    collision_with_people = DoneTerm(
        func=collision_with_people_check,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot")
        }
    )
    
@configclass
class DynExploreEventCfg:
    """动态探索的事件配置"""
    reset_pose = EventTerm(
        func=dynexplore_reset,
        mode='reset',
        params={}
    )


@configclass
class DingoDynExploreCfg(ManagerBasedRLEnvCfg):
    """Dingo动态探索任务配置 - 在有人环境下探索"""
    scene: DynExploreSceneCfg = MISSING
    observations = DynExploreObservationsCfg()
    actions = DingoActionsCfg()
    terminations = DynExploreTerminationsCfg()
    events = DynExploreEventCfg()
    rewards = RewardsCfg()
    
    def __post_init__(self):
        self.sim.render_interval = 15
        self.decimation = 15
        self.episode_length_s = 120.0
        self.sim.dt = 0.01
        self.sim.disable_contact_processing = True
        self.people_simulation = True


# === Residual RL Reward Functions ===

def social_distance_reward(
    env: ManagerBasedEnv,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_distance: float = 0.45,
    comfort_distance: float = 1.2,
):
    """行人距离奖励：太近给惩罚，舒适距离以外无惩罚。

    用于 SocialNav 和 DynExplore，不用于 DynPointGoal。
    """
    from omni.anim.people.scripts.global_character_position_manager import (
        GlobalCharacterPositionManager,
    )

    robot_asset = env.scene[robot_asset_cfg.name]
    robot_pos = robot_asset.data.root_pos_w

    char_manager = GlobalCharacterPositionManager.get_instance()
    all_chars = char_manager.get_all_managed_characters()

    if len(all_chars) == 0:
        return torch.zeros(robot_pos.shape[0], device=robot_pos.device)

    rewards = torch.zeros(robot_pos.shape[0], device=robot_pos.device)

    for char_path in all_chars:
        pos = char_manager.get_character_current_pos(char_path)
        person_pos = torch.tensor(
            [float(pos[0]), float(pos[1])],
            dtype=torch.float32,
            device=robot_pos.device,
        )
        for i in range(robot_pos.shape[0]):
            dist = torch.norm(robot_pos[i, :2] - person_pos)
            if dist < min_distance:
                rewards[i] += -10.0
            elif dist < comfort_distance:
                # 线性从 -1（边界）到 0（舒适距离）
                rewards[i] += -1.0 * (comfort_distance - dist) / (
                    comfort_distance - min_distance
                )

    return rewards


def anti_stuck_reward(
    env: ManagerBasedEnv,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    window_size: int = 30,
    threshold: float = 0.1,
):
    """卡住惩罚：复用 stuck_terminal_check 的判断逻辑，返回 tensor。

    stuck_terminal_check 返回 bool（scalar），这里包装成 (num_envs,) tensor。
    """
    is_stuck = stuck_terminal_check(
        env,
        robot_asset_cfg=robot_asset_cfg,
        window_size=window_size,
        threshold=threshold,
    )
    robot_asset = env.scene[robot_asset_cfg.name]
    num_envs = robot_asset.data.root_pos_w.shape[0]
    penalty = torch.zeros(num_envs, device=robot_asset.data.root_pos_w.device)
    if is_stuck:
        penalty[:] = -20.0
    return penalty


# ------------------------------------------------------------------
# 三个任务各自独立的 RewardsCfg
# ------------------------------------------------------------------

@configclass
class DynPointGoalRewardsCfg:
    """DingoDynPointGoalCfg 的奖励配置。

    目标是跟随行人并靠近（1m内），所以：
    - 不加 social_distance（会和 success 产生对抗梯度）
    - success = 到达动态行人附近
    - anti_stuck 防止原地不动
    - alive 提供微小 dense 信号稳定训练
    """

    arrive_goal = RewTerm(
        func=arrival_dynamic_target_check,
        weight=100.0,
        params={"robot_asset_cfg": SceneEntityCfg("robot"), "threshold": 1.0},
    )

    anti_stuck = RewTerm(
        func=anti_stuck_reward,
        weight=1.0,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "window_size": 30,
            "threshold": 0.1,
        },
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.01)


@configclass
class SocialNavRewardsCfg:
    """DingoSocialNavCfg 的奖励配置。

    目标是到达静态 Goal，行人是障碍物，需要维持社交距离：
    - success = 到达静态 Goal prim
    - social_distance = 保持礼貌距离
    - anti_stuck
    - alive
    """

    arrive_goal = RewTerm(
        func=arrival_terminal_check,
        weight=100.0,
        params={"robot_asset_cfg": SceneEntityCfg("robot")},
    )

    social_distance = RewTerm(
        func=social_distance_reward,
        weight=1.0,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "min_distance": 0.45,
            "comfort_distance": 1.2,
        },
    )

    anti_stuck = RewTerm(
        func=anti_stuck_reward,
        weight=1.0,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "window_size": 30,
            "threshold": 0.1,
        },
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.01)


@configclass
class DynExploreRewardsCfg:
    """DingoDynExploreCfg 的奖励配置。

    没有导航目标，核心是在有人环境中安全移动：
    - 无 arrive_goal（没有目标）
    - social_distance = 核心约束，不能撞行人
    - anti_stuck
    - alive
    """

    social_distance = RewTerm(
        func=social_distance_reward,
        weight=1.0,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "min_distance": 0.45,
            "comfort_distance": 1.2,
        },
    )

    anti_stuck = RewTerm(
        func=anti_stuck_reward,
        weight=1.0,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "window_size": 30,
            "threshold": 0.1,
        },
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.01)


# ------------------------------------------------------------------
# 三个任务的最终 Cfg 类（使用各自独立的 RewardsCfg）
# ------------------------------------------------------------------

@configclass
class DingoSocialNavCfg(ManagerBasedRLEnvCfg):
    """Dingo Social Navigation：到达静态目标，同时维持社交距离。"""

    scene: SocialNavSceneCfg = MISSING
    observations = SocialNavObservationsCfg()
    actions = DingoActionsCfg()
    terminations = SocialNavTerminationsCfg()
    events = SocialNavEventCfg()
    rewards = SocialNavRewardsCfg()        # ← 独立 cfg

    def __post_init__(self):
        self.sim.render_interval = 15
        self.decimation = 15
        self.episode_length_s = 120.0
        self.sim.dt = 0.01
        self.sim.disable_contact_processing = True
        self.people_simulation = True


@configclass
class DingoDynPointGoalCfg(ManagerBasedRLEnvCfg):
    """Dingo Dynamic Point Goal：跟随并靠近移动的行人。"""

    scene: DynPointGoalSceneCfg = MISSING
    observations = DynPointGoalObservationsCfg()
    actions = DingoActionsCfg()
    terminations = DynPointGoalTerminationsCfg()
    events = DynPointGoalEventCfg()
    rewards = DynPointGoalRewardsCfg()     # ← 独立 cfg

    def __post_init__(self):
        self.sim.render_interval = 15
        self.decimation = 15
        self.episode_length_s = 120.0
        self.sim.dt = 0.01
        self.sim.disable_contact_processing = True
        self.people_simulation = True


@configclass
class DingoDynExploreCfg(ManagerBasedRLEnvCfg):
    """Dingo Dynamic Explore：在有行人的环境中安全探索。"""

    scene: DynExploreSceneCfg = MISSING
    observations = DynExploreObservationsCfg()
    actions = DingoActionsCfg()
    terminations = DynExploreTerminationsCfg()
    events = DynExploreEventCfg()
    rewards = DynExploreRewardsCfg()       # ← 独立 cfg

    def __post_init__(self):
        self.sim.render_interval = 15
        self.decimation = 15
        self.episode_length_s = 120.0
        self.sim.dt = 0.01
        self.sim.disable_contact_processing = True
        self.people_simulation = True