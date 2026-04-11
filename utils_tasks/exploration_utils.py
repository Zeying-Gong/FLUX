from __future__ import annotations

import numpy as np


def update_occupancy(global_pcd, camera_int, current_pos, current_rot, robot_rgb, robot_depth):
    """Update voxelized global pointcloud + compute explore area.

    This is copied from the eval_*_nogoal scripts to keep behavior identical.
    """
    from isaaclab.sensors.camera.utils import create_pointcloud_from_rgbd
    import torch

    from utils_tasks.basic_utils import cpu_pointcloud_from_array

    filter_rgb = torch.tensor(robot_rgb, device=robot_rgb.device)
    filter_depth = torch.tensor(robot_depth, device=robot_depth.device)
    filter_depth[filter_depth > 5.0] = 0
    points, colors = create_pointcloud_from_rgbd(
        camera_int,
        filter_depth,
        filter_rgb,
        position=current_pos,
        orientation=current_rot,
    )
    current_pcd = cpu_pointcloud_from_array(points.cpu().numpy(), colors.cpu().numpy())
    global_pcd = (global_pcd + current_pcd).voxel_down_sample(0.05)
    point_values = np.array(global_pcd.points)
    navigable_pcd = global_pcd.select_by_index(
        np.where(point_values[:, 2] < np.quantile(point_values[:, 2], 0.25) + 0.1)[0]
    )
    navigable_values = np.array(navigable_pcd.points)
    occupancy_dimension = np.ceil((navigable_values.max(axis=0) - navigable_values.min(axis=0)) / 0.1).astype(np.int32)
    occupancy_dimension[0] = max(occupancy_dimension[0], 1)
    occupancy_dimension[1] = max(occupancy_dimension[1], 1)
    occupancy_dimension[2] = max(occupancy_dimension[2], 1)
    occupancy_grid = np.zeros(occupancy_dimension)
    occupancy_index = np.floor((navigable_values - navigable_values.min(axis=0)) / 0.1).astype(np.int32)
    occupancy_grid[occupancy_index[:, 0], occupancy_index[:, 1], occupancy_index[:, 2]] = 1
    explore_area = occupancy_grid.sum() * 0.01
    return global_pcd, navigable_pcd, explore_area

