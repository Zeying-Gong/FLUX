import numpy as np
import cv2
from collections import deque
from scipy.ndimage import binary_dilation

from typing import List
from collections import deque
# ===== Social Distance Thresholds (Hall's Proxemics) =====
INTIMATE_SPACE = 0.45    # < 0.45m: 碰撞/亲密空间
PERSONAL_SPACE = 1.2     # 0.45-1.2m: 个人空间
SOCIAL_SPACE = 3.6       # 1.2-3.6m: 社交空间

# Draw all trajectories with colors based on values
# Define color mapping function from value to color (blue to red gradient)
def value_to_color(value, values_min, values_max):
    if not np.isfinite(value):
        return (128, 128, 128)
    # Normalize value to [0, 1] based on a fixed range [-2, 0.5]
    fixed_min = -1.2
    fixed_max = 0.2
    value = np.clip(value, fixed_min, fixed_max)
    normalized = (value - fixed_min) / (fixed_max - fixed_min)
    # 低分(危险)=红, 中=黄, 高分(安全)=绿
    if normalized < 0.5:
        b = 255
        g = int(255 * 2 * normalized)
        r = 0
    else:
        b = int(255 * (2 - 2 * normalized))
        g = 255
        r = 0
    return (b, g, r)  # BGR

# Helper function to transform world points to vis_coords
def transform_to_vis_coords(world_pts, current_pose, res, offset, size):
    if world_pts.size == 0:
        return np.array([])
        
    # Transform world points to yaw=0 frame centered at current robot position
    dx = world_pts[:, 0] - current_pose[0]
    dy = world_pts[:, 1] - current_pose[1]
    
    current_rotation = np.array([
        [np.cos(0), -np.sin(0)],
        [np.sin(0), np.cos(0)]
    ])
    transformed_points = (current_rotation @ np.vstack([dx, dy])).T
    
    # Convert to grid coordinates relative to center
    center_coords = (transformed_points / res).astype(int)
    
    # Filter points within visualization range
    valid_mask = (np.abs(center_coords[:, 0]) < size//2) & (np.abs(center_coords[:, 1]) < size//2)
    center_coords = center_coords[valid_mask]
    
    # Convert to visualization coordinates (adjust for image coordinate system)
    vis_coords = np.zeros_like(center_coords)
    vis_coords[:, 0] = -center_coords[:, 0] + offset  # Flip x axis
    vis_coords[:, 1] = -center_coords[:, 1] + offset   # Keep y axis
    
    # Final boundary check
    valid_mask = (vis_coords[:, 0] >= 0) & (vis_coords[:, 0] < size) & \
                (vis_coords[:, 1] >= 0) & (vis_coords[:, 1] < size)
    vis_coords = vis_coords[valid_mask]
    return vis_coords
    
class VisualizationManager:
    def __init__(self, history_size=5, global_map_size=40.0):
        self.history_size = history_size
        self.occupancy_history = deque(maxlen=history_size)  # Will store (grid, min_coords, robot_pose)
        
        # 新增：全局地图存储（不限制大小）
        self.global_occupancy_history = []  # 存储整个episode的所有帧
        
        self.resolution = 0.05  # 5cm per pixel
        self.inflation = 5      # inflation radius in pixels

        # ========== 新增：全局地图参数 ==========
        self.global_map_size = global_map_size  # 40m×40m (±20m)
        self.global_map_origin = None  # 机器人初始位置（世界坐标）
        self.global_grid_size = int(global_map_size / self.resolution)  # 800×800 pixels
        self.global_occupancy_map = np.zeros((self.global_grid_size, self.global_grid_size), dtype=np.uint8)
        # ========== 新增：行人历史轨迹存储 ==========
        self.people_history = {}  # {person_id: deque([(x, y), ...], maxlen=30)}
        self.people_history_length = 20  # 记录最近5帧

    def reset(self, initial_robot_pose=None):
        """重置时清空所有历史"""
        self.occupancy_history.clear()
        self.global_occupancy_history.clear()  # 清空全局历史

        # ========== 新增：设置全局地图原点 ==========
        if initial_robot_pose is not None:
            self.global_map_origin = initial_robot_pose[:2].copy()  # 记录初始位置(x, y)
        
        # 清空全局占据栅格
        self.global_occupancy_map = np.zeros((self.global_grid_size, self.global_grid_size), dtype=np.uint8)
        self.people_history.clear() # 清空行人历史
        
    def build_occupancy_grid(self, depth_map, intrinsic, camera_roll=0):
        try:
            """Convert depth image to occupancy grid in BEV"""
            if len(depth_map.shape) == 3:
                depth_map = depth_map[:,:,0]
            height, width = depth_map.shape
            uu, vv = np.meshgrid(np.arange(width), np.arange(height))
            z = depth_map
            x = (uu - intrinsic[0, 2]) * z / intrinsic[0, 0]
            y = (vv - intrinsic[1, 2]) * z / intrinsic[1, 1]
            
            # Filter valid points
            valid_mask = (z > 0) & np.isfinite(z) & (z < 10)
            points_3d = np.stack((x[valid_mask], y[valid_mask], z[valid_mask]), axis=-1)
            
            # Apply camera roll
            roll = camera_roll * np.pi / 180
            rotation_matrix_x = np.array([[1, 0, 0], 
                                        [0, np.cos(roll), -np.sin(roll)], 
                                        [0, np.sin(roll), np.cos(roll)]])
            point_3d_flat = (rotation_matrix_x @ points_3d.transpose()).transpose()
            
            # Transform to world coordinates
            point_3d_world = np.zeros((point_3d_flat.shape[0], 3))
            point_3d_world[:, 0] = point_3d_flat[:, 2]
            point_3d_world[:, 1] = -point_3d_flat[:, 0]
            point_3d_world[:, 2] = -point_3d_flat[:, 1]
            bins = np.arange(np.min(point_3d_world[:, 2]), np.max(point_3d_world[:, 2]), 0.05)
            try:
                hist, bin_edges = np.histogram(point_3d_world[:, 2], bins=bins)
                max_freq_index = np.argmax(hist)
                point_3d_world[:, 2] -= bin_edges[max_freq_index]
                # print(f"bin_edges[max_freq_index] {bin_edges[max_freq_index]}")
            except:
                point_3d_world[:, 2] -= -0.5
            
            # Filter points within height range
            filtered_points = point_3d_world[(point_3d_world[:, 2] >= 0.2) & (point_3d_world[:, 2] <= 1.5)]
            if filtered_points.shape[0] == 0:
                min_coords = np.array([-5.0,-5.0,-5.0])
                max_coords = np.array([5.0,5.0,5.0])
                grid_size = np.ceil((max_coords - min_coords) / self.resolution + 1).astype(int)
                occupancy_grid = np.zeros(grid_size[:2], dtype=np.int8)
                return occupancy_grid, min_coords
                
            # Create occupancy grid
            min_coords = np.min(filtered_points, axis=0)
            max_coords = np.max(filtered_points, axis=0)
            grid_size = np.ceil((max_coords - min_coords) / self.resolution + 1).astype(int)
            occupancy_grid = np.zeros(grid_size[:2], dtype=np.int8)
            
            grid_coords = ((filtered_points[:, :2] - min_coords[:2]) / self.resolution).astype(int)
            occupancy_grid[grid_coords[:, 0], grid_coords[:, 1]] = 1
            
        except:
            occupancy_grid = np.zeros((100,100),dtype=np.int8)
            min_coords = np.array([0,0])
        
        return occupancy_grid, min_coords
        
    def visualize_trajectory(self, rgb_image, depth_image, intrinsic, trajectory_points, robot_pose, camera_roll=0, all_trajectories_points=None, all_trajectories_values=None):
        # Calculate visualization size based on 10m×10m range
        grid_size = int(10.0 / self.resolution)  # 20m in grid cells
        vis_image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

        # Resize visualization to match RGB image height with better interpolation
        vis_resized = cv2.resize(vis_image, (int(rgb_image.shape[0]), int(rgb_image.shape[0])), interpolation=cv2.INTER_CUBIC)
        # Apply slight Gaussian blur to smooth pixelated edges (adjust sigma as needed)
        vis_resized = cv2.GaussianBlur(vis_resized, (3, 3), 0.5)
        
        # Concatenate images
        combined_image = np.concatenate((rgb_image, vis_resized), axis=1)
         
        # Build current occupancy grid
        occupancy_grid, min_coords = self.build_occupancy_grid(depth_image[..., 0], intrinsic, camera_roll)
        if occupancy_grid is None:
            return combined_image
        
        # Add to history with robot pose
        self.occupancy_history.append((occupancy_grid, min_coords, robot_pose))
        
        # Calculate center offset (assuming robot is at center)
        center_offset = grid_size // 2
        
        # Draw historical occupancy grids
        all_hist_world_points_list = []
        current_world_points = np.array([])

        # Process historical frames first
        for i, (hist_grid, hist_min_coords, hist_pose) in enumerate(self.occupancy_history):
            # Get occupied points in the grid's local frame
            grid_coords = np.where(hist_grid > 0)
            points = np.array([
                grid_coords[0] * self.resolution + hist_min_coords[0],
                grid_coords[1] * self.resolution + hist_min_coords[1]
            ]).T
            
            # Transform points from the grid's local frame to world frame
            hist_rotation = np.array([
                [np.cos(hist_pose[2]), -np.sin(hist_pose[2])],
                [np.sin(hist_pose[2]), np.cos(hist_pose[2])]
            ])
            world_points = (hist_rotation @ points.T).T + hist_pose[:2]

            if i == len(self.occupancy_history) - 1:  # Current frame
                current_world_points = world_points
            else:  # Historical frame
                if world_points.size > 0:
                    all_hist_world_points_list.append(world_points)

        # Combine all historical points
        if all_hist_world_points_list:
            all_hist_world_points = np.concatenate(all_hist_world_points_list, axis=0)
        else:
            all_hist_world_points = np.array([])

        # Helper function to transform world points to vis_coords
        def transform_to_vis_coords(world_pts, current_pose, res, offset, size):
            if world_pts.size == 0:
                return np.array([])
                
            # Transform world points to yaw=0 frame centered at current robot position
            dx = world_pts[:, 0] - current_pose[0]
            dy = world_pts[:, 1] - current_pose[1]
            
            current_rotation = np.array([
                [np.cos(0), -np.sin(0)],
                [np.sin(0), np.cos(0)]
            ])
            transformed_points = (current_rotation @ np.vstack([dx, dy])).T
            
            # Convert to grid coordinates relative to center
            center_coords = (transformed_points / res).astype(int)
            
            # Filter points within visualization range
            valid_mask = (np.abs(center_coords[:, 0]) < size//2) & (np.abs(center_coords[:, 1]) < size//2)
            center_coords = center_coords[valid_mask]
            
            # Convert to visualization coordinates (adjust for image coordinate system)
            vis_coords = np.zeros_like(center_coords)
            vis_coords[:, 0] = -center_coords[:, 0] + offset  # Flip x axis
            vis_coords[:, 1] = -center_coords[:, 1] + offset   # Keep y axis
            
            # Final boundary check
            valid_mask = (vis_coords[:, 0] >= 0) & (vis_coords[:, 0] < size) & \
                        (vis_coords[:, 1] >= 0) & (vis_coords[:, 1] < size)
            vis_coords = vis_coords[valid_mask]
            return vis_coords

        # Draw historical points (Gray)
        vis_coords_hist = transform_to_vis_coords(all_hist_world_points, robot_pose, self.resolution, center_offset, grid_size)
        if vis_coords_hist.size > 0:
            vis_image[vis_coords_hist[:, 0], vis_coords_hist[:, 1]] = (128, 128, 128) # Gray

        # Draw current points (Red)
        vis_coords_current = transform_to_vis_coords(current_world_points, robot_pose, self.resolution, center_offset, grid_size)
        if vis_coords_current.size > 0:
            vis_image[vis_coords_current[:, 0], vis_coords_current[:, 1]] = (0, 0, 255) # Red
        
        # Draw trajectory
        if trajectory_points is not None:
            # Transform trajectory points to yaw=0 frame centered at current robot position
            dx = trajectory_points[:, 0] - robot_pose[0]
            dy = trajectory_points[:, 1] - robot_pose[1]
            
            # Rotate points to align with yaw=0 frame
            current_rotation = np.array([
                [np.cos(0), np.sin(0)],
                [np.sin(0), np.cos(0)]
            ])
            transformed_points = (current_rotation @ np.vstack([dx, dy])).T
            
            # Convert to grid coordinates
            grid_points = (transformed_points / self.resolution).astype(int)
            
            # Filter points within range
            valid_mask = (np.abs(grid_points[:, 0]) < grid_size//2) & (np.abs(grid_points[:, 1]) < grid_size//2)
            grid_points = grid_points[valid_mask]
            
            # Convert to visualization coordinates (adjust for image coordinate system)
            vis_points = np.zeros_like(grid_points)
            vis_points[:, 0] = -grid_points[:, 1] + center_offset  # Flip x axis
            vis_points[:, 1] = -grid_points[:, 0] + center_offset   # Keep y axis
            
            # Draw trajectory with anti-aliased lines
            for i in range(len(vis_points) - 1):
                cv2.line(vis_image, tuple(vis_points[i]), tuple(vis_points[i+1]), (0, 255, 0), 2, cv2.LINE_AA)
            # Draw start and end points
            if len(vis_points) > 0:
                # Use larger circles with anti-aliasing for smoother appearance
                cv2.circle(vis_image, tuple(vis_points[0]), 3, (255, 0, 0), -1, cv2.LINE_AA)  # Blue for start
                
                # Draw robot rectangle at start position
                rect_length = 10  # pixels
                rect_width = 5   # pixels
                start_point = (center_offset, center_offset)
                # Get yaw angle from trajectory points
                yaw = -robot_pose[2]
                
                # Calculate rectangle corners
                cos_yaw = np.cos(yaw)
                sin_yaw = np.sin(yaw)
                corners = np.array([
                    [-rect_width/2, -rect_length/2],
                    [rect_width/2, -rect_length/2],
                    [rect_width/2, rect_length/2],
                    [-rect_width/2, rect_length/2]
                ])
                # Rotate corners
                rot_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
                rotated_corners = (rot_matrix @ corners.T).T + start_point
                
                # Draw rectangle with anti-aliasing
                corners_int = rotated_corners.astype(np.int32)
                cv2.polylines(vis_image, [corners_int], True, (255, 255, 255), 1, cv2.LINE_AA)  # Blue rectangle
                cv2.circle(vis_image, tuple(vis_points[-1]), 3, (0, 0, 255), -1, cv2.LINE_AA)  # Red for end
        
        # Resize visualization to match RGB image height with better interpolation
        vis_resized = cv2.resize(vis_image, (int(rgb_image.shape[0]), int(rgb_image.shape[0])), interpolation=cv2.INTER_CUBIC)
        # Apply slight Gaussian blur to smooth pixelated edges (adjust sigma as needed)
        vis_resized = cv2.GaussianBlur(vis_resized, (3, 3), 0.5)
        # Concatenate images
        combined_image = np.concatenate((rgb_image, vis_resized), axis=1)
        
        # If no all_trajectories_points, return original combined image
        if all_trajectories_points is None or len(all_trajectories_points) == 0:
            return combined_image
        # print(f"all_trajectories_points: {len(all_trajectories_points)}")
        # --- Create additional visualization for all trajectories ---
        # Create a new image for all trajectories visualization
        vis_image_all = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        
        # Draw the same occupancy grid
        if vis_coords_hist.size > 0:
            vis_image_all[vis_coords_hist[:, 0], vis_coords_hist[:, 1]] = (128, 128, 128) # Gray
        if vis_coords_current.size > 0:
            vis_image_all[vis_coords_current[:, 0], vis_coords_current[:, 1]] = (0, 0, 255) # Red
        
        # Set default colors if no values provided
        if all_trajectories_values is None:
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
            trajectory_colors = [colors[idx % len(colors)] for idx in range(len(all_trajectories_points))]
        else:
            # Get min and max values for normalization
            values_min = np.min(all_trajectories_values)
            values_max = np.max(all_trajectories_values)
            # Generate color for each trajectory
            trajectory_colors = [value_to_color(v, values_min, values_max) for v in all_trajectories_values]
        
        for idx, traj in enumerate(all_trajectories_points):
            color = trajectory_colors[idx]
            
            # Transform trajectory points
            dx = traj[:, 0] - robot_pose[0]
            dy = traj[:, 1] - robot_pose[1]
            
            # Rotate points
            transformed_points = (current_rotation @ np.vstack([dx, dy])).T
            
            # Convert to grid coordinates
            grid_points = (transformed_points / self.resolution).astype(int)
            
            # Filter points within range
            valid_mask = (np.abs(grid_points[:, 0]) < grid_size//2) & (np.abs(grid_points[:, 1]) < grid_size//2)
            grid_points = grid_points[valid_mask]
            
            # Convert to visualization coordinates
            vis_points_all = np.zeros_like(grid_points)
            vis_points_all[:, 0] = -grid_points[:, 1] + center_offset
            vis_points_all[:, 1] = -grid_points[:, 0] + center_offset
            
            # Draw trajectory with anti-aliased lines
            for i in range(len(vis_points_all) - 1):
                cv2.line(vis_image_all, tuple(vis_points_all[i]), tuple(vis_points_all[i+1]), color, 1, cv2.LINE_AA)
                
            # Draw start and end points with anti-aliasing
            if len(vis_points_all) > 0:
                cv2.circle(vis_image_all, tuple(vis_points_all[0]), 2, color, -1, cv2.LINE_AA)
        
        # Draw robot position with anti-aliasing
        if len(vis_points) > 0:
            corners_int = rotated_corners.astype(np.int32)
            cv2.polylines(vis_image_all, [corners_int], True, (255, 255, 255), 1, cv2.LINE_AA)  # White robot outline
        
        # Resize all trajectories visualization - align width with rgb_image
        # Get target width (same as rgb_image width)
        target_width = rgb_image.shape[1]
        # Calculate the height to maintain aspect ratio
        target_height = int(vis_image_all.shape[0] * (target_width / vis_image_all.shape[1]))
        # Resize with the calculated dimensions using better interpolation
        vis_resized_all = cv2.resize(vis_image_all, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        # Apply slight Gaussian blur to smooth pixelated edges
        vis_resized_all = cv2.GaussianBlur(vis_resized_all, (3, 3), 0.5)
        
        # Create black padding to match combined_image width
        if combined_image.shape[1] > target_width:
            # Add black padding to the right
            padding_width = combined_image.shape[1] - target_width
            padding = np.zeros((target_height, padding_width, 3), dtype=np.uint8)
            vis_resized_all = np.concatenate((vis_resized_all, padding), axis=1)
        
        # Stack vertically: combined_image (top) and vis_resized_all (bottom)
        final_combined_image = np.concatenate((combined_image, vis_resized_all), axis=0)
        
        return final_combined_image 

    def visualize_trajectory_global_with_people(self, rgb_image, depth_image, intrinsic, trajectory_points, 
                                                robot_pose, goal_position=None, camera_roll=0, 
                                                all_trajectories_points=None, all_trajectories_values=None,
                                                people_positions=None,people_positions_dict=None):
        """
        带行人可视化的全局轨迹可视化
        
        新增参数:
            people_positions: List[np.ndarray] - 行人位置列表 [[x, y, z], ...]
            people_positions_dict: List[str] - 对应的角色路径（稳定ID）
        """
        
        # ========== RGB尺寸定义 ==========
        rgb_height, rgb_width = rgb_image.shape[:2]
        local_map_size = rgb_height
        bottom_map_size = 500

        grid_size = int(10.0 / self.resolution)
        vis_image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

        # Build current occupancy grid
        occupancy_grid, min_coords = self.build_occupancy_grid(depth_image[..., 0], intrinsic, camera_roll)
        if occupancy_grid is None:
            combined_image = np.concatenate((rgb_image, cv2.resize(vis_image, (rgb_image.shape[0], rgb_image.shape[0]))), axis=1)
            return combined_image
        
        self.occupancy_history.append((occupancy_grid, min_coords, robot_pose))
        self.global_occupancy_history.append((occupancy_grid, min_coords, robot_pose))
        
        center_offset = grid_size // 2
        
        # ==================== 右上角：近期5帧地图 + 最优轨迹 + 行人 ====================
        all_hist_world_points_list = []
        current_world_points = np.array([])

        for i, (hist_grid, hist_min_coords, hist_pose) in enumerate(self.occupancy_history):
            grid_coords = np.where(hist_grid > 0)
            points = np.array([
                grid_coords[0] * self.resolution + hist_min_coords[0],
                grid_coords[1] * self.resolution + hist_min_coords[1]
            ]).T
            
            hist_rotation = np.array([
                [np.cos(hist_pose[2]), -np.sin(hist_pose[2])],
                [np.sin(hist_pose[2]), np.cos(hist_pose[2])]
            ])
            world_points = (hist_rotation @ points.T).T + hist_pose[:2]

            if i == len(self.occupancy_history) - 1:
                current_world_points = world_points
            else:
                if world_points.size > 0:
                    all_hist_world_points_list.append(world_points)

        if all_hist_world_points_list:
            all_hist_world_points = np.concatenate(all_hist_world_points_list, axis=0)
        else:
            all_hist_world_points = np.array([])

        vis_coords_hist = transform_to_vis_coords(all_hist_world_points, robot_pose, self.resolution, center_offset, grid_size)
        if vis_coords_hist.size > 0:
            vis_image[vis_coords_hist[:, 0], vis_coords_hist[:, 1]] = (128, 128, 128)

        vis_coords_current = transform_to_vis_coords(current_world_points, robot_pose, self.resolution, center_offset, grid_size)
        if vis_coords_current.size > 0:
            vis_image[vis_coords_current[:, 0], vis_coords_current[:, 1]] = (0, 0, 255)
        
        current_rotation = np.array([
            [np.cos(0), np.sin(0)],
            [np.sin(0), np.cos(0)]
        ])
        
        # ===== 在右上角地图绘制行人 =====
        vis_image = self._draw_people_on_local_map(vis_image, robot_pose, people_positions, 
                                                    grid_size, center_offset)
        
        # Draw trajectory
        if trajectory_points is not None:
            dx = trajectory_points[:, 0] - robot_pose[0]
            dy = trajectory_points[:, 1] - robot_pose[1]
            
            transformed_points = (current_rotation @ np.vstack([dx, dy])).T
            grid_points = (transformed_points / self.resolution).astype(int)
            
            valid_mask = (np.abs(grid_points[:, 0]) < grid_size//2) & (np.abs(grid_points[:, 1]) < grid_size//2)
            grid_points = grid_points[valid_mask]
            
            vis_points = np.zeros_like(grid_points)
            vis_points[:, 0] = -grid_points[:, 1] + center_offset
            vis_points[:, 1] = -grid_points[:, 0] + center_offset
            
            for i in range(len(vis_points) - 1):
                cv2.line(vis_image, tuple(vis_points[i]), tuple(vis_points[i+1]), (0, 128, 0), 2, cv2.LINE_AA)
            
            if len(vis_points) > 0:
                cv2.circle(vis_image, tuple(vis_points[0]), 3, (255, 0, 0), -1, cv2.LINE_AA)
                
                rect_length = 10
                rect_width = 5
                start_point = (center_offset, center_offset)
                yaw = -robot_pose[2]
                
                cos_yaw = np.cos(yaw)
                sin_yaw = np.sin(yaw)
                corners = np.array([
                    [-rect_width/2, -rect_length/2],
                    [rect_width/2, -rect_length/2],
                    [rect_width/2, rect_length/2],
                    [-rect_width/2, rect_length/2]
                ])
                rot_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
                rotated_corners = (rot_matrix @ corners.T).T + start_point
                
                corners_int = rotated_corners.astype(np.int32)
                cv2.polylines(vis_image, [corners_int], True, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(vis_image, tuple(vis_points[-1]), 3, (255, 182, 193), -1, cv2.LINE_AA)
        
        vis_resized = cv2.resize(vis_image, (local_map_size, local_map_size), interpolation=cv2.INTER_CUBIC)
        vis_resized = cv2.GaussianBlur(vis_resized, (3, 3), 0.5)
        combined_image = np.concatenate((rgb_image, vis_resized), axis=1)
        
        # ==================== 左下角：所有候选轨迹 + 行人 ====================
        if all_trajectories_points is None or len(all_trajectories_points) == 0:
            vis_global = self._create_global_map_with_people(robot_pose, goal_position, people_positions)
            vis_global_resized = cv2.resize(vis_global, (bottom_map_size, bottom_map_size), interpolation=cv2.INTER_CUBIC)
            vis_global_resized = cv2.GaussianBlur(vis_global_resized, (3, 3), 0.5)
            
            target_width = rgb_image.shape[1]
            if combined_image.shape[1] > bottom_map_size:
                padding_width = combined_image.shape[1] - bottom_map_size
                padding = np.zeros((bottom_map_size, padding_width, 3), dtype=np.uint8)
                vis_global_resized = np.concatenate((vis_global_resized, padding), axis=1)
            
            final_combined_image = np.concatenate((combined_image, vis_global_resized), axis=0)
            return final_combined_image
        
        vis_image_all = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        
        if vis_coords_hist.size > 0:
            vis_image_all[vis_coords_hist[:, 0], vis_coords_hist[:, 1]] = (128, 128, 128)
        if vis_coords_current.size > 0:
            vis_image_all[vis_coords_current[:, 0], vis_coords_current[:, 1]] = (0, 0, 255)
        
        # ===== 在左下角地图绘制行人 =====
        vis_image_all = self._draw_people_on_local_map(vis_image_all, robot_pose, people_positions,
                                                        grid_size, center_offset)
        
        if all_trajectories_values is None:
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
            trajectory_colors = [colors[idx % len(colors)] for idx in range(len(all_trajectories_points))]
        else:
            values_min = np.min(all_trajectories_values)
            values_max = np.max(all_trajectories_values)
            trajectory_colors = [value_to_color(v, values_min, values_max) for v in all_trajectories_values]
        
        for idx, traj in enumerate(all_trajectories_points):
            color = trajectory_colors[idx]
            
            dx = traj[:, 0] - robot_pose[0]
            dy = traj[:, 1] - robot_pose[1]
            
            transformed_points = (current_rotation @ np.vstack([dx, dy])).T
            grid_points = (transformed_points / self.resolution).astype(int)
            
            valid_mask = (np.abs(grid_points[:, 0]) < grid_size//2) & (np.abs(grid_points[:, 1]) < grid_size//2)
            grid_points = grid_points[valid_mask]
            
            vis_points_all = np.zeros_like(grid_points)
            vis_points_all[:, 0] = -grid_points[:, 1] + center_offset
            vis_points_all[:, 1] = -grid_points[:, 0] + center_offset
            
            for i in range(len(vis_points_all) - 1):
                cv2.line(vis_image_all, tuple(vis_points_all[i]), tuple(vis_points_all[i+1]), color, 1, cv2.LINE_AA)
                
            if len(vis_points_all) > 0:
                cv2.circle(vis_image_all, tuple(vis_points_all[0]), 2, color, -1, cv2.LINE_AA)
        
        if trajectory_points is not None and len(vis_points) > 0:
            corners_int = rotated_corners.astype(np.int32)
            cv2.polylines(vis_image_all, [corners_int], True, (255, 255, 255), 1, cv2.LINE_AA)
        
        vis_resized_all = cv2.resize(vis_image_all, (bottom_map_size, bottom_map_size), interpolation=cv2.INTER_CUBIC)
        vis_resized_all = cv2.GaussianBlur(vis_resized_all, (3, 3), 0.5)

        # ==================== 右下角：全局地图 + 行人 ====================
        # ===== 新增：更新行人历史轨迹（基于稳定char_path ID）=====
        if people_positions is not None and len(people_positions) > 0:
            # people_positions_dict 是 {char_path: position} 的字典
            for char_path, person_pos in zip(people_positions_dict, people_positions):
                # 使用 char_path 作为稳定的 person_id
                if char_path not in self.people_history:
                    self.people_history[char_path] = deque(maxlen=self.people_history_length)
                
                self.people_history[char_path].append(person_pos[:2].copy())
            
            # 清除不存在的行人（已经消失的char_path）
            current_char_paths = set(people_positions_dict)
            disappeared_paths = set(self.people_history.keys()) - current_char_paths
            for char_path in disappeared_paths:
                del self.people_history[char_path]

        # ==================== 为子图添加白色边框 ====================
        # 定义边框颜色  和粗细
        border_color = (180, 180, 180) 
        thickness = 2

        # ==================== 右下角：全局地图 + 行人 + 规划轨迹 ====================
        # 修改这里，传入trajectory_points参数
        vis_global = self._create_global_map_with_people(robot_pose, goal_position, people_positions, trajectory_points)
        vis_global_resized = cv2.resize(vis_global, (bottom_map_size, bottom_map_size), interpolation=cv2.INTER_CUBIC)
        vis_global_resized = cv2.GaussianBlur(vis_global_resized, (3, 3), 0.5)

        # 给右下角子图画框以区分
        cv2.rectangle(vis_global_resized, (0, 0), (bottom_map_size-1, bottom_map_size-1), border_color, thickness)

        bottom_row = np.concatenate((vis_resized_all, vis_global_resized), axis=1)
        final_combined_image = np.concatenate((combined_image, bottom_row), axis=0)
        
        return final_combined_image



    def _draw_people_on_local_map(self, vis_image, robot_pose, people_positions, grid_size, center_offset):
        """在局部地图上绘制行人（带距离颜色编码）"""
        if people_positions is None or len(people_positions) == 0:
            return vis_image
        
        robot_pos_2d = robot_pose[:2]
        
        for person_pos in people_positions:
            # 计算相对位置
            dx = person_pos[0] - robot_pose[0]
            dy = person_pos[1] - robot_pose[1]
            
            # 转换到可视化坐标
            current_rotation = np.array([
                [np.cos(0), -np.sin(0)],
                [np.sin(0), np.cos(0)]
            ])
            transformed = current_rotation @ np.array([dx, dy])
            grid_coord = (transformed / self.resolution).astype(int)
            
            # 检查是否在范围内
            if np.abs(grid_coord[0]) >= grid_size//2 or np.abs(grid_coord[1]) >= grid_size//2:
                continue
            
            vis_x = -grid_coord[0] + center_offset
            vis_y = -grid_coord[1] + center_offset
            
            if not (0 <= vis_x < grid_size and 0 <= vis_y < grid_size):
                continue
            
            # 计算距离并选择颜色
            dist = np.linalg.norm(person_pos[:2] - robot_pos_2d)
            
            if dist < INTIMATE_SPACE:
                color = (255, 0, 0)      # 红色: 碰撞危险
                radius = 8
            elif dist < PERSONAL_SPACE:
                color = (255, 165, 0)    # 橙色: personal space
                radius = 7
            elif dist < SOCIAL_SPACE:
                color = (255, 255, 0)    # 黄色: social space
                radius = 6
            else:
                color = (0, 255, 0)    # 绿色: public space
                radius = 5
            
            # 绘制行人（圆形 + 距离文字）
            cv2.circle(vis_image, (vis_y, vis_x), radius, color, -1, cv2.LINE_AA)
            cv2.circle(vis_image, (vis_y, vis_x), radius + 2, color, 1, cv2.LINE_AA)
            
            # 显示距离
            dist_text = f"{dist:.1f}"
            cv2.putText(vis_image, dist_text, (vis_y + radius + 2, vis_x + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
        
        return vis_image

    def _create_global_map_with_people(self, robot_pose, goal_position, people_positions, trajectory_points=None):
        """创建带行人的全局累积地图"""
        grid_size = self.global_grid_size
        vis_global = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        center_offset = grid_size // 2
        
        # ========== 收集所有历史帧的点 ==========
        all_global_world_points_list = []
        
        for hist_grid, hist_min_coords, hist_pose in self.global_occupancy_history:
            grid_coords = np.where(hist_grid > 0)
            points = np.array([
                grid_coords[0] * self.resolution + hist_min_coords[0],
                grid_coords[1] * self.resolution + hist_min_coords[1]
            ]).T
            
            hist_rotation = np.array([
                [np.cos(hist_pose[2]), -np.sin(hist_pose[2])],
                [np.sin(hist_pose[2]), np.cos(hist_pose[2])]
            ])
            world_points = (hist_rotation @ points.T).T + hist_pose[:2]
            
            if world_points.size > 0:
                all_global_world_points_list.append(world_points)
        
        if all_global_world_points_list:
            all_global_world_points = np.concatenate(all_global_world_points_list, axis=0)
        else:
            all_global_world_points = np.array([])
        
        # ========== 绘制障碍物 ==========
        vis_coords_global = transform_to_vis_coords(all_global_world_points, robot_pose, 
                                                    self.resolution, center_offset, grid_size)
        if vis_coords_global.size > 0:
            vis_global[vis_coords_global[:, 0], vis_coords_global[:, 1]] = (200, 200, 200)
        
        # ========== 新增：绘制规划轨迹 ==========
        pink_color = (255, 182, 193)  # 粉色
        
        if trajectory_points is not None and len(trajectory_points) > 0:
            # 计算轨迹点相对于机器人的位置
            dx_traj = trajectory_points[:, 0] - robot_pose[0]
            dy_traj = trajectory_points[:, 1] - robot_pose[1]
            
            # 转换到机器人局部坐标系（yaw=0）
            traj_rotation = np.array([
                [np.cos(0), -np.sin(0)],
                [np.sin(0), np.cos(0)]
            ])
            traj_local = (traj_rotation @ np.vstack([dx_traj, dy_traj])).T
            
            # 转换到栅格坐标
            traj_grid = (traj_local / self.resolution).astype(int)
            
            # 转换到可视化坐标
            traj_vis_coords = np.zeros_like(traj_grid)
            traj_vis_coords[:, 0] = -traj_grid[:, 0] + center_offset
            traj_vis_coords[:, 1] = -traj_grid[:, 1] + center_offset
            
            # 过滤在地图范围内的点
            valid_mask = (traj_vis_coords[:, 0] >= 0) & (traj_vis_coords[:, 0] < grid_size) & \
                        (traj_vis_coords[:, 1] >= 0) & (traj_vis_coords[:, 1] < grid_size)
            traj_vis_coords = traj_vis_coords[valid_mask]
            
            # 绘制轨迹线
            for i in range(len(traj_vis_coords) - 1):
                pt1 = (traj_vis_coords[i, 1], traj_vis_coords[i, 0])  # (y, x) for cv2
                pt2 = (traj_vis_coords[i+1, 1], traj_vis_coords[i+1, 0])
                cv2.line(vis_global, pt1, pt2, (0, 128, 0), 3, cv2.LINE_AA)
            
            # 绘制轨迹终点
            if len(traj_vis_coords) > 0:
                end_point = (traj_vis_coords[-1, 1], traj_vis_coords[-1, 0])
                cv2.circle(vis_global, end_point, 6, pink_color, -1, cv2.LINE_AA)
                cv2.circle(vis_global, end_point, 8, (255, 255, 255), 1, cv2.LINE_AA)  # 白色外圈
        
        # ========== 新增：绘制行人历史轨迹 ==========
        for person_id, history in self.people_history.items():
            if len(history) < 2:
                continue
            
            # 转换历史轨迹到可视化坐标
            history_array = np.array(list(history))
            
            dx_hist = history_array[:, 0] - robot_pose[0]
            dy_hist = history_array[:, 1] - robot_pose[1]
            
            hist_rotation = np.array([
                [np.cos(0), -np.sin(0)],
                [np.sin(0), np.cos(0)]
            ])
            hist_local = (hist_rotation @ np.vstack([dx_hist, dy_hist])).T
            hist_grid = (hist_local / self.resolution).astype(int)
            
            hist_vis_coords = np.zeros_like(hist_grid)
            hist_vis_coords[:, 0] = -hist_grid[:, 0] + center_offset
            hist_vis_coords[:, 1] = -hist_grid[:, 1] + center_offset
            
            # 过滤在地图范围内的点
            valid_mask = (hist_vis_coords[:, 0] >= 0) & (hist_vis_coords[:, 0] < grid_size) & \
                        (hist_vis_coords[:, 1] >= 0) & (hist_vis_coords[:, 1] < grid_size)
            
            # 绘制历史轨迹点（浅灰色圆点效果，替代直线）
            for i in range(len(hist_vis_coords)):
                # 跳过无效的轨迹点
                if not valid_mask[i]:
                    continue
                
                # 获取当前轨迹点坐标（注意坐标顺序：y, x）
                pt = (hist_vis_coords[i, 1], hist_vis_coords[i, 0])
                
                # 渐变透明度效果（越旧越淡）
                alpha = (i + 1) / len(hist_vis_coords)  # 越新的点alpha越接近1，颜色越深
                color_intensity = int(150 * alpha + 50)  # 颜色强度范围：50(淡灰) - 200(深灰)
                
                # 绘制圆形轨迹点（替换原来的line）
                # 参数说明：图像、圆心、半径、颜色、填充(-1)、抗锯齿
                cv2.circle(
                    img=vis_global,
                    center=pt,
                    radius=8,  # 圆点半径
                    color=(color_intensity, color_intensity, color_intensity),
                    thickness=-1,  # -1表示填充圆形，0/正数是描边
                    lineType=cv2.LINE_AA
                )

        # ========== 绘制行人（在全局地图上）==========
        if people_positions is not None and len(people_positions) > 0:
            robot_pos_2d = robot_pose[:2]
            
            for person_pos in people_positions:
                dx = person_pos[0] - robot_pose[0]
                dy = person_pos[1] - robot_pose[1]
                
                goal_rotation = np.array([
                    [np.cos(0), -np.sin(0)],
                    [np.sin(0), np.cos(0)]
                ])
                person_local = goal_rotation @ np.array([dx, dy])
                person_grid = (person_local / self.resolution).astype(int)
                
                person_vis_x = -person_grid[0] + center_offset
                person_vis_y = -person_grid[1] + center_offset
                
                if 0 <= person_vis_x < grid_size and 0 <= person_vis_y < grid_size:
                    dist = np.linalg.norm(person_pos[:2] - robot_pos_2d)
                    
                    if dist < INTIMATE_SPACE:
                        color = (255, 0, 0)      # 红色: 碰撞危险
                        radius = 10
                    elif dist < PERSONAL_SPACE:
                        color = (255, 165, 0)    # 橙色: personal space
                        radius = 9
                    elif dist < SOCIAL_SPACE:
                        color = (255, 255, 0)    # 黄色: social space
                        radius = 8
                    else:
                        color = (0, 255, 0)    # 绿色: public space
                        radius = 7
                    
                    cv2.circle(vis_global, (person_vis_y, person_vis_x), radius, color, -1, cv2.LINE_AA)
                    cv2.circle(vis_global, (person_vis_y, person_vis_x), radius + 2, (255, 255, 255), 1, cv2.LINE_AA)
                    
        # ========== 绘制机器人箭头 ==========
        robot_center = (center_offset, center_offset)
        
        rect_length = 10
        rect_width = 5
        yaw = -robot_pose[2]
        
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        arrow_points_local = np.array([
            [0, -rect_length],
            [-rect_width, rect_length/2],
            [rect_width, rect_length/2]
        ])
        
        rot_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        rotated_arrow = (rot_matrix @ arrow_points_local.T).T + robot_center
        
        arrow_int = rotated_arrow.astype(np.int32)
        cv2.fillPoly(vis_global, [arrow_int], (0, 0, 0))
        cv2.polylines(vis_global, [arrow_int], True, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(vis_global, robot_center, 3, (128, 0, 0), -1, cv2.LINE_AA)
                    
        # ========== Goal位置绘制 ==========
        if goal_position is not None:
            dx_goal = goal_position[0] - robot_pose[0]
            dy_goal = goal_position[1] - robot_pose[1]
            
            goal_rotation = np.array([
                [np.cos(0), -np.sin(0)],
                [np.sin(0), np.cos(0)]
            ])
            goal_local = goal_rotation @ np.array([dx_goal, dy_goal])
            goal_grid = (goal_local / self.resolution).astype(int)
            
            goal_vis_x = -goal_grid[0] + center_offset
            goal_vis_y = -goal_grid[1] + center_offset
            
            if goal_vis_x < 0 or goal_vis_x >= grid_size or goal_vis_y < 0 or goal_vis_y >= grid_size:
                goal_distance = np.sqrt(dx_goal**2 + dy_goal**2)
                if goal_distance > 0:
                    dir_x = dx_goal / goal_distance
                    dir_y = dy_goal / goal_distance
                    
                    dir_local = goal_rotation @ np.array([dir_x, dir_y])
                    
                    edge_distance = grid_size * 0.4
                    edge_x = int(center_offset - dir_local[0] * edge_distance)
                    edge_y = int(center_offset - dir_local[1] * edge_distance)
                    
                    edge_x = np.clip(edge_x, 20, grid_size - 20)
                    edge_y = np.clip(edge_y, 20, grid_size - 20)
                    
                    arrow_angle = np.arctan2(-dir_local[1], -dir_local[0])
                    arrow_len = 15
                    tip_x = int(edge_x - arrow_len * np.cos(arrow_angle))
                    tip_y = int(edge_y - arrow_len * np.sin(arrow_angle))
                    
                    cv2.arrowedLine(vis_global, (edge_y, edge_x), (tip_y, tip_x), 
                                (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.4)
                    
                    distance_text = f"{goal_distance:.1f}m"
                    cv2.putText(vis_global, distance_text, (edge_y + 10, edge_x), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                vis_goal = (goal_vis_y, goal_vis_x)
                cv2.circle(vis_global, vis_goal, 12, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(vis_global, vis_goal, 15, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(vis_global, vis_goal, 18, (255, 255, 255), 1, cv2.LINE_AA)

                # 新增：目标点G字母标注（核心优化）
                text = 'G'
                font = cv2.FONT_HERSHEY_SIMPLEX  # 经典无衬线字体，清晰易读（推荐）
                font_scale = 0.6                 # 字体大小，适配12px内圆
                font_thickness = 2               # 字体粗细，保证文字清晰
                text_color = (255, 255, 255)     # 白色文字，与红色内圆强对比

                # 计算文字绘制位置：让G字母**居中**在红色内圆中（关键步骤）
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_origin = (
                    vis_goal[0] - text_size[0] // 2,  # 水平居中
                    vis_goal[1] + text_size[1] // 2   # 垂直居中
                )

                # 绘制G字母（抗锯齿，与圆环风格统一）
                cv2.putText(
                    vis_global, text, text_origin,
                    font, font_scale, text_color,
                    font_thickness, cv2.LINE_AA
                )
        
        # ========== 添加图例（已放大） ==========
        legend_x = 10
        # 调整图例起始y坐标，留出更多空间
        legend_y = grid_size - 150
        
        # 1. 标题文字：放大字体大小和粗细
        cv2.putText(vis_global, "D_human:", (legend_x, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # 2. 亲密距离：放大圆形半径、字体大小和粗细，增加间距
        cv2.circle(vis_global, (legend_x + 20, legend_y + 25), 10, (255, 0, 0), -1)
        cv2.putText(vis_global, f"<{INTIMATE_SPACE}m", (legend_x + 40, legend_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # 3. 个人距离：同样放大
        cv2.circle(vis_global, (legend_x + 20, legend_y + 60), 10, (255, 165, 0), -1)
        cv2.putText(vis_global, f"<{PERSONAL_SPACE}m", (legend_x + 40, legend_y + 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        
        # 4. 社交距离：同样放大
        cv2.circle(vis_global, (legend_x + 20, legend_y + 95), 10, (255, 255, 0), -1)
        cv2.putText(vis_global, f"<{SOCIAL_SPACE}m", (legend_x + 40, legend_y + 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # 5. 公共距离：同样放大
        cv2.circle(vis_global, (legend_x + 20, legend_y + 130), 10, (0, 255, 0), -1)
        cv2.putText(vis_global, "public", (legend_x + 40, legend_y + 135), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return vis_global