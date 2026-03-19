"""
Linear-path RF Agent（独立版本，无继承）
所有方法直接复制自 baselines/cfm/policy_agent.py
仅 __init__ 使用本目录的 CFM_Policy（cond_pos_embed=+4，支持归一化）
"""
import torch
import numpy as np
import cv2
from PIL import Image
from matplotlib import colormaps as cm
from policy_network import CFM_Policy


class CFM_Agent:
    """
    Linear RF Agent.
    所有预处理/帧管理/推理逻辑复制自 baselines/cfm/policy_agent.py，
    仅 __init__ 使用本目录的 CFM_Policy（cond_pos_embed=+4，支持归一化）。
    """

    def __init__(self,
                 image_intrinsic,
                 image_size=224,
                 memory_size=8,
                 predict_size=24,
                 temporal_depth=16,
                 heads=8,
                 token_dim=384,
                 navi_model="./100.ckpt",
                 cfm_num_steps=5,
                 device='cuda:0',
                 normalization_config=None):
        self.image_intrinsic = image_intrinsic
        self.device = device
        self.predict_size = predict_size
        self.image_size = image_size
        self.memory_size = memory_size

        # 使用本目录的 CFM_Policy（cond_pos_embed=+4，支持归一化）
        self.navi_former = CFM_Policy(
            image_size, memory_size, predict_size,
            temporal_depth, heads, token_dim,
            device=device,
            normalization_config=normalization_config,
            cfm_num_steps=cfm_num_steps,
        )

        ckpt = torch.load(navi_model, map_location=self.device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        incompatible = self.navi_former.load_state_dict(ckpt, strict=False)
        print(f"[RF Agent] load_state_dict: missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}")
        self.navi_former.to(self.device)
        self.navi_former.eval()

    def reset(self, batch_size, threshold):
        """重置所有环境的状态"""
        self.batch_size = batch_size
        self.stop_threshold = threshold
        self.memory_queue = [[] for i in range(batch_size)]

    def reset_env(self, i):
        """重置单个环境的历史帧队列"""
        self.memory_queue[i] = []
    
    def project_trajectory(self, images, n_trajectories, n_values):
        """
        将轨迹投影到图像上进行可视化
        
        Args:
            images: 原始图像 (batch, H, W, 3)
            n_trajectories: 所有轨迹 (batch, 16, 24, 3)，相机坐标系
            n_values: 所有轨迹的价值 (batch, 16)
        
        Returns:
            拼接后的可视化图像 (H, W*batch, 3)
        """
        trajectory_masks = []
        
        for i in range(images.shape[0]):  # 遍历 batch
            trajectory_mask = np.array(images[i])  # 复制图像作为画布
            # trajectory_mask = trajectory_mask[..., ::-1]
            n_trajectory = n_trajectories[i, :, :, 0:2]  # 取 xy 坐标 (16, 24, 2)
            n_value = n_values[i]  # (16,)
            
            # 遍历每条轨迹
            for waypoints, value in zip(n_trajectory, n_value):
                # ===== 根据价值映射颜色 =====
                # norm_value = np.clip(-value * 0.1, 0, 1)  # 归一化到[0,1]，价值越高颜色越"热"
                fixed_min = -1.2
                fixed_max = 0.2

                # Clamp the value to be within the fixed range
                value = np.clip(value, fixed_min, fixed_max)

                # Normalize value to [0, 1]
                norm_value = (value - fixed_min) / (fixed_max - fixed_min)
                
                # colormap = cm.get('jet')  # 使用 jet 色图（蓝->红）
                colormap = cm.get_cmap('RdYlGn') 
                color = np.array(colormap(norm_value)[0:3] ) * 255.0  # RGB颜色
                
                # ===== 准备3D点（添加z=-0.2作为地面高度）=====
                input_points = np.zeros((waypoints.shape[0], 3)) - 0.2
                input_points[:, 0:2] = waypoints  # xy 来自轨迹
                input_points[:, 1] = -input_points[:, 1]  # 翻转y（相机坐标系转换）
                
                # ===== 投影到图像坐标系（针孔相机模型）=====
                # camera_z = v（图像纵坐标）
                camera_z = images[0].shape[0] - 1 - \
                          self.image_intrinsic[1][1] * input_points[:, 2] / (input_points[:, 0] + 1e-8) - \
                          self.image_intrinsic[1][2]
                # camera_x = u（图像横坐标）
                camera_x = self.image_intrinsic[0][0] * input_points[:, 1] / (input_points[:, 0] + 1e-8) + \
                          self.image_intrinsic[0][2]
                
                # ===== 绘制轨迹线段 =====
                for i in range(camera_x.shape[0] - 1):
                    try:
                        # 只绘制在图像范围内的点
                        if camera_x[i] > 0 and camera_z[i] > 0 and \
                           camera_x[i+1] > 0 and camera_z[i+1] > 0:
                            trajectory_mask = cv2.line(
                                trajectory_mask,
                                (int(camera_x[i]), int(camera_z[i])),      # 起点
                                (int(camera_x[i+1]), int(camera_z[i+1])),  # 终点
                                color.astype(np.uint8).tolist(),           # 颜色
                                5  # 线宽
                            )
                    except:
                        pass  # 忽略投影错误
            
            trajectory_masks.append(trajectory_mask)
        
        # 横向拼接所有 batch 的图像
        return np.concatenate(trajectory_masks, axis=1)

    def project_trajectory_depth(self, depth_images, n_trajectories, n_values):
        """
        将轨迹投影到深度图像上进行可视化
        
        Args:
            depth_images: 原始深度图像 (batch, H, W, 1)
            n_trajectories: 所有轨迹 (batch, 16, 24, 3)，相机坐标系
            n_values: 所有轨迹的价值 (batch, 16)
        
        Returns:
            拼接后的可视化图像 (H, W*batch, 3)
        """
        trajectory_masks = []
        
        for i in range(depth_images.shape[0]):  # 遍历 batch
            # 裁剪深度图像到[0.1, 3]范围
            depth_image_clipped = np.clip(depth_images[i], 0.1, 3)
            
            # 归一化到[0, 1]范围
            depth_image_normalized = (depth_image_clipped - 0.1) / (3 - 0.1)  # 线性归一化
            
            # 将深度图像转换为RGB格式
            depth_image_rgb = cv2.cvtColor((depth_image_normalized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            n_trajectory = n_trajectories[i, :, :, 0:2]  # 取 xy 坐标 (16, 24, 2)
            n_value = n_values[i]  # (16,)
            
            # 遍历每条轨迹
            for waypoints, value in zip(n_trajectory, n_value):
                # ===== 根据价值映射颜色 =====
                # norm_value = np.clip(-value * 0.1, 0, 1)  # 归一化到[0,1]，价值越高颜色越"热"
                # colormap = cm.get('jet')  # 使用 jet 色图（蓝->红）
                
                fixed_min = -1.2
                fixed_max = 0.2

                # Clamp the value to be within the fixed range
                value = np.clip(value, fixed_min, fixed_max)

                # Normalize value to [0, 1]
                norm_value = (value - fixed_min) / (fixed_max - fixed_min)
                
                colormap = cm.get_cmap('RdYlGn') 
                color = np.array(colormap(norm_value)[0:3] ) * 255.0  # RGB颜色
                
                # ===== 准备3D点（添加z=-0.2作为地面高度）=====
                input_points = np.zeros((waypoints.shape[0], 3)) - 0.2
                input_points[:, 0:2] = waypoints  # xy 来自轨迹
                input_points[:, 1] = -input_points[:, 1]  # 翻转y（相机坐标系转换）
                
                # ===== 投影到图像坐标系（针孔相机模型）=====
                # camera_z = v（图像纵坐标）
                camera_z = depth_images[i].shape[0] - 1 - \
                        self.image_intrinsic[1][1] * input_points[:, 2] / (input_points[:, 0] + 1e-8) - \
                        self.image_intrinsic[1][2]
                # camera_x = u（图像横坐标）
                camera_x = self.image_intrinsic[0][0] * input_points[:, 1] / (input_points[:, 0] + 1e-8) + \
                        self.image_intrinsic[0][2]
                
                # ===== 绘制轨迹线段 =====
                for j in range(camera_x.shape[0] - 1):
                    try:
                        # 只绘制在图像范围内的点
                        if camera_x[j] > 0 and camera_z[j] > 0 and \
                        camera_x[j+1] > 0 and camera_z[j+1] > 0 and \
                        camera_z[j] < depth_images[i].shape[0] and camera_z[j+1] < depth_images[i].shape[0]:
                            depth_image_rgb = cv2.line(
                                depth_image_rgb,
                                (int(camera_x[j]), int(camera_z[j])),      # 起点
                                (int(camera_x[j+1]), int(camera_z[j+1])),  # 终点
                                color.astype(np.uint8).tolist(),            # 颜色
                                2  # 线宽
                            )
                    except:
                        pass  # 忽略投影错误
                
            trajectory_masks.append(depth_image_rgb)
        
        # 横向拼接所有 batch 的图像
        return np.concatenate(trajectory_masks, axis=1)

    def process_image(self, images):
        """预处理 RGB 图像：resize到224x224并归一化"""
        assert len(images.shape) == 4
        H, W, C = images.shape[1], images.shape[2], images.shape[3]
        prop = self.image_size / max(H, W)

        return_images = []
        for img in images:
            resize_image = cv2.resize(img, (-1, -1), fx=prop, fy=prop)
            pad_width = max((self.image_size - resize_image.shape[1]) // 2, 0)
            pad_height = max((self.image_size - resize_image.shape[0]) // 2, 0)
            pad_image = np.pad(
                resize_image,
                ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
                mode='constant',
                constant_values=0
            )
            resize_image = cv2.resize(pad_image, (self.image_size, self.image_size))
            resize_image = np.array(resize_image)
            resize_image = resize_image.astype(np.float32) / 255.0
            return_images.append(resize_image)

        return np.array(return_images)

    def process_depth(self, depths):
        """预处理深度图：resize到224x224并过滤异常值"""
        assert len(depths.shape) == 4
        depths[depths == np.inf] = 0
        H, W, C = depths.shape[1], depths.shape[2], depths.shape[3]
        prop = self.image_size / max(H, W)

        return_depths = []
        for depth in depths:
            resize_depth = cv2.resize(depth, (-1, -1), fx=prop, fy=prop)
            pad_width = max((self.image_size - resize_depth.shape[1]) // 2, 0)
            pad_height = max((self.image_size - resize_depth.shape[0]) // 2, 0)
            pad_depth = np.pad(
                resize_depth,
                ((pad_height, pad_height), (pad_width, pad_width)),
                mode='constant',
                constant_values=0
            )
            resize_depth = cv2.resize(pad_depth, (self.image_size, self.image_size))
            resize_depth[resize_depth > 5.0] = 0
            resize_depth[resize_depth < 0.1] = 0
            return_depths.append(resize_depth[:, :, np.newaxis])

        return np.array(return_depths)

    def process_pixel(self, pixel_coords, input_images):
        """处理像素目标：创建一个mask标记目标像素位置"""
        return_pixels = []
        H, W, C = input_images.shape[1], input_images.shape[2], input_images.shape[3]
        prop = self.image_size / max(H, W)

        for pixel_coord, input_image in zip(pixel_coords, input_images):
            panel_image = np.zeros_like(input_image, dtype=np.uint8)
            min_x = pixel_coord[0] - 10
            min_y = pixel_coord[1] - 10
            max_x = pixel_coord[0] + 10
            max_y = pixel_coord[1] + 10

            if min_x <= 0:
                panel_image[:, 0:10] = 255
            elif min_y <= 0:
                panel_image[0:10, :] = 255
            elif max_x >= panel_image.shape[1]:
                panel_image[:, panel_image.shape[1]-10:] = 255
            elif max_y >= panel_image.shape[0]:
                panel_image[panel_image.shape[0]-10:, :] = 255
            elif min_x > 0 and min_y > 0 and max_x < panel_image.shape[1] and max_y < panel_image.shape[0]:
                panel_image[min_y:max_y, min_x:max_x] = 255

            resize_image = cv2.resize(
                panel_image, (-1, -1),
                fx=prop, fy=prop,
                interpolation=cv2.INTER_NEAREST
            )
            pad_width = max((self.image_size - resize_image.shape[1]) // 2, 0)
            pad_height = max((self.image_size - resize_image.shape[0]) // 2, 0)
            pad_image = np.pad(
                resize_image,
                ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
                mode='constant',
                constant_values=0
            )
            resize_image = cv2.resize(pad_image, (self.image_size, self.image_size))
            resize_image = np.array(resize_image)
            resize_image = resize_image.astype(np.float32) / 255.0
            return_pixels.append(resize_image)

        return np.array(return_pixels).mean(axis=-1)

    def process_pointgoal(self, goals):
        """处理点目标坐标：裁剪到合理范围"""
        clip_goals = goals.clip(-10, 10)
        clip_goals[:, 0] = np.clip(clip_goals[:, 0], 0, 10)
        return clip_goals

    def step_nogoal(self, images, depths):
        """无目标探索推理"""
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)

        input_images = []
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size:
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
                input_image = np.pad(
                    input_image,
                    ((self.memory_size - input_image.shape[0], 0), (0, 0), (0, 0), (0, 0))
                )
            else:
                del self.memory_queue[i][0]
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
            input_images.append(input_image)

        input_image = np.array(input_images)
        input_depth = process_depths

        all_trajectory, all_values, good_trajectory, bad_trajectory = \
            self.navi_former.predict_nogoal_action(input_image, input_depth)

        if all_values.max() < self.stop_threshold:
            good_trajectory[:, :, :, 0] = 0.0
            good_trajectory[:, :, :, 1] = np.sign(good_trajectory[:, :, :, 1].mean())

        trajectory_mask = self.project_trajectory(images, all_trajectory, all_values)
        trajectory_depth_mask =  self.project_trajectory_depth(depths, all_trajectory, all_values)

        return (
            good_trajectory[:, 0],
            all_trajectory,
            all_values,
            trajectory_mask,
            trajectory_depth_mask,
        )

    def step_pointgoal(self, goals, images, depths):
        """点目标导航推理"""
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)

        input_images = []
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size:
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
                input_image = np.pad(
                    input_image,
                    ((self.memory_size - input_image.shape[0], 0), (0, 0), (0, 0), (0, 0))
                )
            else:
                del self.memory_queue[i][0]
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
            input_images.append(input_image)

        input_image = np.array(input_images)
        input_depth = process_depths
        input_goals = self.process_pointgoal(goals)

        all_trajectory, all_values, good_trajectory, bad_trajectory = \
            self.navi_former.predict_pointgoal_action(input_goals, input_image, input_depth)

        if all_values.max() < self.stop_threshold:
            good_trajectory[:, :, :, 0] = 0.0
            good_trajectory[:, :, :, 1] = np.sign(good_trajectory[:, :, :, 1].mean())

        print(all_values.max(), all_values.min())
        trajectory_mask = self.project_trajectory(images, all_trajectory, all_values)
        trajectory_depth_mask =  self.project_trajectory_depth(depths, all_trajectory, all_values)

        return (
            good_trajectory[:, 0],
            all_trajectory,
            all_values,
            trajectory_mask,
            trajectory_depth_mask
        )

    def step_imagegoal(self, goals, images, depths):
        """图像目标导航推理"""
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)

        input_images = []
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size:
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
                input_image = np.pad(
                    input_image,
                    ((self.memory_size - input_image.shape[0], 0), (0, 0), (0, 0), (0, 0))
                )
            else:
                del self.memory_queue[i][0]
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
            input_images.append(input_image)

        input_image = np.array(input_images)
        input_depth = process_depths
        input_goals = self.process_image(goals)

        all_trajectory, all_values, good_trajectory, bad_trajectory = \
            self.navi_former.predict_imagegoal_action(input_goals, input_image, input_depth)

        if all_values.max() < self.stop_threshold:
            good_trajectory[:, :, :, 0] = 0.0
            good_trajectory[:, :, :, 1] = np.sign(good_trajectory[:, :, :, 1].mean())

        print(all_values.max(), all_values.min())
        trajectory_mask = self.project_trajectory(images, all_trajectory, all_values)
        trajectory_depth_mask =  self.project_trajectory_depth(depths, all_trajectory, all_values)

        return good_trajectory[:, 0], all_trajectory, all_values, trajectory_mask, trajectory_depth_mask

    def step_pixelgoal(self, goals, images, depths):
        """像素目标导航推理"""
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)

        input_images = []
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size:
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
                input_image = np.pad(
                    input_image,
                    ((self.memory_size - input_image.shape[0], 0), (0, 0), (0, 0), (0, 0))
                )
            else:
                del self.memory_queue[i][0]
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
            input_images.append(input_image)

        input_image = np.array(input_images)
        input_depth = process_depths
        input_goals = self.process_pixel(goals, images)

        all_trajectory, all_values, good_trajectory, bad_trajectory = \
            self.navi_former.predict_pixelgoal_action(input_goals, input_image, input_depth)

        if all_values.max() < self.stop_threshold:
            good_trajectory[:, :, :, 0] = 0.0
            good_trajectory[:, :, :, 1] = np.sign(good_trajectory[:, :, :, 1].mean())

        trajectory_mask = self.project_trajectory(images, all_trajectory, all_values)
        trajectory_depth_mask =  self.project_trajectory_depth(depths, all_trajectory, all_values)

        return good_trajectory[:, 0], all_trajectory, all_values, trajectory_mask, trajectory_depth_mask

    def step_point_image_goal(self, pointgoal, imagegoal, images, depths):
        """混合目标导航推理（点目标 + 图像目标）"""
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)

        input_images = []
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size:
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
                input_image = np.pad(
                    input_image,
                    ((self.memory_size - input_image.shape[0], 0), (0, 0), (0, 0), (0, 0))
                )
            else:
                del self.memory_queue[i][0]
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
            input_images.append(input_image)

        input_image = np.array(input_images)
        input_depth = process_depths
        input_pointgoal = self.process_pointgoal(pointgoal)
        input_imagegoal = self.process_image(imagegoal)

        all_trajectory, all_values, good_trajectory, bad_trajectory = \
            self.navi_former.predict_ip_action(
                input_pointgoal, input_imagegoal, input_image, input_depth
            )

        if all_values.max() < self.stop_threshold:
            good_trajectory[:, :, :, 0] = 0.0
            good_trajectory[:, :, :, 1] = torch.sign(good_trajectory[:, :, :, 1].mean())

        trajectory_mask = self.project_trajectory(images, all_trajectory, all_values)
        trajectory_depth_mask =  self.project_trajectory_depth(depths, all_trajectory, all_values)

        return good_trajectory[:, 0], all_trajectory, all_values, trajectory_mask, trajectory_depth_mask
