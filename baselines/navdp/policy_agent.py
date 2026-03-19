"""
NavDP Agent 类
负责数据预处理、历史帧管理、调用策略网络推理
"""
import torch
import numpy as np
import cv2
from PIL import Image
from matplotlib import colormaps as cm
from policy_network import NavDP_Policy

class NavDP_Agent:
    """
    NavDP 导航代理
    封装了完整的导航推理流程：
    1. 图像预处理（resize到224x224，归一化）
    2. 历史帧队列管理（维护8帧时序记忆）
    3. 调用扩散策略网络生成轨迹
    4. 轨迹可视化投影
    """
    def __init__(self,
                 image_intrinsic,       # 相机内参矩阵 (3x3)
                 image_size=224,        # 输入图像尺寸
                 memory_size=8,         # 历史帧数量（时序记忆长度）
                 predict_size=24,       # 预测的轨迹点数
                 temporal_depth=16,     # Transformer 解码器层数
                 heads=8,               # 多头注意力的头数
                 token_dim=384,         # Token 嵌入维度
                 navi_model="./100.ckpt",  # 模型权重文件路径
                 device='cuda:0'):      # 运行设备
        # ===== 保存配置参数 =====
        self.image_intrinsic = image_intrinsic  # 相机内参，用于轨迹投影
        self.device = device
        self.predict_size = predict_size  # 24个轨迹点
        self.image_size = image_size      # 224x224
        self.memory_size = memory_size    # 8帧历史
        
        # ===== 创建策略网络 =====
        self.navi_former = NavDP_Policy(
            image_size, memory_size, predict_size, 
            temporal_depth, heads, token_dim, device
        )
        
        # ===== 加载预训练权重 =====
        self.navi_former.load_state_dict(
            torch.load(navi_model, map_location=self.device), 
            strict=False  # 允许部分权重不匹配（灵活性）
        )
        self.navi_former.to(self.device)
        self.navi_former.eval()  # 设置为评估模式（关闭dropout等）
    
    def reset(self, batch_size, threshold):
        """
        重置所有环境的状态
        
        Args:
            batch_size: 并行环境数量
            threshold: 停止阈值，当所有轨迹价值 < threshold 时停止前进
        """
        self.batch_size = batch_size      # 批次大小
        self.stop_threshold = threshold   # 停止阈值（如 -3.0）
        # 为每个环境创建空的历史帧队列
        self.memory_queue = [[] for i in range(batch_size)]
    
    def reset_env(self, i):
        """
        重置单个环境的历史帧队列
        
        Args:
            i: 环境索引
        """
        self.memory_queue[i] = []  # 清空该环境的历史
    
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
        """
        预处理 RGB 图像：resize到224x224并归一化
        
        Args:
            images: 输入图像 (batch, H, W, 3)，值域 [0, 255]
        
        Returns:
            处理后的图像 (batch, 224, 224, 3)，值域 [0.0, 1.0]
        
        处理步骤：
        1. 等比例缩放到最大边为224
        2. Padding到正方形224x224
        3. 归一化到[0,1]
        """
        assert len(images.shape) == 4  # 确保是4维 (batch, H, W, C)
        H, W, C = images.shape[1], images.shape[2], images.shape[3]
        
        # ===== 计算缩放比例（保持宽高比）=====
        prop = self.image_size / max(H, W)  # 例如：224/640 = 0.35
        
        return_images = []
        for img in images:
            # ===== 等比例缩放 =====
            resize_image = cv2.resize(img, (-1, -1), fx=prop, fy=prop)
            # 例如：(480, 640, 3) -> (168, 224, 3)
            
            # ===== 计算需要padding的宽度和高度 =====
            pad_width = max((self.image_size - resize_image.shape[1]) // 2, 0)
            pad_height = max((self.image_size - resize_image.shape[0]) // 2, 0)
            # 例如：(224-224)/2=0, (224-168)/2=28
            
            # ===== Padding到正方形 =====
            pad_image = np.pad(
                resize_image,
                ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
                mode='constant',
                constant_values=0
            )
            # 例如：(168, 224, 3) -> (224, 224, 3)
            
            # ===== 最终 resize（确保精确到224x224）=====
            resize_image = cv2.resize(pad_image, (self.image_size, self.image_size))
            
            # ===== 归一化到[0,1] =====
            resize_image = np.array(resize_image)
            resize_image = resize_image.astype(np.float32) / 255.0
            
            return_images.append(resize_image)
        
        return np.array(return_images)  # (batch, 224, 224, 3)

    def process_depth(self, depths):
        """
        预处理深度图：resize到224x224并过滤异常值
        
        Args:
            depths: 输入深度图 (batch, H, W, 1)，单位：米
        
        Returns:
            处理后的深度图 (batch, 224, 224, 1)
        
        处理步骤：
        1. 过滤无穷值
        2. 等比例缩放到最大边为224
        3. Padding到正方形224x224
        4. 过滤过远(>5m)和过近(<0.1m)的值
        """
        assert len(depths.shape) == 4
        
        # ===== 过滤无穷值 =====
        depths[depths == np.inf] = 0
        
        H, W, C = depths.shape[1], depths.shape[2], depths.shape[3]
        prop = self.image_size / max(H, W)
        
        return_depths = []
        for depth in depths:
            # ===== 等比例缩放 =====
            resize_depth = cv2.resize(depth, (-1, -1), fx=prop, fy=prop)
            
            # ===== Padding到正方形 =====
            pad_width = max((self.image_size - resize_depth.shape[1]) // 2, 0)
            pad_height = max((self.image_size - resize_depth.shape[0]) // 2, 0)
            pad_depth = np.pad(
                resize_depth,
                ((pad_height, pad_height), (pad_width, pad_width)),
                mode='constant',
                constant_values=0
            )
            
            # ===== 最终 resize =====
            resize_depth = cv2.resize(pad_depth, (self.image_size, self.image_size))
            
            # ===== 过滤异常深度值 =====
            resize_depth[resize_depth > 5.0] = 0    # 超过5米设为0
            resize_depth[resize_depth < 0.1] = 0   # 小于0.1米设为0（太近，可能是噪声）
            
            return_depths.append(resize_depth[:, :, np.newaxis])  # 添加通道维度
        
        return np.array(return_depths)  # (batch, 224, 224, 1)
    
    def process_pixel(self, pixel_coords, input_images):
        """
        处理像素目标：创建一个mask标记目标像素位置
        
        Args:
            pixel_coords: 像素坐标 (batch, 2)，[x, y]
            input_images: 原始图像 (batch, H, W, 3)，用于获取尺寸
        
        Returns:
            像素mask (batch, 224, 224)，目标位置为1，其他为0
        
        处理逻辑：
        1. 在原图尺寸上创建全0 mask
        2. 在目标像素周围20x20区域标记为255
        3. Resize和Padding到224x224
        4. 归一化到[0,1]
        """
        return_pixels = []
        H, W, C = input_images.shape[1], input_images.shape[2], input_images.shape[3]
        prop = self.image_size / max(H, W)
        
        for pixel_coord, input_image in zip(pixel_coords, input_images):
            # ===== 创建全0 mask =====
            panel_image = np.zeros_like(input_image, dtype=np.uint8)
            
            # ===== 计算目标区域边界 =====
            min_x = pixel_coord[0] - 10
            min_y = pixel_coord[1] - 10
            max_x = pixel_coord[0] + 10
            max_y = pixel_coord[1] + 10
            
            # ===== 标记目标区域（处理边界情况）=====
            if min_x <= 0:
                # 目标在左边界
                panel_image[:, 0:10] = 255
            elif min_y <= 0:
                # 目标在上边界
                panel_image[0:10, :] = 255
            elif max_x >= panel_image.shape[1]:
                # 目标在右边界
                panel_image[:, panel_image.shape[1]-10:] = 255
            elif max_y >= panel_image.shape[0]:
                # 目标在下边界
                panel_image[panel_image.shape[0]-10:, :] = 255
            elif min_x > 0 and min_y > 0 and max_x < panel_image.shape[1] and max_y < panel_image.shape[0]:
                # 目标在图像内部
                panel_image[min_y:max_y, min_x:max_x] = 255
            
            # ===== Resize（使用最近邻插值保持mask的二值性）=====
            resize_image = cv2.resize(
                panel_image, (-1, -1), 
                fx=prop, fy=prop, 
                interpolation=cv2.INTER_NEAREST
            )
            
            # ===== Padding到正方形 =====
            pad_width = max((self.image_size - resize_image.shape[1]) // 2, 0)
            pad_height = max((self.image_size - resize_image.shape[0]) // 2, 0)
            pad_image = np.pad(
                resize_image,
                ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
                mode='constant',
                constant_values=0
            )
            
            # ===== 最终 resize 和归一化 =====
            resize_image = cv2.resize(pad_image, (self.image_size, self.image_size))
            resize_image = np.array(resize_image)
            resize_image = resize_image.astype(np.float32) / 255.0
            
            return_pixels.append(resize_image)
        
        # 取RGB三通道的平均，得到单通道mask (batch, 224, 224)
        return np.array(return_pixels).mean(axis=-1)
    
    def process_pointgoal(self, goals):
        """
        处理点目标坐标：裁剪到合理范围
        
        Args:
            goals: 点目标坐标 (batch, 3)，[x, y, z]，单位：米
        
        Returns:
            裁剪后的目标 (batch, 3)
        
        裁剪策略：
        - x（前进）：裁剪到 [0, 10]，不允许后退
        - y（左右）：裁剪到 [-10, 10]
        - z：裁剪到 [-10, 10]
        """
        clip_goals = goals.clip(-10, 10)  # 所有维度先裁剪到[-10, 10]
        clip_goals[:, 0] = np.clip(clip_goals[:, 0], 0, 10)  # x维度进一步限制为[0, 10]
        return clip_goals
    
    def step_nogoal(self, images, depths):
        """
        无目标探索推理
        
        Args:
            images: RGB 图像 (batch, H, W, 3)
            depths: 深度图 (batch, H, W, 1)
        
        Returns:
            execute_trajectory: 最优轨迹 (batch, 24, 3)
            all_trajectory: 所有16条候选轨迹 (batch, 16, 24, 3)
            all_values: 所有16条轨迹的价值 (batch, 16)
            trajectory_mask: 可视化图像
        
        流程：
        1. 预处理当前帧
        2. 更新历史帧队列（维护8帧时序记忆）
        3. 调用扩散策略网络生成16条候选轨迹
        4. 如果所有轨迹价值都很低，强制原地转向（避免卡住）
        5. 生成可视化
        """
        # ===== 步骤1：预处理 =====
        process_images = self.process_image(images)    # (batch, 224, 224, 3)
        process_depths = self.process_depth(depths)    # (batch, 224, 224, 1)
        
        # ===== 步骤2：更新历史帧队列 =====
        input_images = []
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size:
                # 队列未满，添加当前帧
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
                # Padding到8帧（前面补0）
                input_image = np.pad(
                    input_image,
                    ((self.memory_size - input_image.shape[0], 0), (0, 0), (0, 0), (0, 0))
                )
            else:
                # 队列已满，删除最旧的帧，添加当前帧
                del self.memory_queue[i][0]
                self.memory_queue[i].append(process_images[i])    
                input_image = np.array(self.memory_queue[i])
            
            input_images.append(input_image)
        
        input_image = np.array(input_images)  # (batch, 8, 224, 224, 3)
        input_depth = process_depths          # (batch, 224, 224, 1)
        
        # ===== 步骤3：调用策略网络 =====
        all_trajectory, all_values, good_trajectory, bad_trajectory = \
            self.navi_former.predict_nogoal_action(input_image, input_depth)
        # all_trajectory: (batch, 16, 24, 3) - 所有候选轨迹
        # all_values: (batch, 16) - 每条轨迹的价值评分
        # good_trajectory: (batch, 2, 24, 3) - 最优的2条
        # bad_trajectory: (batch, 2, 24, 3) - 最差的2条
        
        # ===== 步骤4：处理停止条件 =====
        if all_values.max() < self.stop_threshold:
            # 所有轨迹价值都低于阈值，可能前方无路或遇到障碍
            # 强制不前进，只原地转向
            good_trajectory[:, :, :, 0] = 0.0  # x方向（前进）置0
            good_trajectory[:, :, :, 1] = np.sign(good_trajectory[:, :, :, 1].mean())  # y方向保持转向趋势
        
        # ===== 步骤5：生成可视化 =====
        trajectory_mask = self.project_trajectory(images, all_trajectory, all_values)
        trajectory_depth_mask =  self.project_trajectory_depth(depths, all_trajectory, all_values)
        # ===== 返回结果 =====
        return (
            good_trajectory[:, 0],  # 只返回最优的那一条 (batch, 24, 3)
            all_trajectory,         # 所有候选 (batch, 16, 24, 3)
            all_values,             # 所有价值 (batch, 16)
            trajectory_mask,         # 可视化图像
            trajectory_depth_mask
        )
    
    def step_pointgoal(self, goals, images, depths):
        """
        点目标导航推理
        
        Args:
            goals: 目标点坐标 (batch, 3)，[x, y, z]，单位：米
            images: RGB 图像 (batch, H, W, 3)
            depths: 深度图 (batch, H, W, 1)
        
        Returns:
            execute_trajectory: 最优轨迹 (batch, 24, 3)
            all_trajectory: 所有16条候选轨迹 (batch, 16, 24, 3)
            all_values: 所有16条轨迹的价值 (batch, 16)
            trajectory_mask: 可视化图像
        
        流程与 step_nogoal 类似，但额外输入了目标点坐标
        """
        # ===== 步骤1：预处理 =====
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        
        # ===== 步骤2：更新历史帧队列 =====
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
        
        input_image = np.array(input_images)  # (batch, 8, 224, 224, 3)
        input_depth = process_depths          # (batch, 224, 224, 1)
        input_goals = self.process_pointgoal(goals)  # (batch, 3)，裁剪到合理范围
        
        # ===== 步骤3：调用策略网络（点目标版本）=====
        all_trajectory, all_values, good_trajectory, bad_trajectory = \
            self.navi_former.predict_pointgoal_action(input_goals, input_image, input_depth)
        
        # ===== 步骤4：处理停止条件 =====
        if all_values.max() < self.stop_threshold:
            # 所有轨迹价值都很低，强制原地转向
            good_trajectory[:, :, :, 0] = 0.0
            good_trajectory[:, :, :, 1] = np.sign(good_trajectory[:, :, :, 1].mean())
        
        # 打印价值范围（用于调试）
        print(all_values.max(), all_values.min())
        
        # ===== 步骤5：生成可视化 =====
        trajectory_mask = self.project_trajectory(images, all_trajectory, all_values)
        trajectory_depth_mask =  self.project_trajectory_depth(depths, all_trajectory, all_values)
        
        return (
            good_trajectory[:, 0],  # 最优轨迹 (batch, 24, 3)
            all_trajectory,         # 所有候选 (batch, 16, 24, 3)
            all_values,             # 所有价值 (batch, 16)
            trajectory_mask,         # 可视化图像
            trajectory_depth_mask
        )
    
    def step_imagegoal(self, goals, images, depths):
        """
        图像目标导航推理
        
        Args:
            goals: 目标图像 (batch, H, W, 3)
            images: 当前 RGB 图像 (batch, H, W, 3)
            depths: 当前深度图 (batch, H, W, 1)
        
        Returns:
            execute_trajectory: 最优轨迹 (batch, 24, 3)
            all_trajectory: 所有16条候选轨迹 (batch, 16, 24, 3)
            all_values: 所有16条轨迹的价值 (batch, 16)
            trajectory_mask: 可视化图像
        
        核心逻辑与 step_pointgoal 相同，但目标是图像而非坐标
        """
        # ===== 预处理和队列管理 =====
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
        
        input_image = np.array(input_images)      # (batch, 8, 224, 224, 3)
        input_depth = process_depths              # (batch, 224, 224, 1)
        input_goals = self.process_image(goals)   # (batch, 224, 224, 3)，目标图像同样预处理
        
        # ===== 调用策略网络（图像目标版本）=====
        all_trajectory, all_values, good_trajectory, bad_trajectory = \
            self.navi_former.predict_imagegoal_action(input_goals, input_image, input_depth)
        
        # ===== 处理停止条件 =====
        if all_values.max() < self.stop_threshold:
            good_trajectory[:, :, :, 0] = 0.0
            good_trajectory[:, :, :, 1] = np.sign(good_trajectory[:, :, :, 1].mean())
        
        print(all_values.max(), all_values.min())
        trajectory_mask = self.project_trajectory(images, all_trajectory, all_values)
        
        return good_trajectory[:, 0], all_trajectory, all_values, trajectory_mask

    def step_pixelgoal(self, goals, images, depths):
        """
        像素目标导航推理
        
        Args:
            goals: 像素坐标 (batch, 2)，[x, y]，图像坐标系
            images: 当前 RGB 图像 (batch, H, W, 3)
            depths: 当前深度图 (batch, H, W, 1)
        
        Returns:
            execute_trajectory: 最优轨迹 (batch, 24, 3)
            all_trajectory: 所有16条候选轨迹 (batch, 16, 24, 3)
            all_values: 所有16条轨迹的价值 (batch, 16)
            trajectory_mask: 可视化图像
        
        像素目标：在图像中指定一个像素位置作为导航目标
        """
        # ===== 预处理和队列管理 =====
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
        
        input_image = np.array(input_images)                # (batch, 8, 224, 224, 3)
        input_depth = process_depths                        # (batch, 224, 224, 1)
        input_goals = self.process_pixel(goals, images)     # (batch, 224, 224)，像素mask
        
        # ===== 可选：生成像素目标可视化（用于调试）=====
        # pixel_vis_image = input_image[0, -1].copy() * 255
        # pixel_vis_image[np.where(input_goals[0] == 1)] = np.array([255, 0, 0])
        # cv2.imwrite("pixel_goal.jpg", pixel_vis_image)
        
        # ===== 调用策略网络（像素目标版本）=====
        all_trajectory, all_values, good_trajectory, bad_trajectory = \
            self.navi_former.predict_pixelgoal_action(input_goals, input_image, input_depth)
        
        # ===== 处理停止条件 =====
        if all_values.max() < self.stop_threshold:
            good_trajectory[:, :, :, 0] = 0.0
            good_trajectory[:, :, :, 1] = np.sign(good_trajectory[:, :, :, 1].mean())
        
        trajectory_mask = self.project_trajectory(images, all_trajectory, all_values)
        
        return good_trajectory[:, 0], all_trajectory, all_values, trajectory_mask
    
    def step_point_image_goal(self, pointgoal, imagegoal, images, depths):
        """
        混合目标导航推理（点目标 + 图像目标）
        
        Args:
            pointgoal: 点目标坐标 (batch, 3)，[x, y, z]
            imagegoal: 图像目标 (batch, H, W, 3)
            images: 当前 RGB 图像 (batch, H, W, 3)
            depths: 当前深度图 (batch, H, W, 1)
        
        Returns:
            execute_trajectory: 最优轨迹 (batch, 24, 3)
            all_trajectory: 所有16条候选轨迹 (batch, 16, 24, 3)
            all_values: 所有16条轨迹的价值 (batch, 16)
            trajectory_mask: 可视化图像
        
        混合目标：同时使用点目标和图像目标的信息进行导航
        两种目标会被分配到不同的 goal slot，由模型融合
        """
        # ===== 预处理和队列管理 =====
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
        
        input_image = np.array(input_images)                  # (batch, 8, 224, 224, 3)
        input_depth = process_depths                          # (batch, 224, 224, 1)
        input_pointgoal = self.process_pointgoal(pointgoal)   # (batch, 3)
        input_imagegoal = self.process_image(imagegoal)       # (batch, 224, 224, 3)
        
        # ===== 可选：保存历史帧用于调试 =====
        # cv2.imwrite("input_image.jpg", np.concatenate(self.memory_queue[0], axis=0) * 255)
        
        # ===== 调用策略网络（混合目标版本）=====
        all_trajectory, all_values, good_trajectory, bad_trajectory = \
            self.navi_former.predict_ip_action(
                input_pointgoal, input_imagegoal, input_image, input_depth
            )
        
        # ===== 处理停止条件 =====
        if all_values.max() < self.stop_threshold:
            good_trajectory[:, :, :, 0] = 0.0
            good_trajectory[:, :, :, 1] = torch.sign(good_trajectory[:, :, :, 1].mean())
        
        trajectory_mask = self.project_trajectory(images, all_trajectory, all_values)
        
        return good_trajectory[:, 0], all_trajectory, all_values, trajectory_mask