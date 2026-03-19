"""
NavDP 策略网络
基于 Diffusion Policy + Transformer 的导航策略
核心功能：
1. DDPM 扩散策略生成轨迹
2. Critic 网络评估轨迹价值
3. 支持多种导航任务（无目标/点目标/图像目标/像素目标）
"""
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from policy_backbone import *

class NavDP_Policy(nn.Module):
    """
    NavDP 策略网络主类
    
    架构：
    - 输入编码器：RGBD + 多种目标编码器
    - 核心网络：Transformer Decoder（16层）
    - 输出头：Action Head（生成轨迹）+ Critic Head（评估价值）
    - 采样策略：DDPM 去噪过程（10步）
    """
    def __init__(self,
                 image_size=224,        # 输入图像尺寸
                 memory_size=8,         # 历史帧数（时序记忆）
                 predict_size=24,       # 预测轨迹点数
                 temporal_depth=8,      # Transformer Decoder 层数
                 heads=8,               # 多头注意力头数
                 token_dim=384,         # Token 嵌入维度
                 channels=3,            # 输入图像通道数
                 device='cuda:0'):      # 运行设备
        super().__init__()
        # ===== 保存配置 =====
        self.device = device
        self.image_size = image_size
        self.memory_size = memory_size          # 8帧
        self.predict_size = predict_size        # 24个点
        self.temporal_depth = temporal_depth    # 16层
        self.attention_heads = heads            # 8头
        self.input_channels = channels
        self.token_dim = token_dim              # 384维
        
        # ===== 输入编码器（多模态目标支持）=====
        self.rgbd_encoder = NavDP_RGBD_Backbone(
            image_size, token_dim, memory_size=memory_size, device=device
        )  # RGBD 编码器：(batch, 8, 224, 224, 4) -> (batch, 128, 384)
        
        self.point_encoder = nn.Linear(3, self.token_dim)  # 点目标编码器：(x,y,z) -> 384维
        self.pixel_encoder = NavDP_PixelGoal_Backbone(image_size, token_dim, device=device)  # 像素目标
        self.image_encoder = NavDP_ImageGoal_Backbone(image_size, token_dim, device=device)  # 图像目标
        
        # ===== Transformer Decoder（核心网络）=====
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_dim,              # 384维
            nhead=heads,                    # 8头注意力
            dim_feedforward=4 * token_dim,  # FFN 隐藏层 1536维
            activation='gelu',              # 激活函数
            batch_first=True,               # batch 维度在最前
            norm_first=True                 # Pre-LN（LayerNorm在前）
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=self.temporal_depth  # 堆叠16层
        )
        
        # ===== 嵌入层 =====
        self.input_embed = nn.Linear(3, token_dim)  # 轨迹点(x,y,θ)编码为token
        
        # 条件位置编码：memory_size*16(RGBD tokens) + 4(time + 3个goal slots)
        self.cond_pos_embed = LearnablePositionalEncoding(token_dim, memory_size * 16 + 4)
        
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)  # 24个点的位置编码
        self.time_emb = SinusoidalPosEmb(token_dim)  # DDPM 时间步编码（正弦）
        self.layernorm = nn.LayerNorm(token_dim)
        
        # ===== 输出头 =====
        self.action_head = nn.Linear(token_dim, 3)    # 预测轨迹增量 (Δx, Δy, Δθ)
        self.critic_head = nn.Linear(token_dim, 1)    # 预测轨迹价值（标量）
        
        # ===== DDPM 噪声调度器 =====
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=10,           # 10步去噪
            beta_schedule='squaredcos_cap_v2', # 余弦噪声调度
            clip_sample=True,                  # 裁剪样本
            prediction_type='epsilon'          # 预测噪声（而非直接预测样本）
        )
        
        # ===== Attention Mask（因果mask，防止看到未来）=====
        # 创建下三角mask：每个点只能看到自己和之前的点
        self.tgt_mask = (torch.triu(torch.ones(predict_size, predict_size)) == 1).transpose(0, 1)
        self.tgt_mask = self.tgt_mask.float().masked_fill(
            self.tgt_mask == 0, float('-inf')  # 未来的位置用 -inf mask掉
        ).masked_fill(self.tgt_mask == 1, float(0.0))
        
        # Critic 的 memory mask：屏蔽掉目标信息（只用观测）
        self.cond_critic_mask = torch.zeros((predict_size, 4 + memory_size * 16))
        self.cond_critic_mask[:, 0:4] = float('-inf')  # 屏蔽前4个token（time+3个goal）
    
    def predict_noise(self, last_actions, timestep, goal_embed, rgbd_embed):
        """
        预测噪声（DDPM 去噪的核心函数）
        
        Args:
            last_actions: 当前带噪的轨迹 (batch*16, 24, 3)
            timestep: DDPM 时间步 (1,)，范围 [0, 9]
            goal_embed: 目标编码 (batch*16, 1, 384)
            rgbd_embed: RGBD 观测编码 (batch*16, 128, 384)
        
        Returns:
            noise: 预测的噪声 (batch*16, 24, 3)
        
        工作流程：
        1. 编码轨迹点 -> token
        2. 编码时间步 -> token
        3. 拼接条件 token（time + 3个goal + rgbd）
        4. Transformer Decoder 融合
        5. 输出噪声预测
        """
        # ===== 步骤1：编码当前轨迹 =====
        action_embeds = self.input_embed(last_actions)
        # 输入: (batch*16, 24, 3)
        # 输出: (batch*16, 24, 384)
        
        # ===== 步骤2：编码时间步 =====
        time_embeds = self.time_emb(timestep.to(self.device))
        # 输入: (1,)，如 timestep=5
        # 输出: (1, 384)，正弦位置编码
        
        time_embeds = time_embeds.unsqueeze(1).tile((last_actions.shape[0], 1, 1))
        # unsqueeze: (1, 1, 384)
        # tile: (batch*16, 1, 384)，复制给每个样本
        
        # ===== 步骤3：拼接条件编码 =====
        # 注意：goal_embed 重复3次是为了支持多目标槽位
        cond_tokens = torch.cat([
            time_embeds,    # (batch*16, 1, 384)   - 时间步
            goal_embed,     # (batch*16, 1, 384)   - 目标槽位1
            goal_embed,     # (batch*16, 1, 384)   - 目标槽位2
            goal_embed,     # (batch*16, 1, 384)   - 目标槽位3
            rgbd_embed      # (batch*16, 128, 384) - RGBD观测
        ], dim=1)
        # 输出: (batch*16, 132, 384)  # 1+1+1+1+128=132
        
        cond_embedding = cond_tokens + self.cond_pos_embed(cond_tokens)
        # 添加可学习位置编码
        
        # ===== 步骤4：轨迹位置编码 =====
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        # 添加轨迹点的位置编码（区分24个点的顺序）
        
        # ===== 步骤5：Transformer Decoder =====
        output = self.decoder(
            tgt=input_embedding,              # Query: (batch*16, 24, 384)
            memory=cond_embedding,            # Key/Value: (batch*16, 132, 384)
            tgt_mask=self.tgt_mask.to(self.device)  # 因果mask: (24, 24)
        )
        # 输出: (batch*16, 24, 384)
        # 16层 Transformer Decoder，每层做：
        #   1. Self-Attention（轨迹点之间，带因果mask）
        #   2. Cross-Attention（轨迹点 attend to 条件）
        #   3. FFN
        
        # ===== 步骤6：输出噪声 =====
        output = self.layernorm(output)       # LayerNorm
        output = self.action_head(output)     # Linear(384 -> 3)
        # 输出: (batch*16, 24, 3)，预测的噪声 ε
        
        return output
    
    def predict_mix_noise(self, last_actions, timestep, goal_embeds, rgbd_embed):
        """
        混合目标的噪声预测（用于多目标任务）
        
        与 predict_noise 的区别：
        - goal_embeds 是一个列表 [goal1, goal2, goal3]
        - 分别填充3个目标槽位
        
        Args:
            last_actions: 当前带噪的轨迹 (batch*16, 24, 3)
            timestep: DDPM 时间步 (1,)
            goal_embeds: 目标编码列表 [embed1, embed2, embed3]
                         每个 (batch*16, 1, 384)
            rgbd_embed: RGBD 观测编码 (batch*16, 128, 384)
        
        Returns:
            noise: 预测的噪声 (batch*16, 24, 3)
        
        示例：点目标+图像目标混合任务
            goal_embeds = [imagegoal_embed, pointgoal_embed, imagegoal_embed]
        """
        # 编码和逻辑与 predict_noise 相同，只是目标槽位填充不同
        action_embeds = self.input_embed(last_actions)
        time_embeds = self.time_emb(timestep.to(self.device)).unsqueeze(1).tile((last_actions.shape[0], 1, 1))
        
        # 关键区别：使用不同的 goal_embeds
        cond_tokens = torch.cat([
            time_embeds,      # 时间步
            goal_embeds[0],   # 目标槽位1（如 image goal）
            goal_embeds[1],   # 目标槽位2（如 point goal）
            goal_embeds[2],   # 目标槽位3（如 image goal）
            rgbd_embed        # RGBD观测
        ], dim=1)
        
        cond_embedding = cond_tokens + self.cond_pos_embed(cond_tokens)
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        output = self.decoder(
            tgt=input_embedding, 
            memory=cond_embedding, 
            tgt_mask=self.tgt_mask.to(self.device)
        )
        output = self.layernorm(output)
        output = self.action_head(output)
        return output
    
    def predict_critic(self, predict_trajectory, rgbd_embed):
        """
        Critic 网络：评估轨迹的价值
        
        Args:
            predict_trajectory: 预测的轨迹 (batch*16, 24, 3)
            rgbd_embed: RGBD 观测编码 (batch*16, 128, 384)
        
        Returns:
            critic_values: 轨迹价值 (batch*16,)，标量评分
        
        关键设计：
        1. 不使用目标信息（全0），只基于观测评估轨迹安全性
        2. 用 memory_mask 屏蔽目标槽位，强制只看 RGBD
        3. 平均池化24个点的特征后输出标量
        
        作用：
        - 从16条候选轨迹中选出最优的
        - 评估标准：避障能力、可行性、探索价值
        """
        # ===== 步骤1：创建空目标（Critic不看目标）=====
        nogoal_embed = torch.zeros_like(rgbd_embed[:, 0:1])
        # 输出: (batch*16, 1, 384)，全0向量
        # 设计理念：Critic 应该基于观测判断轨迹安全性，不依赖目标
        
        # ===== 步骤2：编码轨迹 =====
        action_embeddings = self.input_embed(predict_trajectory)
        # 输入: (batch*16, 24, 3)
        # 输出: (batch*16, 24, 384)
        
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)
        # 添加位置编码
        
        # ===== 步骤3：拼接条件（4个nogoal + RGBD）=====
        cond_tokens = torch.cat([
            nogoal_embed,   # 时间步位置（用0填充）
            nogoal_embed,   # 目标槽位1（用0填充）
            nogoal_embed,   # 目标槽位2（用0填充）
            nogoal_embed,   # 目标槽位3（用0填充）
            rgbd_embed      # RGBD观测（真实信息）
        ], dim=1)
        # 输出: (batch*16, 132, 384)
        
        cond_embeddings = cond_tokens + self.cond_pos_embed(cond_tokens)
        
        # ===== 步骤4：Transformer Decoder（带 memory mask）=====
        critic_output = self.decoder(
            tgt=action_embeddings,                            # Query: (batch*16, 24, 384)
            memory=cond_embeddings,                           # Key/Value: (batch*16, 132, 384)
            memory_mask=self.cond_critic_mask.to(self.device) # Mask掉前4个token（目标）
        )
        # memory_mask: (24, 132)
        # 前4列为 -inf，屏蔽目标信息
        # 后128列为 0，允许 attend to RGBD
        
        # 输出: (batch*16, 24, 384)
        
        # ===== 步骤5：聚合并输出标量 =====
        critic_output = self.layernorm(critic_output)        # LayerNorm
        critic_output = critic_output.mean(dim=1)            # 平均池化: (batch*16, 384)
        critic_output = self.critic_head(critic_output)[:, 0] # Linear + 取标量: (batch*16,)
        
        return critic_output  # 每条轨迹一个分数
    
    def predict_pointgoal_action(self, goal_point, input_images, input_depths, sample_num=16):
        """
        点目标导航推理
        
        Args:
            goal_point: 目标点坐标 (batch, 3)，[x, y, z]，单位：米
            input_images: 历史RGB图像 (batch, 8, 224, 224, 3)
            input_depths: 当前深度图 (batch, 224, 224, 1)
            sample_num: 采样轨迹数量，默认16条
        
        Returns:
            all_trajectory: 所有候选轨迹 (batch, 16, 24, 3)
            all_values: 所有轨迹的价值 (batch, 16)
            good_trajectory: 最优的2条轨迹 (batch, 2, 24, 3)
            bad_trajectory: 最差的2条轨迹 (batch, 2, 24, 3)
        
        完整流程：
        1. 编码 RGBD 和 目标
        2. 复制16份用于并行采样
        3. DDPM 去噪10步生成轨迹
        4. Critic 评分
        5. 选择最优/最差轨迹
        """
        with torch.no_grad():  # 推理模式，不计算梯度
            # ===== 步骤1：编码输入 =====
            tensor_point_goal = torch.as_tensor(goal_point, dtype=torch.float32, device=self.device)
            # goal_point: (batch, 3)
            
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            # 输入: images (batch, 8, 224, 224, 3), depths (batch, 224, 224, 1)
            # 输出: (batch, 128, 384)
            
            pointgoal_embed = self.point_encoder(tensor_point_goal).unsqueeze(1)
            # Linear(3 -> 384): (batch, 3) -> (batch, 384)
            # unsqueeze: (batch, 1, 384)
            
            # ===== 步骤2：复制16份用于并行采样 =====
            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
            # (batch, 128, 384) -> (batch*16, 128, 384)
            
            pointgoal_embed = torch.repeat_interleave(pointgoal_embed, sample_num, dim=0)
            # (batch, 1, 384) -> (batch*16, 1, 384)
            
            # ===== 步骤3：初始化噪声 =====
            noisy_action = torch.randn(
                (sample_num * goal_point.shape[0], self.predict_size, 3), 
                device=self.device
            )
            # 纯高斯噪声: (batch*16, 24, 3)
            # 每条轨迹从不同的随机噪声开始
            
            naction = noisy_action
            
            # ===== 步骤4：DDPM 去噪循环（10步）=====
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            # 设置时间步序列: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
            
            for k in self.noise_scheduler.timesteps[:]:
                # 4.1 预测噪声
                noise_pred = self.predict_noise(naction, k.unsqueeze(0), pointgoal_embed, rgbd_embed)
                # 输入: naction (batch*16, 24, 3), k (1,)
                # 输出: noise_pred (batch*16, 24, 3)
                
                # 4.2 去噪一步
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
                # 根据预测的噪声更新样本
                # 公式: x_{t-1} = √(α_{t-1}) * (x_t - √(1-α_t) * ε_θ) / √(α_t) + σ_t * z
            
            # 循环结束后，naction 是去噪后的轨迹增量: (batch*16, 24, 3)
            
            # ===== 步骤5：Critic 评分 =====
            critic_values = self.predict_critic(naction, rgbd_embed)
            # 输入: naction (batch*16, 24, 3)
            # 输出: (batch*16,)
            
            critic_values = critic_values.reshape(goal_point.shape[0], sample_num)
            # reshape 回: (batch, 16)
            
            # ===== 步骤6：累积求和得到绝对轨迹 =====
            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            # naction 是增量 Δx, Δy, Δθ
            # cumsum 累加得到绝对坐标
            # 除以4.0是训练时的归一化系数
            # 输出: (batch*16, 24, 3)
            
            all_trajectory = all_trajectory.reshape(
                goal_point.shape[0], sample_num, self.predict_size, 3
            )
            # reshape: (batch, 16, 24, 3)
            
            # ===== 步骤7：过滤短轨迹 =====
            trajectory_length = all_trajectory[:, :, -1, 0:2].norm(dim=-1)
            # 计算每条轨迹终点的距离: (batch, 16)
            
            all_trajectory[trajectory_length < 0.5] = \
                all_trajectory[trajectory_length < 0.5] * torch.tensor([[[0, 0, 1.0]]], device=all_trajectory.device)
            # 如果轨迹太短(<0.5米)，保留角度但清零位移
            # 避免"原地不动"的无效轨迹
            
            # ===== 步骤8：选择最优轨迹 =====
            sorted_indices = (-critic_values).argsort(dim=1)
            # 价值从大到小排序
            
            topk_indices = sorted_indices[:, 0:2]
            # 选前2名: (batch, 2)
            
            batch_indices = torch.arange(goal_point.shape[0]).unsqueeze(1).expand(-1, 2)
            # 创建batch索引: [[0,0], [1,1], ...]
            
            positive_trajectory = all_trajectory[batch_indices, topk_indices]
            # 高级索引: (batch, 2, 24, 3)
            
            # ===== 步骤9：选择最差轨迹 =====
            sorted_indices = (critic_values).argsort(dim=1)
            # 价值从小到大排序
            
            topk_indices = sorted_indices[:, 0:2]
            batch_indices = torch.arange(goal_point.shape[0]).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]
            # 最差的2条: (batch, 2, 24, 3)
            
            # ===== 步骤10：返回结果 =====
            return (
                all_trajectory.cpu().numpy(),       # (batch, 16, 24, 3)
                critic_values.cpu().numpy(),        # (batch, 16)
                positive_trajectory.cpu().numpy(),  # (batch, 2, 24, 3)
                negative_trajectory.cpu().numpy()   # (batch, 2, 24, 3)
            )
    
    def predict_imagegoal_action(self, goal_image, input_images, input_depths, sample_num=16):
        """
        图像目标导航推理
        
        Args:
            goal_image: 目标图像 (batch, 224, 224, 3)
            input_images: 历史RGB图像 (batch, 8, 224, 224, 3)
            input_depths: 当前深度图 (batch, 224, 224, 1)
            sample_num: 采样轨迹数量，默认16条
        
        Returns:
            all_trajectory, all_values, good_trajectory, bad_trajectory
        
        与 predict_pointgoal_action 的区别：
        - 目标不是坐标，而是图像
        - 使用 image_encoder 编码目标
        - 其余流程完全相同
        """
        with torch.no_grad():
            # ===== 编码 RGBD =====
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            # (batch, 128, 384)
            
            # ===== 编码图像目标 =====
            # 拼接目标图像和当前图像: (batch, 224, 224, 6)
            concat_images = np.concatenate((goal_image, input_images[:, -1]), axis=-1)
            imagegoal_embed = self.image_encoder(concat_images).unsqueeze(1)
            # image_encoder: (batch, 224, 224, 6) -> (batch, 384)
            # unsqueeze: (batch, 1, 384)
            
            # ===== 复制16份并行采样 =====
            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
            imagegoal_embed = torch.repeat_interleave(imagegoal_embed, sample_num, dim=0)
            
            # ===== DDPM 去噪过程（与 pointgoal 相同）=====
            noisy_action = torch.randn((sample_num * goal_image.shape[0], self.predict_size, 3), device=self.device)
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction, k.unsqueeze(0), imagegoal_embed, rgbd_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
            
            # ===== Critic 评分 =====
            critic_values = self.predict_critic(naction, rgbd_embed)
            critic_values = critic_values.reshape(goal_image.shape[0], sample_num)
            
            # ===== 累积轨迹并过滤短轨迹 =====
            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(goal_image.shape[0], sample_num, self.predict_size, 3)
            trajectory_length = all_trajectory[:, :, -1, 0:2].norm(dim=-1)
            all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * torch.tensor([[[0, 0, 1.0]]], device=all_trajectory.device)
            
            # ===== 选择最优/最差轨迹 =====
            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:, 0:2]
            batch_indices = torch.arange(goal_image.shape[0]).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]
            
            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:, 0:2]
            batch_indices = torch.arange(goal_image.shape[0]).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]
            
            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()
    
    def predict_pixelgoal_action(self, goal_image, input_images, input_depths, sample_num=16):
        """
        像素目标导航推理
        
        Args:
            goal_image: 像素mask (batch, 224, 224)，目标位置=1，其他=0
            input_images: 历史RGB图像 (batch, 8, 224, 224, 3)
            input_depths: 当前深度图 (batch, 224, 224, 1)
            sample_num: 采样轨迹数量，默认16条
        
        Returns:
            all_trajectory, all_values, good_trajectory, bad_trajectory
        
        与其他任务的区别：
        - 目标是像素mask（单通道）
        - 使用 pixel_encoder 编码
        """
        with torch.no_grad():
            rgbd_embed = self.rgbd_encoder(input_images,input_depths)
            pixelgoal_embed = self.pixel_encoder(np.concatenate((goal_image[:,:,:,None],input_images[:,-1]),axis=-1)).unsqueeze(1)
    
            rgbd_embed = torch.repeat_interleave(rgbd_embed,sample_num,dim=0)
            pixelgoal_embed = torch.repeat_interleave(pixelgoal_embed,sample_num,dim=0)
            
            noisy_action = torch.randn((sample_num * goal_image.shape[0], self.predict_size, 3), device=self.device)
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction,k.unsqueeze(0),pixelgoal_embed,rgbd_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred,timestep=k,sample=naction).prev_sample
            
            critic_values = self.predict_critic(naction,rgbd_embed)
            critic_values = critic_values.reshape(goal_image.shape[0],sample_num)
            
            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(goal_image.shape[0],sample_num,self.predict_size,3)
            trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)
            all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * torch.tensor([[[0,0,1.0]]],device=all_trajectory.device)
            
            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0]).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]
            
            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0]).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]
            
            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()
    
    def predict_nogoal_action(self, input_images, input_depths, sample_num=16):
        """
        无目标探索推理
        
        Args:
            input_images: 历史RGB图像 (batch, 8, 224, 224, 3)
            input_depths: 当前深度图 (batch, 224, 224, 1)
            sample_num: 采样轨迹数量，默认16条
        
        Returns:
            all_trajectory, all_values, good_trajectory, bad_trajectory
        
        特点：
        - 没有明确目标，目标编码为全0
        - 依靠 Critic 选择探索价值高的轨迹
        - 额外惩罚短轨迹（<1米），鼓励探索
        """
        with torch.no_grad():
            # ===== 编码 RGBD =====
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            # (batch, 128, 384)
            
            # ===== 创建空目标（无目标探索）=====
            nogoal_embed = torch.zeros_like(rgbd_embed[:, 0:1])
            # 全0向量: (batch, 1, 384)
            # 表示"没有明确目标"
            
            # ===== 复制16份 =====
            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
            nogoal_embed = torch.repeat_interleave(nogoal_embed, sample_num, dim=0)
           
            # ===== DDPM 去噪过程 =====
            noisy_action = torch.randn((sample_num * input_images.shape[0], self.predict_size, 3), device=self.device)
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction, k.unsqueeze(0), nogoal_embed, rgbd_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
            
            # ===== Critic 评分 =====
            critic_values = self.predict_critic(naction, rgbd_embed)
            critic_values = critic_values.reshape(input_images.shape[0], sample_num)
            
            # ===== 累积轨迹 =====
            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(input_images.shape[0], sample_num, self.predict_size, 3)

            # ===== 惩罚短轨迹（探索专用策略）=====
            trajectory_length = all_trajectory[:, :, -1, 0:2].norm(dim=-1)
            # 计算终点距离: (batch, 16)
            
            print(trajectory_length.shape, trajectory_length.max(), trajectory_length.min())
            
            # 如果轨迹太短（<1米），降低其价值
            # 鼓励探索更远的地方
            critic_values[torch.where(trajectory_length < 1.0)] -= 10.0
            
            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(input_images.shape[0]).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]
            
            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(input_images.shape[0]).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]
            
            #import pdb
            #pdb.set_trace()
            
            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()
        
    def predict_ip_action(self, goal_point, goal_image, input_images, input_depths, sample_num=16):
        """
        混合目标导航推理（Image + Point）
        
        Args:
            goal_point: 点目标坐标 (batch, 3)
            goal_image: 图像目标 (batch, 224, 224, 3)
            input_images: 历史RGB图像 (batch, 8, 224, 224, 3)
            input_depths: 当前深度图 (batch, 224, 224, 1)
            sample_num: 采样轨迹数量，默认16条
        
        Returns:
            all_trajectory, all_values, good_trajectory, bad_trajectory
        
        关键设计：
        - 同时使用两种目标信息
        - 分配到不同的 goal slot: [image, point, image]
        - 模型自动融合两种目标的信息
        """
        with torch.no_grad():
            # ===== 编码输入 =====
            tensor_point_goal = torch.as_tensor(goal_point, dtype=torch.float32, device=self.device)
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            
            # ===== 编码两种目标 =====
            imagegoal_embed = self.image_encoder(
                np.concatenate((goal_image, input_images[:, -1]), axis=-1)
            ).unsqueeze(1)
            # 图像目标: (batch, 1, 384)
            
            pointgoal_embed = self.point_encoder(tensor_point_goal).unsqueeze(1)
            # 点目标: (batch, 1, 384)
            
            # ===== 复制16份 =====
            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
            pointgoal_embed = torch.repeat_interleave(pointgoal_embed, sample_num, dim=0)
            imagegoal_embed = torch.repeat_interleave(imagegoal_embed, sample_num, dim=0)
            
            # ===== DDPM 去噪（使用 predict_mix_noise）=====
            noisy_action = torch.randn((sample_num * goal_image.shape[0], self.predict_size, 3), device=self.device)
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                # 关键：传入混合目标列表 [image, point, image]
                noise_pred = self.predict_mix_noise(
                    naction, k.unsqueeze(0), 
                    [imagegoal_embed, pointgoal_embed, imagegoal_embed],  # 3个槽位
                    rgbd_embed
                )
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
            
            # ===== Critic 评分 =====
            critic_values = self.predict_critic(naction, rgbd_embed)
            critic_values = critic_values.reshape(goal_image.shape[0], sample_num)
            
            # ===== 累积轨迹并过滤 =====
            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(goal_image.shape[0], sample_num, self.predict_size, 3)
            trajectory_length = all_trajectory[:, :, -1, 0:2].norm(dim=-1)
            all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * torch.tensor([[[0, 0, 1.0]]], device=all_trajectory.device)
            
            # ===== 选择最优/最差轨迹 =====
            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:, 0:2]
            batch_indices = torch.arange(goal_image.shape[0]).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]
            
            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:, 0:2]
            batch_indices = torch.arange(goal_image.shape[0]).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]
            
            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()
    
    