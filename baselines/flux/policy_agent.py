"""
GRPO (Group Relative Policy Optimization) 训练 Agent。
"""

from __future__ import annotations

import os
import sys
import cv2
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from matplotlib import colormaps as cm
from typing import Optional, List, Tuple

from policy_network import CFM_RL_Policy


class RolloutBuffer:
    """
    存储一个 update_interval 内的 rollout 数据。

    关键设计：buffer 只存 numpy 观测数据，不持有任何计算图。
    GRPO update 时重新做 forward 获取有梯度的 critic_scores。
    这样避免：
      1. 计算图跨越多步累积导致 OOM
      2. optimizer inplace 更新权重后旧计算图版本号冲突

    每条记录：
        goal_np:    (3,) np.ndarray，当前 step 的目标（pointgoal）或 None（nogoal）
        image_np:   (memory_size, H, W, 3) np.ndarray，预处理后的图像序列
        depth_np:   (H, W, 1) np.ndarray，预处理后的深度图
        task:       str，"pointgoal" 或 "nogoal"
        reward:     float，从环境获得的真实奖励
        done:       bool，episode 是否结束
    """

    def __init__(self, maxlen: int = 64):
        self.maxlen = maxlen
        self.clear()

    def clear(self):
        self.goals:    List[Optional[np.ndarray]] = []
        self.images:   List[np.ndarray] = []
        self.depths:   List[np.ndarray] = []
        self.tasks:    List[str] = []
        self.rewards:  List[float] = []
        self.dones:    List[bool] = []

    def add(self, goal_np, image_np, depth_np, task, reward, done):
        self.goals.append(goal_np)
        self.images.append(image_np)
        self.depths.append(depth_np)
        self.tasks.append(task)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def __len__(self):
        return len(self.rewards)

    @property
    def full(self) -> bool:
        # 达到maxlen触发，或者遇到done也在外部强制触发
        return len(self) >= self.maxlen
    # 说明：done时的强制触发已移到train_grpo.py的episode结束处理里
    # 这里保持不变，两个触发条件并存


class GRPO_Agent:
    """
    GRPO 训练 Agent，供主训练脚本调用。

    接口设计与 FLUX base 的 CFM_Agent 保持一致（step_pointgoal 返回相同格式），
    方便在 eval 脚本中替换。
    """

    def __init__(
        self,
        checkpoint_path: str,
        image_intrinsic: np.ndarray,
        image_size: int = 224,
        memory_size: int = 8,
        predict_size: int = 24,
        temporal_depth: int = 16,
        heads: int = 8,
        token_dim: int = 384,
        cfm_num_steps: int = 5,
        device: str = "cuda:0",
        normalization_config: Optional[str] = None,
        # GRPO 超参
        lr: float = 3e-5,
        update_interval: int = 32,       # 每 N 步做一次梯度更新
        sample_num: int = 16,            # 每步候选轨迹数
        unfreeze_decoder_last_n: int = 4,
        grad_clip: float = 1.0,
        entropy_coef: float = 0.01,      # critic 输出多样性正则
        save_dir: str = "./checkpoints_rl",
        save_interval: int = 100,        # 每 N 个 episode 保存一次
    ):
        self.device = device
        self.image_intrinsic = image_intrinsic
        self.image_size = image_size
        self.memory_size = memory_size
        self.predict_size = predict_size
        self.sample_num = sample_num
        self.update_interval = update_interval
        self.grad_clip = grad_clip
        self.entropy_coef = entropy_coef
        self.save_dir = save_dir
        self.save_interval = save_interval
        os.makedirs(save_dir, exist_ok=True)

        # ---- 构建网络 ----
        self.policy = CFM_RL_Policy(
            image_size=image_size,
            memory_size=memory_size,
            predict_size=predict_size,
            temporal_depth=temporal_depth,
            heads=heads,
            token_dim=token_dim,
            device=device,
            normalization_config=normalization_config,
            cfm_num_steps=cfm_num_steps,
        )

        # ---- 加载预训练权重 ----
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        incompatible = self.policy.load_state_dict(ckpt, strict=False)
        print(f"[GRPO_Agent] Loaded checkpoint: {checkpoint_path}")
        print(f"  missing={len(incompatible.missing_keys)}, "
              f"unexpected={len(incompatible.unexpected_keys)}")

        self.policy.to(device)

        # ---- 冻结策略 ----
        self.policy.freeze_for_rl(unfreeze_decoder_last_n=unfreeze_decoder_last_n)

        # ---- 优化器（只优化可训练参数）----
        trainable = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=5000, eta_min=lr * 0.1
        )

        # ---- Rollout buffer ----
        self.buffer = RolloutBuffer(maxlen=update_interval)

        # ---- 状态 ----
        self.memory_queue: List[List] = []
        self.batch_size = 1
        self.stop_threshold = -3.0
        self.step_count = 0
        self.episode_count = 0
        self.total_updates = 0

        # ---- 训练日志 ----
        self.train_log: List[dict] = []

        # ---- 视频 writer（用于可视化）----
        self._fps_writer = None

        self._return_running_mean = 0.0
        self._return_running_std  = 1.0
        self._return_ema_alpha    = 0.05   # EMA 衰减系数

        # 尝试恢复 EMA 统计量（resume 训练时）
        ema_path = os.path.join(save_dir, "ema_stats.json")
        if os.path.exists(ema_path):
            with open(ema_path) as f:
                ema = json.load(f)
            self._return_running_mean = ema["return_running_mean"]
            self._return_running_std  = ema["return_running_std"]
            print(f"[GRPO] Restored EMA stats: mean={self._return_running_mean:.3f}, "
                f"std={self._return_running_std:.3f}")
    # ------------------------------------------------------------------
    # 接口：reset
    # ------------------------------------------------------------------

    def reset(self, batch_size: int, stop_threshold: float):
        self.batch_size = batch_size
        self.stop_threshold = stop_threshold
        self.memory_queue = [[] for _ in range(batch_size)]

    def reset_env(self, i: int):
        """重置单个环境的历史帧队列"""
        if i < len(self.memory_queue):
            self.memory_queue[i] = []

    # ------------------------------------------------------------------
    # 图像预处理（与 FLUX base 完全相同）
    # ------------------------------------------------------------------

    def process_image(self, images: np.ndarray) -> np.ndarray:
        assert len(images.shape) == 4
        H, W = images.shape[1], images.shape[2]
        prop = self.image_size / max(H, W)
        result = []
        for img in images:
            r = cv2.resize(img, (-1, -1), fx=prop, fy=prop)
            ph = max((self.image_size - r.shape[0]) // 2, 0)
            pw = max((self.image_size - r.shape[1]) // 2, 0)
            r = np.pad(r, ((ph, ph), (pw, pw), (0, 0)), constant_values=0)
            r = cv2.resize(r, (self.image_size, self.image_size))
            result.append(r.astype(np.float32) / 255.0)
        return np.array(result)

    def process_depth(self, depths: np.ndarray) -> np.ndarray:
        assert len(depths.shape) == 4
        depths = depths.copy()
        depths[depths == np.inf] = 0
        H, W = depths.shape[1], depths.shape[2]
        prop = self.image_size / max(H, W)
        result = []
        for depth in depths:
            r = cv2.resize(depth, (-1, -1), fx=prop, fy=prop)
            ph = max((self.image_size - r.shape[0]) // 2, 0)
            pw = max((self.image_size - r.shape[1]) // 2, 0)
            r = np.pad(r, ((ph, ph), (pw, pw)), constant_values=0)
            r = cv2.resize(r, (self.image_size, self.image_size))
            r[r > 5.0] = 0
            r[r < 0.1] = 0
            result.append(r[:, :, np.newaxis])
        return np.array(result)

    def process_pointgoal(self, goals: np.ndarray) -> np.ndarray:
        clip = goals.clip(-10, 10)
        clip[:, 0] = np.clip(clip[:, 0], 0, 10)
        return clip

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

    def _build_input_images(self, proc_images: np.ndarray) -> np.ndarray:
        """维护 memory queue，返回 (batch, memory_size, H, W, 3)"""
        input_images = []
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size:
                self.memory_queue[i].append(proc_images[i])
                buf = np.array(self.memory_queue[i])
                buf = np.pad(buf, ((self.memory_size - buf.shape[0], 0), (0,0), (0,0), (0,0)))
            else:
                del self.memory_queue[i][0]
                self.memory_queue[i].append(proc_images[i])
                buf = np.array(self.memory_queue[i])
            input_images.append(buf)
        return np.array(input_images)

    # ------------------------------------------------------------------
    # 推理接口：step_pointgoal
    # 与 FLUX base 的 CFM_Agent 返回格式完全相同，可直接替换
    # ------------------------------------------------------------------

    def step_pointgoal(
        self,
        goals: np.ndarray,
        images: np.ndarray,
        depths: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行一步点目标导航推理（纯推理，不保留计算图）。

        观测数据会暂存到 _pending_obs，等 record_reward 调用时
        再连同 reward 一起写入 buffer。

        Returns:
            execute_trajectory:  (batch, predict_size, 3)，最优轨迹
            all_trajectory:      (batch, sample_num, predict_size, 3)
            all_values:          (batch, sample_num)
        """
        proc_images = self.process_image(images)
        proc_depths = self.process_depth(depths)
        input_images = self._build_input_images(proc_images)
        input_goals = self.process_pointgoal(goals)

        # 纯推理，不保留计算图，节省显存
        with torch.inference_mode():
            all_trajectory, critic_scores, _ = self.policy.forward_train_pointgoal(
                input_goals, input_images, proc_depths, sample_num=self.sample_num
            )

        # 暂存当前 step 的 numpy 观测，供 record_reward 写入 buffer
        # 注意：存 numpy 不存 Tensor，彻底断开计算图
        self._pending_obs = {
            "goal":   input_goals.copy(),            # (B, 3) numpy
            "image":  input_images.copy(),           # (B, memory_size, H, W, 3) numpy
            "depth":  proc_depths.copy(),            # (B, H, W, 1) numpy
            "task":   "pointgoal",
        }

        # 选择最优轨迹
        best_idx = critic_scores.argmax(dim=1)
        execute_trajectory = all_trajectory[
            torch.arange(self.batch_size), best_idx
        ]

        # stop 判断
        if critic_scores.max().item() < self.stop_threshold:
            execute_trajectory[:, :, 0] = 0.0
            execute_trajectory[:, :, 1] = (
                execute_trajectory[:, :, 1].mean().sign().item()
            )

        self.step_count += 1
        
        trajectory_mask = self.project_trajectory(images, all_trajectory.cpu().numpy(), critic_scores.cpu().numpy())
        trajectory_depth_mask =  self.project_trajectory_depth(depths, all_trajectory.cpu().numpy(), critic_scores.cpu().numpy())
        
        return (
            execute_trajectory.cpu().numpy(),
            all_trajectory.cpu().numpy(),
            critic_scores.cpu().numpy(),
            trajectory_mask,
            trajectory_depth_mask
        )

    def step_nogoal(
        self,
        images: np.ndarray,
        depths: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行一步无目标探索推理（dynnogoal 任务）。
        goal 位置传全零，与预训练的 nogoal 接口一致。
        """
        proc_images = self.process_image(images)
        proc_depths = self.process_depth(depths)
        input_images = self._build_input_images(proc_images)

        with torch.inference_mode():
            all_trajectory, critic_scores, _ = self.policy.forward_train_nogoal(
                input_images, proc_depths, sample_num=self.sample_num
            )

        # 暂存观测，goal=None 表示 nogoal 任务
        self._pending_obs = {
            "goal":  None,
            "image": input_images.copy(),
            "depth": proc_depths.copy(),
            "task":  "nogoal",
        }

        # nogoal专用：强制在有效长度的轨迹里选best
        B, S, T, _ = all_trajectory.shape
        best_idx = torch.zeros(B, dtype=torch.long, device=all_trajectory.device)
        for b in range(B):
            traj_lens = all_trajectory[b, :, -1, :2].norm(dim=-1)  # (S,)
            valid_mask = traj_lens > 0.3
            masked_scores = critic_scores[b].clone()
            if valid_mask.any():
                masked_scores[~valid_mask] = -1e9
            best_idx[b] = masked_scores.argmax()
        
        execute_trajectory = all_trajectory[
            torch.arange(self.batch_size), best_idx
        ]

        if critic_scores.max().item() < self.stop_threshold:
            execute_trajectory[:, :, 0] = 0.0
            execute_trajectory[:, :, 1] = (
                execute_trajectory[:, :, 1].mean().sign().item()
            )

        self.step_count += 1

        return (
            execute_trajectory.cpu().numpy(),
            all_trajectory.cpu().numpy(),
            critic_scores.cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # 奖励记录 & GRPO 更新
    # ------------------------------------------------------------------

    def record_reward(
        self,
        reward: float,
        done: bool,
        env_idx: int = 0,
    ):
        """
        将上一步 step_pointgoal / step_nogoal 暂存的观测 + 本步奖励写入 buffer。

        不再存储任何 Tensor 或计算图，彻底避免跨步 OOM 和 inplace 版本冲突。

        Args:
            reward:   当前 step 获得的总奖励（标量）
            done:     episode 是否结束
            env_idx:  环境索引（目前只支持单 env，env_idx=0）
        """
        if hasattr(self, "_pending_obs"):
            obs = self._pending_obs
            # 取第 env_idx 个环境的观测（目前 batch_size=1，env_idx=0）
            goal_np  = obs["goal"][env_idx].copy()  if obs["goal"]  is not None else None
            image_np = obs["image"][env_idx].copy()
            depth_np = obs["depth"][env_idx].copy()
            self.buffer.add(goal_np, image_np, depth_np, obs["task"], reward, done)

        if done:
            self.episode_count += 1
            # if self.episode_count % self.save_interval == 0:
            #     self._save_checkpoint()

    def maybe_update(self) -> Optional[dict]:
        """
        如果 buffer 满了，执行一次 GRPO 更新。

        Returns:
            log_dict: 训练日志（如果触发了更新），否则 None
        """
        if not self.buffer.full:
            return None

        log = self._grpo_update()
        self.buffer.clear()
        return log

    def _grpo_update(self) -> dict:
        self.policy.train()
        T = len(self.buffer)

        # ---- T=0 或 T=1 的边界情况 ----
        if T == 0:
            self.total_updates += 1
            self.policy.eval()
            return {
                "update": self.total_updates, "loss": 0.0,
                "regression": 0.0, "ranking": 0.0, "entropy": 0.0,
                "mean_return": 0.0, "mean_reward": 0.0,
                "lr": self.scheduler.get_last_lr()[0],
            }

        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        returns = self._compute_returns(rewards, gamma=0.99)

        alpha = self._return_ema_alpha
        raw_mean = returns.mean().item()
        # ↓ 关键修复：T=1 时 std() = NaN，改用 unbiased=False 或加保护
        raw_std = returns.std(unbiased=False).item()   # unbiased=False → 除以N而非N-1
        if not np.isfinite(raw_std) or raw_std < 1e-8:
            raw_std = 1e-8  # fallback

        self._return_running_mean = (1-alpha)*self._return_running_mean + alpha*raw_mean
        self._return_running_std  = (1-alpha)*self._return_running_std  + alpha*(raw_std + 1e-8)

        # running_std 也需要检查（历史污染恢复）
        if not np.isfinite(self._return_running_std) or self._return_running_std < 1e-8:
            self._return_running_std = 1.0  # 重置为安全值

        returns_norm = (returns - self._return_running_mean) / (self._return_running_std + 1e-8)
        returns_norm = returns_norm.clamp(-3.0, 3.0)

        # 最终防御：如果 returns_norm 仍含 NaN（running_mean 被污染）
        if not torch.isfinite(returns_norm).all():
            print(f"[GRPO] WARNING: returns_norm contains NaN, resetting EMA stats")
            self._return_running_mean = returns.mean().item()
            self._return_running_std  = max(returns.std(unbiased=False).item(), 1e-8)
            returns_norm = torch.zeros_like(returns)  # 这次 update 用 0 advantage

        total_loss_val    = 0.0
        total_reg_val     = 0.0
        total_ranking_val = 0.0
        total_ent_val     = 0.0

        # ← 关键修复：每步单独 zero_grad + step，避免梯度累积爆炸
        for t in range(T):
            self.optimizer.zero_grad()          # ← 移到循环内

            task    = self.buffer.tasks[t]
            goal_np = self.buffer.goals[t]
            img_np  = self.buffer.images[t][np.newaxis]
            dep_np  = self.buffer.depths[t][np.newaxis]

            if task == "pointgoal" and goal_np is not None:
                goal_in = goal_np[np.newaxis]
                _, scores_t, _ = self.policy.forward_train_pointgoal(
                    goal_in, img_np, dep_np, sample_num=self.sample_num
                )
            else:
                _, scores_t, _ = self.policy.forward_train_nogoal(
                    img_np, dep_np, sample_num=self.sample_num
                )
            scores_t = scores_t.squeeze(0)

            return_t = returns_norm[t]

            best_score = scores_t.max()
            regression_loss = F.mse_loss(best_score, return_t.detach())

            with torch.no_grad():
                scores_detach = scores_t.detach()
                grp_mean = scores_detach.mean()
                grp_std  = scores_detach.std() + 1e-8
                grp_adv  = (scores_detach - grp_mean) / grp_std

            log_probs_t  = torch.log_softmax(scores_t, dim=-1)
            ranking_loss = -(grp_adv * log_probs_t).mean()

            probs_t  = log_probs_t.exp()
            ent_loss = -self.entropy_coef * (-(probs_t * log_probs_t).sum())

            loss_t = regression_loss + 0.3 * ranking_loss + ent_loss

            # NaN 守卫：跳过异常步
            if not torch.isfinite(loss_t):
                print(f"[GRPO] step {t}: non-finite loss={loss_t.item()}, skipping")
                del scores_t, log_probs_t, probs_t
                torch.cuda.empty_cache()
                continue

            loss_t.backward()

            nn.utils.clip_grad_norm_(
                [p for p in self.policy.parameters() if p.requires_grad],
                self.grad_clip,
            )
            self.optimizer.step()       # ← 每步都更新，梯度不累积

            total_loss_val    += loss_t.item()
            total_reg_val     += regression_loss.item()
            total_ranking_val += ranking_loss.item()
            total_ent_val     += (-(probs_t * log_probs_t).sum()).item()

            del scores_t, log_probs_t, probs_t, loss_t
            torch.cuda.empty_cache()

        self.scheduler.step()           # ← scheduler 仍然每次 update 调用一次
        self.total_updates += 1
        self.policy.eval()
        torch.cuda.empty_cache()

        # 日志除以实际有效步数
        valid_steps = max(T, 1)
        log = {
            "update":      self.total_updates,
            "loss":        total_loss_val / valid_steps,
            "regression":  total_reg_val / valid_steps,
            "ranking":     total_ranking_val / valid_steps,
            "entropy":     total_ent_val / valid_steps,
            "mean_return": returns_norm.mean().item(),
            "mean_reward": rewards.mean().item(),
            "lr":          self.scheduler.get_last_lr()[0],
        }
        self.train_log.append(log)
        print(
            f"[GRPO] update={self.total_updates:4d} | "
            f"loss={log['loss']:.4f} | reg={log['regression']:.4f} | "
            f"rank={log['ranking']:.4f} | ent={log['entropy']:.4f} | "
            f"rew={log['mean_reward']:.3f} | lr={log['lr']:.2e}"
        )
        if self.total_updates % self.save_interval == 0:
            self._save_checkpoint()
        return log

    @staticmethod
    def _compute_returns(rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
        """Monte-Carlo 折扣回报"""
        T = len(rewards)
        returns = torch.zeros_like(rewards)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t].item() + gamma * G
            returns[t] = G
        return returns

    # ------------------------------------------------------------------
    # 保存 / 加载
    # ------------------------------------------------------------------

    def _save_checkpoint(self):
        path = os.path.join(
            self.save_dir,
            f"rl_checkpoint_update{self.total_updates:05d}.ckpt"
        )
        torch.save({"state_dict": self.policy.state_dict()}, path)

        ema_path = os.path.join(self.save_dir, "ema_stats.json")
        with open(ema_path, "w") as f:
            json.dump({
                "return_running_mean": self._return_running_mean,
                "return_running_std":  self._return_running_std,
            }, f)

        log_path = os.path.join(self.save_dir, "train_log.json")
        with open(log_path, "w") as f:
            json.dump(self.train_log, f, indent=2)

        print(f"[GRPO] Saved checkpoint → {path}  (update={self.total_updates})")

    def save_final(self, name: str = "rl_final.ckpt"):
        path = os.path.join(self.save_dir, name)
        torch.save({"state_dict": self.policy.state_dict()}, path)
        print(f"[GRPO] Saved final checkpoint → {path}")
