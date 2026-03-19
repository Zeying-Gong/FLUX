"""
flux/policy_network.py

1. CFM_Policy: 基础 Rectified Flow 推理策略
2. CFM_RL_Policy: 在 CFM_Policy 基础上添加 GRPO 训练支持
"""

from __future__ import annotations

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from policy_backbone import (
    NavDP_RGBD_Backbone,
    NavDP_PixelGoal_Backbone,
    NavDP_ImageGoal_Backbone,
    LearnablePositionalEncoding,
    SinusoidalPosEmb,
)


class CFM_Policy(nn.Module):
    """
    Linear-path RF 推理版策略网络。
    架构与原始 NavDP / CFM 版本完全相同（RGBD + 目标编码器 +
    Transformer Decoder + Action/Critic Head），推理时用 RF ODE Euler 积分。
    """

    def __init__(
        self,
        image_size=224,
        memory_size=8,
        predict_size=24,
        temporal_depth=8,
        heads=8,
        token_dim=384,
        channels=3,
        device="cuda:0",
        normalization_config=None,
        cfm_num_steps=5,
    ):
        super().__init__()
        self.cfm_num_steps = cfm_num_steps
        self.device = device
        self.image_size = image_size
        self.memory_size = memory_size
        self.predict_size = predict_size
        self.temporal_depth = temporal_depth
        self.attention_heads = heads
        self.input_channels = channels
        self.token_dim = token_dim

        # ===== 输入编码器 =====
        self.rgbd_encoder = NavDP_RGBD_Backbone(
            image_size, token_dim, memory_size=memory_size, device=device
        )
        self.point_encoder = nn.Linear(3, token_dim)
        self.pixel_encoder = NavDP_PixelGoal_Backbone(image_size, token_dim, device=device)
        self.image_encoder = NavDP_ImageGoal_Backbone(image_size, token_dim, device=device)

        # ===== Transformer Decoder =====
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=4 * token_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=self.temporal_depth,
        )

        # ===== 嵌入层 =====
        self.input_embed = nn.Linear(3, token_dim)

        # 条件位置编码：memory_size*16 + 4 = 132
        # （1 time token + 3 goal slots + 128 RGBD tokens）
        self.cond_pos_embed = LearnablePositionalEncoding(token_dim, memory_size * 16 + 4)
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)
        self.time_emb = SinusoidalPosEmb(token_dim)
        self.layernorm = nn.LayerNorm(token_dim)

        # ===== 输出头 =====
        self.action_head = nn.Linear(token_dim, 3)
        self.critic_head = nn.Linear(token_dim, 1)

        # ===== DDPM 噪声调度器（RF 推理仍保留，供 predict_ip_action 使用）=====
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=10,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        # ===== Attention Mask =====
        self.tgt_mask = (
            torch.triu(torch.ones(predict_size, predict_size)) == 1
        ).transpose(0, 1)
        self.tgt_mask = self.tgt_mask.float().masked_fill(
            self.tgt_mask == 0, float("-inf")
        ).masked_fill(self.tgt_mask == 1, float(0.0))

        self.cond_critic_mask = torch.zeros((predict_size, 4 + memory_size * 16))
        self.cond_critic_mask[:, 0:4] = float("-inf")

        # ===== 动作归一化配置 =====
        self.action_norm_scales = None
        if normalization_config and os.path.isfile(normalization_config):
            try:
                with open(normalization_config, "r") as f:
                    cfg = json.load(f)
                scales = cfg.get("normalization", {}).get("scales", {})
                if all(k in scales for k in ["x", "y", "theta"]):
                    self.action_norm_scales = torch.tensor(
                        [float(scales["x"]), float(scales["y"]), float(scales["theta"])],
                        dtype=torch.float32,
                    ).view(1, 1, 3)
            except Exception:
                pass

    def denormalize_action_tensor(self, action):
        """反归一化：[-1,1] → 真实尺度（cumsum 前调用）"""
        if self.action_norm_scales is None:
            return action
        return action * self.action_norm_scales.to(action.device)

    def predict_noise(self, last_actions, timestep, goal_embed, rgbd_embed):
        """预测噪声 / 速度场 v(x_t, t)（RF 和 CFM 共用同一前向结构）"""
        action_embeds = self.input_embed(last_actions)

        time_embeds = self.time_emb(timestep.to(self.device))
        time_embeds = time_embeds.unsqueeze(1).tile((last_actions.shape[0], 1, 1))

        cond_tokens = torch.cat(
            [time_embeds, goal_embed, goal_embed, goal_embed, rgbd_embed], dim=1
        )  # (B, 132, D)
        cond_embedding = cond_tokens + self.cond_pos_embed(cond_tokens)
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)

        output = self.decoder(
            tgt=input_embedding,
            memory=cond_embedding,
            tgt_mask=self.tgt_mask.to(self.device),
        )
        output = self.layernorm(output)
        return self.action_head(output)

    def predict_velocity(self, x_t, timestep, goal_embed, rgbd_embed):
        """RF 速度场（与 predict_noise 共享同一网络结构）"""
        return self.predict_noise(x_t, timestep, goal_embed, rgbd_embed)

    def predict_mix_noise(self, last_actions, timestep, goal_embeds, rgbd_embed):
        """混合目标噪声预测（goal_embeds = [embed1, embed2, embed3]）"""
        action_embeds = self.input_embed(last_actions)
        time_embeds = (
            self.time_emb(timestep.to(self.device))
            .unsqueeze(1)
            .tile((last_actions.shape[0], 1, 1))
        )
        cond_tokens = torch.cat(
            [time_embeds, goal_embeds[0], goal_embeds[1], goal_embeds[2], rgbd_embed],
            dim=1,
        )
        cond_embedding = cond_tokens + self.cond_pos_embed(cond_tokens)
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        output = self.decoder(
            tgt=input_embedding,
            memory=cond_embedding,
            tgt_mask=self.tgt_mask.to(self.device),
        )
        output = self.layernorm(output)
        return self.action_head(output)

    def predict_critic(self, predict_trajectory, rgbd_embed):
        """Critic：只基于 RGBD 观测评分（屏蔽目标 token）"""
        nogoal_embed = torch.zeros_like(rgbd_embed[:, 0:1])
        action_embeddings = self.input_embed(predict_trajectory)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)

        cond_tokens = torch.cat(
            [nogoal_embed, nogoal_embed, nogoal_embed, nogoal_embed, rgbd_embed], dim=1
        )
        cond_embeddings = cond_tokens + self.cond_pos_embed(cond_tokens)

        critic_output = self.decoder(
            tgt=action_embeddings,
            memory=cond_embeddings,
            memory_mask=self.cond_critic_mask.to(self.device),
        )
        critic_output = self.layernorm(critic_output)
        critic_output = critic_output.mean(dim=1)
        return self.critic_head(critic_output)[:, 0]

    def sample_ode_euler(self, goal_embed, rgbd_embed, sample_num, num_steps=5):
        """RF ODE Euler 积分：t=1（纯噪声）→ t=0（数据）。"""
        T = self.noise_scheduler.config.num_train_timesteps  # 10
        x = torch.randn(
            (goal_embed.shape[0], self.predict_size, 3), device=self.device
        )
        t_seq = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)

        for i in range(num_steps):
            t_cur = t_seq[i]
            dt = t_seq[i] - t_seq[i + 1]
            k = torch.clamp(torch.round(t_cur * (T - 1)), 1, T - 1)
            v = self.predict_velocity(x, k.unsqueeze(0), goal_embed, rgbd_embed)
            x = x + v * dt

        return x

    def _select_trajectories(self, all_trajectory, critic_values, batch_size, sample_num):
        """从 all_trajectory 中选出价值最高 / 最低的各 2 条。"""
        sorted_desc = (-critic_values).argsort(dim=1)
        sorted_asc = critic_values.argsort(dim=1)
        bi = torch.arange(batch_size).unsqueeze(1).expand(-1, 2)
        positive = all_trajectory[bi, sorted_desc[:, 0:2]]
        negative = all_trajectory[bi, sorted_asc[:, 0:2]]
        return positive, negative

    def _build_trajectory(self, naction, batch_size, sample_num, short_threshold=0.5):
        """反归一化 → /4.0 → cumsum，过滤过短轨迹（保留角度）。"""
        naction_denorm = self.denormalize_action_tensor(naction)
        traj = torch.cumsum(naction_denorm / 4.0, dim=1)
        traj = traj.reshape(batch_size, sample_num, self.predict_size, 3)
        length = traj[:, :, -1, 0:2].norm(dim=-1)
        mask = length < short_threshold
        traj[mask] = traj[mask] * torch.tensor(
            [[[0.0, 0.0, 1.0]]], device=traj.device
        )
        return traj, length

    def predict_pointgoal_action(self, goal_point, input_images, input_depths, sample_num=16):
        with torch.no_grad():
            B = goal_point.shape[0]
            tensor_goal = torch.as_tensor(goal_point, dtype=torch.float32, device=self.device)

            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            goal_embed = self.point_encoder(tensor_goal).unsqueeze(1)

            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
            goal_embed = torch.repeat_interleave(goal_embed, sample_num, dim=0)

            naction = self.sample_ode_euler(goal_embed, rgbd_embed, B * sample_num, self.cfm_num_steps)

            critic_values = self.predict_critic(naction, rgbd_embed).reshape(B, sample_num)
            traj, _ = self._build_trajectory(naction, B, sample_num)
            positive, negative = self._select_trajectories(traj, critic_values, B, sample_num)

            return (
                traj.cpu().numpy(),
                critic_values.cpu().numpy(),
                positive.cpu().numpy(),
                negative.cpu().numpy(),
            )

    def predict_nogoal_action(self, input_images, input_depths, sample_num=16):
        with torch.no_grad():
            B = input_images.shape[0]

            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            goal_embed = torch.zeros_like(rgbd_embed[:, 0:1])

            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
            goal_embed = torch.repeat_interleave(goal_embed, sample_num, dim=0)

            naction = self.sample_ode_euler(goal_embed, rgbd_embed, B * sample_num, self.cfm_num_steps)

            critic_values = self.predict_critic(naction, rgbd_embed).reshape(B, sample_num)
            traj, length = self._build_trajectory(naction, B, sample_num, short_threshold=0.5)

            # 探索专用：惩罚终点不足 1m 的轨迹
            critic_values[length < 1.0] -= 10.0

            positive, negative = self._select_trajectories(traj, critic_values, B, sample_num)

            return (
                traj.cpu().numpy(),
                critic_values.cpu().numpy(),
                positive.cpu().numpy(),
                negative.cpu().numpy(),
            )

    def predict_imagegoal_action(self, goal_image, input_images, input_depths, sample_num=16):
        with torch.no_grad():
            B = goal_image.shape[0]

            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            concat = np.concatenate((goal_image, input_images[:, -1]), axis=-1)
            goal_embed = self.image_encoder(concat).unsqueeze(1)

            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
            goal_embed = torch.repeat_interleave(goal_embed, sample_num, dim=0)

            naction = self.sample_ode_euler(goal_embed, rgbd_embed, B * sample_num, self.cfm_num_steps)

            critic_values = self.predict_critic(naction, rgbd_embed).reshape(B, sample_num)
            traj, _ = self._build_trajectory(naction, B, sample_num)
            positive, negative = self._select_trajectories(traj, critic_values, B, sample_num)

            return (
                traj.cpu().numpy(),
                critic_values.cpu().numpy(),
                positive.cpu().numpy(),
                negative.cpu().numpy(),
            )

    def predict_pixelgoal_action(self, goal_image, input_images, input_depths, sample_num=16):
        with torch.no_grad():
            B = goal_image.shape[0]

            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            concat = np.concatenate((goal_image[:, :, :, None], input_images[:, -1]), axis=-1)
            goal_embed = self.pixel_encoder(concat).unsqueeze(1)

            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
            goal_embed = torch.repeat_interleave(goal_embed, sample_num, dim=0)

            naction = self.sample_ode_euler(goal_embed, rgbd_embed, B * sample_num, self.cfm_num_steps)

            critic_values = self.predict_critic(naction, rgbd_embed).reshape(B, sample_num)
            traj, _ = self._build_trajectory(naction, B, sample_num)
            positive, negative = self._select_trajectories(traj, critic_values, B, sample_num)

            return (
                traj.cpu().numpy(),
                critic_values.cpu().numpy(),
                positive.cpu().numpy(),
                negative.cpu().numpy(),
            )

    def predict_ip_action(self, goal_point, goal_image, input_images, input_depths, sample_num=16):
        with torch.no_grad():
            B = goal_image.shape[0]
            tensor_goal = torch.as_tensor(goal_point, dtype=torch.float32, device=self.device)

            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            image_embed = self.image_encoder(
                np.concatenate((goal_image, input_images[:, -1]), axis=-1)
            ).unsqueeze(1)
            point_embed = self.point_encoder(tensor_goal).unsqueeze(1)

            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
            point_embed = torch.repeat_interleave(point_embed, sample_num, dim=0)
            image_embed = torch.repeat_interleave(image_embed, sample_num, dim=0)

            naction = torch.randn((B * sample_num, self.predict_size, 3), device=self.device)
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.predict_mix_noise(
                    naction, k.unsqueeze(0),
                    [image_embed, point_embed, image_embed],
                    rgbd_embed,
                )
                naction = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample

            critic_values = self.predict_critic(naction, rgbd_embed).reshape(B, sample_num)
            traj, _ = self._build_trajectory(naction, B, sample_num)
            positive, negative = self._select_trajectories(traj, critic_values, B, sample_num)

            return (
                traj.cpu().numpy(),
                critic_values.cpu().numpy(),
                positive.cpu().numpy(),
                negative.cpu().numpy(),
            )


class CFM_RL_Policy(CFM_Policy):
    """
    GRPO 可训练版的 CFM_Policy。
    在基础 CFM_Policy 基础上添加：
    - freeze_for_rl(): 冻结除训练目标之外的所有参数
    - forward_train_pointgoal(): 可微分前向，返回 (trajectories, critic_scores)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rl_frozen = False

    def freeze_for_rl(self, unfreeze_decoder_last_n: int = 0):
        for p in self.parameters():
            p.requires_grad_(False)

        if unfreeze_decoder_last_n > 0:
            total_layers = len(self.decoder.layers)
            unfreeze_start = max(0, total_layers - unfreeze_decoder_last_n)
            for i, layer in enumerate(self.decoder.layers):
                if i >= unfreeze_start:
                    for p in layer.parameters():
                        p.requires_grad_(True)
            print(f"[CFM_RL_Policy] Also unfreezing decoder layers {unfreeze_start}-{total_layers-1}")

        for p in self.critic_head.parameters():
            p.requires_grad_(True)

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[CFM_RL_Policy] Critic-only mode: trainable={n_train:,} / total={n_total:,} "
            f"({100*n_train/n_total:.2f}%)")

    def forward_train_pointgoal(
        self,
        goal_point: np.ndarray,
        input_images: np.ndarray,
        input_depths: np.ndarray,
        sample_num: int = 16,
    ):
        tensor_point_goal = torch.as_tensor(goal_point, dtype=torch.float32, device=self.device)
        rgbd_embed = self.rgbd_encoder(input_images, input_depths)          # (B, 128, 384)
        pointgoal_embed = self.point_encoder(tensor_point_goal).unsqueeze(1)  # (B, 1, 384)

        B = goal_point.shape[0]
        rgbd_embed_rep = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)       # (B*S, 128, 384)
        goal_embed_rep = torch.repeat_interleave(pointgoal_embed, sample_num, dim=0)  # (B*S, 1, 384)

        naction_raw = self._sample_ode_euler_grad(
            goal_embed_rep, rgbd_embed_rep, num_steps=self.cfm_num_steps
        )  # (B*S, predict_size, 3)

        critic_scores_flat = self._predict_critic_grad(naction_raw, rgbd_embed_rep)
        critic_scores = critic_scores_flat.reshape(B, sample_num)                    # (B, S)

        naction_denorm = self.denormalize_action_tensor(naction_raw)
        all_trajectory_flat = torch.cumsum(naction_denorm / 4.0, dim=1)             # (B*S, 24, 3)
        all_trajectory = all_trajectory_flat.reshape(B, sample_num, self.predict_size, 3)

        trajectory_length = all_trajectory[:, :, -1, 0:2].norm(dim=-1)
        mask = trajectory_length < 0.5
        all_trajectory[mask] = all_trajectory[mask] * torch.tensor(
            [[[0, 0, 1.0]]], device=all_trajectory.device
        )

        return all_trajectory, critic_scores, naction_raw

    def forward_train_nogoal(
        self,
        input_images: np.ndarray,
        input_depths: np.ndarray,
        sample_num: int = 16,
    ):
        rgbd_embed = self.rgbd_encoder(input_images, input_depths)   # (B, 128, 384)
        B = rgbd_embed.shape[0]
        goal_embed = torch.zeros(B, 1, self.token_dim, device=self.device)

        rgbd_embed_rep = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
        goal_embed_rep = torch.repeat_interleave(goal_embed, sample_num, dim=0)

        naction_raw = self._sample_ode_euler_grad(
            goal_embed_rep, rgbd_embed_rep, num_steps=self.cfm_num_steps
        )
        critic_scores_flat = self._predict_critic_grad(naction_raw, rgbd_embed_rep)
        critic_scores = critic_scores_flat.reshape(B, sample_num)

        naction_denorm = self.denormalize_action_tensor(naction_raw)
        all_trajectory_flat = torch.cumsum(naction_denorm / 4.0, dim=1)
        all_trajectory = all_trajectory_flat.reshape(B, sample_num, self.predict_size, 3)

        trajectory_length = all_trajectory[:, :, -1, 0:2].norm(dim=-1)
        mask = trajectory_length < 0.5
        all_trajectory[mask] = all_trajectory[mask] * torch.tensor(
            [[[0, 0, 1.0]]], device=all_trajectory.device
        )

        return all_trajectory, critic_scores, naction_raw

    def _sample_ode_euler_grad(self, goal_embed, rgbd_embed, num_steps: int = 5):
        x = torch.randn(
            (goal_embed.shape[0], self.predict_size, 3), device=self.device
        )
        T = self.noise_scheduler.config.num_train_timesteps  # 10
        t_seq = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)

        for i in range(num_steps):
            t_cur = t_seq[i]
            dt = t_seq[i] - t_seq[i + 1]
            k = torch.clamp(torch.round(t_cur * (T - 1)), 1, T - 1)
            k_batch = k.unsqueeze(0)
            v = self.predict_velocity(x, k_batch, goal_embed, rgbd_embed)
            x = x + v * dt

        return x

    def _predict_critic_grad(self, predict_trajectory, rgbd_embed, goal_embed=None):
        nogoal_embed = torch.zeros_like(rgbd_embed[:, 0:1])
        action_embeddings = self.input_embed(predict_trajectory)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)

        cond_tokens = torch.cat([
            nogoal_embed, nogoal_embed, nogoal_embed, nogoal_embed, rgbd_embed,
        ], dim=1)
        cond_embeddings = cond_tokens + self.cond_pos_embed(cond_tokens)

        critic_output = self.decoder(
            tgt=action_embeddings,
            memory=cond_embeddings,
            memory_mask=self.cond_critic_mask.to(self.device),
        )
        critic_output = self.layernorm(critic_output)
        critic_output = critic_output.mean(dim=1)
        critic_output = self.critic_head(critic_output)[:, 0]
        return critic_output
