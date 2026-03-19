"""
cfm_rf_rl/policy_network.py

在 cfm_rf 的基础上添加：
1. forward_train_pointgoal: 可微分前向（不用 torch.no_grad），供 GRPO 训练使用
2. 冻结策略：rgb_model / depth_model / imagegoal / pixelgoal encoder + decoder 前12层全部冻结
3. 可训练参数：decoder 后4层、action_head、critic_head、point_encoder、pos_embed
4. 其余推理逻辑直接复用 cfm_rf 的 CFM_Policy（通过继承）

网络结构与 cfm_rf 完全一致，checkpoint 可直接加载。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

# ---- 动态加载 cfm_rf 版的 CFM_Policy ----
_CFM_RF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cfm_rf"))
_CFM_RF_POLICY_PATH = os.path.join(_CFM_RF_DIR, "policy_network.py")


def _load_cfm_rf_policy():
    if _CFM_RF_DIR not in sys.path:
        sys.path.insert(0, _CFM_RF_DIR)
    spec = importlib.util.spec_from_file_location("cfm_rf_policy_network", _CFM_RF_POLICY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load cfm_rf policy_network from: {_CFM_RF_POLICY_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_rf_mod = _load_cfm_rf_policy()
_BaseCFMRFPolicy = _rf_mod.CFM_Policy


class CFM_RL_Policy(_BaseCFMRFPolicy):
    """
    GRPO 可训练版的 CFM_Policy。

    在 cfm_rf 的 CFM_Policy 基础上：
    - 新增 freeze_for_rl(): 冻结除训练目标之外的所有参数
    - 新增 forward_train_pointgoal(): 可微分前向，返回 (trajectories, critic_scores)
    - 保留所有 cfm_rf 的推理接口，可直接用于测试
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 默认不冻结，调用 freeze_for_rl() 后生效
        self._rl_frozen = False

    # ------------------------------------------------------------------
    # 冻结策略
    # ------------------------------------------------------------------

    def freeze_for_rl(self, unfreeze_decoder_last_n: int = 0,
                       unfreeze_conditioning: bool = False):
        # 冻结所有
        for p in self.parameters():
            p.requires_grad_(False)

        if unfreeze_decoder_last_n > 0:
            # 保留原来的选项，以后想做完整finetune时还能用
            total_layers = len(self.decoder.layers)
            unfreeze_start = max(0, total_layers - unfreeze_decoder_last_n)
            for i, layer in enumerate(self.decoder.layers):
                if i >= unfreeze_start:
                    for p in layer.parameters():
                        p.requires_grad_(True)
            print(f"[CFM_RL_Policy] Also unfreezing decoder layers {unfreeze_start}-{total_layers-1}")

        # 只解冻critic_head，不解冻action_head
        # action_head影响轨迹生成，冻结它保护CFM的生成分布
        for p in self.critic_head.parameters():
            p.requires_grad_(True)

        # action_head完全冻结
        # （原来是解冻的，现在去掉）

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[CFM_RL_Policy] Critic-only mode: trainable={n_train:,} / total={n_total:,} "
            f"({100*n_train/n_total:.2f}%)")

    # ------------------------------------------------------------------
    # 可微分前向（供 GRPO 训练使用）
    # ------------------------------------------------------------------

    def forward_train_pointgoal(
        self,
        goal_point: np.ndarray,
        input_images: np.ndarray,
        input_depths: np.ndarray,
        sample_num: int = 16,
    ):
        """
        可微分的点目标前向传播，保留梯度。

        与 predict_pointgoal_action 的区别：
        - 不包裹 torch.no_grad()
        - 返回 torch.Tensor 而非 numpy，保留计算图
        - 返回 (all_trajectory, critic_scores, raw_naction) 三元组

        Args:
            goal_point:    (batch, 3) numpy，目标坐标
            input_images:  (batch, 8, 224, 224, 3) numpy
            input_depths:  (batch, 224, 224, 1) numpy
            sample_num:    候选轨迹数量，默认 16

        Returns:
            all_trajectory:  (batch, sample_num, predict_size, 3) Tensor，绝对轨迹
            critic_scores:   (batch, sample_num) Tensor，critic 打分
            naction_raw:     (batch*sample_num, predict_size, 3) Tensor，RF 原始输出（用于 loss 计算）
        """
        # ---- 编码输入（rgbd_encoder 内部有 no_grad，不影响外层）----
        tensor_point_goal = torch.as_tensor(goal_point, dtype=torch.float32, device=self.device)
        rgbd_embed = self.rgbd_encoder(input_images, input_depths)          # (B, 128, 384)
        pointgoal_embed = self.point_encoder(tensor_point_goal).unsqueeze(1)  # (B, 1, 384)

        # ---- 复制 sample_num 份并行采样 ----
        B = goal_point.shape[0]
        rgbd_embed_rep = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)       # (B*S, 128, 384)
        goal_embed_rep = torch.repeat_interleave(pointgoal_embed, sample_num, dim=0)  # (B*S, 1, 384)

        # ---- RF ODE 采样（有梯度版本）----
        naction_raw = self._sample_ode_euler_grad(
            goal_embed_rep, rgbd_embed_rep, num_steps=self.cfm_num_steps
        )  # (B*S, predict_size, 3)

        # ---- Critic 评分（有梯度）----
        # critic_scores_flat = self._predict_critic_grad(naction_raw, rgbd_embed_rep)  # (B*S,)
        critic_scores_flat = self._predict_critic_grad(naction_raw, rgbd_embed_rep)
        critic_scores = critic_scores_flat.reshape(B, sample_num)                    # (B, S)

        # ---- 反归一化 + cumsum 得到绝对轨迹 ----
        naction_denorm = self.denormalize_action_tensor(naction_raw)
        all_trajectory_flat = torch.cumsum(naction_denorm / 4.0, dim=1)             # (B*S, 24, 3)
        all_trajectory = all_trajectory_flat.reshape(B, sample_num, self.predict_size, 3)

        # ---- 过滤短轨迹（和推理一致）----
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
        """
        无目标探索的可微分前向（dynnogoal 任务）。
        goal_embed 全零，与预训练 nogoal_step 接口一致。
        返回格式与 forward_train_pointgoal 完全相同。
        """
        rgbd_embed = self.rgbd_encoder(input_images, input_depths)   # (B, 128, 384)
        B = rgbd_embed.shape[0]

        # nogoal: goal_embed 全零，与 predict_nogoal_action 一致
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

    # ------------------------------------------------------------------
    # 有梯度的内部方法（不加 no_grad）
    # ------------------------------------------------------------------

    def _sample_ode_euler_grad(self, goal_embed, rgbd_embed, num_steps: int = 5):
        """
        ODE Euler 积分（有梯度版）。

        与 sample_ode_euler 完全相同的数值逻辑，去掉了 no_grad。
        """
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

        return x  # (B*S, predict_size, 3)

    def _predict_critic_grad(self, predict_trajectory, rgbd_embed, goal_embed=None):
        # critic 设计本来就不看 goal，与基类 predict_critic 完全一致
        nogoal_embed = torch.zeros_like(rgbd_embed[:, 0:1])
        
        action_embeddings = self.input_embed(predict_trajectory)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)

        cond_tokens = torch.cat([
            nogoal_embed,
            nogoal_embed,
            nogoal_embed,
            nogoal_embed,
            rgbd_embed,
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