# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""
差速控制器
功能：将高层控制指令 (v, ω) 转换为底层关节速度 (v_left, v_right)

差速机器人运动学：
    机器人中心的线速度 v 和角速度 ω 与左右轮速度的关系：
    
    v = (v_left + v_right) / 2 * r           # 中心线速度
    ω = (v_right - v_left) / b * r           # 角速度
    
    反解得到：
    v_left = (2v - ωb) / (2r)
    v_right = (2v + ωb) / (2r)
    
    其中：
    - v: 机器人中心线速度 (m/s)
    - ω: 机器人角速度 (rad/s)
    - v_left, v_right: 左右轮角速度 (rad/s)
    - r: 轮子半径 (m)
    - b: 轮距（两轮之间的距离，m）
"""
import numpy as np
from .base_controller import BaseController
from omni.isaac.core.utils.types import ArticulationAction
import torch

class DifferentialController(BaseController):
    """
    差速机器人控制器
    
    功能：
    - 输入：(v, ω) - 期望的中心线速度和角速度
    - 输出：(v_left, v_right) - 左右轮的角速度
    
    运动学模型：
        ω_R = (2V + ωb) / (2r)
        ω_L = (2V - ωb) / (2r)
    
    其中：
    - ω_R, ω_L: 右轮和左轮角速度 (rad/s)
    - V: 机器人中心线速度 (m/s)
    - ω: 机器人角速度 (rad/s)
    - r: 轮子半径 (m)
    - b: 轮距 (m)
    
    Args:
        name (str): 控制器名称
        wheel_radius (float): 轮子半径 (m)
        wheel_base (float): 轮距 (m)
        max_linear_speed (float): 最大线速度限制 (m/s)
        max_angular_speed (float): 最大角速度限制 (rad/s)
        max_wheel_speed (float): 最大轮速限制 (rad/s)
    """
    def __init__(
        self,
        name: str,
        wheel_radius: float,
        wheel_base: float,
        max_linear_speed: float = 1.0e20,
        max_angular_speed: float = 1.0e20,
        max_wheel_speed: float = 1.0e20,
    ) -> None:
        super().__init__(name)
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.max_wheel_speed = max_wheel_speed

        assert self.max_linear_speed >= 0
        assert self.max_angular_speed >= 0
        assert self.max_wheel_speed >= 0

    def forward(self, command: np.ndarray) -> ArticulationAction:
        """
        正向运动学：(v, ω) → (v_left, v_right)
        
        Args:
            command (np.ndarray): 控制指令 [v, ω]
                v: 线速度 (m/s)
                ω: 角速度 (rad/s)
        
        Returns:
            ArticulationAction: 关节控制动作（左右轮速度）
        
        计算流程：
        1. 限制输入速度在允许范围内
        2. 根据差速运动学公式计算左右轮速度
        3. 限制轮速在允许范围内
        
        示例：
            command = [0.5, 0.2]  # v=0.5m/s, ω=0.2rad/s
            → v_left = 4.2 rad/s, v_right = 5.8 rad/s
        """
        # ===== 步骤1：输入检查 =====
        if isinstance(command, list):
            command = np.array(command)
        if command.shape[0] != 2:
            raise Exception("控制指令应为长度2的数组：[v, ω]")

        # ===== 步骤2：限制输入速度 =====
        command = np.clip(
            command,
            a_min=[-self.max_linear_speed, -self.max_angular_speed],
            a_max=[self.max_linear_speed, self.max_angular_speed],
        )
        
        # ===== 步骤3：差速运动学公式 =====
        # v_left = (2v - ωb) / (2r)
        # v_right = (2v + ωb) / (2r)
        #
        # 其中：
        # - command[0] = v (线速度)
        # - command[1] = ω (角速度)
        # - self.wheel_base = b (轮距)
        # - self.wheel_radius = r (轮子半径)
        
        joint_velocities = [0.0, 0.0]
        
        # 左轮角速度
        joint_velocities[0] = ((2 * command[0]) - (command[1] * self.wheel_base)) / (2 * self.wheel_radius)
        
        # 右轮角速度
        joint_velocities[1] = ((2 * command[0]) + (command[1] * self.wheel_base)) / (2 * self.wheel_radius)
        
        # ===== 步骤4：限制轮速 =====
        joint_velocities = np.clip(
            joint_velocities,
            a_min=[-self.max_wheel_speed, -self.max_wheel_speed],
            a_max=[self.max_wheel_speed, self.max_wheel_speed],
        )
        
        return ArticulationAction(joint_velocities=joint_velocities)

    def forward_batch(self, commands: np.ndarray) -> torch.Tensor:
        """
        批处理版本：将多个控制指令同时转换为轮速
        
        Args:
            commands (np.ndarray): 批次控制指令 (batch, 2)
                每行为 [v, ω]
        
        Returns:
            torch.Tensor: 批次关节速度 (batch, 2)
                每行为 [v_left, v_right]
        
        示例：
            commands = [[0.5, 0.2],   # 机器人1: v=0.5, ω=0.2
                       [0.3, -0.1]]  # 机器人2: v=0.3, ω=-0.1
            → [[v_left1, v_right1],
               [v_left2, v_right2]]
        """
        # ===== 批量计算轮速（向量化操作）=====
        joint_velocities = np.zeros((commands.shape[0], 2))
        
        # 左轮角速度（batch）
        joint_velocities[:, 0] = ((2 * commands[:, 0]) - (commands[:, 1] * self.wheel_base)) / (2 * self.wheel_radius)
        
        # 右轮角速度（batch）
        joint_velocities[:, 1] = ((2 * commands[:, 0]) + (commands[:, 1] * self.wheel_base)) / (2 * self.wheel_radius)

        return torch.tensor(joint_velocities, dtype=torch.float32)