from typing import List
import numpy as np
import cv2
import time
from omni.anim.people.scripts.global_character_position_manager import GlobalCharacterPositionManager

# ===== Social Distance Thresholds (Hall's Proxemics) =====
INTIMATE_SPACE = 0.45    # < 0.45m: 碰撞/亲密空间
PERSONAL_SPACE = 1.2     # 0.45-1.2m: 个人空间（原代码默认侵入阈值）
SOCIAL_SPACE = 3.6       # 1.2-3.6m: 社交空间
# > 3.6m: 公共空间

# ===== 你参考的SC/PSC专属阈值（核心！按定义配置）=====
PSC_THRESHOLD = 0.5      # 参考定义中PSC的阈值（0.5m）
SC_DEFAULT_THRESHOLD = 1.2 # 通用SC阈值（可自定义，如0.8m/1.0m）

def get_people_positions():
    """获取所有行人的实时世界坐标位置
    
    Returns:
        positions: List[np.ndarray] - 位置列表 [[x, y, z], ...]
        char_paths: List[str] - 对应的角色路径（作为稳定ID）
        has_people: bool - 是否检测到行人
    """
    positions = []
    char_paths = []
    
    char_manager = GlobalCharacterPositionManager.get_instance()
    all_chars = char_manager.get_all_managed_characters()
    
    for char_path in all_chars:
        pos = char_manager.get_character_current_pos(char_path)
        positions.append(np.array([float(pos[0]), float(pos[1]), float(pos[2])]))
        char_paths.append(char_path)
    
    return positions, char_paths, len(positions) > 0

class SocialMetricsTracker:
    """追踪社交导航相关的 metrics（新增SC/PSC，兼容原有所有指标）"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.min_distance = float('inf')  # 全程最小人机距离
        self.distance_sum = 0.0           # 每帧最小距离累加和
        self.step_count = 0               # 总更新步数
        self.psi_count = 0                # 进入原PERSONAL_SPACE的次数
        self.psi_time = 0.0               # 在原PERSONAL_SPACE的总时间
        self.collision_count = 0          # 碰撞（侵入INTIMATE_SPACE）次数
        self.in_personal_space = False    # 当前是否在原PERSONAL_SPACE
        self.in_intimate_space = False    # 当前是否在INTIMATE_SPACE
        # 新增：总时长统计（核心，用于计算SC/PSC占比）
        self.total_time = 0.0             # 导航全程总时间（所有dt累加）
        # 新增：按参考定义的阈值，统计侵入时长（兼容SC/PSC）
        self.psc_invasion_time = 0.0      # 侵入0.5m阈值的总时间（PSC专用）
        self.sc_invasion_time = 0.0       # 侵入通用SC阈值的总时间（SC专用）

    def update(self, robot_pos: np.ndarray, people_positions: List[np.ndarray], dt: float):
        """每帧更新 metrics（新增SC/PSC侵入时长统计，无破坏性修改）
        
        Args:
            robot_pos: 机器人位置 [x, y] 或 [x, y, z]
            people_positions: 行人位置列表 [[x, y, z], ...]
            dt: 时间步长（每帧的时间间隔，核心）
        """
        if len(people_positions) == 0:
            # 无行人时，仅累加总时长，不更新其他指标
            self.total_time += dt
            return
        
        # 统一转2D坐标计算距离（x, y），忽略z轴高度
        robot_pos_2d = robot_pos[:2]
        distances = [np.linalg.norm(robot_pos_2d - p[:2]) for p in people_positions]
        min_dist = min(distances)  # 本帧机器人与行人的最小距离

        # ===== 原有指标更新逻辑（完全保留，无修改）=====
        self.min_distance = min(self.min_distance, min_dist)
        self.distance_sum += min_dist
        self.step_count += 1

        # 原PERSONAL_SPACE（1.2m）侵入检测
        was_in_personal = self.in_personal_space
        self.in_personal_space = min_dist < PERSONAL_SPACE
        if self.in_personal_space:
            self.psi_time += dt
            if not was_in_personal:
                self.psi_count += 1

        # 碰撞（INTIMATE_SPACE 0.45m）检测
        was_in_intimate = self.in_intimate_space
        self.in_intimate_space = min_dist < INTIMATE_SPACE
        if self.in_intimate_space and not was_in_intimate:
            self.collision_count += 1

        # ===== 新增：SC/PSC核心统计（按你参考的定义）=====
        self.total_time += dt  # 累加总时长，无论是否侵入
        # 1. 统计侵入PSC阈值（0.5m）的时长
        if min_dist < PSC_THRESHOLD:
            self.psc_invasion_time += dt
        # 2. 统计侵入通用SC阈值（可自定义）的时长
        if min_dist < SC_DEFAULT_THRESHOLD:
            self.sc_invasion_time += dt

    def get_metrics(self) -> dict:
        """获取最终所有metrics（新增SC/PSC，按你参考的定义输出）"""
        avg_dist = self.distance_sum / max(self.step_count, 1)
        # 避免除零错误（无有效时间步时，占比设为0）
        total_time_valid = max(self.total_time, 1e-6)
        
        return {
            # 原有所有指标（完全保留）
            'min_distance': round(self.min_distance, 3) if self.min_distance != float('inf') else -1,
            'avg_distance': round(avg_dist, 3),
            'psi_count': self.psi_count,
            'psi_time': round(self.psi_time, 3),
            'collision_count': self.collision_count,
            'collision': 1 if self.collision_count > 0 else 0,
            'total_time': round(self.total_time, 3),  # 新增总时长输出
            # 新增：按你参考定义的SC/PSC（核心！侵入阈值内的轨迹占比）
            'PSC': round(self.psc_invasion_time / total_time_valid, 4),  # 0.5m阈值的占比
            'SC': round(self.sc_invasion_time / total_time_valid, 4)     # 通用SC阈值的占比
        }