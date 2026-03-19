"""
MPC 轨迹跟踪控制器
功能：将NavDP规划的轨迹转换为平滑的控制指令

核心算法：
- 模型预测控制（MPC）：优化有限时域内的控制序列
- 使用 CasADi 进行非线性优化
- 差速机器人运动学模型

优化目标：
- 最小化轨迹跟踪误差
- 最小化控制输入变化（平滑性）
- 满足速度和角速度约束
"""
import casadi as ca
import numpy as np
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scipy.interpolate import interp1d
from typing import Optional, List, Tuple
from dataclasses import dataclass
from queue import Queue

@dataclass
class PlanningInput:
    """规划线程的输入缓存"""
    current_goal: Optional[np.ndarray] = None      # 目标 (batch, 2/3)
    current_image: Optional[np.ndarray] = None     # RGB (batch, H, W, 3)
    current_depth: Optional[np.ndarray] = None     # Depth (batch, H, W)
    camera_pos: Optional[np.ndarray] = None        # 相机位置 (batch, 3)
    camera_rot: Optional[np.ndarray] = None        # 相机旋转矩阵 (batch, 3, 3)

@dataclass
class PlanningOutput:
    """规划线程的输出缓存"""
    trajectory_points_world: Optional[np.ndarray] = None       # 最优轨迹（世界坐标）
    all_trajectories_world: Optional[List[np.ndarray]] = None  # 所有候选轨迹
    all_values_camera: Optional[np.ndarray] = None             # Critic 价值
    is_planning: bool = False                                   # 是否正在规划
    planning_error: Optional[str] = None                        # 错误信息

class MPC_Controller:
    """
    模型预测控制器（MPC）
    
    功能：根据规划轨迹优化控制指令，实现平滑跟踪
    
    优化问题：
        min  Σ (state - ref_state)^T Q (state - ref_state) + u^T R u
        s.t. state[k+1] = state[k] + f(state[k], u[k]) * dt
             0 <= v <= v_max
             -w_max <= w <= w_max
    
    状态空间：state = [x, y, θ] (位置和朝向)
    控制空间：u = [v, ω] (线速度和角速度)
    """
    def __init__(self, global_planed_traj, N=15, desired_v=0.5, v_max=0.5, w_max=0.5, ref_gap=3):
        """
        Args:
            global_planed_traj: 规划轨迹 (num_points, 2)，世界坐标系下的 (x, y)
            N: 预测时域长度（优化15个时间步）
            desired_v: 期望速度 (m/s)
            v_max: 最大线速度 (m/s)
            w_max: 最大角速度 (rad/s)
            ref_gap: 参考点间隔（每3步取一个参考点）
        """
        self.N = N                  # 预测时域 = 15
        self.desired_v = desired_v  # 期望速度 = 0.5 m/s
        self.ref_gap = ref_gap      # 参考点间隔 = 3
        self.T = 0.1                # 控制周期 = 0.1s (10Hz)
        
        # ===== 步骤1：加密参考轨迹 =====
        # 将24个轨迹点插值到更密集的点（24*50=1200个点）
        # 便于实时查找最近的参考点
        self.ref_traj = self.make_ref_denser(global_planed_traj)
        self.ref_traj_len = N // ref_gap + 1  # 参考点数量 = 15//3+1 = 6

        # ===== 步骤2：构建优化问题（使用 CasADi）=====
        opti = ca.Opti()  # 优化对象
        
        # 决策变量1：控制序列 (N, 2)
        opt_controls = opti.variable(N, 2)
        v, w = opt_controls[:, 0], opt_controls[:, 1]  # v: 线速度, w: 角速度
        
        # 决策变量2：状态序列 (N+1, 3)
        opt_states = opti.variable(N+1, 3)
        x, y, theta = opt_states[:, 0], opt_states[:, 1], opt_states[:, 2]  # x, y, θ
        
        # 参数1：初始状态 (3,)
        opt_x0 = opti.parameter(3)
        
        # 参数2：参考轨迹 (ref_traj_len * 3,)
        # 存储格式：[x0, y0, θ0, x1, y1, θ1, ...]
        opt_xs = opti.parameter(3 * self.ref_traj_len)
        
        # ===== 步骤3：定义运动学模型 =====
        # 差速机器人运动学：
        #   ẋ = v * cos(θ)
        #   ẏ = v * sin(θ)
        #   θ̇ = ω
        f = lambda x_, u_: ca.vertcat(*[
            u_[0] * ca.cos(x_[2]),  # ẋ
            u_[0] * ca.sin(x_[2]),  # ẏ
            u_[1]                    # θ̇
        ])
        
        # ===== 步骤4：添加约束 =====
        # 约束1：初始状态约束
        opti.subject_to(opt_states[0, :] == opt_x0.T)
        
        # 约束2：动力学约束（欧拉积分）
        for i in range(N):
            x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T * self.T
            opti.subject_to(opt_states[i+1, :] == x_next)
        
        # ===== 步骤5：定义代价函数 =====
        # Q: 状态误差权重矩阵
        Q = np.diag([10.0, 10.0, 0.0])  # 只惩罚 (x, y)，不惩罚 θ
        
        # R: 控制输入权重矩阵
        R = np.diag([0.02, 0.15])  # 惩罚控制变化，鼓励平滑
        
        obj = 0
        for i in range(N):
            # 控制输入代价：u^T R u
            obj = obj + ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])
            
            # 跟踪误差代价：每隔 ref_gap 步与参考点比较
            if i % ref_gap == 0:
                nn = i // ref_gap
                state_error = opt_states[i, :] - opt_xs[nn*3:nn*3+3].T
                obj = obj + ca.mtimes([state_error, Q, state_error.T])
        
        opti.minimize(obj)
        
        # ===== 步骤6：添加边界约束 =====
        opti.subject_to(opti.bounded(0.0, v, v_max))        # 0 <= v <= v_max
        opti.subject_to(opti.bounded(-w_max, w, w_max))     # -w_max <= w <= w_max
        
        # ===== 步骤7：配置求解器（IPOPT）=====
        opts_setting = {
            'ipopt.max_iter': 100,              # 最大迭代次数
            'ipopt.print_level': 0,             # 关闭打印
            'print_time': 0,                    # 关闭时间打印
            'ipopt.acceptable_tol': 1e-8,       # 可接受容差
            'ipopt.acceptable_obj_change_tol': 1e-6
        }
        opti.solver('ipopt', opts_setting)
        
        # 保存优化对象
        self.opti = opti
        self.opt_xs = opt_xs
        self.opt_x0 = opt_x0
        self.opt_controls = opt_controls
        self.opt_states = opt_states
        self.last_opt_x_states = None    # 上一次的最优状态序列（热启动）
        self.last_opt_u_controls = None  # 上一次的最优控制序列（热启动）
    def make_ref_denser(self, ref_traj, ratio=50):
        """
        加密参考轨迹
        
        功能：通过线性插值将稀疏轨迹点变密集，便于实时查找最近点
        
        Args:
            ref_traj: 原始轨迹 (N, 2)，如 (24, 2)
            ratio: 加密比例，默认50倍
        
        Returns:
            dense_traj: 加密后的轨迹 (N*ratio, 2)，如 (1200, 2)
        
        原理：
            原始24个点 → 插值到1200个点 → 查找时更准确
        """
        x_orig = np.arange(len(ref_traj))  # [0, 1, 2, ..., 23]
        new_x = np.linspace(0, len(ref_traj) - 1, num=len(ref_traj) * ratio)  # [0, 0.02, 0.04, ..., 23]
        
        # 对 x, y 分别插值
        interp_func_x = interp1d(x_orig, ref_traj[:, 0], kind='linear')
        interp_func_y = interp1d(x_orig, ref_traj[:, 1], kind='linear')
        
        uniform_x = interp_func_x(new_x)
        uniform_y = interp_func_y(new_x)
        ref_traj = np.stack((uniform_x, uniform_y), axis=1)
        return ref_traj
    
    def solve(self, x00):
        """
        求解MPC优化问题
        
        Args:
            x00: 当前状态 (3,)，[x, y, θ]
        
        Returns:
            opt_u_controls: 最优控制序列 (N, 2)，[v, ω]
            opt_x_states: 最优状态序列 (N+1, 3)，[x, y, θ]
        
        流程：
        1. 查找参考轨迹（从当前位置开始的N个点）
        2. 设置优化参数（初始状态、参考轨迹）
        3. 热启动（用上一次的解作为初始猜测）
        4. 求解优化问题
        5. 保存结果用于下一次热启动
        """
        # ===== 步骤1：查找参考轨迹 =====
        ref_traj = self.find_reference_traj(x00, self.ref_traj)
        # ref_traj: (ref_traj_len, 2)，从当前位置开始的参考点
        
        # 添加角度维度（设为0，因为MPC中角度权重为0）
        ref_traj = np.concatenate((ref_traj, np.zeros((ref_traj.shape[0], 1))), axis=1).reshape(-1, 1)
        # ref_traj: (ref_traj_len*3, 1)，格式 [x0,y0,0, x1,y1,0, ...]
        
        # ===== 步骤2：设置参数 =====
        self.opti.set_value(self.opt_xs, ref_traj.reshape(-1, 1))  # 参考轨迹
        self.opti.set_value(self.opt_x0, x00)                       # 初始状态
        
        # ===== 步骤3：热启动（用上一次的解） =====
        u0 = np.zeros((self.N, 2)) if self.last_opt_u_controls is None else self.last_opt_u_controls
        x0 = np.zeros((self.N+1, 3)) if self.last_opt_x_states is None else self.last_opt_x_states
        self.opti.set_initial(self.opt_controls, u0)
        self.opti.set_initial(self.opt_states, x0)
        sol = self.opti.solve()
        self.last_opt_u_controls = sol.value(self.opt_controls)
        self.last_opt_x_states = sol.value(self.opt_states)

        return self.last_opt_u_controls, self.last_opt_x_states
    def reset(self):
        self.last_opt_x_states = None
        self.last_opt_u_controls = None
        
    def find_reference_traj(self, x0, global_planed_traj):
        ref_traj_pts = []
        # find the nearest point in global_planed_traj
        nearest_idx = np.argmin(np.linalg.norm(global_planed_traj - x0[:2].reshape((1, 2)), axis=1))
        desire_arc_length = self.desired_v * self.ref_gap * self.T 
        cum_dist = np.cumsum(np.linalg.norm(np.diff(global_planed_traj, axis=0), axis=1))

        # select the reference points from the nearest point to the end of global_planed_traj
        for i in range(nearest_idx, len(global_planed_traj) - 1):
            if cum_dist[i] - cum_dist[nearest_idx] >= desire_arc_length * len(ref_traj_pts):
                ref_traj_pts.append(global_planed_traj[i, :])
                if len(ref_traj_pts) == self.ref_traj_len:
                    break
        # if the target is reached before the reference trajectory is complete, add the last point of global_planed_traj 
        while len(ref_traj_pts) < self.ref_traj_len:
            ref_traj_pts.append(global_planed_traj[-1, :])
        return np.array(ref_traj_pts)