# NavDP 工程实现总结

## 📋 目录
1. [整体架构](#整体架构)
2. [客户端实现](#客户端实现)
3. [服务端实现](#服务端实现)
4. [控制流程](#控制流程)
5. [关键技术点](#关键技术点)
6. [性能优化](#性能优化)

---

## 🏗️ 整体架构

### 系统拓扑

```
┌────────────────────────────────────────────────────────────────┐
│                     NavDP 导航系统                              │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────── Client (Isaac Sim) ──────────┐                   │
│  │                                          │                   │
│  │  Main Thread (100Hz)                    │                   │
│  │  ┌────────────────────────────────┐    │                   │
│  │  │ 1. 物理仿真 (0.01s/step)      │    │                   │
│  │  │ 2. 获取观测 (RGB, Depth, Pose)│    │                   │
│  │  │ 3. 更新共享缓存               │    │                   │
│  │  │ 4. 读取规划结果               │    │                   │
│  │  │ 5. MPC 控制求解               │    │                   │
│  │  │ 6. 差速控制转换               │    │                   │
│  │  │ 7. 执行动作                   │    │                   │
│  │  │ 8. 可视化                     │    │                   │
│  │  └────────────────────────────────┘    │                   │
│  │           ↕ (Lock)                      │                   │
│  │  Planning Thread (10Hz)                 │                   │
│  │  ┌────────────────────────────────┐    │                   │
│  │  │ 1. 读取共享缓存               │    │                   │
│  │  │ 2. 编码数据 (JPEG/PNG/JSON)  │    │   HTTP POST        │
│  │  │ 3. HTTP 请求 ─────────────────┼────┼──────────→        │
│  │  │ 4. 解析响应                   │    │   HTTP Response    │
│  │  │ 5. 坐标转换 (Camera→World)   │ ←──┼──────────         │
│  │  │ 6. 初始化 MPC                 │    │                   │
│  │  │ 7. 更新共享缓存               │    │                   │
│  │  └────────────────────────────────┘    │                   │
│  └──────────────────────────────────────────┘                  │
│                                                                  │
│  ┌─────────── Server (Flask) ──────────────┐                   │
│  │                                          │                   │
│  │  HTTP Server (Flask, 多线程)            │                   │
│  │  ┌────────────────────────────────┐    │                   │
│  │  │ 1. 接收请求                   │    │                   │
│  │  │ 2. 解码数据 (JPEG→numpy)     │    │                   │
│  │  │ 3. 调用 NavDP_Agent           │    │                   │
│  │  │ 4. 图像预处理                 │    │                   │
│  │  │ 5. 历史队列管理               │    │                   │
│  │  │ 6. 模型推理 (GPU)             │    │                   │
│  │  │ 7. 后处理                     │    │                   │
│  │  │ 8. 编码响应 (JSON)            │    │                   │
│  │  └────────────────────────────────┘    │                   │
│  │           ↓                              │                   │
│  │  NavDP_Policy (DDPM + Transformer)      │                   │
│  │  ┌────────────────────────────────┐    │                   │
│  │  │ RGBD 编码 → memory_token      │    │                   │
│  │  │ Goal 编码 → goal_embed        │    │                   │
│  │  │ DDPM 采样 (10步去噪)          │    │                   │
│  │  │ Critic 评分                   │    │                   │
│  │  │ 选择最优轨迹                  │    │                   │
│  │  └────────────────────────────────┘    │                   │
│  └──────────────────────────────────────────┘                  │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

---

## 💻 客户端实现

### 文件结构

```
eval_pointgoal_wheeled.py        # 主评估脚本
├── utils_tasks/
│   ├── client_utils.py           # HTTP 客户端封装
│   ├── tracking_utils.py         # MPC 控制器
│   └── basic_utils.py            # 通用工具
├── wheeled_robots/controllers/
│   └── differential_controller.py # 差速控制器
└── configs/
    ├── robots.py                 # 机器人配置 (Dingo)
    ├── scenes.py                 # 场景配置
    └── tasks/
        └── wheeled_task.py       # 任务配置
```

### 核心组件

#### 1. 异步规划线程 (`planning_thread`)

**功能**：异步调用 NavDP Server，不阻塞主仿真循环

```python
def planning_thread(env, camera_intrinsic):
    while not stop_event.is_set():
        # 1. 读取共享输入（加锁）
        with input_lock:
            goal = planning_input.current_goal.copy()
            image = planning_input.current_image.copy()
            depth = planning_input.current_depth.copy()
            camera_pos = planning_input.camera_pos.copy()
            camera_rot = planning_input.camera_rot.copy()
        
        # 2. HTTP 请求（80ms，不阻塞主线程）
        trajectory_camera, all_traj, all_values = \
            pointgoal_step(goal, image, depth, port=8888)
        
        # 3. 坐标转换（相机 → 世界）
        trajectory_world = camera_pos + camera_rot @ trajectory_camera
        
        # 4. 初始化 MPC
        mpc = MPC_Controller(trajectory_world, desired_v=0.5)
        
        # 5. 更新共享输出（加锁）
        with output_lock:
            planning_output.trajectory_points_world = trajectory_world
            planning_output.all_trajectories_world = all_traj_world
            planning_output.all_values = all_values
        
        time.sleep(0.1)  # 10Hz 规划频率
```

**线程安全设计**：
- 使用 `threading.Lock()` 保护共享数据
- 输入锁：保护 `planning_input`
- 输出锁：保护 `planning_output`
- 主线程写入输入、读取输出
- 规划线程读取输入、写入输出

#### 2. 主仿真循环

**功能**：100Hz 物理仿真 + 10Hz 控制更新

```python
while simulation_app.is_running():
    # ===== 步骤1：获取观测（100Hz）=====
    goals = infos['observations']['goal_pose']
    images = infos['observations']['rgb']
    depths = infos['observations']['depth']
    camera_pos = env.scene.sensors['camera_sensor'].data.pos_w
    camera_rot = R.from_quat(camera_rot_quat).as_matrix()
    
    # ===== 步骤2：更新规划输入（加锁）=====
    with input_lock:
        planning_input.current_goal = goals.copy()
        planning_input.current_image = images.copy()
        planning_input.current_depth = depths.copy()
        planning_input.camera_pos = camera_pos.copy()
        planning_input.camera_rot = camera_rot.copy()
    
    # ===== 步骤3：读取规划输出（加锁）=====
    with output_lock:
        current_trajectory = planning_output.trajectory_points_world
        current_all_trajectories = planning_output.all_trajectories_world
        current_all_values = planning_output.all_values
    
    # ===== 步骤4：MPC 控制（如果轨迹可用）=====
    if current_trajectory is not None:
        # 4.1 MPC 求解最优控制
        x0 = [camera_pos[0], camera_pos[1], theta, robot_vel, robot_ang_vel]
        opt_u_controls, opt_x_states = mpc.solve(x0[:3])
        v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]
        
        # 4.2 差速控制转换
        joint_velocities = controller.forward([v, w]).joint_velocities
        
        # 4.3 执行动作
        action = torch.tensor(joint_velocities, device="cuda:0")
        obs, rewards, dones, infos = env.step(action)
    else:
        # 零动作（轨迹未就绪）
        env.step(torch.zeros(2, device="cuda:0"))
    
    # ===== 步骤5：检查回合结束 =====
    if dones:
        # 计算评估指标
        success = (distance_to_goal < 1.5)
        spl = (euclidean / trajectory_length) * success
        evaluation_metrics.append({'success': success, 'spl': spl})
        
        # 重置环境
        navigator_reset(env_id=i, port=8888)
```

**时序分析**：
```
时间轴 (ms):
0────10───20───30───40───50───60───70───80───90───100

Main Thread (100Hz):
├─ Step: ●────●────●────●────●────●────●────●────●────●  (10ms/step)
├─ Obs:  ●────●────●────●────●────●────●────●────●────●  (获取观测)
└─ Ctrl: ●────●────●────●────●────●────●────●────●────●  (MPC + 执行)

Planning Thread (10Hz):
└─ Plan: ●─────────────────────────●────────────────────── (100ms/plan)
         ├─ HTTP: 80ms (NavDP推理)
         └─ Trans: 20ms (坐标转换 + MPC初始化)
```

#### 3. HTTP 客户端 (`client_utils.py`)

**功能**：封装与 NavDP Server 的通信

```python
def pointgoal_step(point_goals, rgb_images, depth_images, port=8888):
    """
    点目标导航推理
    
    数据编码：
    - RGB: JPEG 压缩（节省带宽）
    - Depth: PNG 16位，深度 = 米 * 10000
    - Goal: JSON 格式
    
    返回：
    - trajectory: (batch, 24, 3) 最优轨迹
    - all_trajectory: (batch, 16, 24, 3) 所有候选
    - all_values: (batch, 16) Critic 价值
    """
    # 1. 拼接 batch
    concat_images = np.concatenate([img for img in rgb_images], axis=0)
    concat_depths = np.concatenate([img for img in depth_images], axis=0)
    
    # 2. 编码 RGB (JPEG)
    _, rgb_image = cv2.imencode('.jpg', concat_images)
    
    # 3. 编码 Depth (PNG 16位)
    depth_uint16 = np.clip(concat_depths * 10000, 0, 65535).astype(np.uint16)
    _, depth_image = cv2.imencode('.png', depth_uint16)
    
    # 4. 发送 HTTP POST
    url = f"http://localhost:{port}/pointgoal_step"
    files = {
        'image': ('image.jpg', rgb_image, 'image/jpeg'),
        'depth': ('depth.png', depth_image, 'image/png'),
    }
    data = {
        'goal_data': json.dumps({
            'goal_x': point_goals[:, 0].tolist(),
            'goal_y': point_goals[:, 1].tolist()
        })
    }
    response = requests.post(url, files=files, data=data)
    
    # 5. 解析 JSON 响应
    trajectory = json.loads(response.text)['trajectory']
    all_trajectory = json.loads(response.text)['all_trajectory']
    all_values = json.loads(response.text)['all_values']
    
    return np.array(trajectory), np.array(all_trajectory), np.array(all_values)
```

#### 4. MPC 控制器 (`tracking_utils.py`)

**功能**：将规划轨迹转换为平滑控制指令

```python
class MPC_Controller:
    """
    模型预测控制器
    
    优化目标：
        min Σ (state - ref_state)^T Q (state - ref_state) + u^T R u
        s.t. state[k+1] = state[k] + f(state, u) * dt
             0 <= v <= v_max
             -w_max <= w <= w_max
    
    参数：
    - N = 15: 预测时域（1.5秒）
    - T = 0.1s: 控制周期
    - Q = diag([10, 10, 0]): 状态权重（只惩罚 x, y）
    - R = diag([0.02, 0.15]): 控制权重（鼓励平滑）
    """
    def __init__(self, global_planed_traj, N=15, desired_v=0.5):
        # 1. 加密参考轨迹 (24点 → 1200点)
        self.ref_traj = self.make_ref_denser(global_planed_traj, ratio=50)
        
        # 2. 构建优化问题 (CasADi)
        opti = ca.Opti()
        opt_controls = opti.variable(N, 2)  # 决策变量：u = [v, ω]
        opt_states = opti.variable(N+1, 3)  # 预测状态：x = [x, y, θ]
        
        # 3. 运动学约束
        f = lambda x, u: ca.vertcat(
            u[0] * ca.cos(x[2]),  # ẋ = v * cos(θ)
            u[0] * ca.sin(x[2]),  # ẏ = v * sin(θ)
            u[1]                  # θ̇ = ω
        )
        for i in range(N):
            opti.subject_to(
                opt_states[i+1, :] == opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]) * self.T
            )
        
        # 4. 代价函数
        obj = 0
        for i in range(N):
            # 控制代价
            obj += opt_controls[i, :] @ R @ opt_controls[i, :].T
            # 跟踪误差代价（每隔3步）
            if i % 3 == 0:
                error = opt_states[i, :] - ref_state[i//3, :]
                obj += error @ Q @ error.T
        opti.minimize(obj)
        
        # 5. 边界约束
        opti.subject_to(opti.bounded(0, v, v_max))
        opti.subject_to(opti.bounded(-w_max, w, w_max))
        
        # 6. 配置求解器
        opti.solver('ipopt', {'ipopt.max_iter': 100, 'ipopt.print_level': 0})
    
    def solve(self, x0):
        """
        求解 MPC 优化问题
        
        Args:
            x0: 当前状态 [x, y, θ]
        
        Returns:
            opt_u: 最优控制序列 (N, 2)
            opt_x: 最优状态序列 (N+1, 3)
        """
        # 1. 查找参考轨迹
        ref_traj = self.find_reference_traj(x0, self.ref_traj)
        
        # 2. 设置参数
        self.opti.set_value(self.opt_x0, x0)
        self.opti.set_value(self.opt_xs, ref_traj.flatten())
        
        # 3. 热启动（用上一次的解）
        if self.last_opt_u is not None:
            self.opti.set_initial(self.opt_controls, self.last_opt_u)
            self.opti.set_initial(self.opt_states, self.last_opt_x)
        
        # 4. 求解
        sol = self.opti.solve()
        opt_u = sol.value(self.opt_controls)
        opt_x = sol.value(self.opt_states)
        
        # 5. 保存用于下次热启动
        self.last_opt_u = opt_u
        self.last_opt_x = opt_x
        
        return opt_u, opt_x
```

**MPC 性能分析**：
- 求解时间：5-10ms (IPOPT)
- 预测时域：1.5秒 (15步 × 0.1s)
- 控制频率：100Hz（每步都调用，但只用第2步的控制量）
- 热启动效果：减少50%求解时间

#### 5. 差速控制器 (`differential_controller.py`)

**功能**：(v, ω) → (v_left, v_right)

```python
class DifferentialController:
    """
    差速运动学模型：
        v_left = (2v - ωb) / (2r)
        v_right = (2v + ωb) / (2r)
    
    其中：
    - v: 中心线速度 (m/s)
    - ω: 角速度 (rad/s)
    - r: 轮子半径 (m)
    - b: 轮距 (m)
    """
    def forward(self, command):
        v, w = command[0], command[1]
        v_left = ((2 * v) - (w * self.wheel_base)) / (2 * self.wheel_radius)
        v_right = ((2 * v) + (w * self.wheel_base)) / (2 * self.wheel_radius)
        return ArticulationAction(joint_velocities=[v_left, v_right])
```

**Dingo 机器人参数**：
```python
DINGO_WHEEL_RADIUS = 0.1  # 轮子半径 10cm
DINGO_WHEEL_BASE = 0.5    # 轮距 50cm

# 示例计算：
command = [0.5, 0.2]  # v=0.5m/s, ω=0.2rad/s
v_left = ((2*0.5) - (0.2*0.5)) / (2*0.1) = 4.5 rad/s
v_right = ((2*0.5) + (0.2*0.5)) / (2*0.1) = 5.5 rad/s
```

---

## 🖥️ 服务端实现

### 核心流程

```python
@app.route("/pointgoal_step", methods=['POST'])
def navdp_step_xy():
    # 1. 接收数据
    image_file = request.files['image']      # JPEG
    depth_file = request.files['depth']      # PNG
    goal_data = json.loads(request.form['goal_data'])
    
    # 2. 解码
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    depth_uint16 = cv2.imdecode(np.frombuffer(depth_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    depth = depth_uint16.astype(np.float32) / 10000.0  # uint16 → 米
    
    # 3. 调用 Agent
    trajectory, all_trajectory, all_values, _ = \
        navdp_navigator.step_pointgoal(
            goal_data['goal_x'],
            goal_data['goal_y'],
            image,
            depth
        )
    
    # 4. 返回 JSON
    return jsonify({
        'trajectory': trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist()
    })
```

**数据流**：
```
HTTP Request (80KB)
    ├─ image.jpg: 50KB (480×640×3 → JPEG压缩)
    ├─ depth.png: 30KB (480×640 → PNG 16位压缩)
    └─ goal_data: <1KB (JSON)

NavDP Agent (80ms)
    ├─ 预处理: 5ms (resize, normalize)
    ├─ 推理: 70ms (DDPM + Transformer)
    └─ 后处理: 5ms (cumsum, 选择最优)

HTTP Response (10KB)
    ├─ trajectory: 0.5KB (24×3 floats)
    ├─ all_trajectory: 8KB (16×24×3 floats)
    └─ all_values: 0.1KB (16 floats)
```

---

## 🎮 控制流程

### 完整控制链

```
NavDP 规划 (10Hz, 80ms)
    ↓ trajectory (24, 3)
MPC 优化 (100Hz, 5-10ms)
    ↓ (v, ω)
差速控制 (100Hz, <1ms)
    ↓ (v_left, v_right)
Isaac Sim 执行 (100Hz, 10ms)
    ↓ 物理仿真
新的观测 (RGB, Depth, Pose)
    ↓
循环...
```

### 层次化控制

```
┌────────────────────────────────────────────┐
│  Layer 3: Planning (10Hz)                  │
│  NavDP: RGB-D → Trajectory (24 points)     │
│  输出: [(x,y,θ)_0, ..., (x,y,θ)_23]       │
└─────────────────┬──────────────────────────┘
                  ↓
┌────────────────────────────────────────────┐
│  Layer 2: Control (100Hz)                  │
│  MPC: Trajectory → (v, ω)                  │
│  优化目标: 最小化跟踪误差 + 控制平滑       │
└─────────────────┬──────────────────────────┘
                  ↓
┌────────────────────────────────────────────┐
│  Layer 1: Actuation (100Hz)                │
│  Differential: (v, ω) → (v_L, v_R)         │
│  运动学模型                                │
└─────────────────┬──────────────────────────┘
                  ↓
┌────────────────────────────────────────────┐
│  Layer 0: Simulation (100Hz)               │
│  Isaac Sim: 物理引擎仿真                   │
└────────────────────────────────────────────┘
```

---

## 🔑 关键技术点

### 1. 异步架构设计

**问题**：NavDP 推理慢 (80ms)，但控制需要实时 (10ms)

**解决方案**：规划-控制分离
```python
# 规划线程 (10Hz): 调用 NavDP
planning_thread():
    trajectory = navdp_step(obs)  # 80ms
    with lock:
        shared_output.trajectory = trajectory

# 主线程 (100Hz): 控制执行
main_loop():
    with lock:
        current_traj = shared_output.trajectory
    if current_traj:
        action = mpc.solve(current_traj)  # 5ms
        env.step(action)
```

**优势**：
- 规划慢不影响控制频率
- 利用 MPC 的预测能力平滑轨迹
- 即使规划失败，也能用上一次的轨迹继续执行

### 2. 坐标系转换

**转换链**：
```
NavDP 输出 (Camera Frame)
    ↓ [Δx, Δy, Δθ] in camera frame
cumsum 累积
    ↓ [x, y, θ] in camera frame (相对于初始点)
World Transform
    ↓ P_world = P_camera + R_camera @ P_local
World Frame
    ↓ [x, y] in world frame (MPC 使用)
```

**代码实现**：
```python
# 1. NavDP 输出：相机坐标系下的增量
trajectory_camera = [[0.1, 0.0, 0.0],   # 第1步: 前进0.1米
                     [0.1, 0.0, 0.0],   # 第2步: 前进0.1米
                     [0.0, 0.1, 0.1]]   # 第3步: 右转

# 2. 累积求和：得到相对轨迹
trajectory_cumsum = cumsum(trajectory_camera / 4.0)
# [[0.025, 0.0, 0.0],
#  [0.050, 0.0, 0.0],
#  [0.050, 0.025, 0.1]]

# 3. 转换到世界坐标系
for point in trajectory_cumsum:
    point_local = [point[0], point[1], 0.0]  # 只取 x, y
    point_world = camera_pos + camera_rot @ point_local
    trajectory_world.append(point_world[:2])  # 只保留 x, y
```

### 3. MPC 参数调优

**权重矩阵选择**：
```python
# 状态误差权重 Q = diag([Q_x, Q_y, Q_θ])
Q = np.diag([10.0, 10.0, 0.0])
# - 位置误差惩罚大 (10.0)
# - 角度误差不惩罚 (0.0)，因为 MPC 会自动调整朝向

# 控制输入权重 R = diag([R_v, R_ω])
R = np.diag([0.02, 0.15])
# - 线速度变化惩罚小 (0.02)，允许加速/减速
# - 角速度变化惩罚大 (0.15)，鼓励平滑转向
```

**预测时域选择**：
```python
N = 15              # 15步预测
T = 0.1             # 0.1秒/步
horizon = N * T = 1.5秒

# 太短 (N<10): 轨迹跟踪不稳定，频繁振荡
# 太长 (N>20): 计算量大，实时性差
# N=15: 平衡性能和实时性
```

### 4. 数据编码优化

**深度图编码**：
```python
# 浮点深度 → uint16 PNG
depth_float = 0.5  # 0.5米
depth_uint16 = int(0.5 * 10000) = 5000  # PNG 存储

# 优势：
# - PNG 无损压缩，保留精度
# - uint16 范围 [0, 65535]，最大深度 6.5535米
# - 压缩比 ~3:1（相比浮点数组）
```

**RGB 编码**：
```python
# numpy → JPEG
_, jpeg_bytes = cv2.imencode('.jpg', rgb_image)

# 优势：
# - 有损压缩，压缩比 ~10:1
# - 导航任务对 RGB 细节要求不高
# - 大幅减少网络传输量
```

### 5. 热启动加速

**MPC 热启动**：
```python
# 第1次求解（冷启动）
u0 = np.zeros((15, 2))  # 初始猜测：全0
sol = opti.solve()      # 求解时间: 20ms
opt_u = sol.value(opt_controls)

# 第2次求解（热启动）
self.opti.set_initial(self.opt_controls, opt_u)  # 用上一次的解
sol = opti.solve()      # 求解时间: 10ms (减少50%)

# 原理：
# - IPOPT 是局部优化器，好的初始点→更快收敛
# - 相邻时刻的最优解通常相近
```

---

## ⚡ 性能优化

### 时间分析

```
单步推理耗时 (Total: ~100ms)

Client Side (30ms):
├─ 获取观测: 5ms (GPU→CPU拷贝)
├─ 数据编码: 5ms (JPEG + PNG)
├─ HTTP 传输: 10ms (localhost, ~100KB)
└─ 坐标转换: 10ms (numpy 计算)

Server Side (80ms):
├─ HTTP 解码: 5ms (JPEG→numpy)
├─ 预处理: 5ms (resize, normalize)
├─ RGBD 编码: 20ms (ViT forward)
├─ DDPM 采样: 45ms (10步 × 4.5ms/步)
├─ Critic 评分: 3ms (单次 forward)
└─ 后处理: 2ms (cumsum, argsort)

Control Side (5-10ms):
├─ MPC 求解: 5-10ms (IPOPT)
└─ 差速转换: <1ms
```

### 瓶颈分析

**最大瓶颈：DDPM 采样 (45ms)**
- 10步去噪，每步需要 Transformer forward
- 16条轨迹并行，batch=16×24=384个点
- 优化方向：
  - 减少去噪步数 (10→5)：加速2倍，但质量下降
  - 模型量化 (FP16)：加速1.5倍
  - 批处理优化：GPU利用率提升

**次要瓶颈：RGBD 编码 (20ms)**
- ViT-Small forward 两次 (RGB + Depth)
- 优化方向：
  - 模型蒸馏：ViT-Small → ViT-Tiny
  - 特征缓存：只编码新帧，历史帧复用

### 内存占用

```
Server Side (GPU):
├─ 模型权重: 200MB (ViT×2 + Transformer + Critic)
├─ 激活值: 500MB (batch=16, 24点, 16层)
└─ 中间结果: 100MB (all_trajectories 缓存)
Total: ~800MB

Client Side (GPU):
├─ 仿真场景: 1.5GB (USD + 纹理)
├─ 物理引擎: 500MB
└─ 渲染缓冲: 200MB (RGB + Depth)
Total: ~2.2GB

建议配置: RTX 3090 (24GB) 可同时运行 Server + Client
```

### 并发能力

**单 Server 支持多 Client**：
```python
# Flask 多线程模式
app.run(host='0.0.0.0', port=8888, threaded=True)

# 并发能力：
# - 1个推理: 80ms
# - 10个并发: 800ms (串行)
# - 理论 QPS: 12.5

# 优化：批处理
# - 收集多个请求，batch 推理
# - 10个请求一起推理: 100ms
# - 理论 QPS: 100
```

---

## 📊 评估指标

### 成功率 (Success Rate)

```python
success = (distance_to_goal < 1.5)  # 终点距离 < 1.5米
success_rate = sum(success) / num_episodes
```

### SPL (Success weighted by Path Length)

```python
spl = (shortest_path / actual_path) * success

# 示例：
# - 最短路径: 5米
# - 实际路径: 7米
# - SPL = (5/7) * 1 = 0.714

# 意义：
# - SPL = 1.0: 完美路径（最短且成功）
# - SPL = 0.5: 绕了一倍的路
# - SPL = 0.0: 失败或路径极差
```

### 碰撞率 (Collision Rate)

```python
collision = contact_sensor.is_in_contact()
collision_rate = sum(collision) / num_steps
```

---

## 🔍 调试技巧

### 1. 可视化轨迹

```python
# 在图像上绘制轨迹
for i, point in enumerate(trajectory):
    # 世界坐标 → 像素坐标
    pixel = camera_intrinsic @ [point[0], point[1], 1.0]
    u, v = int(pixel[0]/pixel[2]), int(pixel[1]/pixel[2])
    
    # 绘制点
    cv2.circle(image, (u, v), radius=3, color=(0, 255, 0), thickness=-1)
    
    # 绘制朝向箭头
    arrow_end = [u + 10*cos(point[2]), v + 10*sin(point[2])]
    cv2.arrowedLine(image, (u, v), tuple(map(int, arrow_end)), (255, 0, 0), 2)
```

### 2. 监控控制性能

```python
# 跟踪误差
tracking_error = np.linalg.norm(current_pos - reference_pos)
print(f"Tracking error: {tracking_error:.3f}m")

# 控制平滑性
control_smoothness = np.abs(current_vel - last_vel) / dt
print(f"Control smoothness: {control_smoothness:.3f}")

# 实际 vs 期望速度
print(f"Desired: v={v:.2f}, w={w:.2f}")
print(f"Actual: v={robot_vel:.2f}, w={robot_ang_vel:.2f}")
```

### 3. 日志记录

```python
# 记录关键指标
log_data = {
    'step': step_count,
    'planning_time': planning_time,
    'mpc_time': mpc_time,
    'tracking_error': tracking_error,
    'critic_max': np.max(all_values),
    'critic_min': np.min(all_values),
    'success': success
}
with open('evaluation_log.json', 'a') as f:
    json.dump(log_data, f)
    f.write('\n')
```

---

## 🎯 总结

### 工程亮点

1. **异步架构**：规划-控制分离，保证实时性
2. **层次化控制**：Planning → MPC → Differential → Simulation
3. **坐标系管理**：清晰的 Camera → World 转换
4. **数据编码优化**：JPEG/PNG 压缩，减少传输量
5. **热启动加速**：MPC 求解时间减少 50%
6. **线程安全设计**：Lock 保护共享数据

### 可改进点

1. **批处理推理**：多客户端请求合并，提升吞吐量
2. **模型量化**：FP32 → FP16，加速推理
3. **特征缓存**：历史帧特征复用，减少计算
4. **减少去噪步数**：10步 → 5步，牺牲质量换速度
5. **GPU 流水线**：重叠 CPU-GPU 传输和计算

### 适用场景

- ✅ 室内导航（家居、仓库）
- ✅ 动态避障
- ✅ 实时性要求 <100ms
- ✅ GPU 可用（推理加速）
- ❌ 高速运动（>2m/s）
- ❌ 完全无 GPU 环境
