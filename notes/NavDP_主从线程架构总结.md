# NavDP 主从线程架构与异步规划机制

## 一、整体架构概览

NavDP 采用**异步规划 + 同步控制**的架构，通过主从线程分离规划和执行，实现高频控制和稳定导航。

```
┌─────────────────────────────────────────────────────────────┐
│                      规划线程 (Planning Thread)               │
│                         频率: ~10 Hz                          │
│  ─────────────────────────────────────────────────────────   │
│  ① 读取观测（goal, RGB, depth）                               │
│  ② HTTP 调用 NavDP 服务器                                     │
│  ③ 获取轨迹（24 点，相机坐标系）                               │
│  ④ 坐标转换（相机 → 世界）                                    │
│  ⑤ 创建 MPC 对象                                              │
│  ⑥ 更新共享变量                                               │
│  ⑦ 休眠 100ms                                                │
└─────────────────────────────────────────────────────────────┘
                          ↓ 
            通过 planning_output（线程安全）
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  共享变量 (Shared State)                      │
│  ─────────────────────────────────────────────────────────   │
│  • planning_input:  goal, image, depth（主线程写，规划线程读） │
│  • planning_output: trajectory, mpc（规划线程写，主线程读）    │
│  • input_lock:  保护 planning_input                          │
│  • output_lock: 保护 planning_output                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
            通过 planning_output（线程安全）
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                      主线程 (Main Thread)                     │
│                      频率: 100 Hz (Isaac Sim)                │
│  ─────────────────────────────────────────────────────────   │
│  ① 获取最新观测（camera pose, RGB, depth）                    │
│  ② 更新 planning_input                                       │
│  ③ 读取 planning_output（trajectory, mpc）                   │
│  ④ MPC 求解（基于当前状态）                                    │
│  ⑤ 差速运动学转换                                             │
│  ⑥ 执行动作（env.step）                                       │
│  ⑦ 检查回合结束                                               │
└─────────────────────────────────────────────────────────────┘
```

## 二、主从线程详细工作流程

### 2.1 规划线程（Planning Thread）

**功能**：异步生成导航轨迹和 MPC 控制器

**频率**：~10 Hz（每 100ms 一次）

**代码位置**：`eval_pointgoal_wheeled.py` 第 261-319 行

```python
def planning_thread():
    while not shutdown_event.is_set():
        # ===== 步骤1：读取输入数据 =====
        with input_lock:  # 🔒 读取共享输入
            current_goal = planning_input.current_goal.copy()
            current_image = planning_input.current_image.copy()
            current_depth = planning_input.current_depth.copy()
            camera_pos = planning_input.camera_pos.copy()
            camera_rot = planning_input.camera_rot.copy()
        
        # ===== 步骤2：调用 NavDP 服务器 =====
        pred_data = pointgoal_step(
            point_goals=current_goal,
            rgb_images=current_image,
            depth_images=current_depth,
            port=args_cli.port
        )
        # pred_data['trajectory']: (batch, 24, 3)，相机坐标系
        
        # ===== 步骤3：坐标转换（相机 → 世界）=====
        trajectory_cam = pred_data['trajectory'][i, :, :3]  # (24, 3)
        trajectory_world = transform_trajectory_to_world(
            trajectory_cam, camera_pos, camera_rot
        )
        
        # ===== 步骤4：创建 MPC 控制器 =====
        mpc = MPC_Controller(
            global_planed_traj=trajectory_world,  # 新轨迹
            N=15,           # 预测步数
            desired_v=0.5,  # 期望速度
            v_max=0.5,      # 最大线速度
            w_max=0.5,      # 最大角速度
            ref_gap=3       # 参考点间隔
        )
        
        # ===== 步骤5：更新共享输出 =====
        with output_lock:  # 🔒 写入共享输出
            planning_output.trajectory = trajectory_world
            planning_output.mpc = mpc  # ← 更新全局 MPC 对象
            planning_output.ready = True
        
        # ===== 步骤6：控制频率 =====
        time.sleep(0.1)  # 100ms，约 10 Hz
```

**关键特性**：
- **异步执行**：不阻塞主线程，独立运行
- **低频更新**：10 Hz 足够导航任务，降低服务器负载
- **全量更新**：每次生成新的完整 MPC 对象

### 2.2 主线程（Main Thread）

**功能**：高频控制和仿真执行

**频率**：100 Hz（每 10ms 一次，由 Isaac Sim 决定）

**代码位置**：`eval_pointgoal_wheeled.py` 第 327-491 行

```python
while simulation_app.is_running():
    with torch.inference_mode():
        # ===== 步骤1：获取最新观测 =====
        goals = infos['observations']['goal_pose'].cpu().numpy()[:, 0:2]
        images = infos['observations']['rgb'].cpu().numpy()[:, :, :, 0:3]
        depths = infos['observations']['depth'].cpu().numpy()[:, :, :]
        camera_pos, camera_rot = get_camera_pose(robot)
        
        # ===== 步骤2：更新规划线程的输入缓存 =====
        with input_lock:  # 🔒 写入共享输入
            planning_input.current_goal = goals.copy()
            planning_input.current_image = images.copy()
            planning_input.current_depth = depths.copy()
            planning_input.camera_pos = camera_pos.copy()
            planning_input.camera_rot = camera_rot.copy()
        
        # ===== 步骤3：构建当前状态 =====
        x0 = np.zeros((num_envs, 5))
        x0[:, 0] = camera_pos[:, 0]  # x 位置
        x0[:, 1] = camera_pos[:, 1]  # y 位置
        x0[:, 2] = yaw_angles         # θ 航向角
        x0[:, 3] = linear_velocities  # v 线速度
        x0[:, 4] = angular_velocities # ω 角速度
        
        # ===== 步骤4：从规划线程获取最新轨迹和 MPC =====
        with output_lock:  # 🔒 读取共享输出
            current_trajectory = planning_output.trajectory
            current_mpc = planning_output.mpc  # ← 可能是旧的 MPC
        
        # ===== 步骤5：控制执行 =====
        if current_trajectory is not None and current_mpc is not None:
            # 5.1 MPC 求解
            opt_u_controls, opt_x_states = current_mpc.solve(x0[i, :3])
            # opt_u_controls: (15, 2) = [[v0, ω0], ..., [v14, ω14]]
            
            # 5.2 只取第2步（索引1）
            v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]
            
            # 5.3 差速运动学转换
            wheel_velocities = diff_controller.forward([v, w])
            
            # 5.4 执行动作
            action = torch.as_tensor(wheel_velocities, device="cuda:0")
            obs, rewards, dones, infos = env.step(action)
        else:
            # 轨迹未就绪，使用零动作（机器人静止）
            action = torch.zeros((num_envs, 2), device="cuda:0")
            obs, rewards, dones, infos = env.step(action)
        
        # ===== 步骤6：检查回合结束 =====
        for i in range(num_envs):
            if dones[i]:
                navigator_reset(env_id=i, port=args_cli.port)
                # 计算并保存评估指标
```

**关键特性**：
- **高频控制**：100 Hz 保证平滑运动
- **实时性强**：每步基于最新状态
- **容错设计**：MPC 未就绪时使用零动作

## 三、线程同步机制

### 3.1 共享数据结构

```python
# 规划输入（主线程写，规划线程读）
@dataclass
class PlanningInput:
    current_goal: np.ndarray = None      # (batch, 2) 目标位置
    current_image: np.ndarray = None     # (batch, H, W, 3) RGB图像
    current_depth: np.ndarray = None     # (batch, H, W) 深度图
    camera_pos: np.ndarray = None        # (batch, 3) 相机位置
    camera_rot: np.ndarray = None        # (batch, 4) 相机旋转（四元数）

# 规划输出（规划线程写，主线程读）
@dataclass
class PlanningOutput:
    trajectory: np.ndarray = None        # (24, 3) 世界坐标系轨迹
    mpc: MPC_Controller = None           # MPC 控制器对象
    ready: bool = False                  # 是否就绪
```

### 3.2 锁机制（Thread Locks）

```python
# 全局锁对象
input_lock = threading.Lock()   # 保护 planning_input
output_lock = threading.Lock()  # 保护 planning_output

# 使用示例：主线程写入观测
with input_lock:  # 🔒 获取锁 → 写入 → 释放锁
    planning_input.current_goal = goals.copy()
    planning_input.current_image = images.copy()
    # ... 写入其他字段

# 使用示例：规划线程读取观测
with input_lock:  # 🔒 获取锁 → 读取 → 释放锁
    current_goal = planning_input.current_goal.copy()
    current_image = planning_input.current_image.copy()
    # ... 读取其他字段

# 使用示例：规划线程写入结果
with output_lock:  # 🔒 获取锁 → 写入 → 释放锁
    planning_output.trajectory = trajectory_world
    planning_output.mpc = mpc

# 使用示例：主线程读取结果
with output_lock:  # 🔒 获取锁 → 读取 → 释放锁
    current_trajectory = planning_output.trajectory
    current_mpc = planning_output.mpc
```

**锁的作用**：
1. **互斥访问**：同一时刻只有一个线程可以访问共享变量
2. **数据一致性**：防止读写冲突（如读到一半被写入）
3. **线程安全**：避免竞态条件（race condition）

**为什么需要 `.copy()`**：
```python
# ❌ 错误：直接引用（两个线程共享同一块内存）
with input_lock:
    current_goal = planning_input.current_goal  # 浅拷贝

# ✓ 正确：深拷贝（规划线程有独立的数据副本）
with input_lock:
    current_goal = planning_input.current_goal.copy()  # 深拷贝
```

## 四、时序分析

### 4.1 完整时间线（单回合）

```
时间轴（单位：ms）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

规划线程（~10Hz，每 100ms 一次）：
0ms    │ ① 读取 input → ② HTTP 调用（耗时 20-50ms）
       │ ③ 坐标转换 → ④ 创建 MPC_A → ⑤ 写入 output
       │ ⑥ sleep(100ms) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
       │                                                   ↓
100ms  │ ① 读取 input → ② HTTP 调用                      │
       │ ③ 坐标转换 → ④ 创建 MPC_B → ⑤ 写入 output      │
       │ ⑥ sleep(100ms) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
       │                                                   ↓
200ms  │ ① 读取 input → ② HTTP 调用                      │
       │ ③ 坐标转换 → ④ 创建 MPC_C → ⑤ 写入 output      │

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

主线程（100Hz，每 10ms 一次）：
0ms    │ 读取 MPC_A → solve(x0₀) → v₀, ω₀ → step → 等待 10ms
10ms   │ 读取 MPC_A → solve(x0₁) → v₁, ω₁ → step → 等待 10ms
20ms   │ 读取 MPC_A → solve(x0₂) → v₂, ω₂ → step → 等待 10ms
...    │ （用的都是 MPC_A，因为规划线程还没更新）
90ms   │ 读取 MPC_A → solve(x0₉) → v₉, ω₉ → step → 等待 10ms
100ms  │ 读取 MPC_B → solve(x0₁₀) → v₁₀, ω₁₀ → step ✓ 切换！
110ms  │ 读取 MPC_B → solve(x0₁₁) → v₁₁, ω₁₁ → step
...    │ （用的都是 MPC_B）
190ms  │ 读取 MPC_B → solve(x0₁₉) → v₁₉, ω₁₉ → step
200ms  │ 读取 MPC_C → solve(x0₂₀) → v₂₀, ω₂₀ → step ✓ 切换！
```

### 4.2 关键发现

**发现 1：主线程会使用"旧" MPC**
- 规划线程每 100ms 更新一次 MPC
- 主线程每 10ms 读取一次 MPC
- 在两次规划之间（0-100ms），主线程使用同一个 MPC_A

**发现 2：这是合理的设计**
```python
为什么可以用"旧" MPC？

原因 1：MPC 的滚动优化特性
  - 虽然 MPC_A 是基于旧轨迹创建的
  - 但每次 solve(x0) 都基于当前真实位置
  - 会自动调整控制指令，不会"盲目执行"

原因 2：NavDP 轨迹本身平滑
  - 相邻两次规划的轨迹差异不大
  - 100ms 内机器人只移动 5cm（v=0.5m/s × 0.1s）
  - 轨迹变化是渐进的，不是突变的

原因 3：规划频率足够高
  - 10 Hz 对于导航任务已经够快
  - 更高频率会增加计算负担，收益递减
```

**发现 3：解耦设计的优势**
```python
如果规划线程卡住会怎样？

场景：网络延迟，规划线程 300ms 没更新

0ms:   规划 → MPC_A（正常）
       主线程使用 MPC_A ✓
       
100ms: 规划线程卡住（HTTP 超时）
       主线程继续使用 MPC_A ✓ 不会崩溃
       
200ms: 规划线程还在卡住
       主线程继续使用 MPC_A ✓ 保持运行
       
300ms: 规划线程恢复 → MPC_B
       主线程切换到 MPC_B ✓

结果：
✓ 主线程不会阻塞（始终有 MPC 可用）
✓ 机器人继续运动（虽然用旧轨迹）
✗ 可能偏离最优路径（但不会撞墙）
```

## 五、MPC 的滚动优化机制

### 5.1 为什么优化 15 步，只用 1 步？

```python
MPC 求解结果：
opt_u_controls = [
    [v0, ω0],   ← 索引 0：第 1 步
    [v1, ω1],   ← 索引 1：第 2 步 ✓ 只用这个！
    [v2, ω2],   ← 索引 2：第 3 步
    ...
    [v14, ω14]  ← 索引 14：第 15 步
]

看起来很浪费：
- 优化了 15 步（预测 1.5 秒）
- 只使用第 2 步（10ms）
- 只用了 1/15 = 6.7% 的结果

但这是 MPC 的精髓：滚动优化（Receding Horizon）
```

### 5.2 滚动优化动画

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
时刻 t0: 机器人在位置 A
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MPC 优化（预测 15 步，1.5 秒）:
  A ─→ ① ─→ ② ─→ ③ ─→ ④ ─→ ... ─→ ⑮
  |    ↑
  |    └─── 取第 2 步的控制：v=0.5, ω=0.1
  └── 当前位置

执行: env.step([v, ω])

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
时刻 t1: 机器人移动到位置 A'（接近 ①）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MPC 重新优化（又预测 15 步）:
  A' ─→ ①' ─→ ②' ─→ ③' ─→ ④' ─→ ... ─→ ⑮'
  |     ↑
  |     └─── 取第 2 步的控制：v=0.48, ω=0.12（自动纠偏！）
  └── 新的当前位置

执行: env.step([v, ω])

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
时刻 t2: 机器人移动到位置 A''
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MPC 又重新优化...
  A'' ─→ ①'' ─→ ②'' ─→ ...
   |     ↑
   └─────┘ 不断重复
```

### 5.3 为什么不用后面的 14 步？

**关键原因：每步都重新优化，纠正误差**

```python
# ❌ 错误做法：一次优化，用 15 步
opt_u_controls = mpc.solve(x0)
for i in range(15):
    v, w = opt_u_controls[i, 0], opt_u_controls[i, 1]
    env.step([v, w])
    # 问题：
    # 1. 第 2 步开始，实际位置已经偏离预测
    # 2. 后续控制基于错误假设
    # 3. 误差累积，可能撞墙

# ✓ 正确做法（MPC）：每步重新优化
for _ in range(15):
    x0 = get_current_state()        # 获取真实位置
    opt_u_controls = mpc.solve(x0)  # 重新优化
    v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]
    env.step([v, w])
    # 优点：
    # 1. 每步基于真实位置
    # 2. 自动纠偏
    # 3. 鲁棒性强
```

**数值例子：遇到侧向风（扰动）**

```python
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
方案 1：开环控制（用全部 15 步）

t0: 优化 → [0.5, 0] × 15 步
t1: 执行 opt_u[1] → 风吹偏了 y+0.01
t2: 执行 opt_u[2] → 继续偏 y+0.02
t3: 执行 opt_u[3] → 越来越偏 y+0.04
...
t15: 偏离轨迹 y+0.30 → 失败！✗

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
方案 2：MPC（每步重新优化）

t0: 优化 → [0.5, 0.0] × 15 步
t1: 执行 opt_u[1] → 风吹偏了 y+0.01
    重新优化 → 检测到偏差 → [0.5, -0.05]（轻微左转纠偏）
    
t2: 执行 → 纠偏中 y+0.008
    重新优化 → [0.5, -0.03]（继续纠偏）
    
t3: 执行 → 回到正轨 y+0.003
    重新优化 → [0.5, -0.01]（微调）
    
t4: 执行 → 完全回正 y+0.000
    重新优化 → [0.5, 0.0]（恢复直走）

成功跟踪轨迹！✓
```

### 5.4 为什么是第 2 步而不是第 1 步？

```python
v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]  # 为什么是索引 1？
```

**原因：计算延迟**

```python
时间线：
t0 = 0.00s  ← 当前时刻
  ↓ 获取当前状态 x0 = [x, y, θ, v, ω]
  ↓ MPC 求解（耗时 5-10ms）
t1 = 0.01s  ← MPC 求解完成
  ↓ env.step(action)
t2 = 0.02s  ← 动作执行完成，到达下一个状态

优化问题：
  opt_u[0]: 应该在 t0 → t1 执行（但已经过去了！）
  opt_u[1]: 应该在 t1 → t2 执行（现在执行，刚好！）✓
  opt_u[2]: 应该在 t2 → t3 执行（还太早）
```

### 5.5 MPC 的优势总结

```python
对比：

开环控制（用全部 15 步）：
  - 优化次数：1 次 / 1.5 秒 = 0.67 Hz
  - 计算量：低
  - 精度：差（误差累积）
  - 鲁棒性：差（无法应对扰动）
  
MPC（滚动优化）：
  - 优化次数：100 Hz
  - 计算量：高（但现代 CPU 可以承受，5-10ms）
  - 精度：高（每步纠偏）
  - 鲁棒性：强（自动适应扰动）
  
结论：计算开销完全值得！
```

## 六、仿真时间 vs. 真实时间

### 6.1 两种时间的区别

```python
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
仿真时间（Simulation Time）：
  - 定义：虚拟世界中的时间流逝
  - 由 env.step() 控制
  - 每次 step 固定前进 dt = 0.01s（10ms）
  - 与真实世界时间无关
  - 可以快于或慢于真实时间

例子：
  env.step(action)  → 仿真时间 +10ms
  实际耗时：可能是 5ms（快）或 50ms（慢）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
真实时间（Wall-clock Time）：
  - 定义：现实世界中的时间流逝
  - 由 time.sleep() 控制
  - 用于控制程序执行频率
  - 与仿真无关

例子：
  time.sleep(0.1)  → 真实时间等待 100ms
  不影响仿真时间
```

### 6.2 实际应用

```python
# 主线程：仿真时间驱动
while simulation_app.is_running():
    # 处理逻辑（耗时不定，假设 5-20ms）
    process_observations()
    mpc_solve()
    differential_control()
    
    # env.step() 前进固定的仿真时间
    env.step(action)  # 仿真时间 +10ms
    
    # 真实时间：5-20ms（变化的）
    # 仿真时间：10ms（固定的）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 规划线程：真实时间驱动
def planning_thread():
    while not shutdown_event.is_set():
        # HTTP 调用（耗时 20-50ms）
        pred_data = pointgoal_step(...)
        
        # 创建 MPC（耗时 ~5ms）
        mpc = MPC_Controller(...)
        
        # 固定频率：真实时间 100ms
        time.sleep(0.1)  # 真实时间等待
        
    # 仿真时间：不受影响
    # 规划线程只关心真实时间
```

### 6.3 为什么规划线程是 10 Hz？

```python
原因 1：真实时间控制
  time.sleep(0.1)  # 100ms 真实时间
  保证规划频率稳定在 ~10 Hz

原因 2：匹配 HTTP 延迟
  NavDP 推理耗时：20-50ms
  + 数据传输：5-10ms
  + MPC 创建：~5ms
  总计：30-65ms
  留出 35-70ms 余量 → 设为 100ms 合理

原因 3：导航任务特性
  机器人速度：0.5 m/s
  100ms 移动距离：5cm
  对于导航任务，5cm 的规划延迟可接受

原因 4：降低服务器负载
  10 Hz 远低于主线程的 100 Hz
  避免频繁调用 NavDP 服务器
```

### 6.4 如果规划太慢会怎样？

```python
场景：HTTP 调用耗时 200ms（超过 sleep 时间）

规划线程实际频率：
  200ms (HTTP + 处理) + 100ms (sleep) = 300ms
  实际频率：1000ms / 300ms ≈ 3.3 Hz（而不是 10 Hz）

影响：
✓ 不会崩溃（主线程继续用旧 MPC）
✗ 轨迹更新变慢（可能影响避障）
✗ 机器人反应变慢

解决方案：
1. 优化服务器性能（更快的 GPU）
2. 减少 sleep 时间（但不能为负）
3. 降低规划质量（减少 DDPM 采样步数）
```

## 七、分层导航架构

### 7.1 NavDP 在导航系统中的位置

```
┌─────────────────────────────────────────────────────────────┐
│              NavDP Neural Network（学习式局部规划）            │
│  ─────────────────────────────────────────────────────────   │
│  输入：RGB-D 图像、目标位置                                    │
│  输出：轨迹点序列（24 点，2.4m）                               │
│  频率：~10 Hz                                                │
│  特点：端到端感知 + 路径规划                                   │
└─────────────────────────────────────────────────────────────┘
                          ↓ waypoints
┌─────────────────────────────────────────────────────────────┐
│              MPC（轨迹跟踪 + 局部优化）                        │
│  ─────────────────────────────────────────────────────────   │
│  输入：目标轨迹、当前状态                                      │
│  输出：控制指令（v, ω）                                       │
│  频率：100 Hz                                                │
│  特点：滚动优化、动力学约束                                    │
└─────────────────────────────────────────────────────────────┘
                          ↓ (v, ω)
┌─────────────────────────────────────────────────────────────┐
│              Differential Drive（底层控制）                   │
│  ─────────────────────────────────────────────────────────   │
│  输入：线速度 v、角速度 ω                                     │
│  输出：左右轮速（v_left, v_right）                            │
│  频率：100 Hz                                                │
│  特点：运动学转换、经典控制理论                                │
└─────────────────────────────────────────────────────────────┘
                          ↓ wheel velocities
                      【机器人执行】
```

### 7.2 为什么采用混合设计？

```python
NavDP 的设计哲学：

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
上层用学习（NavDP Network）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ 处理复杂感知（RGB-D → 路径）
  ✓ 学习人类导航策略
  ✓ 端到端优化（感知 + 规划）
  ✓ 泛化到新环境
  ✗ 难以保证约束（如速度限制）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
下层用传统（MPC + Differential Drive）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ 数学保证（约束满足）
  ✓ 实时性好（优化算法快）
  ✓ 可解释性（轨迹可视化）
  ✓ 安全性（速度限制、碰撞检测）
  ✓ 复用成熟技术（不重复造轮子）
  ✗ 难以处理高维感知

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
结合两者优点：学习感知 + 传统控制
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  这是当前机器人学习的主流范式！
```

### 7.3 不同层次的数学性质

```python
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
层次              数学性质              适合方法
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Path Planning     离散、组合优化        学习方法
(NavDP)           高维感知             (处理复杂感知)
                  环境理解、避障        

Trajectory        连续、凸优化          传统优化
Tracking (MPC)    低维状态             (有理论保证)
                  动力学约束            

Low-level         运动学/动力学         经典控制
Control           转换                 (已经非常成熟)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 八、关键设计点总结

### 8.1 异步规划的优势

```python
✓ 解耦设计
  - 规划和控制独立，互不阻塞
  - 规划慢不影响控制频率
  - 容错性强（规划失败时用旧轨迹）

✓ 高效资源利用
  - CPU：主线程做 MPC（轻量）
  - GPU：规划线程做深度学习（重量）
  - 网络：异步 HTTP 不阻塞

✓ 灵活的频率控制
  - 主线程：100 Hz（高频控制）
  - 规划线程：~10 Hz（低频规划）
  - 各自优化，互不干扰
```

### 8.2 MPC 的关键作用

```python
✓ 轨迹跟踪
  - 将粗糙的路径点转换为平滑轨迹
  - 考虑机器人动力学约束
  - 优化速度和加速度

✓ 局部优化
  - 基于当前状态重新规划
  - 自动纠偏（应对模型误差和扰动）
  - 实时性强（5-10ms 求解）

✓ 安全保证
  - 显式约束（v_max, ω_max）
  - 碰撞检测（虽然 NavDP 已经避障）
  - 可视化（调试和验证）
```

### 8.3 线程安全的重要性

```python
✓ 数据一致性
  - 锁保证原子操作
  - 防止读写冲突
  - 避免竞态条件

✓ 深拷贝的必要性
  - 每个线程有独立数据副本
  - 避免共享内存导致的问题
  - 虽然有性能开销，但保证正确性

✓ 最小锁粒度
  - 只在必要时加锁
  - 锁内操作尽量简单（读/写）
  - 复杂计算在锁外进行
```

### 8.4 零动作的容错设计

```python
# 主线程中的容错逻辑
if current_trajectory is not None and current_mpc is not None:
    # 正常执行 MPC 控制
    v, w = mpc.solve(x0)
    action = differential_control(v, w)
else:
    # 轨迹未就绪，使用零动作（机器人静止）
    action = torch.zeros((num_envs, 2), device="cuda:0")
    print("轨迹未就绪，使用零动作")

优点：
✓ 不会崩溃（即使规划线程失败）
✓ 安全（静止总比乱动好）
✗ 影响性能（机器人会停顿）

改进方案：
1. 保留上一次的 MPC（而不是 None）
2. 使用简单的避障策略（如远离障碍物）
3. 记录警告日志（用于调试）
```

### 8.5 性能优化技巧

```python
1. Hot-starting MPC
   - 用上一次的解作为初始值
   - 加速收敛（3-5 次迭代 vs. 10-20 次）
   - 代码：baselines/navdp/tracking_utils.py

2. 深拷贝优化
   - 只拷贝必要的数据
   - 使用 NumPy 的 .copy()（快）
   - 避免 Python 的 copy.deepcopy()（慢）

3. 减少锁争用
   - 锁内只做简单读写
   - 复杂计算在锁外
   - 最小化临界区大小

4. 批处理
   - 多个环境同时处理（batch inference）
   - 向量化操作（NumPy/PyTorch）
   - 减少 HTTP 请求次数
```

## 九、调试和可视化

### 9.1 轨迹可视化

```python
# eval_pointgoal_wheeled.py 中的可视化代码
if current_trajectory is not None:
    # 在 Isaac Sim 中绘制轨迹点
    for point in current_trajectory:
        draw_sphere(position=point, radius=0.02, color=[0, 1, 0])
    
    # 绘制 MPC 预测的未来状态
    for state in opt_x_states:
        draw_sphere(position=state[:2], radius=0.015, color=[1, 0, 0])

好处：
✓ 实时查看规划结果
✓ 调试轨迹异常
✓ 验证坐标转换
✓ 演示效果
```

### 9.2 性能监控

```python
# 添加计时器
import time

# 规划线程
start = time.time()
pred_data = pointgoal_step(...)
http_time = time.time() - start
print(f"HTTP 耗时: {http_time*1000:.1f}ms")

# 主线程
start = time.time()
opt_u_controls, opt_x_states = mpc.solve(x0)
mpc_time = time.time() - start
print(f"MPC 耗时: {mpc_time*1000:.1f}ms")

# 监控指标
- HTTP 调用：20-50ms（正常），>100ms（异常）
- MPC 求解：5-10ms（正常），>20ms（异常）
- 主循环：8-12ms（正常），>15ms（异常）
```

### 9.3 常见问题排查

```python
问题 1：机器人不动（零动作）
原因：
  - 规划线程未启动
  - HTTP 连接失败
  - MPC 未初始化
排查：
  - 检查 planning_output.ready
  - 查看控制台错误日志
  - 验证服务器是否运行

问题 2：机器人运动不平滑
原因：
  - MPC 预测步数太少（N < 10）
  - 参考轨迹间隔太大（ref_gap > 5）
  - 权重参数不合理
排查：
  - 可视化 MPC 预测轨迹
  - 调整 MPC 参数（N, ref_gap, Q, R）
  - 检查速度曲线

问题 3：规划频率过低（< 5 Hz）
原因：
  - HTTP 调用太慢（网络延迟）
  - 服务器性能不足
  - 坐标转换耗时
排查：
  - 添加计时器测量各步骤耗时
  - 优化服务器（更快 GPU）
  - 减少 DDPM 采样步数
```

## 十、总结

### 10.1 核心设计原则

```python
1. 分层设计
   学习感知 + 传统控制 = 最佳实践

2. 异步解耦
   规划慢（10 Hz），控制快（100 Hz）

3. 滚动优化
   预测未来（1.5s），只用当前（10ms）

4. 线程安全
   锁 + 深拷贝 = 数据一致性

5. 容错设计
   零动作 + 旧 MPC = 不会崩溃
```

### 10.2 数据流总览

```
传感器 → 观测（RGB-D, Goal）
           ↓
        主线程（更新 planning_input）
           ↓ 共享变量（input_lock）
           ↓
     规划线程（读取 planning_input）
           ↓
    NavDP 服务器（神经网络推理）
           ↓
        轨迹点（24, 3）
           ↓
     坐标转换 + 创建 MPC
           ↓
     规划线程（更新 planning_output）
           ↓ 共享变量（output_lock）
           ↓
     主线程（读取 planning_output）
           ↓
    MPC 求解（基于当前状态）
           ↓
        控制指令（v, ω）
           ↓
    差速运动学转换
           ↓
      轮速（v_left, v_right）
           ↓
     env.step() → 机器人运动
```

### 10.3 关键技术点回顾

| 技术点 | 作用 | 频率 | 代码位置 |
|--------|------|------|----------|
| **规划线程** | 异步生成导航轨迹 | ~10 Hz | `eval_pointgoal_wheeled.py:261-319` |
| **主线程** | 高频控制和仿真 | 100 Hz | `eval_pointgoal_wheeled.py:327-491` |
| **NavDP Network** | 感知 + 路径规划 | ~10 Hz | `navdp_server.py` |
| **MPC** | 轨迹跟踪 + 优化 | 100 Hz | `tracking_utils.py` |
| **Differential Drive** | 运动学转换 | 100 Hz | `differential_controller.py` |
| **Thread Locks** | 线程安全 | N/A | `input_lock`, `output_lock` |

### 10.4 未来可能的改进

```python
1. 动态调频
   根据场景复杂度调整规划频率
   简单场景：5 Hz，复杂场景：15 Hz

2. 预测性规划
   预测机器人未来位置，提前规划
   减少延迟影响

3. 多层 MPC
   全局 MPC（长期规划）+ 局部 MPC（短期跟踪）
   更好的全局最优性

4. 自适应参数
   根据运动状态自动调整 MPC 参数
   提高鲁棒性

5. 异常处理
   规划失败时的备用策略
   而不是简单的零动作
```

---

**文档版本**: v1.0  
**最后更新**: 2026-01-24  
**作者**: NavDP 项目学习总结
