# NavDP 算法完整流程详解

## 📖 目录
1. [整体架构](#整体架构)
2. [核心算法原理](#核心算法原理)
3. [推理流程（端到端）](#推理流程端到端)
4. [网络架构详解](#网络架构详解)
5. [DDPM采样过程](#ddpm采样过程)
6. [Critic评分机制](#critic评分机制)
7. [多任务支持](#多任务支持)
8. [客户端-服务器架构](#客户端服务器架构)

---

## 🏗️ 整体架构

### 系统概览

```
┌─────────────────────────────────────────────────────────────┐
│                    NavDP 导航系统                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Client (Isaac Sim)          Server (Flask)                  │
│  ┌──────────────┐            ┌──────────────┐               │
│  │   仿真环境    │  HTTP      │  NavDP模型   │               │
│  │   ↓         │ ────────→  │   ↓         │               │
│  │  观测+目标   │  Request    │  推理       │               │
│  │   ↓         │            │   ↓         │               │
│  │  MPC控制    │ ←────────  │  轨迹+价值   │               │
│  │   ↓         │  Response   │             │               │
│  │  执行动作    │            │             │               │
│  └──────────────┘            └──────────────┘               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 文件 | 功能 |
|------|------|------|
| **Server** | `navdp_server.py` | Flask HTTP服务，模型推理接口 |
| **Agent** | `policy_agent.py` | 数据预处理，历史队列管理 |
| **Policy** | `policy_network.py` | DDPM+Transformer核心网络 |
| **Backbone** | `policy_backbone.py` | RGBD/目标编码器 |
| **Client** | `eval_*_wheeled.py` | Isaac Sim仿真，MPC控制 |

---

## 🧠 核心算法原理

### NavDP = Diffusion Policy + Transformer + Critic

```
输入: RGBD观测 + 目标
  ↓
[编码器]
  RGBD → memory_token (128, 384)
  Goal → goal_embed (1, 384)
  ↓
[DDPM 扩散策略]
  初始化: 纯高斯噪声 (16, 24, 3)
  循环10步: 逐步去噪
    t=9 → t=8 → ... → t=0
  每步: Transformer预测噪声并去噪
  ↓
输出: 16条候选轨迹 (16, 24, 3)
  ↓
[Critic 评分]
  评估每条轨迹的价值
  选择最优轨迹
  ↓
最终: 最优轨迹 (24, 3)
```

### 关键创新点

1. **Diffusion Policy**：用生成模型而非回归，多样性更强
2. **Critic Network**：从多条轨迹中选最优，鲁棒性更好
3. **Transformer**：处理多模态、时序、长程依赖
4. **多任务统一**：一个模型支持点目标/图像目标/无目标

---

## 🔄 推理流程（端到端）

### 完整数据流（以点目标导航为例）

```
═══════════════════════════════════════════════════════════════
Stage 0: 客户端发起请求
═══════════════════════════════════════════════════════════════

Isaac Sim 环境
  ├─ 获取观测: RGB (480,640,3), Depth (480,640,1)
  ├─ 计算目标: goal_x=2.0, goal_y=1.5 (相对坐标)
  └─ 发送 HTTP POST → /pointgoal_step

═══════════════════════════════════════════════════════════════
Stage 1: 服务器接收并预处理
═══════════════════════════════════════════════════════════════

navdp_server.py
  ├─ 解码图像: JPEG → numpy (480,640,3)
  ├─ 解码深度: PNG → numpy (480,640,1), 转换单位
  └─ 解析目标: JSON → goal_x, goal_y

    ↓ 调用 policy_agent.py

policy_agent.step_pointgoal()
  ├─ 图像预处理:
  │   (480,640,3) → resize → (224,224,3)
  │   归一化到 [0,1]
  │
  ├─ 深度预处理:
  │   (480,640,1) → resize → (224,224,1)
  │   归一化到 [0,1]
  │
  ├─ 历史队列管理:
  │   queue.append(current_rgb)
  │   if len(queue) < 8: 填充历史帧
  │   input_images = stack(queue)  # (8,224,224,3)
  │
  └─ 目标坐标:
      goal_point = [goal_x, goal_y, 0.0]  # (3,)

═══════════════════════════════════════════════════════════════
Stage 2: 网络推理 - 编码阶段
═══════════════════════════════════════════════════════════════

policy_network.predict_pointgoal_action()

【步骤 2.1】编码 RGBD
    ↓
policy_backbone.NavDP_RGBD_Backbone.forward()

    输入:
      images: (1, 8, 224, 224, 3)  # batch=1
      depths: (1, 224, 224, 1)

    处理 RGB:
      ├─ 8帧展平: (8, 224, 224, 3) → (8, 3, 224, 224)
      ├─ ImageNet归一化
      ├─ ViT-Small 提取特征:
      │   self.rgb_model.get_intermediate_layers()
      │   → (8, 256, 384)  # 每帧256个patch
      └─ 拼接: (1, 2048, 384)  # 8*256=2048

    处理 Depth:
      ├─ 单通道→3通道: (1,224,224,1) → (1,3,224,224)
      ├─ ViT-Small 提取特征:
      │   self.depth_model.get_intermediate_layers()
      │   → (1, 256, 384)
      └─ 输出: (1, 256, 384)

    拼接 RGB+Depth:
      former_token = concat([rgb_tokens, depth_tokens])
      → (1, 2048+256, 384) = (1, 2304, 384)
      注意: 实际代码中depth也是8帧，所以是4096

    添加位置编码:
      former_token += self.former_pe(former_token)

    Transformer 聚合:
      Query: self.former_query → (1, 128, 384)
      Key/Value: former_token → (1, 4096, 384)
      ↓
      memory_token = TransformerDecoder(Query, Key/Value)
      → (1, 128, 384)

    输出:
      rgbd_embed = self.project_layer(memory_token)
      → (1, 128, 384)

【步骤 2.2】编码目标
    ↓
    goal_point = [2.0, 1.5, 0.0]  # (1, 3)
    pointgoal_embed = self.point_encoder(goal_point)
    → (1, 384)
    pointgoal_embed.unsqueeze(1)
    → (1, 1, 384)

═══════════════════════════════════════════════════════════════
Stage 3: 网络推理 - DDPM 采样阶段
═══════════════════════════════════════════════════════════════

【步骤 3.1】复制16份用于并行采样

    rgbd_embed: (1, 128, 384) → repeat_interleave(16) 
                → (16, 128, 384)
    
    pointgoal_embed: (1, 1, 384) → repeat_interleave(16)
                     → (16, 1, 384)

【步骤 3.2】初始化噪声

    noisy_action = torch.randn(16, 24, 3)
    # 16条轨迹，每条24个点，每点(Δx, Δy, Δθ)
    # 纯高斯噪声 ~ N(0, 1)

【步骤 3.3】DDPM 去噪循环（10步）

    self.noise_scheduler.set_timesteps(10)
    timesteps = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    
    naction = noisy_action  # (16, 24, 3)
    
    for t in timesteps:  # t = 9, 8, ..., 0
        ┌────────────────────────────────────────┐
        │  DDPM 单步去噪                          │
        └────────────────────────────────────────┘
        
        ① 预测噪声
           noise_pred = self.predict_noise(
               naction,           # (16, 24, 3) 当前带噪样本
               t,                 # (1,) 当前时间步
               pointgoal_embed,   # (16, 1, 384) 目标
               rgbd_embed         # (16, 128, 384) 观测
           )
           → (16, 24, 3)
        
        ② 去噪一步
           naction = self.noise_scheduler.step(
               model_output=noise_pred,
               timestep=t,
               sample=naction
           ).prev_sample
           
           # DDPM 去噪公式:
           # x_{t-1} = √α_{t-1} * (x_t - √(1-α_t)*ε_θ) / √α_t + σ_t*z
    
    最终: naction (16, 24, 3) 是去噪后的轨迹增量

═══════════════════════════════════════════════════════════════
Stage 4: predict_noise() 详解（Transformer推理）
═══════════════════════════════════════════════════════════════

policy_network.predict_noise(naction, t, goal, rgbd)

【步骤 4.1】编码时间步（正弦编码）
    
    timestep = t  # 标量，如 t=5
    time_embeds = self.time_emb(timestep)
    # SinusoidalPosEmb: (1,) → (1, 384)
    # 公式: [sin(t/10000^0), cos(t/10000^0), sin(t/10000^(2/384)), ...]
    
    time_embeds = time_embeds.unsqueeze(1).tile(16, 1, 1)
    → (16, 1, 384)

【步骤 4.2】拼接条件token

    cond_tokens = concat([
        time_embeds,      # (16, 1, 384)   时间步
        pointgoal_embed,  # (16, 1, 384)   目标槽位1
        pointgoal_embed,  # (16, 1, 384)   目标槽位2（重复）
        pointgoal_embed,  # (16, 1, 384)   目标槽位3（重复）
        rgbd_embed        # (16, 128, 384) RGBD观测
    ], dim=1)
    → (16, 132, 384)  # 1+3+128=132

【步骤 4.3】添加条件位置编码（可学习）

    cond_embedding = cond_tokens + self.cond_pos_embed(cond_tokens)
    # 告诉模型每个token的"角色"：
    #   位置0: 时间步信息
    #   位置1-3: 3个目标槽位
    #   位置4-131: RGBD观测

【步骤 4.4】编码轨迹

    action_embeds = self.input_embed(naction)
    # Linear(3→384): (16, 24, 3) → (16, 24, 384)

【步骤 4.5】添加轨迹位置编码（可学习）

    input_embedding = action_embeds + self.out_pos_embed(action_embeds)
    # 告诉模型24个点的顺序

【步骤 4.6】Transformer Decoder（16层）

    output = self.decoder(
        tgt=input_embedding,     # Query: (16, 24, 384)
        memory=cond_embedding,   # Key/Value: (16, 132, 384)
        tgt_mask=self.tgt_mask   # 因果mask: (24, 24) 下三角
    )
    
    每层 TransformerDecoderLayer 做:
      ① Masked Self-Attention:
         24个轨迹点之间的自注意力
         因果mask防止看到未来的点
      
      ② Cross-Attention:
         轨迹点 attend to 条件 (time/goal/rgbd)
         学习"在当前环境+目标下，如何规划轨迹"
      
      ③ FFN:
         前馈网络，增强表达能力
    
    输出: (16, 24, 384)

【步骤 4.7】预测噪声

    output = self.layernorm(output)
    noise = self.action_head(output)
    # Linear(384→3): (16, 24, 384) → (16, 24, 3)
    
    返回: 预测的噪声 ε_θ(x_t, t, c)

═══════════════════════════════════════════════════════════════
Stage 5: Critic 评分阶段
═══════════════════════════════════════════════════════════════

policy_network.predict_critic(naction, rgbd_embed)

【目的】从16条候选轨迹中选出最优的

【步骤 5.1】创建空目标（Critic不看目标）

    nogoal_embed = torch.zeros_like(rgbd_embed[:, 0:1])
    → (16, 1, 384)
    
    设计理念: Critic应该基于观测判断轨迹的安全性
              而不依赖目标（避免走向目标但碰撞的轨迹）

【步骤 5.2】编码轨迹

    action_embeddings = self.input_embed(naction)
    action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)
    → (16, 24, 384)

【步骤 5.3】拼接条件（4个nogoal + RGBD）

    cond_tokens = concat([
        nogoal_embed,  # (16, 1, 384) 时间步位置（用0填充）
        nogoal_embed,  # (16, 1, 384) 目标槽位1（用0填充）
        nogoal_embed,  # (16, 1, 384) 目标槽位2（用0填充）
        nogoal_embed,  # (16, 1, 384) 目标槽位3（用0填充）
        rgbd_embed     # (16, 128, 384) RGBD观测（真实信息）
    ], dim=1)
    → (16, 132, 384)
    
    cond_embeddings = cond_tokens + self.cond_pos_embed(cond_tokens)

【步骤 5.4】Transformer Decoder（带memory mask）

    critic_output = self.decoder(
        tgt=action_embeddings,                      # Query
        memory=cond_embeddings,                     # Key/Value
        memory_mask=self.cond_critic_mask           # Mask掉前4个token
    )
    
    self.cond_critic_mask:
      前4列 = -inf  # 屏蔽目标信息
      后128列 = 0   # 允许attend to RGBD
    
    输出: (16, 24, 384)

【步骤 5.5】聚合并输出标量

    critic_output = self.layernorm(critic_output)
    critic_output = critic_output.mean(dim=1)  # 平均池化24个点
    → (16, 384)
    
    critic_values = self.critic_head(critic_output)[:, 0]
    # Linear(384→1): (16, 384) → (16, 1) → (16,)
    
    返回: 16个价值分数

【步骤 5.6】选择最优/最差轨迹

    critic_values: (16,)  # 例如: [0.8, -0.3, 1.2, ...]
    
    # 价值从大到小排序
    sorted_indices = (-critic_values).argsort()
    # 例如: [2, 0, 5, ..., 1]  (轨迹2最好，轨迹1最差)
    
    # 选前2名
    topk_indices = sorted_indices[0:2]
    positive_trajectory = all_trajectory[topk_indices]
    → (2, 24, 3)  # 最优的2条轨迹
    
    # 选后2名
    bottomk_indices = sorted_indices[-2:]
    negative_trajectory = all_trajectory[bottomk_indices]
    → (2, 24, 3)  # 最差的2条轨迹

═══════════════════════════════════════════════════════════════
Stage 6: 后处理阶段
═══════════════════════════════════════════════════════════════

【步骤 6.1】累积求和得到绝对轨迹

    naction: (16, 24, 3)  # 轨迹增量 Δx, Δy, Δθ
    
    all_trajectory = torch.cumsum(naction / 4.0, dim=1)
    # 除以4.0是训练时的归一化系数
    # cumsum: 累加增量得到绝对坐标
    
    例如:
      naction[0] = [[0.1, 0.0, 0.0],   # 第1步
                     [0.1, 0.0, 0.0],   # 第2步
                     [0.0, 0.1, 0.0],   # 第3步
                     ...]
      
      cumsum后:
      trajectory[0] = [[0.025, 0.000, 0.0],  # 0.1/4
                        [0.050, 0.000, 0.0],  # (0.1+0.1)/4
                        [0.050, 0.025, 0.0],  # (0.1+0.1)/4, 0.1/4
                        ...]

【步骤 6.2】过滤短轨迹

    trajectory_length = all_trajectory[:, -1, 0:2].norm(dim=-1)
    # 计算终点距离: (16,)
    
    # 如果轨迹太短(<0.5米)，保留角度但清零位移
    all_trajectory[trajectory_length < 0.5] *= [0, 0, 1.0]
    # 避免"原地不动"的无效轨迹

【步骤 6.3】整理输出

    返回:
      - all_trajectory: (16, 24, 3) 所有候选轨迹
      - critic_values: (16,) 所有价值分数
      - positive_trajectory: (2, 24, 3) 最优2条
      - negative_trajectory: (2, 24, 3) 最差2条

═══════════════════════════════════════════════════════════════
Stage 7: 轨迹可视化（可选）
═══════════════════════════════════════════════════════════════

policy_agent.project_trajectory()

【目的】将轨迹投影到图像上，用于调试可视化

    轨迹点: (x, y, θ) in camera frame
    
    for each point (x, y, θ):
      ① 转换到相机坐标: [x, y, z] (z是高度，如0.5米)
      ② 投影到像素: pixel = K @ [x, y, z]^T
         K是相机内参矩阵
      ③ 归一化: u = pixel[0]/pixel[2], v = pixel[1]/pixel[2]
      ④ 在图像上画点或箭头

═══════════════════════════════════════════════════════════════
Stage 8: 返回服务器响应
═══════════════════════════════════════════════════════════════

navdp_server.py

    结果编码:
      response = {
          'trajectory': positive_trajectory[0].tolist(),  # (24, 3)
          'all_trajectory': all_trajectory.tolist(),      # (16, 24, 3)
          'all_values': critic_values.tolist()            # (16,)
      }
    
    返回 JSON → HTTP Response

═══════════════════════════════════════════════════════════════
Stage 9: 客户端控制执行
═══════════════════════════════════════════════════════════════

eval_*_wheeled.py

【步骤 9.1】接收轨迹

    trajectory = response['trajectory']  # (24, 3)
    # 例如: [[0.025, 0.0, 0.0], [0.05, 0.0, 0.0], ...]

【步骤 9.2】MPC 轨迹跟踪

    utils_tasks/tracking_utils.py
    
    输入:
      - trajectory: (24, 3) 期望轨迹
      - current_state: [x, y, θ, v, ω] 当前状态
      - dt: 0.1秒 控制周期
    
    MPC 优化目标:
      min Σ ||state - trajectory||² + λ*||control||²
      
      约束:
        - v ∈ [-1.0, 1.0] m/s  线速度范围
        - ω ∈ [-π, π] rad/s   角速度范围
        - 运动学约束: ẋ = v*cos(θ), ẏ = v*sin(θ), θ̇ = ω
    
    CasADi 求解:
      最优控制 → [v*, ω*]
    
    输出:
      - linear_vel: 0.5 m/s
      - angular_vel: 0.2 rad/s

【步骤 9.3】差速控制

    wheeled_robots/controllers/differential_controller.py
    
    差速运动学:
      v_left  = linear_vel - angular_vel * wheel_base / 2
      v_right = linear_vel + angular_vel * wheel_base / 2
    
    输出:
      - left_wheel_vel: 4.8 rad/s
      - right_wheel_vel: 5.2 rad/s

【步骤 9.4】仿真执行

    Isaac Sim:
      设置关节速度 → 物理仿真一步 (0.01s)
      观测更新 → RGB, Depth, Pose
    
    循环: 每10步发送一次新的请求

═══════════════════════════════════════════════════════════════
完整循环
═══════════════════════════════════════════════════════════════

while not done:
    ① 获取观测 (RGB, Depth, Goal)
    ② 发送请求 → NavDP Server
    ③ 接收轨迹 → MPC 控制
    ④ 执行动作 → 仿真更新
    ⑤ 检查终止条件 (到达/碰撞/超时)
    ⑥ 回到 ①
```

---

## 🏛️ 网络架构详解

### 完整架构图

```
                        NavDP Policy Network
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  输入编码器层                                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                                                       │    │
│  │  ┌─────────────┐     ┌─────────────┐               │    │
│  │  │  RGB (8帧)  │     │ Depth (1帧) │               │    │
│  │  │ (8,224,224,3)│     │(224,224,1) │               │    │
│  │  └──────┬──────┘     └──────┬──────┘               │    │
│  │         │                     │                      │    │
│  │         └──────────┬──────────┘                      │    │
│  │                    ↓                                 │    │
│  │         ┌──────────────────────┐                    │    │
│  │         │  NavDP_RGBD_Backbone │                    │    │
│  │         │  - ViT-Small ×2      │                    │    │
│  │         │  - Transformer       │                    │    │
│  │         └──────────┬───────────┘                    │    │
│  │                    ↓                                 │    │
│  │         rgbd_embed (128, 384) ──────────┐           │    │
│  │                                          │           │    │
│  │  ┌─────────────┐                        │           │    │
│  │  │    Goal     │                        │           │    │
│  │  │ Point/Image │                        │           │    │
│  │  └──────┬──────┘                        │           │    │
│  │         │                                │           │    │
│  │         ↓                                │           │    │
│  │  goal_embed (1, 384) ───────────────────┤           │    │
│  │                                          │           │    │
│  └──────────────────────────────────────────┼───────────┘    │
│                                             │                │
│  DDPM 采样层                                 │                │
│  ┌──────────────────────────────────────────┼───────────┐    │
│  │                                          │           │    │
│  │  初始化噪声: (16, 24, 3)                 │           │    │
│  │         ↓                                │           │    │
│  │  for t in [9,8,...,0]:                  │           │    │
│  │    ┌──────────────────────────────┐     │           │    │
│  │    │   Noise Prediction Network   │←────┴───────────┤    │
│  │    │   ┌─────────────────────┐   │                 │    │
│  │    │   │ Time Embedding      │   │  SinusoidalEmb  │    │
│  │    │   │ (正弦编码)          │   │                 │    │
│  │    │   └─────────┬───────────┘   │                 │    │
│  │    │             ↓                │                 │    │
│  │    │   ┌─────────────────────┐   │                 │    │
│  │    │   │ Condition Tokens    │   │                 │    │
│  │    │   │ [time,goal×3,rgbd]  │   │                 │    │
│  │    │   │ (132, 384)          │   │                 │    │
│  │    │   └─────────┬───────────┘   │                 │    │
│  │    │             │                │                 │    │
│  │    │   ┌─────────┴───────────┐   │                 │    │
│  │    │   │ Action Tokens       │   │                 │    │
│  │    │   │ (24, 384)           │   │                 │    │
│  │    │   └─────────┬───────────┘   │                 │    │
│  │    │             ↓                │                 │    │
│  │    │   ┌─────────────────────┐   │                 │    │
│  │    │   │ Transformer Decoder │   │ 16层            │    │
│  │    │   │ - Self-Attention    │   │                 │    │
│  │    │   │ - Cross-Attention   │   │                 │    │
│  │    │   │ - FFN               │   │                 │    │
│  │    │   └─────────┬───────────┘   │                 │    │
│  │    │             ↓                │                 │    │
│  │    │   ┌─────────────────────┐   │                 │    │
│  │    │   │ Action Head         │   │ Linear(384→3)   │    │
│  │    │   └─────────┬───────────┘   │                 │    │
│  │    │             ↓                │                 │    │
│  │    │      noise_pred (16,24,3)   │                 │    │
│  │    └──────────────┬───────────────┘                 │    │
│  │                   ↓                                 │    │
│  │  DDPM去噪: x_{t-1} = Denoise(x_t, noise_pred)      │    │
│  │                                                     │    │
│  └─────────────────────┬───────────────────────────────┘    │
│                        ↓                                    │
│  Critic 评分层         trajectories (16, 24, 3)             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                                                       │    │
│  │  ┌─────────────────────────────────────┐            │    │
│  │  │   Critic Network (相同Transformer)   │            │    │
│  │  │   - 不使用目标信息 (nogoal=0)       │            │    │
│  │  │   - 只基于RGBD评估安全性            │            │    │
│  │  └─────────────────┬───────────────────┘            │    │
│  │                    ↓                                 │    │
│  │         values (16,) 价值分数                        │    │
│  │                    ↓                                 │    │
│  │         ┌──────────┴──────────┐                     │    │
│  │         │  选择最优轨迹        │                     │    │
│  │         └──────────┬──────────┘                     │    │
│  │                    ↓                                 │    │
│  └────────────────────┼─────────────────────────────────┘    │
│                       ↓                                      │
│  输出: best_trajectory (24, 3)                               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 关键参数统计

```
总参数量: ~50M

组件分解:
├─ ViT-Small (RGB):      22M  (冻结)
├─ ViT-Small (Depth):    22M  (可训练)
├─ Transformer Decoder:  ~5M  (16层)
├─ Embedding Layers:     ~500K
└─ Other:                ~500K
```

---

## 🌀 DDPM采样过程

### DDPM 原理

```
前向过程 (训练时):
  x_0 (真实轨迹) → x_1 → x_2 → ... → x_9 (纯噪声)
  逐步加噪: x_t = √α_t * x_0 + √(1-α_t) * ε

反向过程 (推理时):
  x_9 (纯噪声) → x_8 → x_7 → ... → x_0 (生成轨迹)
  逐步去噪: x_{t-1} = Denoise(x_t, ε_θ(x_t, t, c))
```

### NavDP 中的实现

```python
# 初始化
noisy_action = torch.randn(16, 24, 3)  # 纯噪声
timesteps = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

# 去噪循环
for t in timesteps:
    # 1. 网络预测噪声
    noise_pred = model.predict_noise(
        noisy_action, t, goal, rgbd
    )
    
    # 2. DDPM去噪公式
    # x_{t-1} = √α_{t-1} * (x_t - √(1-α_t)*ε_θ) / √α_t + σ_t*z
    noisy_action = scheduler.step(
        model_output=noise_pred,
        timestep=t,
        sample=noisy_action
    ).prev_sample

# 最终: noisy_action 是去噪后的轨迹
```

### 为什么用 DDPM？

| 对比 | 传统回归 | DDPM |
|------|---------|------|
| 输出 | 单一轨迹 | 多样化轨迹 |
| 鲁棒性 | 易过拟合 | 更鲁棒 |
| 多模态 | 难处理 | 自然支持 |
| 推理速度 | 快 | 慢（10步） |

**NavDP 策略**: DDPM生成16条 + Critic选最优 = 平衡多样性和性能

---

## 🎯 Critic评分机制

### 设计理念

```
Actor (DDPM):  生成多样化的候选轨迹
                ↓
Critic:        评估轨迹的"质量"
                ↓
Selection:     选择价值最高的轨迹
```

### Critic 的独特设计

```python
# ❌ 错误: Critic 使用目标信息
critic_values = critic(trajectory, goal, rgbd)
# → 可能选择"朝向目标但会碰撞"的危险轨迹

# ✅ 正确: Critic 不使用目标信息
critic_values = critic(trajectory, nogoal, rgbd)
# → 只基于观测评估"这条轨迹安全吗"
```

### 评分标准（隐式学习）

训练时学到的评分准则：
- ✅ 避障能力：不会撞墙、撞障碍物
- ✅ 可行性：符合机器人运动学约束
- ✅ 平滑性：轨迹不会突然转向
- ✅ 探索价值：（无目标任务）往有信息的方向走
- ❌ 不考虑：是否接近目标（Actor负责）

---

## 🎨 多任务支持

### 4种导航任务

| 任务类型 | 目标输入 | 目标编码 | 槽位分配 |
|---------|---------|---------|---------|
| **No Goal** | 无 | zeros(1,384) | [0, 0, 0] |
| **Point Goal** | (x,y,z) | Linear(3→384) | [goal, goal, goal] |
| **Image Goal** | RGB图像 | ViT(6ch) | [goal, goal, goal] |
| **Pixel Goal** | 像素mask | ViT(4ch) | [goal, goal, goal] |
| **混合任务** | Point+Image | 两个编码器 | [img, point, img] |

### 多目标槽位的妙用

```python
# 单一目标: 重复填充3个槽位
cond = [time, goal, goal, goal, rgbd]

# 混合目标: 不同槽位填不同目标
cond = [time, image_goal, point_goal, image_goal, rgbd]
```

**为什么要3个槽位？**
1. 灵活性：支持多目标组合
2. 表达力：多个槽位增强目标表示
3. 训练策略：可以随机drop某些槽位（增强泛化）

### 跨任务泛化

```
训练:
  点目标数据 + 图像目标数据 + 无目标数据
  → 统一的 Transformer

推理:
  ✅ 单一任务: 点目标 / 图像目标 / 无目标
  ✅ 混合任务: 点目标 + 图像目标
  ✅ Zero-shot: 训练时没见过的目标组合
```

---

## 🌐 客户端-服务器架构

### 架构设计

```
┌─────────────────────────────────────────────────────────┐
│  Client Side (Isaac Sim)                                │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Main Thread                      Planning Thread        │
│  ┌──────────────────┐            ┌──────────────────┐   │
│  │  仿真循环         │            │  异步规划         │   │
│  │  - 物理仿真 (100Hz)│           │  - HTTP请求 (10Hz)│   │
│  │  - 控制执行       │            │  - 轨迹缓存       │   │
│  │  - 观测更新       │            │                  │   │
│  └────────┬─────────┘            └────────┬─────────┘   │
│           │                               │              │
│           └───────────────┬───────────────┘              │
│                           │                              │
│                           ↓                              │
│                    共享缓存 (Queue)                       │
│                           │                              │
└───────────────────────────┼──────────────────────────────┘
                            │ HTTP
┌───────────────────────────┼──────────────────────────────┐
│  Server Side (Flask)      ↓                              │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Flask HTTP Server                                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │  /pointgoal_step                                 │   │
│  │    - 接收 RGB+Depth+Goal                         │   │
│  │    - 调用 NavDP_Agent                            │   │
│  │    - 返回 轨迹+价值                              │   │
│  │                                                   │   │
│  │  /imagegoal_step                                 │   │
│  │  /pixelgoal_step                                 │   │
│  │  /nogoal_step                                    │   │
│  │  /navigator_reset                                │   │
│  └──────────────────────────────────────────────────┘   │
│                           │                              │
│                           ↓                              │
│  NavDP_Agent (策略代理)                                  │
│  ┌──────────────────────────────────────────────────┐   │
│  │  - 图像预处理                                     │   │
│  │  - 历史队列管理                                   │   │
│  │  - 模型推理                                       │   │
│  │  - 轨迹后处理                                     │   │
│  └──────────────────────────────────────────────────┘   │
│                           │                              │
│                           ↓                              │
│  NavDP_Policy (核心网络)                                 │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### 时序控制

```
时间轴 (ms):
0────100───200───300───400───500───600───700───800───900

Client:
├─ Sim Step:  ●────●────●────●────●────●────●────●────●  (10ms/step, 100Hz)
├─ Control:   ●────●────●────●────●────●────●────●────●  (10ms, 读取轨迹)
└─ Request:   ●─────────────────────────●─────────────── (100ms/request, 10Hz)

Server:
└─ Inference: ──●────────────────────●──────────────────  (80ms, GPU推理)
```

### 为什么需要异步架构？

1. **推理慢** (80ms)：DDPM 10步去噪 + Transformer 16层
2. **控制快** (10ms)：仿真需要实时反馈
3. **解决方案**：Planning thread异步请求，Main thread用缓存的轨迹

---

## 📊 性能特点

### 优势

✅ **跨embodiment泛化**：一个模型适配多种机器人
✅ **多任务统一**：一个网络支持4种导航任务
✅ **鲁棒性强**：Critic筛选 + 多样化采样
✅ **端到端**：RGB-D → 轨迹，无需地图

### 局限

❌ **推理慢**：80ms/推理，DDPM去噪10步
❌ **内存大**：并行16条轨迹，GPU显存需求高
❌ **依赖深度**：纯视觉任务效果待验证

---

## 🎓 总结

### 算法核心公式

```
输入: 
  O_t = {RGB_{t-7:t}, Depth_t}  # 观测
  G = {goal}                     # 目标

编码:
  h_obs = Encoder_RGBD(O_t)      # (128, 384)
  h_goal = Encoder_Goal(G)       # (1, 384)

DDPM采样:
  x_T ~ N(0, I)                  # 初始噪声
  for t = T to 1:
    ε = Transformer(x_t, t, h_goal, h_obs)
    x_{t-1} = Denoise(x_t, ε, t)

Critic评分:
  v_i = Critic(x_0^i, h_obs)     # i=1,...,16
  x_best = argmax_i v_i

输出:
  τ = [p_1, ..., p_{24}]         # 最优轨迹
```

### 关键创新

1. **Diffusion Policy**：生成式模型处理多模态
2. **Transformer**：统一多模态、时序、长程依赖
3. **Critic Selection**：多样性 + 鲁棒性
4. **Goal Slots**：灵活的多任务机制

### 适用场景

- ✅ 室内导航（家居、仓库）
- ✅ 动态避障
- ✅ 目标驱动的探索
- ✅ Sim2Real迁移

---

## 📚 参考

- 论文: NavDP (待发布)
- Backbone: DepthAnythingV2
- Diffusion: DDPM (Ho et al., 2020)
- Architecture: Transformer Decoder
