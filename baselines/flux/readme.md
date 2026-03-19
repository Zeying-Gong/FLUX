# cfm_rf_rl — GRPO RL Fine-tuning for NavDP

在 `cfm_rf`（Rectified Flow）预训练权重的基础上，
用 **GRPO**（Group Relative Policy Optimization）做在线强化学习微调，
专门针对动态行人跟随任务（`DynPointGoal`）。

---

## 文件结构

```
baselines/cfm_rf_rl/
├── policy_network.py   # CFM_RL_Policy：继承 cfm_rf，添加可微分前向 + 冻结策略
├── policy_agent.py     # GRPO_Agent：训练逻辑、rollout buffer、checkpoint 管理
├── server.py           # 推理服务器（加载 RL 微调后的权重，接口与 cfm_rf_server 一致）
├── train_grpo.py       # 主训练脚本（替代 eval_dynpointgoal_wheeled.py）
└── README.md
```

---

## 训练

```bash
# 从项目根目录运行
python baselines/cfm_rf_rl/train_grpo.py \
    --checkpoint baselines/cfm_rf/checkpoints/checkpoint-11710navdp.ckpt \
    --scene_dir  assets/dyn_scenes/isaacsim_scene \
    --scene_index 0 \
    --num_episodes 2000 \
    --save_dir  baselines/cfm_rf_rl/checkpoints_rl \
    --gpu_id 0 \
    --train_gpu_id 0 \
    --lr 3e-5 \
    --update_interval 32 \
    --save_interval 100
```

训练过程中会在 `--save_dir` 下生成：
- `rl_checkpoint_ep00100.ckpt`, `rl_checkpoint_ep00200.ckpt`, ...（每 100 个 episode）
- `rl_final.ckpt`（训练完成后）
- `train_log.json`（每次 GRPO 更新的 loss/reward 记录）
- `metrics/dynpointgoal_<scene>/metric.csv`（evaluation metrics）

---

## 推理（测试 RL 微调后的模型）

**方式一：用推理服务器（推荐，兼容原有 eval 脚本）**

```bash
# 启动 RL 权重的服务器（默认端口 8892）
python baselines/cfm_rf_rl/server.py \
    --checkpoint baselines/cfm_rf_rl/checkpoints_rl/rl_final.ckpt \
    --port 8892 \
    --device cuda:0

# 然后跑原来的 eval 脚本，只需改 --port
python eval_dynpointgoal_wheeled.py \
    --scene_dir assets/dyn_scenes/isaacsim_scene \
    --port 8892 \
    ...
```

**方式二：直接加载（进程内）**

```python
from baselines.cfm_rf_rl.policy_agent import GRPO_Agent

agent = GRPO_Agent(
    checkpoint_path="baselines/cfm_rf_rl/checkpoints_rl/rl_final.ckpt",
    image_intrinsic=camera_intrinsic,
    device="cuda:0",
)
agent.reset(batch_size=1, stop_threshold=-3.0)

# 每步
execute_traj, all_traj, all_values = agent.step_pointgoal(goal, image, depth)
```

---

## GRPO 设计说明

### 为什么用 GRPO 而不是标准 PPO

CFM/RF 策略是生成式模型，没有标准的 `log π(a|s)`，无法直接计算重要性采样比。
GRPO 的优势：
- 每步天然采样 16 条轨迹形成一个 **group**，无需额外 rollout
- 组内相对优势估计，不需要 baseline/value 网络
- 等价于用 RL 信号直接调整 **critic 的偏好分布**

### 冻结策略

| 模块 | 状态 | 原因 |
|------|------|------|
| rgb_model（ViT-S）| ❄️ 冻结 | 预训练感知特征，RL 阶段不需要改变 |
| depth_model（ViT-S）| ❄️ 冻结 | 同上 |
| imagegoal/pixelgoal encoder | ❄️ 冻结 | 任务不涉及 |
| decoder 前12层（共16层）| ❄️ 冻结 | 低层语义稳定 |
| **decoder 后4层** | ✅ 训练 | 高层推理，对任务奖励敏感 |
| **action_head** | ✅ 训练 | 轨迹输出 |
| **critic_head** | ✅ 训练 | GRPO 主要优化对象 |
| **point_encoder** | ✅ 训练 | 目标编码 |
| **pos_embed** | ✅ 训练 | 位置感知 |

可训练参数约占总参数的 **8-12%**，训练稳定高效。

### 奖励函数

```
r_t = r_arrive + r_approach + r_social + r_stuck

r_arrive:   +10.0  到达行人 1m 内
r_approach: +Δd×2  每步靠近行人（dense reward）
r_social:   惩罚   <0.45m: -5.0 / 0.45~1.2m: 线性惩罚
r_stuck:    -3.0   卡住惩罚
```

### GRPO Loss

```
L = -E_steps[ A_i · log_softmax(critic_scores)[best_i] ]
  + entropy_reg（熵正则，防止 critic 过早收敛）

A_i = (G_i - mean(G)) / (std(G) + ε)  # 标准化的折扣回报
```

---

## 超参建议

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr` | 3e-5 | 建议 1e-5 ~ 5e-5，太大容易破坏预训练特征 |
| `--update_interval` | 32 | 每 32 步更新一次，平衡样本效率和稳定性 |
| `--sample_num` | 16 | 保持与预训练一致 |
| `--unfreeze_last_n` | 4 | 4层约占 8% 参数，可改为 6（更多自由度）|
| `--save_interval` | 100 | 每 100 episode 保存，可适当减小 |