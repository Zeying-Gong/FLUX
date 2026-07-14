# go2_zed 数据采集说明文档

## 一、脚本功能概述

### `go2_zed.sh`
在 4 张 GPU 上**并行**启动 Docker 容器，每张 GPU 处理 247 个场景（GPU3 处理剩余到 987），容器内只采集 **go2 机器人 + ZED 相机** 组合的数据。

```bash
# 容器配置
docker run -d --name flux_go2_zed_$GPU --rm \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  --entrypoint bash --runtime=nvidia --gpus device=$GPU --network=host \
  -v /mnt/nvme1/zeyingg/FLUX:/workspace/FLUX \
  -v /mnt/ssd1/zeyingg/SAGE-3D_Official:/workspace/SAGE-3D_Official \
  -v /home/zeyingg/run_single_gpu_docker.sh:/workspace/run_single_gpu_docker.sh \
  -w /workspace quay.io/zeyinggong/flux:v2_deploy \
  -c "bash /workspace/run_single_gpu_docker.sh $GPU $S $E go2 zed"
```

### `run_single_gpu_docker.sh`（容器内执行的入口）

**v2 版本新增：场景级超时自动跳过**
- 每场景总超时 **4 小时**，超时后**自动跳过**该场景，采集已完成的 batch 数据
- 无需手动重启容器
- 原版：一个"挂死"场景可能浪费 15 小时
- 新版：最多浪费 4 小时就自动跳过

**调用 Python 脚本:**
```
/workspace/FLUX/sage_utils/tracking_episode_collection.py
```

**关键参数:**
| 参数 | 值 | 说明 |
|------|-----|------|
| `--max_steps` | 100 | 每个 episode 最大步数 |
| `--character_speed` | 0.4 | 目标人物移动速度 |
| `--max_consecutive_lost` | 5 | 最大连续丢失帧数 |
| `--early_abort_tracking_rate` | 0.3 | 跟踪率低于此值提前终止 |
| `--min_tracking_rate` | 0.3 | 最低成功跟踪率要求 |
| `--resume` | - | 断点续采 |
| `--headless` | - | 无头渲染 |
| `--save_images` | - | 保存所有图像数据 |

---

## 二、完整目录结构

```
logs_formal_v2/
│
├── go2_zed.sh                              # 启动脚本 (4卡并行 go2+ZED)
├── go2_zed copy.sh                         # 同上（含其他组合注释）
│
├── go2_zed/                                # ★ go2机器人 + ZED相机 采集数据主目录
│   │
│   ├── 0001_839920/                        # 场景ID（6位编号_6位数字，共约143个场景）
│   │   │
│   │   ├── episode_0000/                   # 第0个episode（共100个: 0000~0099）
│   │   │   ├── camera_info.json            # 相机内外参（ZED: 1280×720, pinhole, ROS系）
│   │   │   ├── task_description_ep0000.txt # 任务描述文本
│   │   │   ├── frames/
│   │   │   │   ├── frame_data.npz          # ★ 核心数据（NumPy压缩包，含全部step数据）
│   │   │   │   └── preview.json            # 摘要预览（tracking率、首末帧信息等）
│   │   │   ├── rgb/                        # 彩色图 1280×720 PNG (00000.png ~ 00099.png)
│   │   │   ├── depth_mm/                   # 深度图（毫米单位PNG）
│   │   │   ├── depth_viz/                  # 深度伪彩色可视化
│   │   │   ├── debug_occ/                  # 调试用占据栅格可视化
│   │   │   └── target_crops/               # 目标检测框裁剪图
│   │   │
│   │   ├── episode_0001/                   # 同上结构
│   │   └── ...                             # episode_0002 ~ episode_0099
│   │
│   ├── 0002_839955/                        # 场景0002（同上结构）
│   ├── 0003_839978/
│   ├── ...
│   └── metrics.csv                         # 所有episode汇总指标（6018行）
│
├── go2_realsense_d435i/                    # go2 + RealSense D435i（结构同上，分辨率640×480）
├── go2_realsense_d435i_archives/           # 上述数据的tar.gz压缩包
│
├── collect_go2_zed_20260701_074624_gpu0.log   # GPU运行日志
├── collect_go2_zed_20260701_074624_gpu1.log
├── collect_go2_zed_20260701_074624_gpu2.log
├── collect_go2_zed_20260701_074625_gpu3.log
│   ...                                         # 不同时间戳的多批次日志
│
├── collect_gpu0.log                           # 综合采集日志（所有robot+camera组合）
├── collect_gpu1.log
├── collect_gpu2.log
├── collect_gpu3.log
├── collect_gpu0_v2.log                        # V2版本日志
├── collect_gpu1_v2.log
├── collect_gpu2_v2.log
├── collect_gpu3_v2.log
│
└── monitor_restart_gpu0.log                   # 监控/自动重启守护进程日志
    monitor_restart_gpu1.log
    monitor_restart_gpu2.log
    monitor_restart_gpu3.log
```

---

## 三、单 episode 文件详细说明

### 3.1 `camera_info.json` — 相机标定信息
```json
{
  "camera": {
    "model": "pinhole",
    "axes": "ros",
    "width": 1280,
    "height": 720,
    "intrinsics": {
      "fx": 521.85,
      "fy": 521.85,
      "cx": 640.0,
      "cy": 360.0,
      "k": [[521.85, 0.0, 640.0], [0.0, 521.85, 360.0], [0.0, 0.0, 1.0]]
    },
    "extrinsics_robot_to_camera": {
      "translation": [0.0, 0.0, 0.3],     // 相机安装在机器人上方 30cm
      "camera_link": "base"
    },
    "clipping_range": [0.01, 100.0],
    "focal_length_mm": 2.12,
    "horizontal_aperture_mm": 5.2
  }
}
```

### 3.2 `frame_data.npz` — 核心时序数据

| 字段 | 维度 | 类型 | 说明 |
|------|------|------|------|
| **step** | (N,) | int64 | 时间步索引 0~99 |
| **timestamp** | (N,) | float64 | 仿真时间戳 |
| **robot_pos_x** | (N,) | float64 | 机器人世界坐标 X (m) |
| **robot_pos_y** | (N,) | float64 | 机器人世界坐标 Y (m) |
| **robot_pos_z** | (N,) | float64 | 机器人世界坐标 Z (m) |
| **robot_yaw** | (N,) | float64 | 机器人偏航角 (rad) |
| **target_pos_x** | (N,) | float64 | 目标人物世界坐标 X (m) |
| **target_pos_y** | (N,) | float64 | 目标人物世界坐标 Y (m) |
| **target_pos_z** | (N,) | float64 | 目标人物世界坐标 Z (m) |
| **target_rel_x** | (N,) | float64 | 目标在机器人坐标系下 X（前向） |
| **target_rel_y** | (N,) | float64 | 目标在机器人坐标系下 Y（左向） |
| **target_rel_z** | (N,) | float64 | 目标在机器人坐标系下 Z（向上） |
| **target_dist** | (N,) | float64 | 机器人到目标欧氏距离 (m) |
| **target_bearing_deg** | (N,) | float64 | 目标相对于机器人朝向的角度 (°) |
| **target_elevation_deg** | (N,) | float64 | 目标相对于机器人的俯仰角 (°) |
| **target_uv_u** | (N,) | float64 | 目标中心在图像上的 u 像素坐标（不可见为 -1） |
| **target_uv_v** | (N,) | float64 | 目标中心在图像上的 v 像素坐标（不可见为 -1） |
| **target_bbox_x1** | (N,) | float64 | 目标检测框左上角 x（不可见为 -1） |
| **target_bbox_y1** | (N,) | float64 | 目标检测框左上角 y |
| **target_bbox_x2** | (N,) | float64 | 目标检测框右下角 x |
| **target_bbox_y2** | (N,) | float64 | 目标检测框右下角 y |
| **target_visible** | (N,) | int64 | 目标是否在视野内 (0/1) |
| **action_linear** | (N,) | float64 | 控制指令：线速度 (m/s) |
| **action_angular** | (N,) | float64 | 控制指令：角速度 (rad/s) |
| **mode** | (N,) | str | 行为模式: "track" / "search" / "recover" |
| **dist_to_target** | (N,) | float64 | 控制器内部距离估计 |
| **heading_error_deg** | (N,) | float64 | 航向误差 (°) |
| **contact_force** | (N,) | float64 | 碰撞接触力 |
| **ped_min_dist** | (N,) | float64 | 与最近行人的最小距离 |
| | | | |
| **observation.state** | (N, 9) | float64 | 拼接状态: [robot_x, y, z, yaw, target_rel_x, y, z, dist, bearing] |
| **action** | (N, 2) | float64 | 拼接动作: [linear_vel, angular_vel] |
| **goal_uv** | (2,) | float64 | 首帧目标UV（one-shot 条件信号） |
| **goal_bbox** | (4,) | float64 | 首帧目标检测框 |
| **goal_rel_xyz** | (3,) | float64 | 首帧目标相对位置 |
| **robot_trajectory** | (N, 3) | float64 | 机器人完整轨迹 [x, y, z] |
| **target_trajectory** | (N, 3) | float64 | 目标完整轨迹 [x, y, z] |
| **task_description** | (bytes,) | bytes | 任务描述 UTF-8 编码 |

> N = 100 steps（时间步数）

### 3.3 `preview.json` — 快速预览
```json
{
  "episode_id": 15,
  "num_steps": 100,
  "task_description": "Track a person wearing silver hat, olive top, gray pants",
  "tracking_rate": 0.86,
  "avg_heading_error_deg": 36.56,
  "initial_dist_m": 1.13,
  "final_dist_m": 2.33,
  "target_uv_first": [642.1, 707.2],
  "target_bbox_first": [385, 0, 900, 719],
  "robot_start_pos": [-0.38, -2.51, 0.4],
  "target_start_pos": [-0.52, -3.55, 0.0],
  "pedestrian_waypoints": { "Character_01": [[...]] }
}
```

### 3.4 `metrics.csv` — 汇总指标（首行示例）
| episode_id | success | had_recovery | episode_length | tracking_rate | collision | initial_dist | final_dist | avg_dist | avg_heading_error_deg | done_reason |
|------------|---------|--------------|----------------|---------------|-----------|--------------|------------|----------|------------------------|-------------|
| 0 | 1 | 0 | 100 | 0.86 | 0 | 1.13 | 2.33 | 1.65 | 36.56 | max_steps |

- `done_reason`: `max_steps` / `low_tracking` / `had_recovery` / `stuck`

---

## 四、跟踪与导航算法

### 4.1 跟踪行为模式

机器人采用**有限状态机（FSM）**控制跟踪行为，每个 step 根据目标可见性、距离、周围障碍物等信息切换模式：

```
                    ┌──────────────┐
                    │   RECOVERY   │ ← 目标刚丢失，快速原地旋转找回
                    └──────┬───────┘
                           │ 找到目标
                           ▼
                    ┌──────────────┐
               ┌───│    TRACK     │ ← 标准跟踪：目标在视野内且距离适中
               │   └──────┬───────┘
               │          │ 距离 > 3m
               │          ▼
               │   ┌──────────────┐
               │   │   APPROACH   │ ← 接近目标，纯比例控制 + Occ栅格避障
               │   └──────┬───────┘
               │          │ 距离 < 1.5m
               │          ▼
               │   ┌──────────────┐
               │   │   STANDOFF   │ ← 保持距离，NavMesh路径规划 + LOS角度保持
               │   └──────┬───────┘
               │          │ 目标快速接近？
               │          ▼
               │   ┌──────────────┐
               │   │    EVADE     │ ← 目标逼近时横向闪避
               │   └──────┬───────┘
               │          │
               │          │ 目标不可见？
               │          ▼
               │   ┌──────────────┐
               └───│   DETOUR     │ ← 目标被遮挡，寻NavMesh可见点绕行
                   └──────┬───────┘
                          │ 完全丢失
                          ▼
                   ┌──────────────┐
                   │   SEARCH     │ ← 搜索模式，沿预测路径查找
                   └──────┬───────┘
                          │ 持续丢失 → RECOVERY
```

### 4.2 控制器实现

**基础控制律**（`compute_control` 函数，`tracking_episode_collection.py:1765`）:
```
linear  = gain_linear  × (dist - desired_distance)   # P控制器
angular = gain_angular × heading_error                # P控制器
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gain_linear` | 1.0 | 线速度比例增益 |
| `gain_angular` | 3.0 | 角速度比例增益 |
| `max_linear_vel` | 1.2 m/s | 最大线速度 |
| `max_angular_vel` | 4.0 rad/s | 最大角速度 |
| `follow_distance` | 2.0 m | 期望跟踪距离 |
| `tracking_dist_min` | 1.0 m | 最小安全距离 |
| `tracking_dist_max` | 3.0 m | 触发APPROACH的距离阈值 |

**高级行为:**
- **APPROACH 模式**: 比例控制 + Occ栅格探测避障（每20°扫描30cm半径内的障碍物）+ 行人反应式避让（`apply_ped_reactive_avoidance`）
- **STANDOFF 模式**: NavMesh 路径规划 + LOS 角度混合（70%面向目标 + 30%面向路径方向），当目标靠近时自动横向闪避
- **DETOUR 模式**: 目标不可见时，在 NavMesh 上找"可见 vantage point"并规划绕行路径
- **EVADE 模式**: 预测目标未来位置（基于 oracle path 前瞻 2 秒），找到垂直方向的 standoff 位置横向闪避
- **SEARCH 模式**: 沿目标 oracle path 规划搜索路径
- **RECOVERY 模式**: 目标完全丢失时原地旋转尝试重新捕获

**机器人驱动类型:**
| 类型 | 机器人 | 驱动方式 |
|------|--------|---------|
| **diff_drive** | dingo | 轮式差速，通过 wheel joint velocity 控制 |
| **kinematic** | go2, g1 | 运动学驱动，直接设置 world pose + 零化所有速度（防止物理扰动积累）|

> kinematic_move 函数每步积分位姿并直接 set_world_pose，零化线/角速度和关节速度，克服四足/人形机器人关节刚度产生的反作用力矩漂移。

### 4.3 行人反应式避让

机器人对场景中所有 NPC 行人进行未来轨迹预测（基于 oracle command path），并在控制量上叠加避让修正：
- 对每个行人预测未来 2 秒位置
- 如果机器人朝向行人的方向角 < 60°，计算避让修正 angular
- 避让幅度 = 角度偏离的加权和，权重与距离成反比
## 五、角色外观多样化（Coloring System）

### 5.1 配色方案

角色外观通过 `character_clothing_profiles.json` 定义，支持 21 种角色资产 × 16 种颜色组合：

**调色板（16色）：**
```
黑色   silver  灰色    白色    栗色    红色    紫色    紫红
green lime    olive   yellow  navy    blue    teal    aqua
```

**角色资产池（21种）：**
| 资产名 | 角色类型 | USD 路径 |
|--------|---------|----------|
| F_Business_02 | 女性商务2 | Isaac/People/Characters/... |
| F_Medical_01 | 女性医务1 | ... |
| M_Medical_01 | 男性医务1 | ... |
| female_adult_police_01~03_new | 女性警察3种 | ... |
| male_adult_construction_01/03/05_new | 男性建筑工3种 | ... |
| male_adult_police_04 | 男性警察 | ... |
| original_female_adult_business_02 | 女性商务（原始） | ... |
| ...（共21种） | | |

### 5.2 角色生成与染色流程

```
1. episode JSON 中指定 "appearance.by_character" 
   → { "Character_01": { "asset": "F_Business_02", "shirt": "red", "pants": "navy", "hat": "white" } }

2. 加载角色 USD → 渲染引擎定位 ClothingAPI mesh

3. recolor_character() 按 parts 逐部分替换材质颜色
   - 每一部分可独立配色（上衣、裤子、帽子、鞋子等）
   - 颜色从 16 色调色板中随机选取

4. 生成 task_description:
   → "Track a person wearing red shirt, navy pants, white hat"
```

### 5.3 多样性组合

每个 episode 随机选取：
- 角色模型：从 21 种资产中选 1~3 个
- 每件衣服颜色：从 16 色调色板中随机抽取
- 目标人物：从场景所有角色中随机指定
- 每场景最多 100 个 episode，保证外观多样性

## 六、Combo 类型与参数

### 6.1 6 种 Combo 组合

```
ALL_COMBOS = "
  go2:realsense_d435i     # Unitree Go2 四足 + Intel RealSense D435i
  go2:zed                 # Unitree Go2 四足 + Stereolabs ZED 2
  dingo:realsense_d435i   # Dingo 轮式底盘 + Intel RealSense D435i
  dingo:zed               # Dingo 轮式底盘 + Stereolabs ZED 2
  g1:realsense_d435i      # Unitree G1 人形 + Intel RealSense D435i
  g1:zed                  # Unitree G1 人形 + Stereolabs ZED 2
"
```

### 6.2 机器人参数

| 参数 | dingo | go2 | g1 |
|------|-------|-----|-----|
| 驱动模式 | diff_drive（轮式差速） | kinematic（运动学） | kinematic（运动学） |
| USD 资产 | dingo_fixed.usd | go2.usd（IsaacLab） | g1.usd（IsaacLab） |
| camera_link | base_link | base | pelvis |
| robot_z_height | 0.1 m | 0.3 m | 0.9 m |
| wheel_radius | 0.0762 m | N/A | N/A |
| wheel_base | 0.32 m | N/A | N/A |

### 6.3 相机参数

| 参数 | default | realsense_d435i | zed |
|------|---------|----------------|-----|
| 分辨率 | 640×360 | 640×480 | **1280×720** |
| HFOV | ~67.8° | ~87° | **~90°** |
| VFOV | ~40° | ~58° | ~60° |
| 焦距 | 1.4 mm | 1.93 mm | **2.12 mm** |
| 光圈 | 1.88 mm | 3.2 mm | **5.2 mm** |
| 相机模型 | pinhole | pinhole | pinhole |
| 安装高度 | 0.3 m | 0.3 m | 0.3 m |

### 6.4 运行命令示例

```bash
# go2 + zed (当前正在跑)
bash /workspace/run_single_gpu_docker.sh 0 0 247 go2 zed

# dingo + realsense
bash /workspace/run_single_gpu_docker.sh 0 0 247 dingo realsense_d435i

# g1 + zed
bash /workspace/run_single_gpu_docker.sh 0 0 247 g1 zed
```

## 七、采集效率与进度评估

### 7.1 当前进度（截至 2026-07-14）

| 组合 | 状态 | 运行天数 | 已采集场景 | 总场景 | 已采episode | 完成度 |
|------|------|----------|-----------|--------|------------|--------|
| **go2:zed** | ✅ **正在运行**（4容器Up 19h） | 14天（7/1~至今） | 152 | 987 | 7,912 | ~8% |
| **go2:realsense_d435i** | ⏸️ 已停止（7/1后无新数据） | 8天（6/23~7/1） | 497 | 987 | 9,545 | ~10% |

### 7.2 脚本修改：场景级超时自动跳过（v2）

2026-07-14 修改 `run_single_gpu_docker.sh`，新增**每场景 4 小时总超时上限**：

```bash
SCENE_TIMEOUT=$((4 * 60 * 60))  # 4小时

for 每个场景:
   记录开始时间
   for 每个 batch:
     已用时间 > 4h → 跳过剩余 batches → 进入下一个场景
     启动 batch 时: 取 min(批次超时30min, 场景剩余时间) 作为实际超时
```

**改造前后的对比：**
| 场景类型 | 改造前 | 改造后 |
|----------|--------|--------|
| 正常场景（~2.5h） | 无影响 | 无影响 |
| 快速失败场景（~1h） | 正常跳过 | 正常跳过 |
| 挂死场景 | **浪费 15h**（10 batch × 3次重试 × 30min） | **最多 4h 自动跳过** |
| 卡住场景后的影响 | 整卡停摆数小时 | 4h 后自动恢复 |

### 7.3 实测生产效率

**正常场景（稳定产出）：**
```
GPU0 典型场景 0035（10 batches × 10 episodes × ~15min）:
  场景总耗时: 2h18min~2h47min
  平均: ~2.5 小时 / 100 episodes
```

**超时场景挂死成本：**
| 版本 | 单场景最大浪费 | 987场景×5%挂死率 × 4卡 |
|------|-------------|----------------------|
| 改造前 | **~15 小时** | 987×0.05×15/4 = **~185h = 7.7天** |
| 改造后 | **~4 小时** | 987×0.05×4/4 = **~49h = 2天** |

### 7.4 完成 1 个 combo 所需时间

**v2 版本（自动跳过）：**

| 场景类型 | 占比 | 每场景耗时 | 耗时分量（4卡并行） |
|---------|------|-----------|-------------------|
| 正常场景 | 80% | ~2.5h | 987×0.8×2.5/4 = **493h** |
| 快速失败（自动跳过） | 15% | ~1h | 987×0.15×1/4 = **37h** |
| 挂死（4h自动跳过） | 5% | ~4h | 987×0.05×4/4 = **49h** |
| **总时间** | | | **~579h ≈ 24 天** |

**调低每场景 episodes 可加速：**

| 每场景 episodes | 1 combo（4卡） | 全部6 combo串行 |
|----------------|---------------|----------------|
| 100（当前） | **~24 天** | **~144 天（~5个月）** |
| 50 | **~12 天** | **~72 天（~2.5月）** |
| 30 | **~7 天** | **~42 天（~1.5月）** |
| 20 | **~5 天** | **~30 天（1个月）** |

### 7.5 6 combo 同时并行（需 24 GPU）

| 配置 | 100ep/场景 | 50ep/场景 |
|------|-----------|-----------|
| 6 combo × 4 GPU（共 24 GPU） | **~24 天** | **~12 天** |

### 7.6 当前容器状态

```
docker ps --filter name=flux_go2_zed
flux_go2_zed_0  Up 19h  → 场景0040（超时循环中，v2部署后4h自动跳过）
flux_go2_zed_1  Up 19h  → 场景0280（超时循环中）
flux_go2_zed_2  Up 19h  → 场景0540（超时循环中）
flux_go2_zed_3  Up 19h  → 场景0798（batch 90-100，接近完成）
```

> 当前容器尚未部署 v2 脚本。部署后挂死场景 4h 后自动跳过，不再需要手动重启。

---

## 十、关键设计说明

1. **ZED 相机**: 1280×720 分辨率，~90° HFOV，ROS 坐标系（X前Y左Z上），安装于机器人上方 30cm
2. **go2 机器人**: Unitree Go2 四足机器人，运动学驱动模式（直接 set_world_pose）
3. **场景来源**: SAGE-3D 的 `v3_tracking_episodes` 数据集，共 987 个场景，每个场景含预先放置的行人、障碍物等
4. **采集策略**: 每场景 100 episode × 100 step，含随机扰动以增加多样性；tracking 率 <30% 提前终止
5. **断点续采**: 已存在 episode 数据的场景自动跳过；单 batch 超时 30min 自动重试
6. **场景差异**: 约 80%+ 场景可正常工作（13~18min/batch），约 5% 场景反复超时挂死，其余快速失败
7. **v2 改进**: 每场景总超时 4h，超时自动跳过，无需手动干预

---

## 十一、数据量估算（1 个 combo 满采）

| 数据类型 | 单 combo 全量（987 场景 × 100 episode） | go2:zed 当前已有 |
|----------|----------------------------------------|-----------------|
| RGB 图 | 98,700 × 100 = **987 万张** | 7912 × 100 = ~79 万张 |
| 深度图 | 同 RGB 数量 | 同 RGB 数量 |
| frame_data.npz | 98,700 个 | 7,912 个 |
| 单 combo 估算总数据量 | **~1~2 TB** | 已采 ~100GB+ |
