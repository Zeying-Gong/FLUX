# FLUX 行人跟踪数据集格式说明

## 目录结构

```
{robot_type}_{camera_type}/
├── metrics.csv                         # 所有场景的聚合指标
├── {scene_id}/
│   ├── episode_0000/
│   │   ├── rgb/
│   │   │   ├── 00000.png               # uint8 RGB
│   │   │   ├── 00001.png
│   │   │   └── ...
│   │   ├── depth_mm/
│   │   │   ├── 00000.png               # uint16 PNG, 深度值 = 毫米
│   │   │   ├── 00001.png
│   │   │   └── ...
│   │   ├── depth_viz/
│   │   │   └── 00000.png               # (仅第一帧) 彩色可视化
│   │   ├── target_crops/
│   │   │   └── 00000.png               # (仅第一帧) 目标行人 RGB 裁剪图
│   │   ├── camera_info.json            # 相机内参 + 外参
│   │   ├── task_description_ep0000.txt # "Track a woman wearing lime top, blue pants, white shoes"
│   │   └── frames/
│   │       ├── frame_data.npz          # 每步训练数据
│   │       └── preview.json            # JSON 格式预览
│   └── episode_0001/
│       └── ...
└── {scene_id}/
    └── ...
```

### 实际示例

```
go2_realsense_d435i/
├── metrics.csv
├── 0001_839920/
│   ├── episode_0000/ (75 帧 RGB + 深度 + NPZ)
│   └── episode_0001/ (95 帧)
└── 0002_839955/
    ├── episode_0000/ (73 帧)
    └── episode_0001/ (100 帧)
```

---

## 数据存放位置速查

| 你要的数据 | 存在哪里 | 格式 |
|---|---|---|
| **RGB 图像** (每步) | `{robot}_{camera}/{scene}/episode_X/rgb/{step}.png` | uint8 PNG |
| **深度图** (每步, 毫米) | `{robot}_{camera}/{scene}/episode_X/depth_mm/{step}.png` | uint16 PNG (I;16) |
| 深度可视化 (仅第一帧) | `{robot}_{camera}/{scene}/episode_X/depth_viz/00000.png` | uint8 彩色 PNG |
| 目标裁剪图 (仅第一帧) | `{robot}_{camera}/{scene}/episode_X/target_crops/00000.png` | uint8 RGB PNG |
| 相机内参 / 外参 | `{robot}_{camera}/{scene}/episode_X/camera_info.json` | JSON |
| **训练核心数据** | **`{robot}_{camera}/{scene}/episode_X/frames/frame_data.npz`** | **NumPy `.npz`** |
| NPZ 预览 (JSON) | `{robot}_{camera}/{scene}/episode_X/frames/preview.json` | JSON |
| 任务文字描述 | `{robot}_{camera}/{scene}/episode_X/task_description_epX.txt` | UTF-8 文本 |
| 所有场景指标汇总 | `{robot}_{camera}/metrics.csv` | CSV |

### `frame_data.npz` 里面有什么

| 你要的 | NPZ 键名 | 维度 | 实际存放位置 |
|---|---|---|---|
| **状态向量** | `observation.state` | (T, 9) | **NPZ 内部** |
| **动作向量** | `action` | (T, 2) | **NPZ 内部** |
| 目标点坐标 (u,v) | `target_uv_u`, `target_uv_v` | (T,) | **NPZ 内部** |
| 目标包围盒 | `target_bbox_x1/y1/x2/y2` | (T,) | **NPZ 内部** |
| 目标可见性 | `target_visible` | (T,) | **NPZ 内部** |
| 机器人轨迹 | `robot_trajectory` | (T, 3) | **NPZ 内部** |
| 目标轨迹 | `target_trajectory` | (T, 3) | **NPZ 内部** |
| 任务描述 | `task_description` | (bytes) | **NPZ 内部** + `.txt` 外部 |
| **RGB 图像** | — | — | `rgb/{step}.png` **(独立文件)** |
| **深度图** | — | — | `depth_mm/{step}.png` **(独立文件)** |
| 目标裁剪图 | — | — | `target_crops/00000.png` **(独立文件)** |
| 相机参数 | — | — | `camera_info.json` **(独立文件)** |

---

## NPZ 格式 (`frame_data.npz`)

共 34 个字段，T = 总步数 = `episode_length`。

### 核心字段（LeRobot 兼容）

| 键 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `observation.state` | (T, 9) | float64 | 机器人 + 目标状态向量 |
| `action` | (T, 2) | float64 | 速度指令 `[线速度, 角速度]` |

### observation.state 的 9 列含义

```
[robot_x, robot_y, robot_z, robot_yaw,
 target_rel_x, target_rel_y, target_rel_z,
 target_dist, target_bearing_deg]
```

| 序号 | 字段 | 单位 | 说明 |
|-------|------|------|-------------|
| 0 | robot_x | m | 机器人世界坐标 X |
| 1 | robot_y | m | 机器人世界坐标 Y |
| 2 | robot_z | m | 机器人基座高度 (Go2: 0.40m, Dingo: 0.10m) |
| 3 | robot_yaw | rad | 机器人朝向 |
| 4 | target_rel_x | m | 目标在机器人坐标系下的前方分量 (ROS 约定) |
| 5 | target_rel_y | m | 目标在机器人坐标系下的左方分量 |
| 6 | target_rel_z | m | 目标在机器人坐标系下的上方分量 |
| 7 | target_dist | m | 到目标的欧氏距离 |
| 8 | target_bearing_deg | deg | 目标方位角（+ = 在机器人左侧） |

### action 的 2 列含义

| 序号 | 字段 | 单位 | 说明 |
|-------|------|------|-------------|
| 0 | action_linear | m/s | 前进线速度指令 |
| 1 | action_angular | rad/s | 角速度指令（+ = 左转） |

### 任务描述

| 键 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `task_description` | (N,) | uint8 | UTF-8 编码文本，如 `"Track a woman wearing lime top, blue pants, white shoes"` |

读取: `task_description.tobytes().decode("utf-8")`

### 轨迹数据

| 键 | 形状 | 说明 |
|---|---|---|
| `robot_trajectory` | (T, 3) | 每步 `[robot_x, robot_y, robot_z]` |
| `target_trajectory` | (T, 3) | 每步 `[target_x, target_y, target_z]` |

### 目标图像投影

| 键 | 形状 | 说明 |
|---|---|---|
| `target_uv_u` | (T,) | 目标中心投影到图像的 X 像素坐标 |
| `target_uv_v` | (T,) | 目标中心投影到图像的 Y 像素坐标 |
| `target_bbox_x1` | (T,) | 2D 包围盒左边界（clamp 到图像内） |
| `target_bbox_y1` | (T,) | 2D 包围盒上边界 |
| `target_bbox_x2` | (T,) | 2D 包围盒右边界 |
| `target_bbox_y2` | (T,) | 2D 包围盒下边界 |
| `target_visible` | (T,) | int64: 1=可见(无遮挡), 0=被遮挡 |

### 原始每步字段

| 键 | 说明 |
|---|---|
| `robot_pos_x/y/z` | 机器人世界位置 (m) |
| `robot_yaw` | 机器人朝向 (rad) |
| `target_pos_x/y/z` | 目标世界位置 (m) |
| `target_rel_x/y/z` | 目标在机器人坐标系下的相对位置 (m) |
| `target_dist` | 欧氏距离 (m) |
| `target_bearing_deg` | 方位角 (度) |
| `target_elevation_deg` | 俯仰角 (度) |
| `step` | int64: 步序号 |
| `timestamp` | 时间 (秒) = `step * physics_dt * decimation` |
| `mode` | str: 当前跟踪模式 (TRACK, STANDOFF, APPROACH, EVADE, DETOUR, SEARCH, RECOVERY) |
| `action_linear` | 线速度指令 (m/s) |
| `action_angular` | 角速度指令 (rad/s) |
| `dist_to_target` | 同 target_dist |
| `heading_error_deg` | 到目标的角度误差 (度) |
| `contact_force` | 机器人最大接触力 (N) |
| `ped_min_dist` | 距非目标行人的最小距离 (m) |

---

## 图像数据

### RGB (`rgb/{step}.png`)

- 格式: uint8 PNG, 3 通道
- 分辨率: 640×360 (默认), `--camera_type` 可修改
- 相机: 机器人前向安装 (ROS 约定: +X 前, +Y 左, +Z 上)

### 深度 (`depth_mm/{step}.png`)

- 格式: uint16 PNG (mode `I;16`), 单通道
- 分辨率: 同 RGB
- 值: 深度值 = **毫米** (mm)
- 约定: **distance to image plane**（垂直于像平面的深度，非欧氏射线距离）
- 背景（无效深度）: 0
- Python 读取:
  ```python
  from PIL import Image
  import numpy as np
  depth_mm = np.asarray(Image.open("depth_mm/00000.png"), dtype=np.uint16)
  depth_m = depth_mm.astype(np.float32) / 1000.0
  ```
- 深度来源: **复合深度** — RTX 深度（行人，像素级精确）+ PhysX 射线（静态场景背景）。
  RTX 有效处用 RTX（行人），无效处用 PhysX 填充（3DGS 背景）。

### 目标裁剪 (`target_crops/00000.png`)

- 仅第一帧保存
- 目标行人的 RGB 包围盒裁剪图
- 可作为视觉目标模板

---

## 相机参数 (`camera_info.json`)

```json
{
  "camera": {
    "model": "pinhole",
    "axes": "ros",
    "width": 640,
    "height": 360,
    "intrinsics": {
      "fx": 476.6,
      "fy": 476.6,
      "cx": 320.0,
      "cy": 180.0,
      "k": [[476.6, 0, 320.0], [0, 476.6, 180.0], [0, 0, 1]]
    },
    "extrinsics_robot_to_camera": {
      "translation": [0.0, 0.0, 0.3],
      "camera_link": "base"
    },
    "clipping_range": [0.01, 100.0],
    "focal_length_mm": 1.4,
    "horizontal_aperture_mm": 1.88
  }
}
```

---

## Metrics CSV (`metrics.csv`)

每 episode 的聚合统计，用于数据质量筛选。

| 列 | 说明 |
|---|---|
| `episode_id` | Episode 序号 |
| `success` | 1=跟踪率≥0.8, 最终距离≤max, 无 recovery, 有效结束原因 |
| `had_recovery` | 1=机器人被卡住并触发了恢复动作 |
| `episode_length` | 步数 |
| `tracking_rate` | 机器人在跟踪窗口内的步数占比 |
| `collision` | 碰撞次数 |
| `initial_dist` | 第 0 步时机器人到目标的距离 (m) |
| `final_dist` | 最后一步时机器人到目标的距离 (m) |
| `avg_dist` | 全程平均距离 (m) |
| `avg_heading_error_deg` | 全程平均角度误差 (度) |
| `done_reason` | 结束原因: `max_steps`, `char_done`, `had_recovery`, `stuck`, `collision`, `planning_failed`, `target_lost` |

---

## 训练兼容性分析

### 目标输入（Goal）支持情况

| 目标类型 | 数据支持 | 说明 |
|---|---|---|
| **点坐标** `(u, v)` | ✅ `target_uv_u/v` 每步都有 | 目标中心投影到图像平面的像素坐标，可做 heatmap / 坐标回归 |
| **图像目标** (bbox crop) | ✅ `target_crops/00000.png` 第一帧 | 可作为视觉模板输入（类似 GOT / 跟踪器模板）。当前仅第一帧，若需每帧需改代码 |
| **混合输入** (点+图像) | ✅ 两者皆有 | 可同时用点和裁剪图作为 policy 输入 |

### 传感器输入（Observation）支持情况

| 模态 | 数据支持 | 说明 |
|---|---|---|
| **RGB** | ✅ `rgb/{step}.png` | 前向第一人称 RGB 图像 |
| **深度** | ✅ `depth_mm/{step}.png` | 毫米级深度图，和 RGB 对齐 |
| **RGBD** | ✅ RGB + Depth | 两者分辨率一致、像素对齐，可直接拼接为 4 通道输入 |
| **机器人状态** | ✅ `observation.state` | 9 维向量含位置/朝向/目标相对位姿 |

### 动作输出（Action）支持情况

| 动作空间 | 数据支持 | 说明 |
|---|---|---|
| **线速度 + 角速度** | ✅ `action` = `[linear, angular]` | 维度 2，连续值，可直接做回归 |

### 任务描述

| 信息 | 数据支持 | 说明 |
|---|---|---|
| **行人穿搭描述** | ✅ `task_description` / `.txt` | "Track a woman wearing lime top, blue pants, white shoes" |
| **行人外观** | ✅ 原 episode JSON 含完整 `appearance` 字段 | 含 asset 名、各部位 RGB 颜色值 |

### 结论

**当前数据完全支持 RGBD + 点/图像目标 + 线速度角速度输出的模仿学习/强化学习训练。** 若要支持每帧都有的目标裁剪图（而非仅第一帧），需要改 `save_rgb_depth` 中的 `is_first_episode_frame` 守卫条件。

---

## 采集参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--robot_type` | `go2` | 机器人: `go2`(四足), `dingo`(差分驱动), `g1`(人形) |
| `--character_speed` | 0.5 | 行人走路速度倍率 [0-1] |
| `--camera_type` | `default` | 相机: `default`(640×360, 67.8° HFOV), `realsense_d435i`, `zed` |
| `--max_steps` | 500 | 每 episode 最大步数 |
| `--allow_recovery` | True | 是否保存有 recovery 的 episode |
| `--min_tracking_rate` | 0.0 | 最低跟踪率 (0.0 = 全保存) |
| `--min_visible_rate` | 0.0 | 最低目标在画面率 (0.0 = 全保存) |
| `--max_consecutive_lost` | 10 | 目标连续多少帧不在画面则终止此 episode |
