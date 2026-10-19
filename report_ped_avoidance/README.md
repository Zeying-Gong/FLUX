# 行人避让问题报告 (Pedestrian Avoidance Problem)

## 问题描述

FLUX 数据生成管线中，行人在路径上会互相穿越（视觉上重叠），不会主动避让。

### 场景
- 场景: `0001_839920`，episode 15
- 两个行人: **Character**（target，机器人跟随的目标）和 **Character_01**（另一个行人）
- 机器人: go2 + zed 相机
- 行人速度: 0.5（walk speed fraction）
- 运行 600 步

### 问题表现

原始代码（未修改）运行结果：
- **最小行人间距: 0.261m**（两人完全重叠，每人半径 ~0.3m）
- 间距 < 1.5m 的帧数: 366/600 帧
- 碰撞检测: 0（说明碰撞检测不涵盖行人之间）

## 系统架构

行人路径规划流程：
1. `generate_episode_json.py` 预计算 A* 路径（基于 2D semantic map）
2. 运行时 `people_utils.py` 加载路径，通过 `PatchedBaseCommand.walk()` 执行
3. `NavigationManager.update_path()` 更新路径点 → `character.set_variable("PathPoints", ...)` 传给 Isaac Sim C++ 侧
4. Isaac Sim 的 `omni.anim.people` extension 负责实际的 NavMesh 路径跟随

关键文件：
- `sage_utils/patched_base_command.py` — walk() 方法，控制行人移动
- `sage_utils/tracking_episode_collection_datagen.py` — 主数据生成脚本
- `sage_utils/navmesh_utils.py:404` — 烘焙 NavMesh 时设置 `DYNAMIC_AVOIDANCE_ENABLED=False`
- `people_simulation.py:154-168` — PeopleSim 配置

### 已有的避让机制（但已关闭）

Isaac Sim 的 NavigationManager 内置了 `update_path()` 动态避让（左右旋转路径点），但：
1. `navmesh_utils.py` 烘焙时设置 `DYNAMIC_AVOIDANCE_ENABLED=False`
2. 即使开启，测试发现两个行人都对称冻结（都不动），因为双方同时避让导致死锁

## 尝试过的方案

### 方案1: Isaac Sim 内置 DYNAMIC_AVOIDANCE
- 操作: 在 `tracking_episode_collection_datagen.py` 和 `people_simulation.py` 设置 `DYNAMIC_AVOIDANCE_ENABLED=True`
- 结果: `ep_min_ped_dist` 从 1.17m 提升到 2.59m，但两个行人都被冻结（target 600 步只走了 ~0.1m）

### 方案2: 优先级冻结（path freeze）
- 操作: 在 `patched_base_command.py` 加 `_apply_priority_yield()`，低优先级（lexicographically 名字大的）行人靠近高优先级行人时，冻结位置（`set_path_points([current_pos])`）和动画（`desired_walk_speed = 0`）
- 结果: 最小间距 0.59m，但低优先级的 Character_01 完全不动了，不真实

### 方案3: 横向偏移（lateral offset）— 有问题的版本
- 操作: 计算垂直方向偏移量，整体偏移 path points，加上近距离减速
- 结果: 最小间距仍然是 0.59m，可能是因为 NavMesh snapping 把偏移后的路径点拉回原始路径

## 需要解决的问题

如何让两个行人在路径交叉时自然地错开（像有半径的圆柱体一样），而不是完全冻结或无视对方？

### 可能的方向
1. 在 A* 路径生成阶段就考虑行人之间的间距
2. 在运行时直接操控行人位置（绕过 NavMesh snap）
3. 使用 Social Force Model / RVO 在 velocity 层面做避让
4. 开启 DYNAMIC_AVOIDANCE 并解决对称冻结（只有低优先级方避让）

## 修复方案（2026-07-21）

采用方向 4，在 `sage_utils/patched_base_command.py` 中实现非对称动态避让：

- 目标行人 `Character` 是最高优先级，调用 `update_path()` 时关闭动态避让，保持正常前进。
- 其他行人调用 `update_path()` 时临时开启 Isaac 内置动态避让，完成后立刻恢复全局设置。
- CharacterBehavior 在 Kit 主线程串行更新，因此单次调用范围内切换全局设置不会发生并发竞争。
- 避免了全员开启时双方同时冻结，也不需要直接修改人物 transform 或绕过 NavMesh。
- collector 记录 `min_inter_pedestrian_distance`；默认小于 `0.6m` 的多人 episode 会以 `inter_pedestrian_clearance` 原因拒收。

默认开启。通过 `run_distributed.sh` 运行时可配置：

```bash
ASYMMETRIC_PED_AVOIDANCE=1 PED_PRIORITY_CHARACTER=Character \
  MIN_INTERPED_DISTANCE=0.6 \
  MAX_EPISODES=1 bash run_distributed.sh
```

如需与旧逻辑做 A/B 对照，设置 `ASYMMETRIC_PED_AVOIDANCE=0`。
