<p align="center">
<h1 align="center"><strong>FLUX：通过修正流匹配与静到动学习加速跨具身生成式导航策略</strong></h1>

  <!-- Badges -->
  <p align="center">
    <a href="https://zeying-gong.github.io/projects/flux/">
      <img src="https://img.shields.io/badge/Web-Flux-deepgreen.svg" alt="Flux Project Web Badge">
    </a>
    <a href="https://www.youtube.com/watch?v=cY6tFvKgTjo">
      <img src="https://img.shields.io/badge/Video-Youtube-red.svg" alt="YouTube Video Badge">
    </a>
    <a href="https://arxiv.org/abs/2603.12806">
      <img src="https://img.shields.io/badge/cs.ai-arxiv:2603.12806-42ba94.svg" alt="arXiv Paper Badge">
    </a>
    <a href="https://github.com/isaac-sim/IsaacSim">
      <img src="https://img.shields.io/static/v1?label=supports&message=IsaacSim&color=informational" alt="IsaacSim Badge">
    </a>
    <a href="https://github.com/Zeying-Gong/habitat-lab/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License Badge">
    </a>
  </p>

  <p align="center">
    <a href='https://zeying-gong.github.io/' target='_blank'>Zeying Gong</a><sup>1*</sup>&emsp;
    <a href='' target='_blank'>Yangyi Zhong</a><sup>1*</sup>&emsp;
    <a href='https://www.linkedin.com/in/ding-justin-08a421266' target='_blank'>Yiyi Ding</a><sup>1*</sup>&emsp;
    <a href='https://hutslib.github.io/' target='_blank'>Tianshuai Hu</a><sup>2</sup>&emsp;
    <a href='https://guoyangzhao.github.io/' target='_blank'>Guoyang Zhao</a><sup>1</sup>&emsp; <br>
    <a href='https://ldkong.com/' target='_blank'>Lingdong Kong</a><sup>3</sup>&emsp; 
    <a href='https://rongli.tech/' target='_blank'>Rong Li</a><sup>1</sup>&emsp;
    <a href='www.linkedin.com/in/jiadi-you-2362a6354' target='_blank'>Jiadi You</a><sup>1</sup>&emsp;
	<a href='https://junweiliang.me/' target='_blank'>Junwei Liang</a><sup>1,2,&#9993;</sup>&emsp;·
    <br>
    <sup>1</sup>HKUST(GZ)&emsp; 
    <sup>2</sup>HKUST&emsp;
    <sup>3</sup>NUS&emsp;
    <br>
    * Equal Contribution
  </p>
</p>

**语言：** [English](README.md) | [中文](readme_zh.md)

# 🏡 简介

我们提出 **FLUX**，首个基于 **流（Flow）** 的 **跨具身（X-platform）** 统一导航策略。FLUX 采用静到动课程并将概率流线性化，以高效生成近似直线的轨迹，从而在轮式、四足与人形机器人上取得领先表现，并可在零样本 sim-to-real 迁移下工作、无需微调。

<div style="text-align: center;">
    <img src="./assets/images/compressed/v2_teaser_02.png" alt="FLUX Teaser" width=100% >
</div>

### 🛠️ 环境与运行（Docker，推荐）

推荐使用 **Docker 镜像** 运行 FLUX，**无需**在宿主机单独创建 conda 环境。镜像内已包含所需依赖；将本仓库通过卷挂载进容器后，在 `/workspace` 下开发即可。

#### 拉取镜像

```bash
docker pull quay.io/zeyinggong/flux:v1
```

**镜像：** `quay.io/zeyinggong/flux:v1`

#### 克隆 GitHub 仓库

```bash
git clone https://github.com/hkustgz-linker/FLUX.git
```

#### 启动容器

将下面命令中的宿主机路径换成你本机克隆的 FLUX 所在目录（**不要**照抄示例占位路径）。

```bash
docker run --name flux_v1 \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -e "DISPLAY=${DISPLAY}" \
    --entrypoint bash \
    --gpus all \
    --network=host \
    --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /path/to/FLUX:/workspace/FLUX \
    -w /workspace \
    -it quay.io/zeyinggong/flux:v1
```

- `/path/to/FLUX` — 本仓库 [Zeying-Gong/FLUX](https://github.com/hkustgz-linker/FLUX) 在本机的克隆目录。

进入容器后，后续 Python 命令请在 **`/workspace/FLUX`** 目录下执行（可先 `cd /workspace/FLUX`）。

### 数据集与场景资产

评测用 **USD 场景与配套资源** 需放在 FLUX 仓库的 `assets/scenes/` 下：容器内为 **`/workspace/FLUX/assets/scenes/`**。

### 机器人资产说明（Dingo 地面“白雾”问题修复）

我们修复了 `FLUX/assets/robots/dingo.usd` 可能导致地面出现**白色雾气**的显示问题。请改用 `FLUX/assets/robots/dingo_fixed.usd`。

#### 🌆 静态场景资产（Prepare Scene Asset）

请从 HuggingFace 上的 **[InternScene-N1](https://huggingface.co/datasets/InternRobotics/Scene-N1/tree/main/n1_eval_scenes)** 下载场景资源，解压后整理为如下目录结构（相对 **FLUX 仓库根目录** 的 `assets/scenes/`）：

```text
FLUX/assets/n1_eval_scenes
├── SkyTexture/
│   ├── belfast_sunset_puresky_4k.hdr
│   ├── citrus_orchard_road_puresky_4k.hdr
│   ├── ...
├── Materials/
│   ├── Carpet/
│   │   ├── textures/
│   │   ├── Carpet_Woven.mdl
│   │   └── ...
│   ├── ...
├── cluttered_easy/
│   └── easy_0/
│       ├── cluttered-0.usd/
│       ├── imagegoal_start_goal_pairs.npy
│       └── pointgoal_start_goal_pairs.npy
│   ├── ...
├── cluttered_hard/
│   └── hard_0/
│       ├── cluttered-0.usd/
│       ├── imagegoal_start_goal_pairs.npy
│       └── pointgoal_start_goal_pairs.npy
│   ├── ...
├── internscenes_commercial/
│   ├── models/
│   ├── Materials/
│   └── scenes_commercial/
│       ├── MV4AFHQKTKJZ2AABAAAAADQ8_usd/
│       │   ├── models/
│       │   ├── Materials/
│       │   ├── metadata.json
│       │   ├── start_result_navigation.usd
│       │   ├── imagegoal_start_goal_pairs.npy
│       │   └── pointgoal_start_goal_pairs.npy
│       ├── ...
├── internscenes_home/
│   ├── models/
│   ├── Materials/
│   └── scenes_home/
│       ├── MV4AFHQKTKJZ2AABAAAAADQ8_usd/
│       │   ├── models/
│       │   ├── Materials/
│       │   ├── metadata.json
│       │   ├── start_result_navigation.usd
│       │   ├── imagegoal_start_goal_pairs.npy
│       │   └── pointgoal_start_goal_pairs.npy
│       ├── ...
```

| 类别 | 资源下载 | Episodes目录 |
|------|----------|---------------------------|
| SkyTexture | [链接](https://huggingface.co/datasets/InternRobotics/Scene-N1/blob/main/n1_eval_scenes/SkyTexture.tar.gz) | - |
| Materials | [链接](https://huggingface.co/datasets/InternRobotics/Scene-N1/blob/main/n1_eval_scenes/Materials.tar.gz) | - |
| Cluttered-Easy | [链接](https://huggingface.co/datasets/InternRobotics/Scene-N1/blob/main/n1_eval_scenes/cluttered_easy.tar.gz) | [目录](FLUX/assets/n1_eval_scenes/cluttered_easy) |
| Cluttered-Hard | [链接](https://huggingface.co/datasets/InternRobotics/Scene-N1/blob/main/n1_eval_scenes/cluttered_hard.tar.gz) | [目录](FLUX/assets/n1_eval_scenes/cluttered_hard) |
| InternScenes-Home | [链接](https://huggingface.co/datasets/InternRobotics/Scene-N1/tree/main/n1_eval_scenes/internscenes_home) | [目录](FLUX/assets/n1_eval_scenes/internscenes_home) |
| InternScenes-Commercial | [链接](https://huggingface.co/datasets/InternRobotics/Scene-N1/blob/main/n1_eval_scenes/internscenes_commercial.tar.gz) | [目录](FLUX/assets/n1_eval_scenes/internscenes_commercial) |


#### 🏃 动态场景（`isaacsim_scene` / DynBench）

请从 HuggingFace 上的 **[DynBench](https://huggingface.co/datasets/zgong313/DynBench/tree/main)** 下载数据。数据集根目录大致分工如下：

- **`Materials/`、`SkyTexture/`**：场景**辅助资源**（材质、天空盒等），供训练场景共用。
- **`dynbench/cluttered_scenes/`**：**训练集**（cluttered 场景相关资产）。
- **`isaacsim_scene/`**：**测试集**（Isaac Sim 动态场景；各子目录内含 USD、`episode_*.json`、行人配置等）。

下载命令为

```bash
export FLUX_ROOT=/path/to/FLUX 
mkdir -p "${FLUX_ROOT}/assets/dynbench"
huggingface-cli download zgong313/DynBench --repo-type dataset --local-dir "${FLUX_ROOT}/assets/dynbench"
```

应看到如下的文件逻辑结构：

```text
FLUX/assets/dynbench
├── Materials/                 # 场景辅助资源
├── SkyTexture/                # 场景辅助资源
├── dynbench/
│   └── cluttered_scenes/      # 训练集（cluttered）
└── isaacsim_scene/            # 测试集（isaacsim）
    ├── Full_Warehouse/
    │   ├── full_warehouse.usd
    │   ├── episode_0.json
    │   ├── episode_1.json
    │   └── ...
    ├── Hospital/
    ├── Jetracer/
    ├── Office/
    ├── Warehouse/
    └── Warehouse_multiple_shelves/
```

### 🗂️ 数据集生成流程（`sage_utils/`）

`sage_utils/` 目录提供一套**完全离线、CPU 并行**的 episode JSON 生成流程，用于构建社会导航与人员跟踪基准数据集。这些脚本**无需 Isaac Sim 即可运行**，支持大规模快速生成。

#### Episode 生成 — 社会导航

| 脚本 | 说明 |
|------|------|
| `sage_utils/generate_episode_fast.py` | 单场景 episode 生成器（纯 Python）。基于 PSDF 避障与可导航岛分配对机器人起/终点及行人位置进行采样。 |
| `sage_utils/batch_episode_fast.py` | CPU 并行批调度器。使用 `ProcessPoolExecutor` 将多个场景的生成任务并行分发，支持 `--resume` 断点续跑与 `--workers` 进程数配置。 |

```bash
# 对所有场景生成共 100 个 episode（16 进程并行，自动断点续跑）
python sage_utils/batch_episode_fast.py \
    --sem_dir /path/to/semantic_maps \
    --output_dir /path/to/episodes \
    --total_episodes 100 --max_people 10 \
    --resume --workers 16
```

#### Episode 生成 — 人员跟踪

| 脚本 | 说明 |
|------|------|
| `sage_utils/generate_episode_fast_tracking.py` | 跟踪 episode 生成器。为机器人指定一名目标行人及多段路径点链。`--easy_mode` 可放宽距离、视线、岛约束并增加随机种子重试次数，适用于小场景或难以采样的场景。 |
| `sage_utils/generate_episode_fast_tracking_hard.py` | 严格约束版本，适用于需要高难度跟踪场景的情形（不启用 easy-mode 宽松逻辑）。 |
| `sage_utils/batch_episode_fast_tracking.py` | 跟踪 episode 并行批调度器。支持 `--easy_mode` 和 `--scene_ids`（仅重跑指定失败场景）。 |

```bash
# 以 easy-mode 生成 100 个跟踪 episode
python sage_utils/batch_episode_fast_tracking.py \
    --sem_dir /path/to/semantic_maps \
    --output_dir /path/to/tracking_episodes \
    --total_episodes 100 --max_people 10 \
    --easy_mode --resume --workers 16
```

#### 角色外观分配

`sage_utils/clothing_appearance.py` 为 episode 内每名行人分配独立的服装资产与逐部位颜色（取自 CSS-16 色板）。目标行人始终随机染色；干扰行人有 20% 概率保留默认外观。同一 episode 内颜色严格区分（不允许混淆组重叠），从而支持基于自然语言的跟踪指令（例如"跟随穿红色上衣和蓝色裤子的行人"）。

#### 工具模块

| 脚本 | 说明 |
|------|------|
| `sage_utils/occupancy_utils.py` | 通过 PhysX 光线投射在碰撞网格上构建 2-D 占用栅格，不依赖 NavMesh 内部接口。 |
| `sage_utils/navmesh_utils.py` | NavMesh 烘焙辅助工具，支持增量层缓存——后续烘焙命中 Isaac Sim 内部缓存，耗时 < 1 秒。 |
| `generate_pedestrian_trajectories.py` | 离线行人轨迹生成（需 Isaac Sim headless 模式进行 NavMesh 烘焙）。 |
| `sage_utils/test_dataset_pipeline.py` | 交互式测试脚本：加载 USDA 场景与预生成 episode JSON，在仿真器中验证角色外观与轨迹。 |

### 📥 预训练权重

在 **FLUX 仓库根目录**（宿主机上你克隆下来的那份，挂载后对应容器内 `/workspace/FLUX`）下 **新建文件夹 `checkpoints`**，将预训练权重文件 **`flux_v1.ckpt`** 下载并保存到该文件夹中。

权重发布页：[Hugging Face · zgong313/FLUX](https://huggingface.co/zgong313/FLUX/tree/main)。

可使用 Hugging Face CLI（需在能访问 Hugging Face 的环境中执行，例如在容器内若已配置好 CLI）：

```bash
cd /path/to/FLUX   # 换成你的本机 FLUX 根目录；若在容器内则: cd /workspace/FLUX
mkdir -p checkpoints
huggingface-cli download zgong313/FLUX flux_v1.ckpt --local-dir checkpoints --local-dir-use-symlinks False
```

也可在浏览器从上述页面手动下载 `flux_v1.ckpt`，再放入 `checkpoints/`。

### 🤖 运行 FLUX 模型

在容器内、**FLUX 仓库根目录**下启动服务：

```bash
cd /workspace/FLUX
# 终端 1：启动服务
python baselines/flux/server.py --port 9999 --checkpoint checkpoints/flux_v1.ckpt
```

在同一容器内另开终端（或 `docker exec -it flux_v1 bash`）用测试脚本检查服务：

```bash
cd /workspace/FLUX
python test_flask_server.py
```

**说明：** 下文「将基线作为服务」「评测」「遥操作」中的命令，同样建议在容器内先执行 `cd /workspace/FLUX` 再运行。

### 💻 将基线作为服务运行

各预置基线目录中通常包含 `server.py`，指定端口与 checkpoint 路径即可启动。以 NavDP 为例：

```bash
# 请先下载 NavDP checkpoint
cd baselines/navdp/
python navdp_server.py --port 9999 --checkpoint ./checkpoints/navdp_checkpoint.ckpt 
```

其它基线请参考 [NavDP](https://github.com/InternRobotics/NavDP) 仓库或对应 README。

### 📊 运行评测

**脚本与任务对应关系：**

| 任务类型 | 目标类型 | 脚本 | 典型场景根目录（`--scene_dir`） |
|------|------|------|--------------------------------|
| 静态 | 点目标 | `eval_pointgoal_wheeled.py` | `n1_eval_scenes` 下某一类场景目录（如 `cluttered_easy`） |
| 静态 | 图像目标 | `eval_imagegoal_wheeled.py` | `n1_eval_scenes` 下某一类场景目录（如 `cluttered_easy`） |
| 静态 | 无目标探索 | `eval_nogoal_wheeled.py` | `n1_eval_scenes` 下某一类场景目录（如 `cluttered_easy`） |
| 动态 | 动态点目标 | `eval_dynpointgoal_wheeled.py` | `DynBench/isaacsim_scene`下某一类场景目录（如 `Hospital`） |
| 动态 | 有人场景无目标探索 | `eval_dynnogoal_wheeled.py` | `DynBench/isaacsim_scene`下某一类场景目录（如 `Hospital`） |
| 动态 | 社会导航 | `eval_socialnav_wheeled.py` | `DynBench/isaacsim_scene`下某一类场景目录（如 `Hospital`） |

**常用参数：** `--port` 与服务器一致（默认 `9999`）；`--scene_dir` 建议**绝对路径**；`--scene_index` 为 `scene_dir` 下列出的子场景序号（从 0 起）。**`--scene_scale`**：InternScenes 类一般为 **`0.01`**，**cluttered** 类一般为 **`1.0`**。动态类脚本另有 **`--gpu_id`**（默认 `0`）。

**示例：**

```bash
cd /workspace/FLUX

# ---------- 静态场景（资产在 FLUX/assets/n1_eval_scenes）----------
# 点目标 · cluttered
python eval_pointgoal_wheeled.py --port 9999 \
  --scene_dir /workspace/FLUX/assets/n1_eval_scenes/cluttered_easy \
  --scene_index 0 --scene_scale 1.0

# 图像目标
python eval_imagegoal_wheeled.py --port 9999 \
  --scene_dir /workspace/FLUX/assets/n1_eval_scenes/cluttered_easy \
  --scene_index 0 --scene_scale 1.0

# 无目标探索
python eval_nogoal_wheeled.py --port 9999 \
  --scene_dir /workspace/FLUX/assets/n1_eval_scenes/cluttered_easy \
  --scene_index 0 --scene_scale 1.0

# ---------- 动态场景（整库下载后的 isaacsim_scene；子目录如 Hospital、Office）----------
# 动态点目标（跟行人）
python eval_dynpointgoal_wheeled.py --port 9999 --gpu_id 0 \
  --scene_dir /workspace/FLUX/assets/dynbench/isaacsim_scene \
  --scene_index 0 --scene_scale 1.0 --num_episodes 100

# 动态无目标探索
python eval_dynnogoal_wheeled.py --port 9999 --gpu_id 0 \
  --scene_dir /workspace/FLUX/assets/dynbench/isaacsim_scene \
  --scene_index 0 --scene_scale 1.0 --num_episodes 100

# 社会导航（点目标 + 行人）
python eval_socialnav_wheeled.py --port 9999 --gpu_id 0 \
  --scene_dir /workspace/FLUX/assets/dynbench/isaacsim_scene \
  --scene_index 0 --scene_scale 1.0 --num_episodes 100
```


### 🕹️ 遥操作

```bash
# 若服务端支持无目标任务
python teleop_nogoal_wheeled.py
# 若支持点目标
python teleop_pointgoal_wheeled.py
# 若支持图像目标
python teleop_imagegoal_wheeled.py 
```

<!-- ### 📈 使用 GRPO 后训练
FLUX 支持使用 **GRPO（Group Relative Policy Optimization）** 在动态场景中进行在线强化学习微调。

```bash
isaacsim-python baselines/flux/train_grpo.py \
    --checkpoint checkpoints/flux_v1.ckpt \
    --scene_dirs assets/dyn_scenes/cluttered_easy assets/dyn_scenes/isaacsim_scene \
    --tasks dynpointgoal dynnogoal socialnav \
    --num_episodes 3000 \
    --save_dir baselines/flux/checkpoints_rl \
    --gpu_id 0 --train_gpu_id 0 \
    --lr 3e-5 --update_interval 32 --save_interval 100
``` -->

# 🔗 引用

若本工作对您有帮助，欢迎引用：

```bibtex
@article{gong2025flux,
    title     = {FLUX: Accelerating Cross-Embodiment Generative Navigation Policies via Rectified Flow and Static-to-Dynamic Learning},
    author    = {Gong, Zeying and Zhong, Yangyi and Ding, Yiyi and Hu, Tianshuai and Zhao, Guoyang and Kong, Lingdong and Li, Rong and You, Jiadi and Liang, Junwei},
    journal   = {arXiv preprint arXiv:2603.12806},
    year      = {2025}
}
```

# 👏 致谢

感谢 [NavDP](https://github.com/InternRobotics/NavDP) 作者的优秀开源代码。
