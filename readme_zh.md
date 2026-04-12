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

推荐使用 **Docker 镜像** 运行 FLUX 与 Isaac Lab 配套环境，**无需**在宿主机单独创建 conda 环境。镜像内已包含所需依赖；将本仓库与 [IsaacLab](https://github.com/Zeying-Gong/IsaacLab) 通过卷挂载进容器后，在 `/workspace` 下开发即可。

#### 拉取镜像

```bash
docker pull quay.io/zeyinggong/flux:v0
```

**镜像：** `quay.io/zeyinggong/flux:v0`

#### 克隆 GitHub 仓库

```bash
git clone https://github.com/Zeying-Gong/FLUX.git
```

（还需克隆 **[IsaacLab](https://github.com/Zeying-Gong/IsaacLab)** 仓库，以便与下述挂载路径一致。）

#### 启动容器

将下面命令中的宿主机路径换成你本机克隆的 FLUX 与 IsaacLab 所在目录（**不要**照抄示例占位路径）。

```bash
docker run --name flux_v0 \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -e "DISPLAY=${DISPLAY}" \
    --entrypoint bash \
    --gpus all \
    --network=host \
    --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /path/to/FLUX:/workspace/FLUX \
    -v /path/to/IsaacLab:/workspace/IsaacLab \
    -w /workspace \
    -it quay.io/zeyinggong/flux:v0
```

- `/path/to/FLUX` — 本仓库 [Zeying-Gong/FLUX](https://github.com/Zeying-Gong/FLUX) 在本机的克隆目录。
- `/path/to/IsaacLab` — [IsaacLab](https://github.com/Zeying-Gong/IsaacLab) 在本机的克隆目录。

进入容器后，后续 Python 命令请在 **`/workspace/FLUX`** 目录下执行（可先 `cd /workspace/FLUX`）。

### 数据集与场景资产

评测用 **USD 场景与配套资源** 需放在 **[IsaacLab](https://github.com/Zeying-Gong/IsaacLab)** 仓库的 `assets/scenes/` 下：宿主机上对应你克隆的 IsaacLab 目录，容器内为 **`/workspace/IsaacLab/assets/scenes/`**。Episode 配置与说明可在 IsaacLab 仓库中查阅。

#### 🌆 静态场景资产（Prepare Scene Asset）

请从 HuggingFace 上的 **[InternScene-N1](https://huggingface.co/datasets/InternRobotics/Scene-N1/tree/main/n1_eval_scenes)** 下载场景资源，解压后整理为如下目录结构（相对 **IsaacLab 仓库根目录** 的 `assets/scenes/`）：

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

在同一容器内另开终端（或 `docker exec -it flux_v0 bash`）用测试脚本检查服务：

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

# ---------- 静态场景（资产在 IsaacLab assets/scenes/n1_eval_scenes，或你同步后的 FLUX/assets/n1_eval_scenes）----------
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
