<p align="center">
<h1 align="center"><strong>FLUX：通过整流流与静到动学习加速跨具身生成式导航策略</strong></h1>

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

<!-- ### 📈 使用 GRPO 训练
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

### 💻 将基线作为服务运行

各预置基线目录中通常包含 `server.py`，指定端口与 checkpoint 路径即可启动。以 NavDP 为例：

```bash
# 请先下载 NavDP checkpoint
cd baselines/navdp/
python navdp_server.py --port 9999 --checkpoint ./checkpoints/navdp_checkpoint.ckpt 
```

其它基线请参考 [NavDP](https://github.com/InternRobotics/NavDP) 仓库或对应 README。

### 📊 运行评测

```bash
python eval_pointgoal_wheeled.py --port {PORT} --scene_dir {ASSET_SCENE}
```

**说明：** `--port` 需与服务器端口一致（默认 9999）；`--scene_dir` 请使用**绝对路径**。对 **internscenes** 请将 `scene_scale` 设为 `0.01`，**cluttered** 类场景一般为 `1.0`。

### 🕹️ 遥操作

```bash
# 若服务端支持无目标任务
python teleop_nogoal_wheeled.py
# 若支持点目标
python teleop_pointgoal_wheeled.py
# 若支持图像目标
python teleop_imagegoal_wheeled.py 
```

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
