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

### 🛠️ 安装

请按下列步骤配置 FLUX 运行环境。

**步骤 0：** 克隆本仓库

```bash
git clone https://github.com/Zeying-Gong/FLUX.git
cd FLUX/
```

**步骤 1：** 创建 conda 环境并安装依赖

```bash
conda create -n flux python=3.10
conda activate flux
pip install -r requirements.txt
```

### 📥 预训练权重

从 [Hugging Face](https://huggingface.co/zgong313/FLUX/tree/main) 下载 FLUX 预训练权重。

```bash
mkdir checkpoints
# 使用 huggingface-cli 下载
huggingface-cli download zgong313/FLUX flux_v1.ckpt --local-dir checkpoints --local-dir-use-symlinks False
```

### 🤖 运行 FLUX 模型

启动 FLUX 服务：

```bash
# 终端 1：启动服务
python baselines/flux/server.py --port 9999 --checkpoint checkpoints/flux_v1.ckpt
```

在新终端中用测试脚本检查服务是否正常：

```bash
# 终端 2：验证脚本
python test_flask_server.py
```

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
