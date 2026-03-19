<p align="center">
<h1 align="center"><strong>FLUX: Accelerating Cross-Embodiment Generative Navigation Policies via Rectified Flow and Static-to-Dynamic Learning</strong></h1>

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

# 🏡 Introduction
We propose **FLUX**, the first **FL**ow-based **U**nified policy for **X**-platform (Cross-Embodiment) navigation. FLUX leverages a static-to-dynamic curriculum and linearizes probability flow for efficient, straight-line trajectory generation. This enables state-of-the-art performance and zero-shot sim-to-real transfer across wheeled, quadrupedal, and humanoid robots without any fine-tuning.
<div style="text-align: center;">
    <img src="./assets/images/compressed/v2_teaser_02.png" alt="FLUX Teaser" width=100% >
</div>

### 🛠️ Installation
Please follow the instructions to config the environment for FLUX.

Step 0: Clone this repository
```bash
git clone https://github.com/Zeying-Gong/FLUX.git
cd FLUX/
```

Step 1: Create conda environment and install the dependency
```bash
conda create -n flux python=3.10
conda activate flux
pip install -r requirements.txt
```

### 📥 Pre-trained Weights
Download the pre-trained FLUX weights from [Hugging Face](https://huggingface.co/zgong313/FLUX/tree/main).

```bash
mkdir checkpoints
# Download via huggingface-cli
huggingface-cli download zgong313/FLUX flux_v1.ckpt --local-dir checkpoints --local-dir-use-symlinks False
```

### 🤖 Run FLUX Model
Run the following line to start the FLUX server:
```bash
# Terminal 1: Start the server
python baselines/flux/server.py --port 9999 --checkpoint checkpoints/flux_v1.ckpt
```

To verify if the server is running correctly, you can use the provided test script in a new terminal:
```bash
# Terminal 2: Run verification script
python test_flask_server.py
```

<!-- ### 📈 Training with GRPO
FLUX supports online reinforcement learning fine-tuning using **Group Relative Policy Optimization (GRPO)** to enhance performance in dynamic environments.

```bash
# Start GRPO training across multiple tasks and scenes
isaacsim-python baselines/flux/train_grpo.py \
    --checkpoint checkpoints/flux_v1.ckpt \
    --scene_dirs assets/dyn_scenes/cluttered_easy assets/dyn_scenes/isaacsim_scene \
    --tasks dynpointgoal dynnogoal socialnav \
    --num_episodes 3000 \
    --save_dir baselines/flux/checkpoints_rl \
    --gpu_id 0 --train_gpu_id 0 \
    --lr 3e-5 --update_interval 32 --save_interval 100
``` -->


### 💻 Running Baselines as Server
For each pre-built baseline methods, each contains a server.py file, just simply run server python script with parsing the server port as well as the checkpoint path. Taking NavDP as an example:
```bash
# please first download the NavDP checkpoint
cd baselines/navdp/
python navdp_server.py --port 9999 --checkpoint ./checkpoints/navdp_checkpoint.ckpt 
```

For other baselines, please refer to [NavDP](https://github.com/InternRobotics/NavDP)'s repository or the corresponding README.md file.

### 📊 Running Evaluation
```bash
# Evaluation commands
python eval_pointgoal_wheeled.py --port {PORT} --scene_dir {ASSET_SCENE}
```
Notes: Please parse the port to match the server port (default is 9999), and always parse the absolute path for the scene_dir. For **internscenes**, please parse scene_scale as 0.01, and 1.0 for **cluttered scenes**.

### 🕹️ Running Teleoperation
```bash
# Teleoperation commands
# if the running server support no-goal task
python teleop_nogoal_wheeled.py
# if the running server support point-goal task
python teleop_pointgoal_wheeled.py
# if the running server support image-goal task
python teleop_imagegoal_wheeled.py 
```

# 🔗 Citation

If you find our work helpful, please cite:
```bibtex
@article{gong2025flux,
    title     = {FLUX: Accelerating Cross-Embodiment Generative Navigation Policies via Rectified Flow and Static-to-Dynamic Learning},
    author    = {Gong, Zeying and Zhong, Yangyi and Ding, Yiyi and Hu, Tianshuai and Zhao, Guoyang and Kong, Lingdong and Li, Rong and You, Jiadi and Liang, Junwei},
    journal   = {arXiv preprint arXiv:2603.12806},
    year      = {2025}
}
```

# 👏 Acknowledgement
We thank the authors of [NavDP](https://github.com/InternRobotics/NavDP) for their excellent open-source codebase.
