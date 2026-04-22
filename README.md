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

**Languages:** [English](README.md) | [中文](readme_zh.md)

# 🏡 Introduction

We propose **FLUX**, the first **FL**ow-based **U**nified policy for **X**-platform (Cross-Embodiment) navigation. FLUX leverages a static-to-dynamic curriculum and linearizes probability flow for efficient, straight-line trajectory generation. This enables state-of-the-art performance and zero-shot sim-to-real transfer across wheeled, quadrupedal, and humanoid robots without any fine-tuning.

<div style="text-align: center;">
    <img src="./assets/images/compressed/v2_teaser_02.png" alt="FLUX Teaser" width=100% >
</div>

### 🛠️ Environment and setup (Docker, recommended)

We recommend using the **Docker image** for FLUX. You **do not** need a separate conda environment on the host; dependencies are preinstalled in the image. Mount this repo into the container and work under `/workspace`.

#### Pull the image

```bash
docker pull quay.io/zeyinggong/flux:v1
```

**Image:** `quay.io/zeyinggong/flux:v1`

#### Clone the repositories

```bash
git clone https://github.com/hkustgz-linker/FLUX.git
```

#### Start the container

Replace the host path in the command with your local clone of FLUX (**do not** copy the placeholder path verbatim).

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

- `/path/to/FLUX` — local clone of [hkustgz-linker/FLUX](https://github.com/hkustgz-linker/FLUX).

Inside the container, run subsequent Python commands from the **FLUX repo root** (e.g. `cd /workspace/FLUX` first).

### Datasets and scene assets

Place evaluation **USD scenes and related assets** under the FLUX repo at `assets/scenes/` — inside the container that is **`/workspace/FLUX/assets/scenes/`**.

### Robot asset note (Dingo ground “white fog” fix)

We fixed an issue in `FLUX/assets/robots/dingo.usd` that could make the ground look like it has **white fog**. Use `FLUX/assets/robots/dingo_fixed.usd` instead.

#### 🌆 Static scene assets (prepare scene asset)

Download scene assets from Hugging Face **[InternScene-N1](https://huggingface.co/datasets/InternRobotics/Scene-N1/tree/main/n1_eval_scenes)**, extract, and arrange them as below relative to the **FLUX repo root** under `assets/scenes/`:

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

| Category | Download | Episodes |
|----------|----------|----------|
| SkyTexture | [link](https://huggingface.co/datasets/InternRobotics/Scene-N1/blob/main/n1_eval_scenes/SkyTexture.tar.gz) | — |
| Materials | [link](https://huggingface.co/datasets/InternRobotics/Scene-N1/blob/main/n1_eval_scenes/Materials.tar.gz) | — |
| Cluttered-Easy | [link](https://huggingface.co/datasets/InternRobotics/Scene-N1/blob/main/n1_eval_scenes/cluttered_easy.tar.gz) | [dir](FLUX/assets/n1_eval_scenes/cluttered_easy) |
| Cluttered-Hard | [link](https://huggingface.co/datasets/InternRobotics/Scene-N1/blob/main/n1_eval_scenes/cluttered_hard.tar.gz) | [dir](FLUX/assets/n1_eval_scenes/cluttered_hard) |
| InternScenes-Home | [link](https://huggingface.co/datasets/InternRobotics/Scene-N1/tree/main/n1_eval_scenes/internscenes_home) | [dir](FLUX/assets/n1_eval_scenes/internscenes_home) |
| InternScenes-Commercial | [link](https://huggingface.co/datasets/InternRobotics/Scene-N1/blob/main/n1_eval_scenes/internscenes_commercial.tar.gz) | [dir](FLUX/assets/n1_eval_scenes/internscenes_commercial) |

#### 🏃 Dynamic scenes (`isaacsim_scene` / DynBench)

Download data from Hugging Face **[DynBench](https://huggingface.co/datasets/zgong313/DynBench/tree/main)**. The dataset layout is roughly:

- **`Materials/`**, **`SkyTexture/`**: shared **auxiliary** assets (materials, skyboxes, etc.) for training scenes.
- **`dynbench/cluttered_scenes/`**: **training** set (cluttered scene assets).
- **`isaacsim_scene/`**: **test** set (Isaac Sim dynamic scenes; each subfolder has USD, `episode_*.json`, pedestrian configs, etc.).

Download command:

```bash
export FLUX_ROOT=/path/to/FLUX 
mkdir -p "${FLUX_ROOT}/assets/dynbench"
huggingface-cli download zgong313/DynBench --repo-type dataset --local-dir "${FLUX_ROOT}/assets/dynbench"
```

You should see a layout like:

```text
FLUX/assets/dynbench
├── Materials/                 # auxiliary scene assets
├── SkyTexture/                # auxiliary scene assets
├── dynbench/
│   └── cluttered_scenes/      # training (cluttered)
└── isaacsim_scene/            # test (isaacsim)
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

### 📥 Pre-trained weights

At the **FLUX repo root** (your host clone; inside the container: `/workspace/FLUX`), create a folder **`checkpoints`** and download **`flux_v1.ckpt`** into it.

Weights page: [Hugging Face · zgong313/FLUX](https://huggingface.co/zgong313/FLUX/tree/main).

With Hugging Face CLI (run where HF is reachable, e.g. inside the container if CLI is configured):

```bash
cd /path/to/FLUX   # your host path; in container: cd /workspace/FLUX
mkdir -p checkpoints
huggingface-cli download zgong313/FLUX flux_v1.ckpt --local-dir checkpoints --local-dir-use-symlinks False
```

You can also download `flux_v1.ckpt` from that page in a browser and place it under `checkpoints/`.

### 🤖 Run FLUX model

Inside the container, from the **FLUX repo root**:

```bash
cd /workspace/FLUX
# Terminal 1: start server
python baselines/flux/server.py --port 9999 --checkpoint checkpoints/flux_v1.ckpt
```

In another terminal in the same container (or `docker exec -it flux_v1 bash`), verify with the test script:

```bash
cd /workspace/FLUX
python test_flask_server.py
```

**Note:** For **Running baselines as server**, **Evaluation**, and **Teleoperation** below, run `cd /workspace/FLUX` first inside the container.

### 💻 Running baselines as server

Each baseline folder typically includes `server.py`; pass port and checkpoint to start. Example (NavDP):

```bash
# Download the NavDP checkpoint first
cd baselines/navdp/
python navdp_server.py --port 9999 --checkpoint ./checkpoints/navdp_checkpoint.ckpt 
```

For other baselines, see the [NavDP](https://github.com/InternRobotics/NavDP) repo or the corresponding README.

### 📊 Running evaluation

**Scripts vs. tasks:**

| Task type | Goal type | Script | Typical `--scene_dir` |
|-----------|-----------|--------|------------------------|
| Static | Point goal | `eval_pointgoal_wheeled.py` | A category under `n1_eval_scenes` (e.g. `cluttered_easy`) |
| Static | Image goal | `eval_imagegoal_wheeled.py` | A category under `n1_eval_scenes` (e.g. `cluttered_easy`) |
| Static | No-goal exploration | `eval_nogoal_wheeled.py` | A category under `n1_eval_scenes` (e.g. `cluttered_easy`) |
| Dynamic | Dynamic point goal | `eval_dynpointgoal_wheeled.py` | A category under `DynBench/isaacsim_scene` (e.g. `Hospital`) |
| Dynamic | No-goal with people | `eval_dynnogoal_wheeled.py` | A category under `DynBench/isaacsim_scene` (e.g. `Hospital`) |
| Dynamic | Social navigation | `eval_socialnav_wheeled.py` | A category under `DynBench/isaacsim_scene` (e.g. `Hospital`) |

**Common flags:** `--port` matches the server (default `9999`); `--scene_dir` should be an **absolute** path; `--scene_index` is the sub-scene index under `scene_dir` (0-based). **`--scene_scale`**: use **`0.01`** for InternScenes-style assets and **`1.0`** for **cluttered** scenes. Dynamic scripts also take **`--gpu_id`** (default `0`).

**Examples:**

```bash
cd /workspace/FLUX

# ---------- Static (assets under FLUX/assets/n1_eval_scenes) ----------
# Point goal · cluttered
python eval_pointgoal_wheeled.py --port 9999 \
  --scene_dir /workspace/FLUX/assets/n1_eval_scenes/cluttered_easy \
  --scene_index 0 --scene_scale 1.0

# Image goal
python eval_imagegoal_wheeled.py --port 9999 \
  --scene_dir /workspace/FLUX/assets/n1_eval_scenes/cluttered_easy \
  --scene_index 0 --scene_scale 1.0

# No-goal exploration
python eval_nogoal_wheeled.py --port 9999 \
  --scene_dir /workspace/FLUX/assets/n1_eval_scenes/cluttered_easy \
  --scene_index 0 --scene_scale 1.0

# ---------- Dynamic (full DynBench download; isaacsim_scene subdirs e.g. Hospital, Office) ----------
# Dynamic point goal (follow pedestrians)
python eval_dynpointgoal_wheeled.py --port 9999 --gpu_id 0 \
  --scene_dir /workspace/FLUX/assets/dynbench/isaacsim_scene \
  --scene_index 0 --scene_scale 1.0 --num_episodes 100

# Dynamic no-goal exploration
python eval_dynnogoal_wheeled.py --port 9999 --gpu_id 0 \
  --scene_dir /workspace/FLUX/assets/dynbench/isaacsim_scene \
  --scene_index 0 --scene_scale 1.0 --num_episodes 100

# Social navigation (point goal + pedestrians)
python eval_socialnav_wheeled.py --port 9999 --gpu_id 0 \
  --scene_dir /workspace/FLUX/assets/dynbench/isaacsim_scene \
  --scene_index 0 --scene_scale 1.0 --num_episodes 100
```

### 🕹️ Teleoperation

```bash
# If the server supports no-goal tasks
python teleop_nogoal_wheeled.py
# If it supports point-goal tasks
python teleop_pointgoal_wheeled.py
# If it supports image-goal tasks
python teleop_imagegoal_wheeled.py 
```

<!-- ### 📈 Post-training with GRPO
FLUX supports online RL fine-tuning in dynamic scenes with **GRPO (Group Relative Policy Optimization)**.

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
