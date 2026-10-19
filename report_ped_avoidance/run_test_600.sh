#!/bin/bash
set -euo pipefail

REPO_DIR=/mnt/nvme1/zeyingg/FLUX
SAGE3D_DIR=/mnt/ssd1/zeyingg/SAGE-3D_Official
ISAAC_CACHE_DIR=/home/zeyingg/.cache/flux-isaac-sim
RUN_DIR=$REPO_DIR/logs_formal_v3/test_runs/dynamic_avoid_test_600_v2

mkdir -p $RUN_DIR/0001_839920

docker run --rm -i \
  --name flux_dynamic_avoid_600_v2 \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e SAVE_VIDEO=1 \
  -e FLUX_ASYMMETRIC_PED_AVOIDANCE=1 \
  -e FLUX_PED_PRIORITY_CHARACTER=Character \
  --entrypoint bash \
  --gpus "device=0" --network=host \
  -v $REPO_DIR:/workspace/FLUX \
  -v $SAGE3D_DIR:/workspace/SAGE-3D_Official \
  -v $ISAAC_CACHE_DIR/kit:/isaac-sim/kit/cache \
  -v $ISAAC_CACHE_DIR/ov:/root/.cache/ov \
  -v $ISAAC_CACHE_DIR/ov-data:/root/.local/share/ov/data \
  -v $ISAAC_CACHE_DIR/pip:/root/.cache/pip \
  -v $ISAAC_CACHE_DIR/glcache:/root/.cache/nvidia/GLCache \
  -v $ISAAC_CACHE_DIR/computecache:/root/.nv/ComputeCache \
  -v $REPO_DIR/sage_utils/patched_base_command.py:/isaac-sim/extscache/omni.anim.people-0.7.9+107.3.3/omni/anim/people/scripts/commands/base_command.py \
  -w /workspace \
  quay.io/zeyinggong/flux:v2_deploy \
  -c '/isaac-sim/python.sh /workspace/FLUX/sage_utils/tracking_episode_collection_datagen.py \
    --episode_dir /workspace/SAGE-3D_Official/SAGE-3D_data/v3_tracking_episodes/0001_839920 \
    --robot_type go2 --camera_type zed \
    --start_idx 15 --end_idx 16 \
    --max_steps 600 --character_speed 0.5 --min_interped_distance 0.6 \
    --save_images --image_save_dir /workspace/FLUX/logs_formal_v3/test_runs/dynamic_avoid_test_600_v2/0001_839920 \
    --headless 2>&1'
