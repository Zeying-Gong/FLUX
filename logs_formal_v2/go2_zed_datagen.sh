#!/bin/bash
# 只跑 datagen follower 控制版 go2:zed
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SAGE3D_DIR="${SAGE3D_DIR:-/mnt/ssd1/zeyingg/SAGE-3D_Official}"

COLLECTOR="$REPO_DIR/sage_utils/tracking_episode_collection_datagen.py"
RUNNER="$REPO_DIR/run_single_gpu_docker_datagen.sh"

[ -f "$COLLECTOR" ] || {
  echo "Missing collector: $COLLECTOR" >&2
  echo "Pull the latest dev branch before running this script." >&2
  exit 1
}
[ -f "$RUNNER" ] || {
  echo "Missing runner: $RUNNER" >&2
  exit 1
}
[ -d "$SAGE3D_DIR" ] || {
  echo "Missing SAGE-3D directory: $SAGE3D_DIR" >&2
  echo "Set it with: SAGE3D_DIR=/path/to/SAGE-3D_Official bash $0" >&2
  exit 1
}

for GPU in 0 1 2 3; do
  S=$(( GPU * 247 ))
  E=$(( S + 247 ))
  [ $GPU -eq 3 ] && E=987
  docker run -d --name flux_go2_zed_datagen_$GPU --rm \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    --entrypoint bash --runtime=nvidia --gpus device=$GPU --network=host \
    -v "$REPO_DIR:/workspace/FLUX" \
    -v "$SAGE3D_DIR:/workspace/SAGE-3D_Official" \
    -w /workspace quay.io/zeyinggong/flux:v2_deploy \
    -c "bash /workspace/run_single_gpu_docker_datagen.sh $GPU $S $E go2 zed"
done
