#!/bin/bash
# run_fast_parallel.sh – Fast CPU parallel generation (16 workers)
set -e
cd "$(dirname "$0")"          # ← 确保从脚本所在目录运行

SEM_DIR="/workspace/SAGE-3D_Official/SAGE-3D_data/semantic_maps_v2"
OUTPUT_DIR="/workspace/SAGE-3D_Official/SAGE-3D_data/v2_episodes"
TOTAL_EP=100
MAX_PEOPLE=10

echo "[INFO] Launching batch_episode_fast.py with 16 workers"
python batch_episode_fast.py \
    --sem_dir "$SEM_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --total_episodes $TOTAL_EP \
    --max_people $MAX_PEOPLE \
    --resume \
    --workers 16