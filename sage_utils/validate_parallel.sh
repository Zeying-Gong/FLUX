#!/bin/bash
# validate_parallel.sh - Run episode validation on multiple GPUs
set -e

USDA_DIR="/workspace/SAGE-3D_Official/SAGE-3D_data/usda"
EPISODE_DIR="/workspace/SAGE-3D_Official/SAGE-3D_data/v1_episodes"
OUTPUT_DIR="$EPISODE_DIR/validation"
ISAAC_PYTHON="/isaac-sim/python.sh"
VALIDATOR="/workspace/FLUX/sage_utils/validate_episodes.py"

# Collect all scene IDs
cd "$USDA_DIR"
scene_ids=()
for f in *.usda; do
    scene_ids+=("${f%.usda}")
done
cd - > /dev/null

# Split into N parts (N = number of GPUs you want to use)
N_GPU=4
n=${#scene_ids[@]}
chunk_size=$(( (n + N_GPU - 1) / N_GPU ))

tmpdir=$(mktemp -d)
for i in $(seq 0 $((N_GPU-1))); do
    start=$(( i * chunk_size ))
    chunk=("${scene_ids[@]:start:chunk_size}")
    if [ ${#chunk[@]} -eq 0 ]; then
        echo "[WARN] Chunk $i empty"
        continue
    fi
    printf "%s\n" "${chunk[@]}" > "$tmpdir/scene_chunk_$i.txt"
done

echo "[INFO] Launching $N_GPU validators..."
for gpu in $(seq 0 $((N_GPU-1))); do
    chunk_file="$tmpdir/scene_chunk_$gpu.txt"
    [ -f "$chunk_file" ] || continue
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        $ISAAC_PYTHON $VALIDATOR \
            --usda_dir "$USDA_DIR" \
            --episode_dir "$EPISODE_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --scene_ids $(cat "$chunk_file") \
            --workers 1 \
            --skip_navmesh_check
        echo "[GPU $gpu] Finished"
    ) &
done

echo "[INFO] Waiting for all validators to complete..."
wait
rm -r "$tmpdir"

# Merge summaries
$ISAAC_PYTHON -c "
import json, glob, os
summary_dir = '$OUTPUT_DIR'
merged = {'total_ok':0, 'total_fail':0, 'scenes':{}}
for f in glob.glob(os.path.join(summary_dir, 'summary_*.json')):
    with open(f) as fh:
        data = json.load(fh)
        merged['total_ok'] += data['total_ok']
        merged['total_fail'] += data['total_fail']
        merged['scenes'].update(data['scenes'])
with open(os.path.join(summary_dir, 'summary.json'), 'w') as out:
    json.dump(merged, out, indent=2)
print(f'[DONE] Pass={merged[\"total_ok\"]} Fail={merged[\"total_fail\"]}')
"