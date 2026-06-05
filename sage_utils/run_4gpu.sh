#!/bin/bash
# run_4gpu.sh – Launch 4 GPU‑parallel batch generators with resume
set -e

# --- Configuration ----------------------------------------------------
USDA_DIR="/workspace/SAGE-3D_Official/SAGE-3D_data/usda"
SEM_DIR="/workspace/SAGE-3D_Official/SAGE-3D_data/semantic_maps"
OUTPUT_DIR="/workspace/SAGE-3D_Official/SAGE-3D_data/episodes"
TOTAL_EP=100
MAX_PEOPLE=10

# --- Collect all scene IDs --------------------------------------------
echo "[INFO] Collecting scene IDs..."
cd "$USDA_DIR"
scene_ids=()
for f in *.usda; do
    scene_ids+=("${f%.usda}")
done
cd - > /dev/null
echo "[INFO] Total scenes found: ${#scene_ids[@]}"

# --- Split into 4 chunks ----------------------------------------------
n=${#scene_ids[@]}
chunk_size=$(( (n + 3) / 4 ))
echo "[INFO] Chunk size: $chunk_size"

tmpdir=$(mktemp -d)
for i in 0 1 2 3; do
    start=$(( i * chunk_size ))
    # slice array
    chunk=("${scene_ids[@]:start:chunk_size}")
    if [ ${#chunk[@]} -eq 0 ]; then
        echo "[WARN] Chunk $i is empty, skipping."
        continue
    fi
    # write IDs to file
    printf "%s\n" "${chunk[@]}" > "$tmpdir/scene_chunk_$i.txt"
done

# --- Launch 4 background processes ------------------------------------
echo "[INFO] Launching 4 GPU workers..."
for gpu in 0 1 2 3; do
    chunk_file="$tmpdir/scene_chunk_$gpu.txt"
    [ -f "$chunk_file" ] || continue
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        echo "[GPU $gpu] Starting on $(wc -l < "$chunk_file") scenes"
        python batch_episode_generator.py \
            --usda_dir    "$USDA_DIR" \
            --sem_dir     "$SEM_DIR" \
            --output_dir  "$OUTPUT_DIR" \
            --total_episodes $TOTAL_EP \
            --max_people  $MAX_PEOPLE \
            --resume \
            --skip_navmesh_check \
            --scene_ids $(cat "$chunk_file")
        echo "[GPU $gpu] Finished."
    ) &
done

# --- Wait for all to finish -------------------------------------------
echo "[INFO] Waiting for all workers to complete..."
wait
rm -r "$tmpdir"
echo "[INFO] All 4 GPUs finished."