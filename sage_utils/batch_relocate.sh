#!/bin/bash
# Batch relocate robot starts for all scenes in v2_tracking_episodes
# Usage: EXP_PATH=/isaac-sim/apps bash batch_relocate.sh [--test N]

INPUT_DIR="/workspace/SAGE-3D_Official/SAGE-3D_data/v2_tracking_episodes"
OUTPUT_DIR="/workspace/SAGE-3D_Official/SAGE-3D_data/v3_tracking_episodes"
SCRIPT="/workspace/FLUX/sage_utils/relocate_robot_start.py"
PYTHON="/isaac-sim/python.sh"

mkdir -p "$OUTPUT_DIR"

# Get all scene dirs
SCENES=($(ls "$INPUT_DIR"))
TOTAL=${#SCENES[@]}

# If --test is given, only process N scenes
if [ "$1" = "--test" ]; then
    TOTAL=$2
fi

echo "Processing $TOTAL scenes..."
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Start:  $(date)"
echo ""

COUNT=0
SKIPPED=0
for SCENE in "${SCENES[@]:0:$TOTAL}"; do
    INPUT_SCENE="$INPUT_DIR/$SCENE"
    if [ ! -d "$INPUT_SCENE" ]; then
        continue
    fi
    # Check if episode files exist
    EP_COUNT=$(ls "$INPUT_SCENE"/episode_*.json 2>/dev/null | wc -l)
    if [ "$EP_COUNT" -eq 0 ]; then
        continue
    fi
    
    # Check if already processed
    if [ -f "$OUTPUT_DIR/$SCENE/episode_0.json" ]; then
        echo "  [$((COUNT+1))/$TOTAL] $SCENE already done, skip"
        COUNT=$((COUNT+1))
        continue
    fi
    
    START=$(date +%s)
    EXP_PATH=/isaac-sim/apps $PYTHON "$SCRIPT" \
        --input_dir "$INPUT_DIR" \
        --scene_id "$SCENE" \
        --output_dir "$OUTPUT_DIR" \
        --min_dist 1.0 --max_dist 3.0 2>/dev/null
    RC=$?
    END=$(date +%s)
    DUR=$((END-START))
    
    if [ $RC -eq 0 ]; then
        echo "  [$((COUNT+1))/$TOTAL] $SCENE done in ${DUR}s"
        COUNT=$((COUNT+1))
    else
        echo "  [$((COUNT+1))/$TOTAL] $SCENE FAILED (exit=$RC, ${DUR}s)"
        SKIPPED=$((SKIPPED+1))
    fi
done

echo ""
echo "Done: $COUNT processed, $SKIPPED failed"
echo "End: $(date)"
