#!/bin/bash
# Datagen follower 控制版单 GPU 数据采集
# 用法: run_single_gpu_docker_datagen.sh <GPU_ID> <START_IDX> <END_IDX>

GPU=${1:?Usage: $0 <GPU_ID> <START> <END> [ROBOT] [CAMERA]}
S=$2
E=$3
ROBOT_ONLY=$4
CAM_ONLY=$5
CHARACTER_SPEED="${CHARACTER_SPEED:-0.5}"

NUM_EPISODES=100
BATCH_SIZE=10
SCENE_DIR="/workspace/SAGE-3D_Official/SAGE-3D_data/v3_tracking_episodes"
SCRIPT_PY="/workspace/FLUX/sage_utils/tracking_episode_collection_datagen.py"
PYTHON="/isaac-sim/python.sh"
ALL_SCENES=($(ls "$SCENE_DIR"))

LOG_DIR="/workspace/FLUX/logs_formal_v2"
LOG_NAME="collect_datagen_${ROBOT_ONLY:-all}_${CAM_ONLY:-all}_$(date '+%Y%m%d_%H%M%S')_gpu${GPU}"
VIDEO_ARGS=()
[ "${SAVE_VIDEO:-0}" = "1" ] && VIDEO_ARGS+=(--save_video)

echo "[GPU$GPU] Processing scenes $S ~ $E (total ${#ALL_SCENES[@]} / BATCH_SIZE=$BATCH_SIZE)"
ALL_COMBOS="go2:realsense_d435i go2:zed dingo:realsense_d435i dingo:zed g1:realsense_d435i g1:zed"
if [ -n "$ROBOT_ONLY" ] && [ -n "$CAM_ONLY" ]; then
  COMBOS="${ROBOT_ONLY}:${CAM_ONLY}"
else
  COMBOS="$ALL_COMBOS"
fi

SCENE_TIMEOUT=$((4 * 60 * 60))  # 每场景最多 4 小时,超时自动跳过

for combo in $COMBOS; do
  robot="${combo%%:*}"
  cam="${combo##*:}"
  echo "[GPU$GPU] Starting ${robot}_${cam} ($((E-S)) scenes)..."
  for ((i=S; i<E; i++)); do
    scene="${ALL_SCENES[$i]}"
    OUT_DIR="$LOG_DIR/${robot}_${cam}_datagen/$scene"
    # Skip if scene already has any output
    if [ -d "$OUT_DIR" ] && ls "$OUT_DIR"/episode_* 2>/dev/null | head -1 | grep -q .; then
      echo "[GPU$GPU] $(date '+%H:%M:%S') SKIP (already done): $scene"
      continue
    fi
    # ── 场景级超时守卫: 超过 SCENE_TIMEOUT 自动跳过 ──
    SCENE_START=$(date +%s)
    SCENE_TIMED_OUT=false
    for ((batch_start=0; batch_start<NUM_EPISODES; batch_start+=BATCH_SIZE)); do
      # 检查场景是否已超时
      NOW=$(date +%s)
      ELAPSED=$((NOW - SCENE_START))
      if [ $ELAPSED -ge $SCENE_TIMEOUT ]; then
        echo "[GPU$GPU] $(date '+%H:%M:%S') $scene SCENE TIMEOUT after ${ELAPSED}s, skipping remaining batches"
        SCENE_TIMED_OUT=true
        break
      fi
      batch_end=$((batch_start + BATCH_SIZE))
      [ $batch_end -gt $NUM_EPISODES ] && batch_end=$NUM_EPISODES
      MAX_RETRIES=3
      BATCH_TIMEOUT=$((30 * 60))  # 30 min per batch
      for ((attempt=0; attempt<MAX_RETRIES; attempt++)); do
        # 每秒检查一次场景超时（在重试间隙）
        if [ $attempt -gt 0 ]; then
          NOW=$(date +%s)
          ELAPSED=$((NOW - SCENE_START))
          if [ $ELAPSED -ge $SCENE_TIMEOUT ]; then
            echo "[GPU$GPU] $(date '+%H:%M:%S') $scene SCENE TIMEOUT during retry, skipping"
            SCENE_TIMED_OUT=true
            break 2  # 跳出 attempt 和 batch 两层循环
          fi
        fi
        echo "[GPU$GPU] $(date '+%H:%M:%S') $scene batch ${batch_start}-${batch_end} (attempt $((attempt+1)))..."
        $PYTHON "$SCRIPT_PY" \
          --episode_dir "$SCENE_DIR/$scene" \
          --robot_type "$robot" --camera_type "$cam" \
          --max_steps 300 --character_speed "$CHARACTER_SPEED" \
          --save_images --start_idx $batch_start --end_idx $batch_end \
          --headless \
          "${VIDEO_ARGS[@]}" \
          --resume \
          >> $LOG_DIR/$LOG_NAME.log 2>&1 &
        BGPID=$!
        # 取批次超时和场景剩余时间的较小值作为kill等待时间
        REMAINING=$((SCENE_TIMEOUT - ELAPSED))
        [ $REMAINING -lt $BATCH_TIMEOUT ] && BATCH_TIMEOUT=$REMAINING
        [ $BATCH_TIMEOUT -le 0 ] && BATCH_TIMEOUT=1
        (sleep $BATCH_TIMEOUT && kill $BGPID 2>/dev/null && sleep 3 && kill -9 $BGPID 2>/dev/null) &
        WATCHER=$!
        wait $BGPID 2>/dev/null
        EXIT_CODE=$?
        kill $WATCHER 2>/dev/null
        if [ $EXIT_CODE -eq 0 ]; then
          echo "[GPU$GPU] $(date '+%H:%M:%S') $scene batch ${batch_start}-${batch_end} done"
          break
        else
          echo "[GPU$GPU] $(date '+%H:%M:%S') $scene batch ${batch_start}-${batch_end} TIMEOUT/FAIL (exit=$EXIT_CODE)"
          if [ $attempt -lt $((MAX_RETRIES - 1)) ]; then
            echo "[GPU$GPU] Retrying batch ${batch_start}-${batch_end} in 5s..."
            sleep 5
          fi
        fi
      done
      if [ "$SCENE_TIMED_OUT" = true ]; then
        break
      fi
    done
    if [ "$SCENE_TIMED_OUT" = true ]; then
      echo "[GPU$GPU] $(date '+%H:%M:%S') $scene SKIPPED due to scene timeout (collected partial data if any)"
    fi
  done
  echo "[GPU$GPU] ${robot}_${cam} DONE ($((E-S)) scenes)"
done
echo "[GPU$GPU] ALL DONE"
