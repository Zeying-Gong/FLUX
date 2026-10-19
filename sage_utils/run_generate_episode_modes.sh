#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# run_generate_episode_modes.sh
# Generate tracking episodes with auto STT/DT/AT mode assignment.
#
# Mode is determined by number of people per episode:
#   1 person  →  STT  (Single‑Target Tracking)
#   >1 person →  50% DT (Distracted Tracking), 50% AT (Ambiguity Tracking)
#
# Usage:
#   ./run_generate_episode_modes.sh --semantic_map_json <path> [options]
#
# Examples:
#   # 100 episodes, 30% STT (1 person), 35% DT, 35% AT
#   ./run_generate_episode_modes.sh --semantic_map_json /path/to/map.json \
#       --num_people 6 --single_ratio 0.3 \
#       --episode_ids $(seq 0 99)
#
#   # Only DT/AT (no STT), larger crowds for DT
#   ./run_generate_episode_modes.sh --semantic_map_json /path/to/map.json \
#       --single_ratio 0.0 --num_people 8 \
#       --episode_ids $(seq 0 99)
#
#   # All STT (debugging / baseline)
#   ./run_generate_episode_modes.sh --semantic_map_json /path/to/map.json \
#       --single_ratio 1.0 --num_people 6 \
#       --episode_ids $(seq 0 19)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"

exec "$PYTHON" "$SCRIPT_DIR/generate_episode_tracking_modes.py" "$@"
