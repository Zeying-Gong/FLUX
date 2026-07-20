#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# migrate_legacy.sh
#
# Migrate old-format run dirs (formal__<scene_id>/) to new structure
# (<short_scene>/, no inner scene folder nesting).
#
# Old:  formal_runs/formal__0001_839920/stt/0001_839920/0/go2_zed/
# New:  formal_runs/0001_83992/stt/0/go2_zed/
#
# This is a pure metadata operation (mv on same filesystem = fast).
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/formal_runs"

MIGRATED=0
SKIPPED=0

for old_dir in formal__*/; do
  [ -d "$old_dir" ] || continue
  old_dir="${old_dir%/}"

  # Extract full scene ID from the old dir name
  full_scene="${old_dir#formal__}"       # e.g. 0001_839920
  short_scene="${full_scene:0:10}"       # e.g. 0001_83992

  new_dir="$short_scene"

  if [ "$old_dir" = "$new_dir" ]; then
    echo "[skip] $old_dir  (already correct name)"
    continue
  fi

  if [ -d "$new_dir" ]; then
    echo "[skip] $old_dir → $new_dir  (target already exists, manual merge needed)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  echo "[migrate] $old_dir → $new_dir"

  # 1. Rename top-level directory
  mv "$old_dir" "$new_dir"

  # 2. Flatten inner scene folder for each mode (stt, dt, at) and logs/rejected
  for prefix in "" "logs/" "rejected/"; do
    inner="$new_dir/${prefix}${full_scene}"
    [ -d "$inner" ] || continue

    # Determine the target parent dir (where the flattened content goes)
    parent_dir="$new_dir/${prefix}"
    parent_dir="${parent_dir%/}"  # remove trailing slash if prefix is empty

    echo "    flatten ${prefix}${full_scene}/ → ${prefix}"

    # Move everything from inner folder to parent
    for item in "$inner"/*; do
      [ -e "$item" ] || continue
      target="${parent_dir}/$(basename "$item")"
      if [ -e "$target" ]; then
        echo "      WARNING: $target exists, skip $(basename "$item")"
      else
        mv "$item" "$parent_dir/"
      fi
    done

    # Remove the now-empty inner folder
    rmdir "$inner" 2>/dev/null || true
  done

  MIGRATED=$((MIGRATED + 1))
done

echo ""
echo "Done: $MIGRATED migrated, $SKIPPED skipped."
