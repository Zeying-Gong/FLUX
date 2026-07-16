#!/usr/bin/env python3
"""Build accepted/rejected manifests from tracking episode quality markers."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Tracking output root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--status", choices=("accepted", "rejected", "all"), default="accepted"
    )
    args = parser.parse_args()

    rows = []
    for quality_path in sorted(args.root.rglob("quality.json")):
        try:
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"WARN: cannot read {quality_path}: {exc}")
            continue
        status = quality.get("status")
        if args.status != "all" and status != args.status:
            continue
        episode_dir = quality_path.parent
        data_name = (
            "frame_data.npz" if status == "accepted"
            else "frame_data.rejected.npz"
        )
        data_path = episode_dir / "frames" / data_name
        rows.append({
            "episode_dir": str(episode_dir.resolve()),
            "data_path": str(data_path.resolve()),
            "status": status,
            "rejection_reasons": quality.get("rejection_reasons", []),
            "metrics": quality.get("metrics", {}),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} {args.status} episodes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
