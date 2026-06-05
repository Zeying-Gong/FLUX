"""Quick standalone check: do semantic_map_v2 coords land on the right
spot in the Isaac stage?

Usage:
    python debug_coord_alignment.py \
        --json /workspace/SAGE-3D_Official/SAGE-3D_data/semantic_maps_v2/2D_Semantic_Map_0044_839926_Complete.json
"""
import argparse, json
from collections import Counter

p = argparse.ArgumentParser()
p.add_argument("--json", required=True)
args = p.parse_args()

with open(args.json) as f:
    raw = json.load(f)

if isinstance(raw, dict) and "instances" in raw:
    meta  = raw.get("meta", {})
    items = raw["instances"]
    print(f"[Format] NEW (has meta).")
    print(f"[Meta]   x∈[{meta.get('x_min'):.3f}, {meta.get('x_max'):.3f}]  "
          f"y∈[{meta.get('y_min'):.3f}, {meta.get('y_max'):.3f}]  "
          f"scale={meta.get('scale')}  size={meta.get('width')}x{meta.get('height')}")
    flip_cx = meta["x_min"] + meta["x_max"]
    flip_cy = meta["y_min"] + meta["y_max"]
else:
    items = raw
    print("[Format] LEGACY (bare list — no meta).")
    all_y, all_x = [], []
    for inst in items:
        for y, x in inst.get("mask_coords_m", []):
            all_y.append(float(y)); all_x.append(float(x))
    flip_cx = min(all_x) + max(all_x)
    flip_cy = min(all_y) + max(all_y)
    print(f"[Stat]   flip_cx={flip_cx:.3f}  flip_cy={flip_cy:.3f}  (from mask stats)")

# Label histogram
cnt = Counter(str(it.get("category_label","")).lower() for it in items)
print(f"\n[Labels] total {len(items)} instances:")
for lbl, n in cnt.most_common():
    print(f"   {n:4d}  {lbl}")

# Walls = your alignment ground truth
walls = [it for it in items if str(it.get("category_label","")).lower() == "wall"]
print(f"\n[Walls]  {len(walls)} wall blocks. First 3 transformed (sm → Isaac, negate_xy=True):")
for w in walls[:3]:
    try:
        xl, yb, xr, yt = [float(v) for v in w["bbox_m"]]
    except Exception:
        continue
    # sm → world: mirror about flip centre
    wxl, wxr = flip_cx - xr, flip_cx - xl      # flip swaps min/max
    wyb, wyt = flip_cy - yt, flip_cy - yb
    # world → Isaac: negate
    ixl, ixr = -wxr, -wxl
    iyb, iyt = -wyt, -wyb
    print(f"   wall id={w.get('instance_id','?'):>10s}  "
          f"sm_bbox=({xl:+.2f},{yb:+.2f},{xr:+.2f},{yt:+.2f})  →  "
          f"isaac_bbox=({ixl:+.2f},{iyb:+.2f},{ixr:+.2f},{iyt:+.2f})  "
          f"centre=({0.5*(ixl+ixr):+.2f},{0.5*(iyb+iyt):+.2f})")

excl = sum(1 for it in items
           if any(k in str(it.get("category_label","")).lower()
                  for k in ("table","chair","sofa","bed","wardrobe","desk","counter","cabinet")))
print(f"\n[Exclude target] {excl} furniture instances would become exclude volumes.")