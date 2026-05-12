"""
compute_scene_areas.py
======================
从 InteriorGS / SAGE-3D 数据集中计算每个场景的面积，
并给出面积分布统计和行人数量推荐。

用法：
    python compute_scene_areas.py --data-root /path/to/InteriorGS

每个场景目录需包含：
    occupancy.json   ← 含 scale、min 字段
    occupancy.png    ← 灰度占用图
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ──────────────────────────────────────────────
# 社交密度 → 行人数量推荐参数（可自行调整）
# ──────────────────────────────────────────────
# 参考标准: HiMari 等人群仿真研究
#   低密度  : 0.05 人/m²  (宽敞、稀疏)
#   中密度  : 0.15 人/m²  (一般居家场景)
#   高密度  : 0.30 人/m²  (繁忙区域、走廊)
DENSITY_LEVELS = {
    "low":    0.05,   # 人/m²
    "medium": 0.15,
    "high":   0.30,
}

# 面积分级（m²）
AREA_BINS = [0, 20, 50, 100, 200, 500, float("inf")]
AREA_LABELS = ["<20", "20-50", "50-100", "100-200", "200-500", "500+"]


def compute_scene_area(scene_dir: Path):
    """
    返回 (total_area_m2, navigable_area_m2) 或 None（若文件缺失）

    - total_area_m2    : 场景总边界框面积 = h * w * scale²
    - navigable_area_m2: 像素值 == 255 的可通行区域面积
    """
    occ_json = scene_dir / "occupancy.json"
    occ_png  = scene_dir / "occupancy.png"

    if not occ_json.exists() or not occ_png.exists():
        return None

    with occ_json.open() as f:
        meta = json.load(f)

    scale = float(meta["scale"])          # 米/像素
    pixel_area = scale ** 2               # 每像素面积 (m²)

    img = Image.open(occ_png).convert("L")
    occ = np.array(img)                   # shape: (H, W), uint8
    h, w = occ.shape

    total_area     = h * w * pixel_area
    navigable_mask = occ == 255
    navigable_area = navigable_mask.sum() * pixel_area

    return total_area, navigable_area


def pedestrian_recommendation(navigable_area: float) -> dict:
    """根据可通行面积，给出各密度等级下推荐的行人数量。"""
    return {
        level: max(1, round(navigable_area * density))
        for level, density in DENSITY_LEVELS.items()
    }


def area_bin(area: float) -> str:
    for i, upper in enumerate(AREA_BINS[1:]):
        if area < upper:
            return AREA_LABELS[i]
    return AREA_LABELS[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True,
                        help="InteriorGS 根目录，每个子目录为一个场景")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="结果输出目录（CSV + 图表）")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="最多处理场景数（调试用）")
    args = parser.parse_args()

    scene_dirs = sorted(p for p in args.data_root.iterdir() if p.is_dir())
    if args.max_scenes:
        scene_dirs = scene_dirs[:args.max_scenes]

    print(f"找到 {len(scene_dirs)} 个场景，开始计算面积…\n")

    results = []
    errors  = []

    for sd in scene_dirs:
        ret = compute_scene_area(sd)
        if ret is None:
            errors.append(sd.name)
            print(f"  [SKIP] {sd.name}：缺少 occupancy.json / occupancy.png")
            continue
        total_area, nav_area = ret
        ped = pedestrian_recommendation(nav_area)
        results.append({
            "scene_id":           sd.name,
            "total_area_m2":      round(total_area, 2),
            "navigable_area_m2":  round(nav_area, 2),
            "area_bin":           area_bin(nav_area),
            "pedestrians_low":    ped["low"],
            "pedestrians_medium": ped["medium"],
            "pedestrians_high":   ped["high"],
        })
        print(f"  {sd.name:>10}  total={total_area:7.1f} m²  "
              f"navigable={nav_area:7.1f} m²  "
              f"行人 low/mid/high = {ped['low']}/{ped['medium']}/{ped['high']}")

    if not results:
        print("未成功处理任何场景，请检查路径。")
        return

    # ── 统计 ──
    nav_areas = np.array([r["navigable_area_m2"] for r in results])
    print(f"""
━━━━━━━━━━━━━━━━━━━ 场景面积统计 ━━━━━━━━━━━━━━━━━━━
  有效场景数  : {len(results)}
  跳过场景数  : {len(errors)}
  最小面积    : {nav_areas.min():.1f} m²
  最大面积    : {nav_areas.max():.1f} m²
  平均面积    : {nav_areas.mean():.1f} m²
  中位数面积  : {np.median(nav_areas):.1f} m²
  标准差      : {nav_areas.std():.1f} m²
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

面积分布（可通行面积 m²）：""")

    from collections import Counter
    bin_counter = Counter(r["area_bin"] for r in results)
    for label in AREA_LABELS:
        cnt = bin_counter.get(label, 0)
        bar = "█" * (cnt * 30 // max(bin_counter.values(), default=1))
        print(f"  {label:>10} m²  | {bar:<30} {cnt}")

    # ── 保存 CSV ──
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "scene_areas.csv"
    import csv
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV 已保存至: {csv_path}")

    # ── 绘图 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 直方图
    ax = axes[0]
    ax.hist(nav_areas, bins=30, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("可通行面积 (m²)", fontsize=12)
    ax.set_ylabel("场景数量", fontsize=12)
    ax.set_title("SAGE-3D 场景可通行面积分布", fontsize=13)
    ax.axvline(nav_areas.mean(),   color="tomato",  linestyle="--", label=f"均值 {nav_areas.mean():.0f} m²")
    ax.axvline(np.median(nav_areas), color="gold", linestyle="--", label=f"中位数 {np.median(nav_areas):.0f} m²")
    ax.legend()

    # 面积分级柱状图 + 行人推荐
    ax2 = axes[1]
    labels_present = [lb for lb in AREA_LABELS if bin_counter.get(lb, 0) > 0]
    counts = [bin_counter[lb] for lb in labels_present]
    bars = ax2.bar(labels_present, counts, color="#55A868", edgecolor="white")
    ax2.set_xlabel("面积区间 (m²)", fontsize=12)
    ax2.set_ylabel("场景数量", fontsize=12)
    ax2.set_title("按面积分级的场景数量", fontsize=13)
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(cnt), ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    chart_path = args.output_dir / "scene_area_distribution.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存至: {chart_path}")

    # ── 行人数量推荐表 ──
    print("""
━━━━━━━━━━━━━━━━━━━ 行人数量推荐参考 ━━━━━━━━━━━━━━━━━━━
社交密度参数（人/m²）：
  低密度  (low)    = 0.05  ← 宽敞住宅、走廊
  中密度  (medium) = 0.15  ← 普通室内场景
  高密度  (high)   = 0.30  ← 繁忙公共区域

面积区间推荐行人数（中密度 0.15 人/m² 估算）：""")
    bin_midpoints = {
        "<20":    10,
        "20-50":  35,
        "50-100": 75,
        "100-200":150,
        "200-500":350,
        "500+":   600,
    }
    for lb, mid in bin_midpoints.items():
        lo = max(1, round(mid * DENSITY_LEVELS["low"]))
        mi = max(1, round(mid * DENSITY_LEVELS["medium"]))
        hi = max(1, round(mid * DENSITY_LEVELS["high"]))
        print(f"  {lb:>10} m²  →  low={lo:>3}  medium={mi:>3}  high={hi:>3} 人")


if __name__ == "__main__":
    main()