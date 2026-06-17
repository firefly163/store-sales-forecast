from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report_assets"
FIG = OUT / "figures"
TAB = OUT / "tables"


PALETTE = {
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "green": "#16a34a",
    "teal": "#0f766e",
    "orange": "#ea580c",
    "red": "#dc2626",
    "rose": "#be123c",
    "purple": "#7c3aed",
    "gray": "#64748b",
    "light_grid": "#e2e8f0",
    "dark": "#0f172a",
}


def ensure_dirs() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)


def write_svg(name: str, body: str, width: int = 1000, height: int = 520) -> None:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<style>
text {{ font-family: Arial, 'Microsoft YaHei', sans-serif; fill: {PALETTE['dark']}; }}
.title {{ font-size: 24px; font-weight: 700; }}
.label {{ font-size: 13px; }}
.small {{ font-size: 11px; fill: #475569; }}
.grid {{ stroke: {PALETTE['light_grid']}; stroke-width: 1; }}
</style>
{body}
</svg>
"""
    path = FIG / name
    path.write_text(svg, encoding="utf-8")
    print(f"[figure] {path}")


def fmt_num(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.2f}"


def read_data():
    train = pd.read_csv(ROOT / "train.csv", parse_dates=["date"])
    test = pd.read_csv(ROOT / "test.csv", parse_dates=["date"])
    stores = pd.read_csv(ROOT / "stores.csv")
    oil = pd.read_csv(ROOT / "oil.csv", parse_dates=["date"]).rename(columns={"dcoilwtico": "oil"})
    transactions = pd.read_csv(ROOT / "transactions.csv", parse_dates=["date"])
    holidays = pd.read_csv(ROOT / "holidays_events.csv", parse_dates=["date"])
    return train, test, stores, oil, transactions, holidays


def line_chart(
    name: str,
    title: str,
    x_labels: list[str],
    y_values: list[float],
    y_label: str,
    color: str = "#2563eb",
    width: int = 1050,
    height: int = 520,
) -> None:
    left, right, top, bottom = 80, 35, 70, 75
    chart_w, chart_h = width - left - right, height - top - bottom
    values = np.array(y_values, dtype=float)
    ymin, ymax = float(np.nanmin(values)), float(np.nanmax(values))
    pad = (ymax - ymin) * 0.08 or 1.0
    ymin, ymax = ymin - pad, ymax + pad

    def x_pos(i: int) -> float:
        return left + (chart_w * i / max(len(values) - 1, 1))

    def y_pos(v: float) -> float:
        return top + chart_h - (v - ymin) / (ymax - ymin) * chart_h

    parts = [f'<text x="{width/2}" y="34" text-anchor="middle" class="title">{escape(title)}</text>']
    for j in range(5):
        y = top + chart_h * j / 4
        val = ymax - (ymax - ymin) * j / 4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+chart_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="small">{fmt_num(val)}</text>')
    points = " ".join(f"{x_pos(i):.1f},{y_pos(v):.1f}" for i, v in enumerate(values))
    parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    step = max(1, len(x_labels) // 8)
    for i in range(0, len(x_labels), step):
        parts.append(f'<text x="{x_pos(i):.1f}" y="{height-38}" text-anchor="middle" class="small">{escape(x_labels[i])}</text>')
    parts.append(f'<text x="{left + chart_w/2}" y="{height-12}" text-anchor="middle" class="label">Month</text>')
    parts.append(f'<text x="18" y="{top + chart_h/2}" transform="rotate(-90 18 {top + chart_h/2})" text-anchor="middle" class="label">{escape(y_label)}</text>')
    write_svg(name, "\n".join(parts), width, height)


def bar_chart(
    name: str,
    title: str,
    labels: list[str],
    values: list[float],
    x_label: str,
    color: str = "#0f766e",
    horizontal: bool = True,
    width: int = 1000,
    height: int = 560,
) -> None:
    left, right, top, bottom = (235, 45, 70, 55) if horizontal else (70, 35, 70, 95)
    chart_w, chart_h = width - left - right, height - top - bottom
    vmax = max(values) if values else 1
    parts = [f'<text x="{width/2}" y="34" text-anchor="middle" class="title">{escape(title)}</text>']
    for j in range(5):
        x = left + chart_w * j / 4
        val = vmax * j / 4
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+chart_h}" class="grid"/>')
        if horizontal:
            parts.append(f'<text x="{x:.1f}" y="{top+chart_h+22}" text-anchor="middle" class="small">{fmt_num(val)}</text>')

    if horizontal:
        bar_h = chart_h / max(len(labels), 1) * 0.68
        for i, (label, value) in enumerate(zip(labels, values)):
            y = top + chart_h * i / len(labels) + (chart_h / len(labels) - bar_h) / 2
            w = chart_w * value / vmax
            parts.append(f'<text x="{left-8}" y="{y+bar_h*0.65:.1f}" text-anchor="end" class="small">{escape(str(label))}</text>')
            parts.append(f'<rect x="{left}" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="2"/>')
            parts.append(f'<text x="{left+w+6:.1f}" y="{y+bar_h*0.65:.1f}" class="small">{fmt_num(value)}</text>')
        parts.append(f'<text x="{left + chart_w/2}" y="{height-10}" text-anchor="middle" class="label">{escape(x_label)}</text>')
    else:
        bar_w = chart_w / max(len(labels), 1) * 0.65
        for i, (label, value) in enumerate(zip(labels, values)):
            x = left + chart_w * i / len(labels) + (chart_w / len(labels) - bar_w) / 2
            h = chart_h * value / vmax
            y = top + chart_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" rx="2"/>')
            parts.append(f'<text x="{x+bar_w/2:.1f}" y="{top+chart_h+18}" text-anchor="middle" class="small">{escape(str(label))}</text>')
            parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle" class="small">{fmt_num(value)}</text>')
        parts.append(f'<text x="{left + chart_w/2}" y="{height-14}" text-anchor="middle" class="label">{escape(x_label)}</text>')
    write_svg(name, "\n".join(parts), width, height)


def scatter_chart(name: str, title: str, x_values, y_values, x_label: str, y_label: str) -> None:
    width, height = 900, 560
    left, right, top, bottom = 85, 35, 70, 70
    chart_w, chart_h = width - left - right, height - top - bottom
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xpad, ypad = (xmax - xmin) * 0.08 or 1.0, (ymax - ymin) * 0.08 or 1.0
    xmin, xmax, ymin, ymax = xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad

    def px(v): return left + (v - xmin) / (xmax - xmin) * chart_w
    def py(v): return top + chart_h - (v - ymin) / (ymax - ymin) * chart_h

    parts = [f'<text x="{width/2}" y="34" text-anchor="middle" class="title">{escape(title)}</text>']
    for j in range(5):
        gx = left + chart_w * j / 4
        gy = top + chart_h * j / 4
        parts.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top+chart_h}" class="grid"/>')
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left+chart_w}" y2="{gy:.1f}" class="grid"/>')
    for xv, yv in zip(x, y):
        parts.append(f'<circle cx="{px(xv):.1f}" cy="{py(yv):.1f}" r="4" fill="{PALETTE["cyan"]}" fill-opacity="0.72"/>')
    parts.append(f'<text x="{left + chart_w/2}" y="{height-12}" text-anchor="middle" class="label">{escape(x_label)}</text>')
    parts.append(f'<text x="18" y="{top + chart_h/2}" transform="rotate(-90 18 {top + chart_h/2})" text-anchor="middle" class="label">{escape(y_label)}</text>')
    write_svg(name, "\n".join(parts), width, height)


def histogram_chart(name: str, title: str, series: list[tuple[str, np.ndarray]]) -> None:
    width, height = 1000, 560
    left, right, top, bottom = 75, 35, 70, 65
    chart_w, chart_h = width - left - right, height - top - bottom
    max_x = max(values.max() for _, values in series)
    bins = np.linspace(0, max_x, 55)
    hist_data = [(label, np.histogram(values, bins=bins, density=True)[0]) for label, values in series]
    ymax = max(hist.max() for _, hist in hist_data)
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["rose"]]
    parts = [f'<text x="{width/2}" y="34" text-anchor="middle" class="title">{escape(title)}</text>']
    for j in range(5):
        y = top + chart_h * j / 4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+chart_w}" y2="{y:.1f}" class="grid"/>')
    for idx, (label, hist) in enumerate(hist_data):
        pts = []
        for i, h in enumerate(hist):
            x = left + chart_w * i / max(len(hist) - 1, 1)
            y = top + chart_h - h / ymax * chart_h
            pts.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[idx % len(colors)]}" stroke-width="2.4"/>')
        lx, ly = left + 15, top + 18 + idx * 20
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+24}" y2="{ly}" stroke="{colors[idx % len(colors)]}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+32}" y="{ly+4}" class="small">{escape(label)}</text>')
    parts.append(f'<text x="{left + chart_w/2}" y="{height-12}" text-anchor="middle" class="label">log1p(predicted sales)</text>')
    parts.append(f'<text x="20" y="{top + chart_h/2}" transform="rotate(-90 20 {top + chart_h/2})" text-anchor="middle" class="label">Density</text>')
    write_svg(name, "\n".join(parts), width, height)


def workflow_diagram() -> None:
    width, height = 1100, 520
    boxes = [
        ("Raw CSVs\\ntrain/test/stores/oil\\ntransactions/holidays", 75, 100, "#dbeafe"),
        ("Feature Processing\\ncalendar, oil interpolation\\nholidays, promotions", 300, 100, "#e0f2fe"),
        ("Self V7 Baseline\\nfeature engineering\\n+ tree models", 525, 100, "#fef3c7"),
        ("Public Repro\\nDarts global models\\nLGBM/XGB", 750, 100, "#dcfce7"),
        ("Optimization\\ncomponent split\\nlog blend, weight search", 525, 325, "#fae8ff"),
        ("Final Submission\\npublic score 0.37927", 750, 325, "#fee2e2"),
    ]
    parts = [f'<text x="{width/2}" y="38" text-anchor="middle" class="title">Experiment Workflow</text>']
    for text, x, y, color in boxes:
        parts.append(f'<rect x="{x}" y="{y}" width="180" height="105" fill="{color}" stroke="#334155" rx="12"/>')
        for j, line in enumerate(text.split("\\n")):
            parts.append(f'<text x="{x+90}" y="{y+32+j*21}" text-anchor="middle" class="label">{escape(line)}</text>')
    arrows = [
        (255, 152, 300, 152),
        (480, 152, 525, 152),
        (705, 152, 750, 152),
        (840, 205, 615, 325),
        (705, 377, 750, 377),
    ]
    for x1, y1, x2, y2 in arrows:
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#334155" stroke-width="2" marker-end="url(#arrow)"/>')
    defs = '<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#334155"/></marker></defs>'
    write_svg("experiment_workflow.svg", defs + "\n".join(parts), width, height)


def write_dataset_overview(train, test, stores, oil, transactions, holidays):
    rows = [
        ["train.csv rows", len(train)],
        ["test.csv rows", len(test)],
        ["stores", stores["store_nbr"].nunique()],
        ["families", train["family"].nunique()],
        ["store-family series", train.groupby(["store_nbr", "family"]).ngroups],
        ["train date min", train["date"].min().date().isoformat()],
        ["train date max", train["date"].max().date().isoformat()],
        ["test date min", test["date"].min().date().isoformat()],
        ["test date max", test["date"].max().date().isoformat()],
        ["oil rows", len(oil)],
        ["transactions rows", len(transactions)],
        ["holidays rows", len(holidays)],
    ]
    path = TAB / "dataset_overview.csv"
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(path, index=False)
    print(f"[table] {path}")


def generate_assets() -> None:
    train, test, stores, oil, transactions, holidays = read_data()
    write_dataset_overview(train, test, stores, oil, transactions, holidays)

    monthly = (
        train.assign(month=train["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .agg(sales=("sales", "sum"), onpromotion=("onpromotion", "sum"))
    )
    monthly.to_csv(TAB / "monthly_sales.csv", index=False)
    line_chart(
        "monthly_sales_trend.svg",
        "Monthly Total Sales Trend",
        monthly["month"].dt.strftime("%Y-%m").tolist(),
        (monthly["sales"] / 1e6).tolist(),
        "Sales (million)",
        color=PALETTE["blue"],
    )

    fam = train.groupby("family", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    fam.to_csv(TAB / "family_sales_total.csv", index=False)
    top = fam.head(15).sort_values("sales")
    bar_chart(
        "top_family_sales.svg",
        "Top 15 Product Families by Total Sales",
        top["family"].tolist(),
        (top["sales"] / 1e6).tolist(),
        "Sales (million)",
        color=PALETTE["teal"],
        horizontal=True,
        height=650,
    )

    type_counts = stores["type"].value_counts().sort_index()
    type_counts.rename_axis("type").reset_index(name="store_count").to_csv(TAB / "store_type_counts.csv", index=False)
    bar_chart(
        "store_type_counts.svg",
        "Store Count by Type",
        type_counts.index.astype(str).tolist(),
        type_counts.values.tolist(),
        "Store type",
        color=PALETTE["purple"],
        horizontal=False,
        width=820,
        height=480,
    )

    cluster_counts = stores["cluster"].value_counts().sort_index()
    cluster_counts.rename_axis("cluster").reset_index(name="store_count").to_csv(TAB / "store_cluster_counts.csv", index=False)
    bar_chart(
        "store_cluster_counts.svg",
        "Store Count by Cluster",
        cluster_counts.index.astype(str).tolist(),
        cluster_counts.values.tolist(),
        "Cluster",
        color=PALETTE["orange"],
        horizontal=False,
        width=1000,
        height=500,
    )

    monthly_oil = oil.assign(month=oil["date"].dt.to_period("M").dt.to_timestamp()).groupby("month", as_index=False)["oil"].mean()
    monthly_combo = monthly.merge(monthly_oil, on="month", how="left")
    monthly_combo.to_csv(TAB / "monthly_sales_promo_oil.csv", index=False)
    line_chart(
        "sales_oil_trend.svg",
        "Average Monthly Oil Price",
        monthly_combo["month"].dt.strftime("%Y-%m").tolist(),
        monthly_combo["oil"].interpolate(limit_direction="both").tolist(),
        "Oil price",
        color=PALETTE["red"],
    )
    scatter_chart(
        "promotion_vs_sales.svg",
        "Monthly Promotion Count vs Sales",
        (monthly_combo["onpromotion"] / 1e6).tolist(),
        (monthly_combo["sales"] / 1e6).tolist(),
        "On-promotion count (million)",
        "Sales (million)",
    )

    holiday_counts = holidays["type"].value_counts().rename_axis("holiday_type").reset_index(name="count")
    holiday_counts.to_csv(TAB / "holiday_type_counts.csv", index=False)
    hp = holiday_counts.sort_values("count")
    bar_chart(
        "holiday_type_counts.svg",
        "Holiday/Event Records by Type",
        hp["holiday_type"].tolist(),
        hp["count"].tolist(),
        "Records",
        color=PALETTE["rose"],
        horizontal=True,
        width=850,
        height=500,
    )

    score_rows = [
        ["Self-developed V7 feature model", 0.40000, "self"],
        ["Chong Darts-LGBM repro", 0.37984, "public_repro"],
        ["Nina h-blend repro", 0.37946, "public_repro"],
        ["xiewenwei public output", 0.37936, "public_repro"],
        ["xiewenwei + hblend log", 0.37932, "blend"],
        ["+ small LGBM component", 0.37931, "blend"],
        ["Local component weight search", 0.37927, "own_optimization"],
    ]
    scores = pd.DataFrame(score_rows, columns=["stage", "public_score", "category"])
    scores.to_csv(TAB / "score_progression.csv", index=False)
    line_chart(
        "score_progression.svg",
        "Public Leaderboard Score Improvement",
        [str(i + 1) for i in range(len(scores))],
        scores["public_score"].tolist(),
        "RMSLE (lower is better)",
        color=PALETTE["green"],
    )

    weights = pd.DataFrame(
        [["lgb_full", 0.325], ["lgb_2015", 0.325], ["xgb_full", 0.350], ["xgb_2015", 0.000]],
        columns=["component", "weight"],
    )
    weights.to_csv(TAB / "best_component_weights.csv", index=False)
    bar_chart(
        "component_weights.svg",
        "Best Local Component Weights",
        weights["component"].tolist(),
        weights["weight"].tolist(),
        "Component",
        color=PALETTE["blue"],
        horizontal=False,
        width=820,
        height=480,
    )

    candidates = {
        "hblend_0.37946": ROOT / "_kaggle_research" / "hblend_outputs_v2" / "submission_hblend_037946.csv",
        "xiewenwei_0.37936": ROOT / "_kaggle_research" / "score_top_xiewenwei_output" / "submission",
        "xw_hb_log_0.37932": ROOT / "score_push_blends" / "submission_xw_hb_log_w05.csv",
    }
    series = []
    stats = []
    for label, path in candidates.items():
        if path.exists():
            df = pd.read_csv(path)
            values = df["sales"].clip(lower=0)
            stats.append(
                {
                    "submission": label,
                    "rows": len(values),
                    "mean_sales": values.mean(),
                    "median_sales": values.median(),
                    "zero_rows": int((values == 0).sum()),
                    "max_sales": values.max(),
                }
            )
            series.append((label, np.log1p(values.to_numpy())))
    if series:
        pd.DataFrame(stats).to_csv(TAB / "submission_distribution_stats.csv", index=False)
        histogram_chart("submission_prediction_distribution.svg", "Submission Prediction Distribution", series)

    workflow_diagram()
    write_notes()


def write_notes() -> None:
    notes = """# Store Sales 实验报告素材说明

## 图片素材

所有图片在 `report_assets/figures/` 目录，格式为 SVG。Word/WPS/Typora 通常可以直接插入 SVG；如果老师要求 PNG，可以用浏览器打开 SVG 后另存/截图。

- `experiment_workflow.svg`：实验流程图，适合放在“解题思路”。
- `monthly_sales_trend.svg`：月度销量趋势，适合放在“数据分析”。
- `top_family_sales.svg`：Top 15 商品品类销量，说明品类差异。
- `store_type_counts.svg`、`store_cluster_counts.svg`：门店结构。
- `sales_oil_trend.svg`：油价趋势，说明外部变量。
- `promotion_vs_sales.svg`：促销数量与销量关系。
- `holiday_type_counts.svg`：节假日记录类型。
- `score_progression.svg`：从 V7 到最终 0.37927 的成绩提升。
- `component_weights.svg`：四组件权重搜索结果。
- `submission_prediction_distribution.svg`：提交预测分布对比。

## 表格素材

所有表格在 `report_assets/tables/` 目录：

- `dataset_overview.csv`：数据集规模和日期范围。
- `score_progression.csv`：成绩迭代表。
- `best_component_weights.csv`：最终组件权重。
- `monthly_sales.csv`：月度销量。
- `family_sales_total.csv`：各 family 总销量。
- `submission_distribution_stats.csv`：提交预测统计。

## 建议报告结构

1. 赛题介绍：说明任务是预测 Ecuador Favorita 门店未来 16 天各商品 family 销量，评价指标为 RMSLE。
2. 数据分析：放数据集概览、月度销量趋势、Top family、门店结构、节假日/促销/油价图。
3. 自研阶段：介绍 V7 表格特征工程模型，最高 public score 约 0.4。
4. 复现与改进：说明参考公开高分方案，拆分 LGBM/XGB full/2015 四组件，使用 log blend 和本地权重搜索。
5. 实验结果：放成绩迭代表和组件权重图，最终 public score 0.37927。
6. 总结：承认参考公开 notebook，但强调自己的工作是复现、工程适配、组件拆解和权重优化。
"""
    path = OUT / "report_notes.md"
    path.write_text(notes, encoding="utf-8")
    print(f"[notes] {path}")


def main() -> None:
    ensure_dirs()
    generate_assets()
    print(f"\nDone. Report assets written to: {OUT}")


if __name__ == "__main__":
    main()
