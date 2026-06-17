"""
探索性数据分析 (EDA)
=====================
对 Store Sales 数据集进行全面的探索性分析，为特征工程和建模提供洞察。

分析维度：
1. 销售趋势分析 — 总体趋势、季节性、增长
2. 商店维度分析 — 不同商店/城市/州的销售模式
3. 商品族分析 — 哪些商品销量高/波动大
4. 促销效果分析 — onpromotion对销售的影响
5. 节假日效应 — 不同节假日对销售的影响
6. 油价影响 — 厄瓜多尔油价与零售的关系
7. 相关性分析 — 各特征之间的关联
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from typing import Optional

from config import FIGURES_DIR, SEED

# 设置中文显示（如果系统支持）
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

# 配色方案
COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]


def run_full_eda(df: pd.DataFrame, raw: dict, save: bool = True):
    """
    运行完整的EDA流程

    Args:
        df: 合并后的训练DataFrame
        raw: 原始数据字典
        save: 是否保存图表
    """
    print("=" * 70)
    print("  Store Sales — 探索性数据分析 (EDA)")
    print("=" * 70)

    analysis_1_sales_overview(df, save)
    analysis_2_store_analysis(df, raw, save)
    analysis_3_family_analysis(df, save)
    analysis_4_promotion_effect(df, save)
    analysis_5_holiday_effect(df, raw, save)
    analysis_6_oil_impact(df, save)
    analysis_7_seasonal_decomposition(df, save)
    analysis_8_correlation(df, save)

    print("\nEDA完成！所有图表已保存至:", FIGURES_DIR)


# ============================================================
# 分析1: 销售总览
# ============================================================
def analysis_1_sales_overview(df: pd.DataFrame, save: bool = True):
    print("\n[1/8] 销售总览...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1.1 每日总销售额趋势
    daily_sales = df.groupby("date")["sales"].sum()
    ax = axes[0, 0]
    ax.plot(daily_sales.index, daily_sales.values,
            color=COLORS[0], linewidth=0.8, alpha=0.8)
    ax.set_title("每日总销售额 (2013-2017)", fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("销售额")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)

    # 1.2 月销售额趋势
    monthly_sales = df.set_index("date").resample("ME")["sales"].sum()
    ax = axes[0, 1]
    ax.bar(range(len(monthly_sales)), monthly_sales.values,
           color=COLORS[1], width=0.8)
    ax.set_title("月度总销售额", fontweight="bold")
    ax.set_xlabel("月份序号")
    ax.set_ylabel("销售额")
    ax.grid(True, alpha=0.3, axis="y")

    # 1.3 销售额分布
    ax = axes[1, 0]
    sample = df["sales"].sample(100000, random_state=SEED)
    ax.hist(sample, bins=100, color=COLORS[2], alpha=0.7, edgecolor="white")
    ax.set_title(f"销售额分布 (采样100K, 原始均值={df['sales'].mean():.1f})", fontweight="bold")
    ax.set_xlabel("销售额")
    ax.set_ylabel("频数")
    ax.set_xlim(0, sample.quantile(0.99))  # 截断极端值

    # 1.4 对数销售额分布
    ax = axes[1, 1]
    log_sales = np.log1p(sample[sample["sales"] > 0])
    ax.hist(log_sales, bins=80, color=COLORS[3], alpha=0.7, edgecolor="white")
    ax.set_title(f"log(1+销售额) 分布 (均值={log_sales.mean():.2f})", fontweight="bold")
    ax.set_xlabel("log(1+sales)")
    ax.set_ylabel("频数")

    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / "eda_01_sales_overview.png")
    plt.close()
    print("  ✓ 完成")


# ============================================================
# 分析2: 商店维度分析
# ============================================================
def analysis_2_store_analysis(df: pd.DataFrame, raw: dict, save: bool = True):
    print("\n[2/8] 商店维度分析...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 2.1 各商店总销售额排名
    store_sales = df.groupby("store_nbr")["sales"].sum().sort_values(ascending=False)
    ax = axes[0, 0]
    colors_bar = [COLORS[0] if i < 5 else COLORS[1] for i in range(len(store_sales))]
    ax.bar(range(len(store_sales)), store_sales.values / 1e6, color=colors_bar)
    ax.set_title("各商店总销售额排名", fontweight="bold")
    ax.set_xlabel("商店编号")
    ax.set_ylabel("销售额 (百万)")
    ax.grid(True, alpha=0.3, axis="y")

    # 2.2 不同城市销售额对比
    city_sales = df.groupby("city")["sales"].sum().sort_values(ascending=True)
    ax = axes[0, 1]
    ax.barh(range(len(city_sales)), city_sales.values / 1e6, color=COLORS[1])
    ax.set_yticks(range(len(city_sales)))
    ax.set_yticklabels(city_sales.index)
    ax.set_title("各城市总销售额", fontweight="bold")
    ax.set_xlabel("销售额 (百万)")

    # 2.3 不同商店类型对比
    type_sales = df.groupby("type")["sales"].mean()
    ax = axes[0, 2]
    ax.bar(type_sales.index, type_sales.values, color=COLORS[2:5])
    ax.set_title("不同商店类型平均日销售额", fontweight="bold")
    ax.set_xlabel("商店类型 (A/B/C/D/E)")
    ax.set_ylabel("平均日销售额")

    # 2.4 不同集群销售额
    cluster_sales = df.groupby("cluster")["sales"].mean().sort_index()
    ax = axes[1, 0]
    ax.bar(cluster_sales.index, cluster_sales.values, color=COLORS[3])
    ax.set_title("不同集群平均日销售额", fontweight="bold")
    ax.set_xlabel("集群编号")
    ax.set_ylabel("平均日销售额")

    # 2.5 商店数最多的城市
    store_counts = raw["stores"]["city"].value_counts().head(10)
    ax = axes[1, 1]
    ax.barh(range(len(store_counts)), store_counts.values, color=COLORS[4])
    ax.set_yticks(range(len(store_counts)))
    ax.set_yticklabels(store_counts.index)
    ax.set_title("商店数量 Top10 城市", fontweight="bold")
    ax.set_xlabel("商店数")

    # 2.6 商店类型分布
    type_counts = raw["stores"]["type"].value_counts()
    ax = axes[1, 2]
    colors_pie = COLORS[:len(type_counts)]
    ax.pie(type_counts.values, labels=type_counts.index, autopct="%1.1f%%",
           colors=colors_pie, startangle=90)
    ax.set_title("商店类型分布", fontweight="bold")

    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / "eda_02_store_analysis.png")
    plt.close()
    print("  ✓ 完成")


# ============================================================
# 分析3: 商品族分析
# ============================================================
def analysis_3_family_analysis(df: pd.DataFrame, save: bool = True):
    print("\n[3/8] 商品族分析...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 3.1 各商品族总销售额
    family_sales = df.groupby("family")["sales"].sum().sort_values(ascending=True)
    ax = axes[0, 0]
    ax.barh(range(len(family_sales)), family_sales.values / 1e6, color=COLORS[1])
    ax.set_yticks(range(len(family_sales)))
    ax.set_yticklabels(family_sales.index, fontsize=9)
    ax.set_title("各商品族总销售额", fontweight="bold")
    ax.set_xlabel("销售额 (百万)")

    # 3.2 各商品族平均日销售额
    family_avg = df.groupby("family")["sales"].mean().sort_values(ascending=True)
    ax = axes[0, 1]
    ax.barh(range(len(family_avg)), family_avg.values, color=COLORS[2])
    ax.set_yticks(range(len(family_avg)))
    ax.set_yticklabels(family_avg.index, fontsize=9)
    ax.set_title("各商品族平均日销售额", fontweight="bold")
    ax.set_xlabel("平均日销售额")

    # 3.3 销售波动（标准差/均值）
    family_cv = (df.groupby("family")["sales"].std() /
                 df.groupby("family")["sales"].mean()).sort_values(ascending=True)
    ax = axes[1, 0]
    ax.barh(range(len(family_cv)), family_cv.values, color=COLORS[0])
    ax.set_yticks(range(len(family_cv)))
    ax.set_yticklabels(family_cv.index, fontsize=9)
    ax.set_title("商品族销售波动 (变异系数)", fontweight="bold")
    ax.set_xlabel("CV = std/mean")

    # 3.4 Top5 商品族的销售趋势
    top5 = family_sales.tail(5).index.tolist()
    ax = axes[1, 1]
    for i, family in enumerate(top5):
        family_daily = df[df["family"] == family].groupby("date")["sales"].sum()
        ax.plot(family_daily.index, family_daily.values,
                label=family, color=COLORS[i], linewidth=1, alpha=0.8)
    ax.set_title("Top5 商品族每日销售趋势", fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("销售额")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / "eda_03_family_analysis.png")
    plt.close()
    print("  ✓ 完成")


# ============================================================
# 分析4: 促销效果
# ============================================================
def analysis_4_promotion_effect(df: pd.DataFrame, save: bool = True):
    print("\n[4/8] 促销效果分析...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 4.1 促销商品占比随时间变化
    promo_rate = df.groupby("date")["onpromotion"].apply(
        lambda x: (x > 0).mean()
    )
    ax = axes[0, 0]
    ax.plot(promo_rate.index, promo_rate.values * 100,
            color=COLORS[0], linewidth=1)
    ax.set_title("促销商品占比变化", fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("促销商品占比 (%)")
    ax.grid(True, alpha=0.3)

    # 4.2 有促销 vs 无促销的销售额分布
    promo_sales = df[df["onpromotion"] > 0]["sales"].sample(50000, random_state=SEED)
    no_promo_sales = df[df["onpromotion"] == 0]["sales"].sample(50000, random_state=SEED)
    ax = axes[0, 1]
    ax.boxplot([no_promo_sales.values, promo_sales.values],
               labels=["无促销", "有促销"],
               patch_artist=True,
               boxprops=dict(facecolor=COLORS[1], alpha=0.6),
               medianprops=dict(color="red", linewidth=2),
               showfliers=False)
    ax.set_title("有/无促销时的销售额分布", fontweight="bold")
    ax.set_ylabel("销售额")
    ax.grid(True, alpha=0.3, axis="y")

    # 4.3 促销强度与销售额的关系
    promo_bins = pd.cut(df["onpromotion"], bins=[0, 1, 5, 10, 20, 50, 100, 800],
                        labels=["0", "1-5", "6-10", "11-20", "21-50", "51-100", "100+"])
    promo_effect = df.groupby(promo_bins, observed=True)["sales"].agg(["mean", "median"])
    ax = axes[1, 0]
    x = range(len(promo_effect))
    ax.plot(x, promo_effect["mean"], "o-", color=COLORS[2], linewidth=2, label="均值")
    ax.plot(x, promo_effect["median"], "s--", color=COLORS[3], linewidth=2, label="中位数")
    ax.set_xticks(x)
    ax.set_xticklabels(promo_effect.index)
    ax.set_title("促销商品数量 vs 销售额", fontweight="bold")
    ax.set_xlabel("促销商品数")
    ax.set_ylabel("销售额")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4.4 促销对各商品族的影响
    family_promo = df.groupby("family").agg(
        promo_effect_ratio=("sales", lambda x:
            x[df.loc[x.index, "onpromotion"] > 0].mean() /
            max(x[df.loc[x.index, "onpromotion"] == 0].mean(), 1))
    ).sort_values("promo_effect_ratio")

    ax = axes[1, 1]
    ax.barh(range(len(family_promo)), family_promo["promo_effect_ratio"].values,
            color=COLORS[4])
    ax.set_yticks(range(len(family_promo)))
    ax.set_yticklabels(family_promo.index, fontsize=8)
    ax.set_title("促销对各商品族的提升倍数", fontweight="bold")
    ax.set_xlabel("促销期/非促销期 销售额比率")
    ax.axvline(x=1, color="red", linestyle="--", alpha=0.7)

    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / "eda_04_promotion_effect.png")
    plt.close()
    print("  ✓ 完成")


# ============================================================
# 分析5: 节假日效应
# ============================================================
def analysis_5_holiday_effect(df: pd.DataFrame, raw: dict, save: bool = True):
    print("\n[5/8] 节假日效应分析...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 5.1 节假日 vs 非节假日销售额
    holiday_sales = df[df["is_holiday"] == 1].groupby("date")["sales"].mean()
    nonholiday_sales = df[df["is_holiday"] == 0].groupby("date")["sales"].mean()

    ax = axes[0, 0]
    ax.boxplot([nonholiday_sales.values, holiday_sales.values],
               labels=["非节假日", "节假日"],
               patch_artist=True,
               boxprops=dict(facecolor=COLORS[1], alpha=0.6),
               medianprops=dict(color="red", linewidth=2))
    ax.set_title("节假日 vs 非节假日日销售额分布", fontweight="bold")
    ax.set_ylabel("日平均销售额")
    ax.grid(True, alpha=0.3, axis="y")

    # 5.2 各节假日类型平均销售额
    holiday_type_avg = df.groupby("holiday_type", observed=True)["sales"].mean().sort_values()
    ax = axes[0, 1]
    ax.barh(range(len(holiday_type_avg)), holiday_type_avg.values, color=COLORS[2])
    ax.set_yticks(range(len(holiday_type_avg)))
    ax.set_yticklabels(holiday_type_avg.index, fontsize=9)
    ax.set_title("不同节假日类型平均销售额", fontweight="bold")
    ax.set_xlabel("平均销售额")

    # 5.3 节假日前后的销售趋势（以圣诞节为例）
    holidays = raw["holidays"]
    christmas = holidays[
        (holidays["description"].str.contains("Navidad|Christmas", case=False, na=False)) &
        (holidays["locale"] == "National")
    ].copy()

    if len(christmas) > 0:
        ax = axes[1, 0]
        for i, christmas_date in enumerate(christmas["date"].iloc[:3]):
            start = christmas_date - pd.Timedelta(days=14)
            end = christmas_date + pd.Timedelta(days=14)
            period_data = df[(df["date"] >= start) & (df["date"] <= end)]
            period_daily = period_data.groupby("date")["sales"].mean()
            days_from = [(d - christmas_date).days for d in period_daily.index]
            ax.plot(days_from, period_daily.values, "o-",
                    color=COLORS[i], linewidth=1.5, markersize=3,
                    label=f'{christmas_date.year}圣诞节')

        ax.set_title("圣诞节前后两周的销售变化", fontweight="bold")
        ax.set_xlabel("距圣诞节天数")
        ax.set_ylabel("平均销售额")
        ax.legend()
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)

    # 5.4 工作日 vs 周末 vs 节假日
    df_temp = df.copy()
    df_temp["day_type"] = "工作日"
    df_temp.loc[df_temp["is_holiday"] == 1, "day_type"] = "节假日"
    df_temp.loc[(df_temp["is_holiday"] == 0) &
                (df_temp["date"].dt.dayofweek >= 5), "day_type"] = "周末"

    day_type_sales = df_temp.groupby("day_type")["sales"].mean()
    ax = axes[1, 1]
    ax.bar(day_type_sales.index, day_type_sales.values,
           color=[COLORS[0], COLORS[5], COLORS[3]])
    ax.set_title("工作日/周末/节假日 平均销售额", fontweight="bold")
    ax.set_xlabel("日期类型")
    ax.set_ylabel("平均销售额")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / "eda_05_holiday_effect.png")
    plt.close()
    print("  ✓ 完成")


# ============================================================
# 分析6: 油价影响
# ============================================================
def analysis_6_oil_impact(df: pd.DataFrame, save: bool = True):
    print("\n[6/8] 油价影响分析...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 6.1 油价走势
    oil_daily = df.groupby("date")["dcoilwtico"].first()
    ax = axes[0, 0]
    ax.plot(oil_daily.index, oil_daily.values,
            color="black", linewidth=1)
    ax.set_title("厄瓜多尔原油价格走势 (WTI)", fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("油价 ($/桶)")
    ax.grid(True, alpha=0.3)

    # 6.2 油价 vs 销售额散点图
    df_sample = df.sample(20000, random_state=SEED)
    ax = axes[0, 1]
    ax.scatter(df_sample["dcoilwtico"], df_sample["sales"],
               c=COLORS[1], alpha=0.3, s=10)
    ax.set_title("油价 vs 销售额 (采样)", fontweight="bold")
    ax.set_xlabel("油价 ($/桶)")
    ax.set_ylabel("销售额")

    # 6.3 按油价分位数的平均销售额
    df_temp = df.copy()
    df_temp["oil_quartile"] = pd.qcut(df_temp["dcoilwtico"].dropna(), q=5,
                                       labels=["Q1(低)", "Q2", "Q3", "Q4", "Q5(高)"])
    oil_sales = df_temp.groupby("oil_quartile", observed=True)["sales"].mean()
    ax = axes[1, 0]
    ax.bar(oil_sales.index, oil_sales.values, color=COLORS[2])
    ax.set_title("不同油价水平下的平均销售额", fontweight="bold")
    ax.set_xlabel("油价分位")
    ax.set_ylabel("平均销售额")
    ax.grid(True, alpha=0.3, axis="y")

    # 6.4 油价与交易量的关系
    oil_monthly = df.set_index("date").resample("ME").agg({
        "dcoilwtico": "first",
        "sales": "mean",
        "transactions": "mean"
    })
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(oil_monthly.index, oil_monthly["dcoilwtico"],
            color="black", linewidth=1.5, label="油价")
    ax2.plot(oil_monthly.index, oil_monthly["transactions"],
             color=COLORS[4], linewidth=1.5, alpha=0.7, label="交易量")
    ax.set_title("油价与交易量的关系", fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("油价 ($/桶)", color="black")
    ax2.set_ylabel("平均交易量", color=COLORS[4])
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / "eda_06_oil_impact.png")
    plt.close()
    print("  ✓ 完成")


# ============================================================
# 分析7: 季节性分解
# ============================================================
def analysis_7_seasonal_decomposition(df: pd.DataFrame, save: bool = True):
    print("\n[7/8] 季节性分解...")

    # 使用移动平均法进行简单的季节性分解
    daily_sales = df.groupby("date")["sales"].sum()

    # 趋势分量（30天移动平均）
    trend = daily_sales.rolling(window=30, center=True).mean()

    # 季节性分量（去除趋势后按周几平均）
    detrended = daily_sales - trend
    seasonal_dayofweek = detrended.groupby(detrended.index.dayofweek).mean()

    # 年度季节性（按月平均）
    monthly_pattern = daily_sales.groupby(daily_sales.index.month).mean()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 7.1 原始序列 + 趋势线
    ax = axes[0, 0]
    ax.plot(daily_sales.index, daily_sales.values,
            color=COLORS[0], linewidth=0.5, alpha=0.5, label="每日销售额")
    ax.plot(trend.index, trend.values,
            color=COLORS[1], linewidth=2, label="30天移动平均(趋势)")
    ax.set_title("销售趋势 (30天移动平均)", fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("销售额")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7.2 周内模式
    days_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    ax = axes[0, 1]
    ax.bar(days_cn, seasonal_dayofweek.values, color=COLORS[2])
    ax.set_title("一周内各天销售偏离趋势的均值", fontweight="bold")
    ax.set_xlabel("星期")
    ax.set_ylabel("偏离趋势的销售额")
    ax.grid(True, alpha=0.3, axis="y")

    # 7.3 月度模式
    months_cn = ["1月", "2月", "3月", "4月", "5月", "6月",
                 "7月", "8月", "9月", "10月", "11月", "12月"]
    ax = axes[1, 0]
    ax.bar(months_cn, monthly_pattern.values, color=COLORS[3])
    ax.set_title("月度销售模式", fontweight="bold")
    ax.set_xlabel("月份")
    ax.set_ylabel("平均销售额")
    ax.grid(True, alpha=0.3, axis="y")

    # 7.4 逐年对比（2013-2017）
    yearly_data = daily_sales.reset_index()
    yearly_data["year"] = yearly_data["date"].dt.year
    yearly_data["dayofyear"] = yearly_data["date"].dt.dayofyear

    ax = axes[1, 1]
    for year in range(2013, 2018):
        year_data = yearly_data[yearly_data["year"] == year]
        year_data = year_data.groupby("dayofyear")["sales"].sum()
        ax.plot(year_data.index, year_data.values,
                label=str(year), linewidth=1.2, alpha=0.8)

    ax.set_title("逐年销售趋势对比", fontweight="bold")
    ax.set_xlabel("一年中的第几天")
    ax.set_ylabel("销售额")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / "eda_07_seasonality.png")
    plt.close()
    print("  ✓ 完成")


# ============================================================
# 分析8: 相关性分析
# ============================================================
def analysis_8_correlation(df: pd.DataFrame, save: bool = True):
    print("\n[8/8] 相关性分析...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 8.1 数值特征相关矩阵
    numeric_cols = ["sales", "onpromotion", "dcoilwtico", "transactions"]
    corr = df[numeric_cols].corr()

    ax = axes[0]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap="RdYlGn",
                center=0, square=True, linewidths=1, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("数值特征相关性矩阵", fontweight="bold")

    # 8.2 特征重要性（通过简单模型评估）
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    ax = axes[1]

    df_sample = df.sample(50000, random_state=SEED).copy()
    features = df_sample[["store_nbr", "onpromotion", "dcoilwtico", "transactions",
                          "cluster", "date"]]
    features["dayofweek"] = features["date"].dt.dayofweek
    features["month"] = features["date"].dt.month
    features["year"] = features["date"].dt.year
    features["is_weekend"] = (features["dayofweek"] >= 5).astype(int)
    features.drop(columns=["date"], inplace=True)

    # 编码类别特征
    for col in ["city", "state", "type", "family"]:
        le = LabelEncoder()
        features[col] = le.fit_transform(df_sample[col].astype(str))

    features = features.fillna(features.median())

    rf = RandomForestRegressor(n_estimators=50, max_depth=10,
                                random_state=SEED, n_jobs=-1)
    rf.fit(features, df_sample["sales"])

    importances = pd.Series(rf.feature_importances_, index=features.columns)
    importances = importances.sort_values()
    ax.barh(range(len(importances)), importances.values, color=COLORS[1])
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(importances.index)
    ax.set_title("Random Forest 特征重要性", fontweight="bold")
    ax.set_xlabel("重要性")

    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / "eda_08_correlation.png")
    plt.close()
    print("  ✓ 完成")


if __name__ == "__main__":
    from data_loader import load_and_merge
    train_df, test_df, raw_data = load_and_merge()
    run_full_eda(train_df, raw_data)
