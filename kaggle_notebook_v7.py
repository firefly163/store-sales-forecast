# ════════════════════════════════════════════════════════════════
# Store Sales V7 — V5 × 0.375 融合版
#
# 保留 V5 的全部成熟架构：
#   - LGB×3 + XGB×2 + CatBoost ensemble
#   - 时间序列特征 (lags/rolling/EWM/momentum)
#   - 油价、节假日、地震特征
#   - 交互统计、促销特征
#   - 后处理 pipeline
#
# 融入 0.375 方案的关键特征：
#   1. Periodogram 驱动傅里叶频率 [3.5, 7, 30, 365]（替代拍脑袋 sin/cos）
#   2. 趋势特征 (Trend + Trend² + Family×Trend + Cluster×Trend)
#   3. Rank 编码 (family_rank + store_rank + cluster_rank)
#   4. STL 风格分解 (trend/seasonal/resid/strength/chg)
#   5. 异常值 clip (per store-family 99.5%)
#   6. 扩展验证窗口 (31天 vs 16天)
#   7. 关店检测 + 节假日类型展开 + 促销×节假日交互
#   8. 发薪日特征 (15号/月末)
# ════════════════════════════════════════════════════════════════

import os
import gc
import warnings
import zipfile

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，Kaggle 必须
import matplotlib.pyplot as plt
import seaborn as sns

# 全局绘图配置
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
sns.set_style("whitegrid")
sns.set_palette("Set2")

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb


# ════════════════════════════════════════════════════════════════
# 0. 基础配置 + 内存优化
# ════════════════════════════════════════════════════════════════

N_TRIALS = 25
USE_GPU = os.environ.get("USE_GPU", "1") == "1"
USE_PSEUDO = os.environ.get("USE_PSEUDO", "0") == "1"
OUT = os.environ.get("STORE_SALES_OUT", "/kaggle/working")
REPORT_DIR = os.path.join(OUT, "report")
os.makedirs(os.path.join(REPORT_DIR, "eda"), exist_ok=True)
os.makedirs(os.path.join(REPORT_DIR, "features"), exist_ok=True)
os.makedirs(os.path.join(REPORT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(REPORT_DIR, "validation"), exist_ok=True)
print(f"Report figures will be saved to: {REPORT_DIR}")
print(f"USE_GPU={USE_GPU}, USE_PSEUDO={USE_PSEUDO}, OUT={OUT}")


def save_and_log(fig, subdir, filename):
    """保存图表到指定子目录，关闭 fig 释放内存"""
    path = os.path.join(REPORT_DIR, subdir, filename)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  [SAVED] {subdir}/{filename}")
    return path


def safe_plot(fn, desc=""):
    """安全绘图包装器：失败不中断主流程"""
    try:
        fn()
    except Exception as e:
        print(f"  [WARN] Plot '{desc}' failed: {e}")


def assert_unique_ids(df, stage):
    """Fail fast when a feature merge duplicates train/test rows."""
    if "id" not in df.columns:
        return
    if df["id"].is_unique:
        return
    dup_cols = [c for c in ["id", "date", "store_nbr", "family"] if c in df.columns]
    dup = df.loc[df["id"].duplicated(keep=False), dup_cols].head(30)
    print(f"\n[DUPLICATE IDS] {stage}")
    print(dup.to_string(index=False))
    raise AssertionError(f"{stage} 导致 id 重复")


def reduce_mem_usage(df, name="df"):
    """将 DataFrame 中 float64→float32, int64→int32（安全下转，跳过 NaN 风险）"""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "float64":
            df[col] = df[col].astype("float32")
        elif col_type == "int64":
            # 只用 int32 下转（安全，不会因为 NaN 报错）
            df[col] = df[col].astype("int32")
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    pct = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
    print(f"  [{name}] {start_mem:.1f}MB → {end_mem:.1f}MB ({pct:.1f}%↓)")
    return df

DATA = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train.csv" in files and "test.csv" in files:
        DATA = root
        break

if DATA is None:
    import kagglehub
    DATA = kagglehub.competition_download("store-sales-time-series-forecasting")

print(f"Data path: {DATA}")


def rmsle(y_true, y_pred):
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def predict_xgb(model, dmatrix):
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None:
        try:
            return model.predict(dmatrix, iteration_range=(0, best_iter + 1))
        except Exception as e:
            print(f"  [WARN] predict_xgb iteration_range failed: {e}, falling back")
            return model.predict(dmatrix)
    return model.predict(dmatrix)


# ════════════════════════════════════════════════════════════════
# 1. 读取数据
# ════════════════════════════════════════════════════════════════

print("=" * 70)
print("Store Sales V7 — V5 × 0.375 融合版")
print("=" * 70)

train = pd.read_csv(
    f"{DATA}/train.csv", parse_dates=["date"],
    dtype={"store_nbr": "int16", "family": "category",
           "sales": "float32", "onpromotion": "int32"}
)
test = pd.read_csv(
    f"{DATA}/test.csv", parse_dates=["date"],
    dtype={"store_nbr": "int16", "family": "category",
           "onpromotion": "int32"}
)
stores = pd.read_csv(f"{DATA}/stores.csv").rename(columns={"type": "store_type"})
oil = pd.read_csv(f"{DATA}/oil.csv", parse_dates=["date"])
holidays = pd.read_csv(f"{DATA}/holidays_events.csv", parse_dates=["date"])

print(f"train: {train.shape}, test: {test.shape}")


# ════════════════════════════════════════════════════════════════
# 2. 合并 + 基础 merge
# ════════════════════════════════════════════════════════════════

print("\n=== 2. 合并数据 ===")

train["is_train"] = 1
test["is_train"] = 0
test["sales"] = np.nan

data = pd.concat([train, test], ignore_index=True)
data = data.merge(stores, on="store_nbr", how="left")
assert_unique_ids(data, "stores merge")
print("combined:", data.shape)
data = reduce_mem_usage(data, "合并后")
gc.collect()


# ════════════════════════════════════════════════════════════════
# 2.5 EDA 可视化
# ════════════════════════════════════════════════════════════════

print("\n--- 2.5 EDA 可视化 ---")


def plot_eda():
    train_only = data[data["is_train"] == 1].copy()
    train_only["dayofweek"] = train_only["date"].dt.dayofweek.astype("int8")
    train_only["year"] = train_only["date"].dt.year.astype("int16")
    train_only["month"] = train_only["date"].dt.month.astype("int8")

    # --- 2.5.1 销量分布 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    sales_plot = train_only["sales"].clip(upper=train_only["sales"].quantile(0.999))
    ax.hist(sales_plot, bins=100, color="#2196F3", alpha=0.7, edgecolor="white", lw=0.3)
    ax.axvline(train_only["sales"].mean(), color="red", ls="--", lw=1.2,
               label=f"Mean={train_only['sales'].mean():.1f}")
    ax.axvline(train_only["sales"].median(), color="green", ls="--", lw=1.2,
               label=f"Median={train_only['sales'].median():.1f}")
    ax.set_xlabel("Sales")
    ax.set_ylabel("Count")
    ax.set_title("Sales Distribution (clipped at 99.9%)")
    ax.legend()

    ax = axes[1]
    log_sales = np.log1p(train_only["sales"].clip(lower=0))
    ax.hist(log_sales, bins=100, color="#4CAF50", alpha=0.7, edgecolor="white", lw=0.3)
    ax.axvline(log_sales.mean(), color="red", ls="--", lw=1.2,
               label=f"Mean={log_sales.mean():.2f}")
    ax.set_xlabel("log(1 + sales)")
    ax.set_ylabel("Count")
    ax.set_title("Log-Sales Distribution")
    ax.legend()

    fig.suptitle("Sales Distribution Analysis", fontsize=14, y=1.01)
    save_and_log(fig, "eda", "01_sales_distribution.png")

    # --- 2.5.2 销量时序图（按 family 聚合） ---
    daily_family = train_only.groupby(["date", "family"])["sales"].sum().reset_index()
    top_families = (daily_family.groupby("family")["sales"].mean()
                    .sort_values(ascending=False).head(6).index.tolist())

    fig, ax = plt.subplots(figsize=(16, 6))
    for fam in top_families:
        fam_daily = daily_family[daily_family["family"] == fam].set_index("date")
        # 7 天平滑
        fam_daily["sales_smooth"] = fam_daily["sales"].rolling(7, center=True, min_periods=1).mean()
        ax.plot(fam_daily.index, fam_daily["sales_smooth"], lw=1.0, alpha=0.8, label=fam)

    # 标记地震
    ax.axvline(pd.Timestamp("2016-04-16"), color="red", ls="--", lw=1.5, alpha=0.7, label="Earthquake")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Sales (7-day smoothed)")
    ax.set_title("Top-6 Families — Sales Over Time")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    save_and_log(fig, "eda", "02_sales_timeseries.png")

    # --- 2.5.3 Store × Family 热力图 ---
    sf_pivot = train_only.pivot_table(
        values="sales", index="store_nbr", columns="family", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(sf_pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Avg Sales"},
                xticklabels=True, yticklabels=True, linewidths=0, rasterized=True)
    ax.set_xlabel("Family")
    ax.set_ylabel("Store")
    ax.set_title("Average Sales — Store × Family Heatmap")
    # 小字体适应大量标签
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    save_and_log(fig, "eda", "03_store_family_heatmap.png")

    # --- 2.5.4 每周销量模式 ---
    train_dow = train_only.groupby(["dayofweek", "family"])["sales"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    for fam in top_families:
        fam_dow = train_dow[train_dow["family"] == fam]
        ax.plot(fam_dow["dayofweek"], fam_dow["sales"], "o-", lw=1.5, ms=5, label=fam)
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Sales")
    ax.set_title("Weekly Sales Pattern — Top Families")
    ax.legend(fontsize=8)
    save_and_log(fig, "eda", "04_weekly_pattern.png")

    # --- 2.5.5 年度/月度模式 ---
    train_month = train_only.groupby(["year", "month"])["sales"].mean().reset_index()
    train_month["ym"] = train_month["year"].astype(str) + "-" + train_month["month"].astype(str).str.zfill(2)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(len(train_month)), train_month["sales"], "o-", color="#2196F3", lw=1.5, ms=5)
    ax.set_xticks(range(0, len(train_month), 3))
    ax.set_xticklabels(train_month["ym"].iloc[::3], rotation=45, fontsize=8)
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Average Sales")
    ax.set_title("Monthly Average Sales Trend")
    ax.axvline(x=len(train_month[train_month["ym"] < "2016-04"]), color="red", ls="--", lw=1.2, alpha=0.7,
               label="Earthquake 2016-04")
    ax.legend()
    save_and_log(fig, "eda", "05_monthly_trend.png")

    print("  EDA 可视化完成 (5 张图)")

safe_plot(plot_eda, "EDA可视化")


# ════════════════════════════════════════════════════════════════
# 3. 油价特征
# ════════════════════════════════════════════════════════════════

print("\n=== 3. 油价特征 ===")

full_dates = pd.DataFrame({
    "date": pd.date_range(data["date"].min(), data["date"].max(), freq="D")
})
oil_all = full_dates.merge(oil, on="date", how="left")
oil_all["dcoilwtico"] = oil_all["dcoilwtico"].interpolate().bfill().ffill()

for w in [7, 30, 90]:
    oil_all[f"oil_ma{w}"] = oil_all["dcoilwtico"].rolling(w, min_periods=1).mean()
oil_all["oil_chg7"] = oil_all["dcoilwtico"].pct_change(7).fillna(0)
oil_all["oil_chg30"] = oil_all["dcoilwtico"].pct_change(30).fillna(0)

data = data.merge(
    oil_all[["date", "dcoilwtico", "oil_ma7", "oil_ma30",
             "oil_ma90", "oil_chg7", "oil_chg30"]],
    on="date", how="left"
)
assert_unique_ids(data, "oil feature merge")


# ════════════════════════════════════════════════════════════════
# 4. 节假日特征
# ════════════════════════════════════════════════════════════════

print("\n=== 4. 节假日特征 ===")

hol = holidays[holidays["transferred"].astype(str).str.lower() != "true"].copy()

# Work Day
wd = hol[hol["type"] == "Work Day"][["date"]].drop_duplicates()
wd["is_work_day"] = 1
data = data.merge(wd, on="date", how="left")
assert_unique_ids(data, "work day merge")
data["is_work_day"] = data["is_work_day"].fillna(0).astype("int8")

# National
nat = hol[(hol["locale"] == "National") & (hol["type"] != "Work Day")
          ][["date", "type"]].copy()
holiday_priority = {
    "Holiday": 0,
    "Transfer": 1,
    "Additional": 2,
    "Bridge": 3,
    "Event": 4,
}
nat["priority"] = nat["type"].map(holiday_priority).fillna(9)
nat = nat.sort_values(["date", "priority"])
nat = nat.drop_duplicates("date", keep="first")
nat = nat.rename(columns={"type": "holiday_type"})[["date", "holiday_type"]]
data = data.merge(nat, on="date", how="left")
assert_unique_ids(data, "national holiday merge")

# Regional
reg = hol[(hol["locale"] == "Regional") & (hol["type"] != "Work Day")
          ][["date", "locale_name"]].copy()
reg = reg.rename(columns={"locale_name": "state"})
reg = reg.drop_duplicates(["date", "state"], keep="first")
reg["is_reg_holiday"] = 1
data = data.merge(reg, on=["date", "state"], how="left")
assert_unique_ids(data, "regional holiday merge")
data["is_reg_holiday"] = data["is_reg_holiday"].fillna(0).astype("int8")

# Local
loc = hol[(hol["locale"] == "Local") & (hol["type"] != "Work Day")
          ][["date", "locale_name"]].copy()
loc = loc.rename(columns={"locale_name": "city"})
loc = loc.drop_duplicates(["date", "city"], keep="first")
loc["is_local_holiday"] = 1
data = data.merge(loc, on=["date", "city"], how="left")
assert_unique_ids(data, "local holiday merge")
data["is_local_holiday"] = data["is_local_holiday"].fillna(0).astype("int8")

# Combined + distance to holiday
data["is_holiday"] = (
    data["holiday_type"].notna() |
    (data["is_reg_holiday"] == 1) |
    (data["is_local_holiday"] == 1)
).astype("int8")
data.loc[data["is_work_day"] == 1, "is_holiday"] = 0

# 距节假日的天数（V7 新增：来自 0.375 EDA 发现节假日影响 ±3天）
hol_dates = sorted(data.loc[data["is_holiday"] == 1, "date"].unique())
data["days_to_holiday"] = 7
data["days_from_holiday"] = 7
for hd in hol_dates:
    for offset in range(-7, 8):
        target = hd + pd.Timedelta(days=offset)
        m = data["date"] == target
        if 0 <= offset <= 7:
            cur_to = data.loc[m, "days_to_holiday"]
            data.loc[m, "days_to_holiday"] = np.minimum(cur_to, offset).values
        if 0 <= -offset <= 7:
            cur_fr = data.loc[m, "days_from_holiday"]
            data.loc[m, "days_from_holiday"] = np.minimum(cur_fr, -offset).values
data["days_to_holiday"] = data["days_to_holiday"].astype("int8")
data["days_from_holiday"] = data["days_from_holiday"].astype("int8")

# Holiday type encode
ht_map = {t: i for i, t in enumerate(data["holiday_type"].dropna().unique())}
data["holiday_type_enc"] = data["holiday_type"].map(ht_map).fillna(-1).astype("int8")
data.drop(columns=["holiday_type"], inplace=True)
assert_unique_ids(data, "holiday feature merge")
print("holiday_type_map:", ht_map)


# --- 4.5 油价 + 节假日 EDA 可视化 ---
print("--- 4.5 油价 & 节假日可视化 ---")


def plot_external_factors():
    train_only = data[data["is_train"] == 1]

    # --- 油价 vs 销量 ---
    fig, ax1 = plt.subplots(figsize=(16, 5))
    daily_sales = train_only.groupby("date")["sales"].mean()
    daily_oil = train_only.groupby("date")["dcoilwtico"].first()

    ax1.plot(daily_sales.index, daily_sales.rolling(30, min_periods=1).mean(),
             color="#2196F3", lw=1.5, label="Avg Sales (30d MA)")
    ax1.set_ylabel("Average Sales", color="#2196F3")
    ax1.tick_params(axis="y", labelcolor="#2196F3")

    ax2 = ax1.twinx()
    ax2.plot(daily_oil.index, daily_oil.rolling(30, min_periods=1).mean(),
             color="#FF5722", lw=1.5, alpha=0.8, label="Oil Price (30d MA)")
    ax2.set_ylabel("Oil Price (dcoilwtico)", color="#FF5722")
    ax2.tick_params(axis="y", labelcolor="#FF5722")

    ax1.set_xlabel("Date")
    fig.suptitle("Oil Price vs Sales (30-day Moving Average)", fontsize=14, y=1.01)
    fig.autofmt_xdate()
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    save_and_log(fig, "eda", "06_oil_vs_sales.png")

    # --- 节假日影响箱线图 ---
    # 对比节假日前后 ±7 天的销量分布
    holiday_dates = sorted(data.loc[data["is_holiday"] == 1, "date"].unique())
    holiday_mask = train_only["date"].isin(holiday_dates)
    non_holiday_mask = ~train_only["date"].isin(holiday_dates)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 节假日 vs 非节假日
    plot_data = [
        train_only.loc[holiday_mask, "sales"].clip(upper=train_only["sales"].quantile(0.99)),
        train_only.loc[non_holiday_mask, "sales"].clip(upper=train_only["sales"].quantile(0.99)),
    ]
    bp = axes[0].boxplot(plot_data, labels=["Holiday", "Non-Holiday"], patch_artist=True,
                         widths=0.5)
    bp["boxes"][0].set_facecolor("#FF5722")
    bp["boxes"][1].set_facecolor("#4CAF50")
    axes[0].set_ylabel("Sales")
    axes[0].set_title("Sales Distribution: Holiday vs Non-Holiday")

    # 距节假日天数的销量
    days_effect = train_only.groupby("days_to_holiday")["sales"].mean()
    axes[1].bar(days_effect.index, days_effect.values, color="#2196F3", alpha=0.8,
                edgecolor="white", lw=0.3)
    axes[1].set_xlabel("Days to Nearest Holiday")
    axes[1].set_ylabel("Average Sales")
    axes[1].set_title("Sales by Days-to-Holiday")

    fig.suptitle("Holiday Effect Analysis", fontsize=14, y=1.01)
    save_and_log(fig, "eda", "07_holiday_effect.png")

    print("  外部因素可视化完成")


safe_plot(plot_external_factors, "外部因素可视化")


# --- 4.6 Periodogram 分析：证明傅里叶频率选择的合理性 ---
print("--- 4.6 Periodogram 频谱分析 ---")


def plot_periodogram():
    """对聚合销量做 Periodogram，标注所选频率"""
    train_only = data[data["is_train"] == 1]

    # 聚合到每日总销量
    daily_total = train_only.groupby("date")["sales"].sum().sort_index()
    sales_vals = daily_total.values.astype("float64")
    # 去趋势（简单的线性去趋势，让周期信号更明显）
    from scipy import signal as scipy_signal
    sales_detrended = scipy_signal.detrend(sales_vals)

    n = len(sales_vals)
    freqs = np.fft.rfftfreq(n, d=1.0)  # 频率: cycles/day
    fft_vals = np.abs(np.fft.rfft(sales_detrended))

    # 只展示有意义的频率范围 (周期 ≥ 2 天)
    mask = freqs > 0
    freqs = freqs[mask]
    fft_vals = fft_vals[mask]

    # 转为周期（天）
    periods = 1.0 / freqs

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 全频谱
    ax = axes[0]
    ax.plot(freqs, fft_vals, color="#2196F3", lw=0.8)
    # 标注所选频率
    chosen_freqs = [1/3.5, 1/7, 1/30, 1/365]
    chosen_labels = ["3.5d", "7d", "30d", "365d"]
    colors = ["#FF5722", "#4CAF50", "#FF9800", "#9C27B0"]
    for cf, cl, cc in zip(chosen_freqs, chosen_labels, colors):
        ax.axvline(cf, color=cc, ls="--", lw=1.5, alpha=0.8, label=f"T={cl}")
    ax.set_xlabel("Frequency (cycles/day)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Periodogram — Full Spectrum")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 0.5)

    # 低频放大部分
    ax = axes[1]
    ax.plot(periods, fft_vals, color="#2196F3", lw=0.8)
    for period, cl, cc in zip([3.5, 7, 30, 365], chosen_labels, colors):
        ax.axvline(period, color=cc, ls="--", lw=1.5, alpha=0.8, label=f"T={cl}")
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Periodogram — Period Domain (log scale)")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.set_xlim(2, 800)

    fig.suptitle("Periodogram Analysis — Frequency Domain Justification", fontsize=14, y=1.01)
    save_and_log(fig, "eda", "08_periodogram.png")
    print("  Periodogram 可视化完成")


safe_plot(plot_periodogram, "Periodogram分析")


# ════════════════════════════════════════════════════════════════
# 5. 日期特征 + Periodogram 驱动傅里叶 + 趋势
#    频率来源: Periodogram 分析 [3.5天(半周), 7天, 30天, 365天]
#    0.375 方案证明这4个频率最有信号
# ════════════════════════════════════════════════════════════════

print("\n=== 5. 日期特征 + Periodogram傅里叶 + 趋势 ===")

d = data["date"]

data["year"] = d.dt.year.astype("int16")
data["month"] = d.dt.month.astype("int8")
data["day"] = d.dt.day.astype("int8")
data["dayofweek"] = d.dt.dayofweek.astype("int8")
data["weekofyear"] = d.dt.isocalendar().week.astype("int16")
data["quarter"] = d.dt.quarter.astype("int8")
data["dayofyear"] = d.dt.dayofyear.astype("int16")

data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype("int8")
data["is_month_start"] = d.dt.is_month_start.astype("int8")
data["is_month_end"] = d.dt.is_month_end.astype("int8")
data["is_quarter_start"] = d.dt.is_quarter_start.astype("int8")
data["is_quarter_end"] = d.dt.is_quarter_end.astype("int8")

data["days_from_2013"] = (d - pd.Timestamp("2013-01-01")).dt.days.astype("int32")
data["half_month"] = (data["day"] <= 15).astype("int8")
data["week_of_month"] = ((data["day"] - 1) // 7 + 1).astype("int8")

# --- V7: 发薪日特征（厄瓜多尔 15号 + 月末） ---
data["days_to_payday"] = np.minimum(
    np.abs(data["day"] - 15),
    np.abs(data["day"] - d.dt.days_in_month)
).astype("int8")
data["is_payday"] = (
    (data["day"] == 15) | (data["day"] == d.dt.days_in_month)
).astype("int8")

# --- 0.375 核心: Periodogram 驱动多阶傅里叶 ---
# 用 days_from_2013 作为连续时间轴（更精确的周期计算）
t = data["days_from_2013"].values.astype("float32")

# 频率: [3.5(半周), 7(周), 30(月), 365(年)]
# 每阶: 2阶 (比单阶能捕捉非对称模式)
fourier_config = [
    (3.5,  3),   # 半周周期, 3阶 (高频细节)
    (7.0,  3),   # 周周期,   3阶
    (30.0, 3),   # 月周期,   3阶
    (365.0, 2),  # 年周期,   2阶
]

fourier_cols = []
for freq, order in fourier_config:
    for k in range(1, order + 1):
        col_sin = f"fourier_sin_{freq}_{k}"
        col_cos = f"fourier_cos_{freq}_{k}"
        data[col_sin] = np.sin(2 * k * np.pi * t / freq).astype("float32")
        data[col_cos] = np.cos(2 * k * np.pi * t / freq).astype("float32")
        fourier_cols.extend([col_sin, col_cos])

print(f"傅里叶特征: {len(fourier_cols)} 个 (来自 4 个频率)")

# --- 0.375 核心: Trend + Trend² ---
data["trend_linear"] = t.astype("float32") / 365.0      # 单位: 年
data["trend_quad"] = (data["trend_linear"] ** 2).astype("float32")

# --- 保留兼容旧名（后处理依赖 month_sin/cos 这些名字） ---
# 这些不再用，但留着避免报错
data["month_sin"] = data["fourier_sin_30.0_1"]
data["month_cos"] = data["fourier_cos_30.0_1"]
data["dow_sin"] = data["fourier_sin_7.0_1"]
data["dow_cos"] = data["fourier_cos_7.0_1"]
data["woy_sin"] = data["fourier_sin_365.0_1"]
data["woy_cos"] = data["fourier_cos_365.0_1"]


# ════════════════════════════════════════════════════════════════
# 6. 地震特征
# ════════════════════════════════════════════════════════════════

print("\n=== 6. 地震特征 ===")

eq = pd.Timestamp("2016-04-16")
data["days_since_quake"] = (data["date"] - eq).dt.days.astype("int32")
data["quake_window"] = (
    (data["date"] >= eq) & (data["date"] <= eq + pd.Timedelta(days=90))
).astype("int8")
data["quake_recovery"] = (
    (data["date"] >= eq + pd.Timedelta(days=91)) &
    (data["date"] <= eq + pd.Timedelta(days=365))
).astype("int8")


# ════════════════════════════════════════════════════════════════
# 6.5. 异常值处理（0.375 思路: clip extreme sales）
# ════════════════════════════════════════════════════════════════

print("\n=== 6.5. 异常值处理 ===")

train_mask = data["is_train"] == 1
sf_upper = data.loc[train_mask].groupby(
    ["store_nbr", "family"], observed=True
)["sales"].quantile(0.995)
global_upper = data.loc[train_mask, "sales"].quantile(0.995)

data["_upper"] = data.set_index(["store_nbr", "family"]).index.map(
    lambda x: sf_upper.get(x, global_upper)
).fillna(global_upper)

data["is_clipped"] = (
    (data["sales"] > data["_upper"]) & train_mask
).astype("int8")

data.loc[train_mask, "sales"] = np.minimum(
    data.loc[train_mask, "sales"].to_numpy(dtype=np.float32),
    data.loc[train_mask, "_upper"].to_numpy(dtype=np.float32),
).astype(np.float32)
n_clip = data.loc[train_mask, "is_clipped"].sum()
print(f"Clipped: {n_clip:,} / {train_mask.sum():,} "
      f"({n_clip / train_mask.sum() * 100:.3f}%)")
data.drop(columns=["_upper"], inplace=True)


# --- 6.6 异常值处理可视化 ---
print("--- 6.6 异常值可视化 ---")

def plot_outlier_analysis():
    train_clipped_mask = data["is_train"] == 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 异常值分布 (per store-family)
    # 找到被 clip 最多的 store-family
    clip_counts = data.loc[train_clipped_mask].groupby(
        ["store_nbr", "family"])["is_clipped"].sum().sort_values(ascending=False)
    top_clip = clip_counts.head(10)

    ax = axes[0]
    labels = [f"S{s}_F{f}" for s, f in top_clip.index]
    ax.barh(range(len(top_clip)), top_clip.values, color="#FF5722", alpha=0.8,
            edgecolor="white")
    ax.set_yticks(range(len(top_clip)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("N Clipped")
    ax.set_title("Top-10 Store-Family with Most Outliers")

    # Clip 前后直方图对比（选一个受影响的 store-family 做示例）
    ax = axes[1]
    worst_sf = top_clip.index[0]
    mask_worst = ((data["store_nbr"] == worst_sf[0]) &
                  (data["family"] == worst_sf[1]) &
                  (data["is_train"] == 1))
    sf_sales_all = data.loc[mask_worst, "sales"].copy()
    # Reconstruct original: clipped ones were at the upper bound
    upper_bound = sf_sales_all.max()  # Since clipped, this is the bound
    ax.hist(sf_sales_all, bins=40, color="#4CAF50", alpha=0.6, edgecolor="white", lw=0.3,
            label="After Clip")
    ax.axvline(upper_bound, color="red", ls="--", lw=1.5,
               label=f"Clip threshold ≈ {upper_bound:.0f}")
    ax.set_xlabel("Sales")
    ax.set_ylabel("Count")
    ax.set_title(f"Store {worst_sf[0]} × {worst_sf[1]} — After Clipping")
    ax.legend(fontsize=7)

    fig.suptitle("Outlier Clipping Analysis (99.5% per store-family)", fontsize=14, y=1.01)
    save_and_log(fig, "features", "04_outlier_analysis.png")
    print("  异常值可视化完成")

safe_plot(plot_outlier_analysis, "异常值可视化")


# ════════════════════════════════════════════════════════════════
# 7. Rank 编码（0.375 思路: 替代复杂交互）
# ════════════════════════════════════════════════════════════════

print("\n=== 7. Rank 编码 ===")

train_mask_all = data["is_train"] == 1
data["sales"] = data["sales"].astype("float32")
global_mean = data.loc[train_mask_all, "sales"].mean()

# family_rank: 按平均销量排名
fam_mean = data.loc[train_mask_all].groupby(
    "family", observed=True)["sales"].mean().sort_values(ascending=False)
data["family_rank"] = data["family"].map(
    {f: i for i, f in enumerate(fam_mean.index)}
).astype("int8")

# store_rank: 按平均销量排名
sto_mean = data.loc[train_mask_all].groupby(
    "store_nbr")["sales"].mean().sort_values(ascending=False)
data["store_rank"] = data["store_nbr"].map(
    {s: i for i, s in enumerate(sto_mean.index)}
).astype("int8")

# cluster_rank
clu_mean = data.loc[train_mask_all].groupby(
    "cluster")["sales"].mean().sort_values(ascending=False)
data["cluster_rank"] = data["cluster"].map(
    {c: i for i, c in enumerate(clu_mean.index)}
).astype("int8")

print(f"family_rank: 0~{data['family_rank'].max()}, "
      f"store_rank: 0~{data['store_rank'].max()}")


# ════════════════════════════════════════════════════════════════
# 8. 交互统计特征（继承 V5）
# ════════════════════════════════════════════════════════════════

print("\n=== 8. 交互统计特征 ===")

# Store × Family
sf_stats = data.loc[train_mask_all].groupby(
    ["store_nbr", "family"], observed=True
)["sales"].agg(["mean", "std", "min", "max"]).add_prefix("sf_").reset_index()
data = data.merge(sf_stats, on=["store_nbr", "family"], how="left")
assert_unique_ids(data, "store-family stats merge")
data["sf_mean"] = data["sf_mean"].fillna(global_mean)
data["sf_std"] = data["sf_std"].fillna(0)
data["sf_min"] = data["sf_min"].fillna(0)
data["sf_max"] = data["sf_max"].fillna(data["sf_mean"])

# Family / Store 全局
for gcol, prefix in [("family", "f"), ("store_nbr", "s")]:
    stats = data.loc[train_mask_all].groupby(
        gcol, observed=True
    )["sales"].agg(["mean", "std"]).add_prefix(f"{prefix}_").reset_index()
    data = data.merge(stats, on=gcol, how="left")
    assert_unique_ids(data, f"{gcol} stats merge")
    data[f"{prefix}_mean"] = data[f"{prefix}_mean"].fillna(global_mean)
    data[f"{prefix}_std"] = data[f"{prefix}_std"].fillna(0)

# 交互均值
interaction_pairs = [
    (["city", "family"],        "f_mean"),
    (["state", "family"],       "f_mean"),
    (["cluster", "family"],     "f_mean"),
    (["store_nbr", "month"],    "s_mean"),
    (["family", "dayofweek"],   "f_mean"),
    (["family", "month"],       "f_mean"),
    (["family", "quarter"],     "f_mean"),
    (["store_nbr", "dayofweek"],"s_mean"),
]
for pair, fill_col in interaction_pairs:
    name = pair[0][:2] + "_" + pair[1][:2] + "_mean"
    tmp = data.loc[train_mask_all].groupby(
        pair, observed=True)["sales"].mean().reset_index()
    tmp.columns = pair + [name]
    data = data.merge(tmp, on=pair, how="left")
    assert_unique_ids(data, f"{pair} interaction stats merge")
    data[name] = data[name].fillna(data[fill_col])

print("交互统计特征完成")


# ════════════════════════════════════════════════════════════════
# 9. 促销特征
# ════════════════════════════════════════════════════════════════

print("\n=== 9. 促销特征 ===")

data["onpromotion"] = data["onpromotion"].fillna(0)
data["promo_log"] = np.log1p(data["onpromotion"]).astype("float32")
data["has_promotion"] = (data["onpromotion"] > 0).astype("int8")


# ════════════════════════════════════════════════════════════════
# 10. 时间序列特征（继承 V5）
# ════════════════════════════════════════════════════════════════

print("\n=== 10. 时间序列特征 ===")

data["sales_log"] = np.log1p(data["sales"].clip(lower=0))
data = data.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
train_mask_all = data["is_train"] == 1
assert_unique_ids(data, "time-series sort")

group_cols = ["store_nbr", "family"]
lag_list = [16, 17, 18, 19, 20, 21, 28, 35, 56, 84, 112, 182, 364]

for lag in lag_list:
    data[f"lag_{lag}"] = data.groupby(
        group_cols, observed=True
    )["sales_log"].shift(lag).astype("float32")

for window in [7, 14, 28, 56, 84, 112, 182]:
    grp = data.groupby(group_cols, observed=True)["sales_log"]
    data[f"rm_{window}"] = grp.transform(
        lambda x: x.shift(16).rolling(window, min_periods=1).mean()
    ).fillna(0).astype("float32")
    data[f"rs_{window}"] = grp.transform(
        lambda x: x.shift(16).rolling(window, min_periods=max(2, window // 4)).std()
    ).fillna(0).astype("float32")

for short, long in [(7, 28), (28, 112), (7, 56), (14, 56), (28, 182)]:
    col = f"mom_{short}_{long}"
    data[col] = np.clip(
        data[f"rm_{short}"] / (data[f"rm_{long}"] + 0.01) - 1, -5, 5
    ).astype("float32").fillna(0)

for span in [7, 14, 28, 56]:
    data[f"ewm_{span}"] = data.groupby(
        group_cols, observed=True
    )["sales_log"].transform(
        lambda x: x.shift(16).ewm(span=span, adjust=False).mean()
    ).fillna(0).astype("float32")

for lag in [16, 28, 56, 112]:
    data[f"plag_{lag}"] = data.groupby(
        group_cols, observed=True
    )["onpromotion"].shift(lag).fillna(0).astype("float32")

for window in [7, 14, 28]:
    data[f"prm_{window}"] = data.groupby(
        group_cols, observed=True
    )["onpromotion"].transform(
        lambda x: x.shift(16).rolling(window, min_periods=1).mean()
    ).fillna(0).astype("float32")

print("时间序列特征完成")


# ════════════════════════════════════════════════════════════════
# 10.5. STL 风格分解特征
# 把 sales_log 拆成 trend / seasonal / residual
# ════════════════════════════════════════════════════════════════

print("\n=== 10.5. STL 风格特征 ===")

data["stl_trend"] = data.groupby(
    group_cols, observed=True
)["sales_log"].transform(
    lambda x: x.shift(16).ewm(span=182, adjust=False).mean()
).fillna(0).astype("float32")

data["stl_seasonal_7"] = (data["rm_7"] - data["stl_trend"]).astype("float32")
data["stl_seasonal_28"] = (data["rm_28"] - data["stl_trend"]).astype("float32")
data["stl_seasonal_364"] = (data["lag_364"] - data["stl_trend"]).astype("float32")

data["stl_resid"] = data.groupby(
    group_cols, observed=True
)["sales_log"].transform(
    lambda x: x.shift(16)
    - x.shift(16).ewm(span=182, adjust=False).mean()
    - (x.shift(16).rolling(7, min_periods=1).mean()
       - x.shift(16).ewm(span=182, adjust=False).mean())
).fillna(0).astype("float32")

data["stl_strength"] = (
    data["stl_seasonal_7"].abs() /
    (data["stl_seasonal_7"].abs() + data["stl_resid"].abs() + 0.01)
).fillna(0.5).clip(0, 1).astype("float32")

data["stl_trend_chg"] = data.groupby(
    group_cols, observed=True
)["stl_trend"].transform(lambda x: x.diff(28).fillna(0)).astype("float32")

print("STL 特征完成")


# --- 10.6 STL 分解可视化 ---
print("--- STL 分解可视化 ---")

def plot_feature_analysis():
    # --- STL Decomposition 示例（选销量最高的 store-family） ---
    train_mask_vis = data["is_train"] == 1
    sf_sales_sum = data.loc[train_mask_vis].groupby(
        ["store_nbr", "family"])["sales"].sum().sort_values(ascending=False)
    top_sf = sf_sales_sum.index[0]

    mask_sf = ((data["store_nbr"] == top_sf[0]) &
               (data["family"] == top_sf[1]) &
               (data["is_train"] == 1))
    sf_data = data.loc[mask_sf].sort_values("date").tail(365).copy()

    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)

    axes[0].plot(sf_data["date"], sf_data["sales"], color="#2196F3", lw=0.8)
    axes[0].set_ylabel("Sales")
    axes[0].set_title(f"Original Sales — Store {top_sf[0]}, Family {top_sf[1]}")

    axes[1].plot(sf_data["date"], sf_data["sales_log"], color="#2196F3", lw=0.8)
    axes[1].set_ylabel("log(1+sales)")
    axes[1].set_title("Log Sales")

    axes[2].plot(sf_data["date"], sf_data["stl_trend"], color="#4CAF50", lw=1.2)
    axes[2].set_ylabel("Trend")
    axes[2].set_title("STL Trend (EWM span=182)")

    axes[3].plot(sf_data["date"], sf_data["stl_seasonal_7"], color="#FF9800", lw=0.8, label="Seasonal_7")
    axes[3].plot(sf_data["date"], sf_data["stl_seasonal_28"], color="#9C27B0", lw=0.8, alpha=0.7, label="Seasonal_28")
    axes[3].set_ylabel("Seasonal")
    axes[3].set_title("STL Seasonal Components")
    axes[3].legend(fontsize=8)

    axes[4].plot(sf_data["date"], sf_data["stl_resid"], color="#F44336", lw=0.5, alpha=0.7)
    axes[4].set_ylabel("Residual")
    axes[4].set_xlabel("Date")
    axes[4].set_title("STL Residual")
    axes[4].axhline(0, color="black", ls="--", lw=0.5)

    fig.autofmt_xdate()
    fig.suptitle("STL-style Decomposition (Example)", fontsize=14, y=1.01)
    save_and_log(fig, "features", "01_stl_decomposition.png")

    # --- Rank 编码分布 ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title, color in [
        (axes[0], "family_rank", "Family Rank", "#4CAF50"),
        (axes[1], "store_rank", "Store Rank", "#2196F3"),
        (axes[2], "cluster_rank", "Cluster Rank", "#FF9800"),
    ]:
        vals = data.loc[train_mask_vis, col]
        ax.hist(vals, bins=50, color=color, alpha=0.7, edgecolor="white", lw=0.3)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.set_title(f"{title} Distribution")
    fig.suptitle("Rank Encoding Distributions", fontsize=14, y=1.01)
    save_and_log(fig, "features", "02_rank_distributions.png")

    # --- 傅里叶特征可视化 ---
    # 选一组合展示周期模式
    t_demo = data.loc[train_mask_vis, "days_from_2013"].values[:1000]
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fourier_demo = [
        ("fourier_sin_7.0_1", "fourier_cos_7.0_1", "Weekly (7d)", axes[0, 0]),
        ("fourier_sin_30.0_1", "fourier_cos_30.0_1", "Monthly (30d)", axes[0, 1]),
        ("fourier_sin_365.0_1", "fourier_cos_365.0_1", "Yearly (365d)", axes[1, 0]),
        ("fourier_sin_3.5_1", "fourier_cos_3.5_1", "Half-week (3.5d)", axes[1, 1]),
    ]
    for sin_col, cos_col, title, ax in fourier_demo:
        sin_vals = data.loc[train_mask_vis, sin_col].values[:1000]
        cos_vals = data.loc[train_mask_vis, cos_col].values[:1000]
        ax.plot(t_demo, sin_vals, lw=0.5, alpha=0.7, label="sin")
        ax.plot(t_demo, cos_vals, lw=0.5, alpha=0.7, label="cos")
        ax.set_xlabel("Days from 2013")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.suptitle("Fourier Features — Periodogram-driven Frequencies", fontsize=14, y=1.01)
    save_and_log(fig, "features", "03_fourier_features.png")

    print("  特征可视化完成")

safe_plot(plot_feature_analysis, "特征可视化")


# ════════════════════════════════════════════════════════════════
# 11. 类别编码
# ════════════════════════════════════════════════════════════════

print("\n=== 11. 类别编码 ===")

for col in ["family", "city", "state", "store_type"]:
    data[col + "_enc"] = LabelEncoder().fit_transform(data[col].astype(str))


# ════════════════════════════════════════════════════════════════
# 11.5. 增强交互特征（0.375 思路 + 安全交互）
# ════════════════════════════════════════════════════════════════

print("\n=== 11.5. 增强交互特征 ===")

# 关店检测（0.375 方案: store 24,25,29,30,42,52,53）
store_total = data.loc[train_mask_all].groupby("store_nbr")["sales"].sum()
closed_stores = store_total[store_total < store_total.quantile(0.05)].index.tolist()
data["store_was_closed"] = data["store_nbr"].isin(closed_stores).astype("int8")
print(f"疑似关店 stores: {closed_stores}")

# Family × Trend（不同品类增长趋势不同）
data["family_trend"] = (
    data["family_rank"].astype("float32") * data["trend_linear"]
).astype("float32")

# Cluster × Trend
data["cluster_trend"] = (
    data["cluster_rank"].astype("float32") * data["trend_linear"]
).astype("float32")

# Promotion × Holiday（节假日促销效果不同）
data["promo_x_holiday"] = (
    data["has_promotion"].astype("int8") * data["is_holiday"].astype("int8")
).astype("int8")

# Weekend × Holiday（周末+节假日叠加效应）
data["weekend_x_holiday"] = (
    data["is_weekend"].astype("int8") * data["is_holiday"].astype("int8")
).astype("int8")

# 节假日类型 One-Hot 展开（区分不同类型假日的影响）
htype_dummies = pd.get_dummies(
    data["holiday_type_enc"].replace(-1, np.nan), prefix="htype"
).astype("int8")
htype_dummies.columns = [c.replace(".0", "") for c in htype_dummies.columns]
data = pd.concat([data, htype_dummies], axis=1)
htype_cols = list(htype_dummies.columns)
print(f"节假日 type dummies: {htype_cols}")

print("增强交互特征完成")

# 特征工程全部完成，最后一次全局压缩
data = reduce_mem_usage(data, "特征工程完成")
gc.collect()


# ════════════════════════════════════════════════════════════════
# 12. 特征列表
# ════════════════════════════════════════════════════════════════

ts_cols = [
    c for c in data.columns
    if c.startswith(("lag_", "rm_", "rs_", "ewm_", "plag_", "prm_", "mom_"))
]

feature_cols = [
    "store_nbr",
    "family_enc",
    "city_enc",
    "state_enc",
    "store_type_enc",
    "cluster",

    "family_rank",
    "store_rank",
    "cluster_rank",
    "store_was_closed",

    "onpromotion",
    "promo_log",
    "has_promotion",
    "promo_x_holiday",
    "weekend_x_holiday",

    "dcoilwtico",
    "oil_ma7",
    "oil_ma30",
    "oil_ma90",
    "oil_chg7",
    "oil_chg30",

    "year",
    "month",
    "day",
    "dayofweek",
    "weekofyear",
    "quarter",
    "dayofyear",
    "days_from_2013",
    "half_month",
    "week_of_month",
    "days_to_payday",
    "is_payday",

    "trend_linear",
    "trend_quad",
    "family_trend",
    "cluster_trend",
] + fourier_cols + htype_cols + [

    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",

    "is_work_day",
    "is_holiday",
    "is_reg_holiday",
    "is_local_holiday",
    "holiday_type_enc",
    "days_to_holiday",
    "days_from_holiday",

    "days_since_quake",
    "quake_window",
    "quake_recovery",

    "is_clipped",

    "sf_mean",
    "sf_std",
    "sf_min",
    "sf_max",
    "f_mean",
    "f_std",
    "s_mean",
    "s_std",

    "stl_trend",
    "stl_seasonal_7",
    "stl_seasonal_28",
    "stl_seasonal_364",
    "stl_resid",
    "stl_strength",
    "stl_trend_chg",

    "ci_fa_mean",
    "st_fa_mean",
    "cl_fa_mean",
    "st_mo_mean",
    "fa_da_mean",
    "fa_mo_mean",
    "fa_qu_mean",
    "st_da_mean",
] + ts_cols

print(f"特征数量: {len(feature_cols)}")


# ════════════════════════════════════════════════════════════════
# 13. 数据划分（V7: 31天验证窗口，和 0.375 方案一致）
# ════════════════════════════════════════════════════════════════

print("\n=== 13. 数据划分：2014+ ===")

train_data = data[
    (data["is_train"] == 1) & (data["date"] >= "2014-01-01")
].copy()
test_data = data[data["is_train"] == 0].copy()


# V7: 31 天验证窗口 (0.375 方案用 31 天)
valid_date = pd.Timestamp("2017-07-16")

train_part = train_data[train_data["date"] < valid_date].copy()
valid_part = train_data[train_data["date"] >= valid_date].copy()

print(f"train_part: {train_part.shape}, valid_part: {valid_part.shape}")
print(f"训练日期: {train_part['date'].min()} ~ {train_part['date'].max()}")
print(f"验证日期: {valid_part['date'].min()} ~ {valid_part['date'].max()}")

X_tr = train_part[feature_cols].replace([np.inf, -np.inf], np.nan)
y_tr = train_part["sales_log"]

X_val = valid_part[feature_cols].replace([np.inf, -np.inf], np.nan)
y_val_log = valid_part["sales_log"]
y_val_real = valid_part["sales"].values

X_test = test_data[feature_cols].replace([np.inf, -np.inf], np.nan)

for df in [X_tr, X_val, X_test]:
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")

# 提前提取关店信息（后面 data 会被回收）
sf_recent_for_zero = data.loc[
    (data["is_train"] == 1) & (data["date"] >= "2017-07-16"),
    ["store_nbr", "family", "sales"]
].groupby(["store_nbr", "family"])["sales"].sum().reset_index()
sf_recent_for_zero.columns = ["store_nbr", "family", "recent_sum"]

# 释放不再需要的中间 DataFrame（data 内存量大，腾给模型训练）
del data
gc.collect()
print(f"X_tr: {X_tr.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")


# ════════════════════════════════════════════════════════════════
# 14. Baseline
# ════════════════════════════════════════════════════════════════

print("\n=== 14. Baseline ===")

b_train = train_data[train_data["date"] < valid_date]
b_dict = b_train.groupby(
    ["store_nbr", "family"], observed=True)["sales"].mean().to_dict()
b_global = b_train["sales"].mean()
b_pred = np.array([
    b_dict.get((r["store_nbr"], r["family"]), b_global)
    for _, r in valid_part.iterrows()
])
print(f"Baseline RMSLE: {rmsle(y_val_real, b_pred):.6f}")


# ════════════════════════════════════════════════════════════════
# 15. LightGBM Optuna
# ════════════════════════════════════════════════════════════════

print(f"\n=== 15. LightGBM Optuna × {N_TRIALS} ===")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

opt_mask = train_part["date"] >= "2017-01-01"
opt_tr = train_part.loc[
    opt_mask & (train_part["date"] < pd.Timestamp("2017-06-15"))
].index
opt_val = train_part.loc[
    opt_mask & (train_part["date"] >= pd.Timestamp("2017-06-15"))
].index


def objective(trial):
    params = {
        "objective": "regression", "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 64, 512),
        "max_depth": trial.suggest_int("max_depth", 6, 16),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
        "random_state": 42, "verbosity": -1, "n_jobs": -1,
        "device": "cpu", "force_col_wise": True,
    }
    dtrain_opt = lgb.Dataset(X_tr.loc[opt_tr], label=y_tr.loc[opt_tr])
    dvalid_opt = lgb.Dataset(
        X_tr.loc[opt_val], label=y_tr.loc[opt_val], reference=dtrain_opt)
    model = lgb.train(
        params=params, train_set=dtrain_opt, num_boost_round=800,
        valid_sets=[dvalid_opt],
        callbacks=[lgb.early_stopping(50, verbose=False)])
    pred_log = model.predict(
        X_tr.loc[opt_val], num_iteration=model.best_iteration)
    return rmsle(
        train_part.loc[opt_val, "sales"].values,
        np.expm1(np.clip(pred_log, 0, None)))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
print("LGB best:", study.best_value)

lgb_base_params = {
    "objective": "regression", "metric": "rmse",
    "boosting_type": "gbdt", **study.best_params,
    "verbosity": -1, "n_jobs": -1, "device": "cpu", "force_col_wise": True,
}


# ════════════════════════════════════════════════════════════════
# 16. LightGBM × 3
# ════════════════════════════════════════════════════════════════

print("\n=== 16. LightGBM × 3 ===")

lgb_models = {}
for seed in [42, 123, 456]:
    print(f"\nLGB seed={seed}")
    params = {**lgb_base_params, "random_state": seed}
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_val, label=y_val_log, reference=dtrain)
    model = lgb.train(
        params=params, train_set=dtrain, num_boost_round=3000,
        valid_sets=[dtrain, dvalid], valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)])
    pred_log = model.predict(X_val, num_iteration=model.best_iteration)
    score = rmsle(y_val_real, np.expm1(np.clip(pred_log, 0, None)))
    lgb_models[seed] = model
    print(f"LGB seed={seed} RMSLE={score:.6f}, best_iter={model.best_iteration}")


# ════════════════════════════════════════════════════════════════
# 17. XGBoost × 3（增加多样性：不同 depth/lr/subsample 组合）
#    0.375 方案核心: max_depth=9, lr=0.05, subsample=0.8
# ════════════════════════════════════════════════════════════════

print("\n=== 17. XGBoost × 3 ===")

xgb_models = {}
xgb_configs = [
    # (depth, lr, subsample, colsample)
    (9,  0.05, 0.8, 0.8),   # 0.375 最优配置
    (7,  0.03, 0.9, 0.7),   # 浅树保守版
    (11, 0.02, 0.7, 0.8),   # 深树低学习率版
]

for i, (depth, lr, ss, cs) in enumerate(xgb_configs):
    print(f"\nXGB depth={depth}, lr={lr}, subsample={ss}")
    params = {
        "objective": "reg:squarederror", "tree_method": "hist",
        "learning_rate": lr, "max_depth": depth,
        "min_child_weight": 10, "subsample": ss,
        "colsample_bytree": cs, "gamma": 0.1,
        "reg_alpha": 0.5, "reg_lambda": 1.0,
        "random_state": 42 + i, "verbosity": 0, "nthread": -1,
    }
    if USE_GPU:
        params.update({"device": "cuda"})
    dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr)
    dvalid_xgb = xgb.DMatrix(X_val, label=y_val_log)
    model = xgb.train(
        params=params, dtrain=dtrain_xgb, num_boost_round=3000,
        evals=[(dtrain_xgb, "train"), (dvalid_xgb, "valid")],
        early_stopping_rounds=150, verbose_eval=200)
    pred_log = predict_xgb(model, dvalid_xgb)
    score = rmsle(y_val_real, np.expm1(np.clip(pred_log, 0, None)))
    xgb_models[(depth, lr)] = model
    print(f"XGB depth={depth} RMSLE={score:.6f}, best_iter={model.best_iteration}")


# ════════════════════════════════════════════════════════════════
# 18. CatBoost
# ════════════════════════════════════════════════════════════════

print("\n=== 18. CatBoost ===")

from catboost import CatBoostRegressor, Pool

cat_params = {
    "iterations": 2500, "learning_rate": 0.03, "depth": 8,
    "l2_leaf_reg": 3.0, "border_count": 254, "random_seed": 42,
    "verbose": 200, "thread_count": -1,
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "early_stopping_rounds": 150, "use_best_model": True,
}
if USE_GPU:
    cat_params.update({"task_type": "GPU", "devices": "0"})
cat_model = CatBoostRegressor(**cat_params)
cat_model.fit(
    Pool(X_tr.values, label=y_tr.values),
    eval_set=Pool(X_val.values, label=y_val_log.values))
cat_pred_log = cat_model.predict(X_val.values)
cat_score = rmsle(y_val_real, np.expm1(np.clip(cat_pred_log, 0, None)))
print(f"CatBoost RMSLE: {cat_score:.6f}")


# ════════════════════════════════════════════════════════════════
# 18.5 模型训练可视化
# ════════════════════════════════════════════════════════════════

print("\n--- 18.5 模型训练可视化 ---")


def plot_model_diagnostics():
    # --- Optuna 优化历史 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 优化历史
    trials_df = study.trials_dataframe()
    ax = axes[0]
    ax.plot(range(len(trials_df)), trials_df["value"], "o-", ms=4, color="#4CAF50",
            alpha=0.7, lw=1)
    ax.axhline(study.best_value, color="red", ls="--", lw=1.2,
               label=f"Best = {study.best_value:.6f}")
    ax.set_xlabel("Trial")
    ax.set_ylabel("RMSLE")
    ax.set_title(f"Optuna Optimization History ({N_TRIALS} trials)")
    ax.legend()

    # 参数重要性
    try:
        param_importances = optuna.importance.get_param_importances(study)
        params_sorted = sorted(param_importances.items(), key=lambda x: x[1], reverse=True)
        ax = axes[1]
        ax.barh([p[0] for p in params_sorted], [p[1] for p in params_sorted],
                color="#2196F3", alpha=0.8, edgecolor="white")
        ax.set_xlabel("Importance")
        ax.set_title("Optuna Parameter Importance")
    except Exception:
        axes[1].text(0.5, 0.5, "Importance not available", ha="center", va="center",
                     transform=axes[1].transAxes)
        axes[1].set_title("Parameter Importance (N/A)")

    fig.suptitle("LightGBM Hyperparameter Tuning", fontsize=14, y=1.01)
    save_and_log(fig, "models", "02_optuna_tuning.png")

    # --- Feature Importance (LGB best model) ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # LGB feature importance
    lgb_imp = lgb_models[42].feature_importance(importance_type="gain")
    lgb_imp_df = pd.DataFrame({
        "feature": X_val.columns, "importance": lgb_imp
    }).sort_values("importance", ascending=False).head(30)
    ax = axes[0]
    ax.barh(range(len(lgb_imp_df)), lgb_imp_df["importance"], color="#4CAF50", alpha=0.8)
    ax.set_yticks(range(len(lgb_imp_df)))
    ax.set_yticklabels(lgb_imp_df["feature"], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Gain")
    ax.set_title("LightGBM Top-30 Feature Importance")

    # XGB feature importance
    xgb_imp = xgb_models[(9, 0.05)].get_score(importance_type="gain")
    xgb_imp_df = pd.DataFrame({
        "feature": list(xgb_imp.keys()), "importance": list(xgb_imp.values())
    }).sort_values("importance", ascending=False).head(30)
    ax = axes[1]
    ax.barh(range(len(xgb_imp_df)), xgb_imp_df["importance"], color="#2196F3", alpha=0.8)
    ax.set_yticks(range(len(xgb_imp_df)))
    ax.set_yticklabels(xgb_imp_df["feature"], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Gain")
    ax.set_title("XGBoost Top-30 Feature Importance")

    # CatBoost feature importance
    cat_imp = cat_model.get_feature_importance()
    cat_imp_df = pd.DataFrame({
        "feature": X_val.columns, "importance": cat_imp
    }).sort_values("importance", ascending=False).head(30)
    ax = axes[2]
    ax.barh(range(len(cat_imp_df)), cat_imp_df["importance"], color="#9C27B0", alpha=0.8)
    ax.set_yticks(range(len(cat_imp_df)))
    ax.set_yticklabels(cat_imp_df["feature"], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("CatBoost Top-30 Feature Importance")

    fig.suptitle("Feature Importance Comparison", fontsize=14, y=1.01)
    save_and_log(fig, "models", "03_feature_importance.png")

    # --- Learning curves (LGB best + XGB best + CatBoost) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # LGB learning curve
    lgb_best = lgb_models[42]
    ax = axes[0]
    evals_result = getattr(lgb_best, "evals_result_", None)
    if evals_result is None:
        try:
            evals_result = lgb_best.evals_result()
        except Exception:
            evals_result = {}
    if "train" in evals_result and "valid" in evals_result:
        train_rmse = evals_result["train"]["rmse"]
        valid_rmse = evals_result["valid"]["rmse"]
        ax.plot(train_rmse, label="Train", color="#4CAF50", alpha=0.6, lw=1)
        ax.plot(valid_rmse, label="Valid", color="#FF5722", alpha=0.8, lw=1.2)
        ax.axvline(lgb_best.best_iteration, color="red", ls="--", lw=1,
                   label=f"Best iter={lgb_best.best_iteration}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMSE")
        ax.set_title("LightGBM Learning Curve")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "LightGBM eval history N/A", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("LightGBM Learning Curve")

    # XGB learning curve
    xgb_best = xgb_models[(9, 0.05)]
    ax = axes[1]
    try:
        evals_result_xgb = xgb_best.evals_result()
    except Exception:
        evals_result_xgb = {}
    if "train" in evals_result_xgb and "valid" in evals_result_xgb:
        train_rmse = evals_result_xgb["train"]["rmse"]
        valid_rmse = evals_result_xgb["valid"]["rmse"]
        ax.plot(train_rmse, label="Train", color="#4CAF50", alpha=0.6, lw=1)
        ax.plot(valid_rmse, label="Valid", color="#FF5722", alpha=0.8, lw=1.2)
        ax.axvline(xgb_best.best_iteration, color="red", ls="--", lw=1,
                   label=f"Best iter={xgb_best.best_iteration}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMSE")
        ax.set_title("XGBoost Learning Curve")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "XGBoost eval history N/A", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("XGBoost Learning Curve")

    # CatBoost learning curve
    ax = axes[2]
    try:
        evals_result_cat = cat_model.get_evals_result()
    except Exception:
        evals_result_cat = {}
    if "learn" in evals_result_cat and "validation" in evals_result_cat:
        train_rmse = evals_result_cat["learn"]["RMSE"]
        valid_rmse = evals_result_cat["validation"]["RMSE"]
        ax.plot(train_rmse, label="Train", color="#4CAF50", alpha=0.6, lw=1)
        ax.plot(valid_rmse, label="Valid", color="#FF5722", alpha=0.8, lw=1.2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMSE")
        ax.set_title("CatBoost Learning Curve")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "CatBoost eval history N/A", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("CatBoost Learning Curve")

    fig.suptitle("Learning Curves — All Models", fontsize=14, y=1.01)
    save_and_log(fig, "models", "04_learning_curves.png")

    print("  模型训练可视化完成")


safe_plot(plot_model_diagnostics, "模型训练可视化")


# ════════════════════════════════════════════════════════════════
# 19. 集成（V5 风格网格搜索）
# ════════════════════════════════════════════════════════════════

print("\n=== 19. 集成 ===")

all_val_raw = {}
for seed, model in lgb_models.items():
    all_val_raw[f"lgb_{seed}"] = model.predict(
        X_val, num_iteration=model.best_iteration)
for (depth, lr), model in xgb_models.items():
    all_val_raw[f"xgb_d{depth}"] = predict_xgb(model, xgb.DMatrix(X_val))
all_val_raw["cat"] = cat_model.predict(X_val.values)

# LGB 内部融合
lgb_best = float("inf")
lgb_w = {"42": 1/3, "123": 1/3, "456": 1/3}
for w42 in np.arange(0.2, 0.61, 0.1):
    for w123 in np.arange(0.2, 0.61, 0.1):
        w456 = 1.0 - w42 - w123
        if w456 < 0.1: continue
        blend_log = (w42 * all_val_raw["lgb_42"] +
                     w123 * all_val_raw["lgb_123"] +
                     w456 * all_val_raw["lgb_456"])
        score = rmsle(y_val_real, np.expm1(np.clip(blend_log, 0, None)))
        if score < lgb_best:
            lgb_best = score
            lgb_w = {"42": w42, "123": w123, "456": w456}
lgb_blend_log = sum(lgb_w[k] * all_val_raw[f"lgb_{k}"] for k in lgb_w)
print(f"LGB blend: {lgb_best:.6f}")

# XGB 内部融合（动态处理 3 个 XGB 模型）
xgb_keys = [f"xgb_d{d}" for d, *_ in xgb_configs]
xgb_best = float("inf")
xgb_w = {k: 1.0 / len(xgb_keys) for k in xgb_keys}

# 简化: 按验证分数反比分配初始权重，再微调
xgb_scores = {}
for k in xgb_keys:
    xgb_scores[k] = rmsle(y_val_real,
                          np.expm1(np.clip(all_val_raw[k], 0, None)))
best_key = min(xgb_scores, key=xgb_scores.get)
second_key = sorted(xgb_scores, key=xgb_scores.get)[1]
third_key = sorted(xgb_scores, key=xgb_scores.get)[2]

for w_best in np.arange(0.3, 0.71, 0.05):
    for w_second in np.arange(0.15, 1.0 - w_best, 0.05):
        w_third = 1.0 - w_best - w_second
        if w_third < 0.1: continue
        test_w = {best_key: w_best, second_key: w_second,
                  third_key: w_third}
        blend_log = sum(test_w[k] * all_val_raw[k] for k in test_w)
        score = rmsle(y_val_real, np.expm1(np.clip(blend_log, 0, None)))
        if score < xgb_best:
            xgb_best = score
            xgb_w = test_w
xgb_blend_log = sum(xgb_w[k] * all_val_raw[k] for k in xgb_w)
print(f"XGB blend: {xgb_best:.6f}")
print(f"XGB weights: {xgb_w}")

# 三路融合
best_ensemble = float("inf")
best_weights = (0.4, 0.3, 0.3)
for w_lgb in np.arange(0.0, 1.01, 0.02):
    for w_xgb in np.arange(0.0, 1.01 - w_lgb, 0.02):
        w_cat = 1.0 - w_lgb - w_xgb
        if w_cat < 0: continue
        blend_log = (w_lgb * lgb_blend_log +
                     w_xgb * xgb_blend_log +
                     w_cat * all_val_raw["cat"])
        score = rmsle(y_val_real, np.expm1(np.clip(blend_log, 0, None)))
        if score < best_ensemble:
            best_ensemble = score
            best_weights = (w_lgb, w_xgb, w_cat)

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)
print(f"Baseline       : {rmsle(y_val_real, b_pred):.6f}")
for name, pred_log in all_val_raw.items():
    print(f"{name:<14}: {rmsle(y_val_real, np.expm1(np.clip(pred_log, 0, None))):.6f}")
print(f"LGB blend      : {lgb_best:.6f}")
print(f"XGB blend      : {xgb_best:.6f}")
print(f"Final ensemble : {best_ensemble:.6f}")
print(f"Final weights  : LGB={best_weights[0]:.2f} "
      f"XGB={best_weights[1]:.2f} CAT={best_weights[2]:.2f}")
print("=" * 60)


# --- 19.5 集成搜索可视化 ---
print("\n--- 19.5 集成可视化 ---")

def plot_ensemble_analysis():
    # 三路融合权重搜索热力图
    # 在 (w_lgb, w_xgb, w_cat) 空间采样并记录 RMSLE
    w_lgb_vals = np.arange(0.0, 1.01, 0.02)
    w_xgb_vals = np.arange(0.0, 1.01, 0.02)
    heatmap_data = np.full((len(w_lgb_vals), len(w_xgb_vals)), np.nan)

    for i, wl in enumerate(w_lgb_vals):
        for j, wx in enumerate(w_xgb_vals):
            wc = 1.0 - wl - wx
            if wc < 0:
                continue
            blend = (wl * lgb_blend_log + wx * xgb_blend_log + wc * all_val_raw["cat"])
            heatmap_data[i, j] = rmsle(y_val_real, np.expm1(np.clip(blend, 0, None)))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap_data, origin="lower", aspect="auto",
                   extent=[0, 1, 0, 1], cmap="YlOrRd_r", vmin=np.nanmin(heatmap_data),
                   vmax=np.nanmin(heatmap_data) * 1.02)
    ax.scatter(best_weights[1], best_weights[0], marker="*", s=300,
               c="blue", edgecolors="white", linewidths=1.5,
               zorder=5, label=f"Best ({best_weights[0]:.2f}, {best_weights[1]:.2f}, {best_weights[2]:.2f})")
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("RMSLE")
    ax.set_xlabel("XGB Weight")
    ax.set_ylabel("LGB Weight")
    ax.set_title(f"3-Way Ensemble Weight Search\nBest RMSLE={best_ensemble:.6f}, "
                 f"W=({best_weights[0]:.2f}, {best_weights[1]:.2f}, {best_weights[2]:.2f})")
    ax.legend(loc="upper right", fontsize=9)
    save_and_log(fig, "validation", "06_ensemble_heatmap.png")

    # 模型对比柱状图
    fig, ax = plt.subplots(figsize=(12, 5))
    names, scores_list = [], []
    for name, pred_log in all_val_raw.items():
        names.append(name)
        scores_list.append(rmsle(y_val_real, np.expm1(np.clip(pred_log, 0, None))))
    names += ["LGB Blend", "XGB Blend", "Final Ensemble"]
    scores_list += [lgb_best, xgb_best, best_ensemble]

    colors = (["#4CAF50"] * 3 + ["#2196F3"] * 3 + ["#9C27B0"] +     # LGB×3 + XGB×3 + Cat
              ["#4CAF50", "#2196F3", "#FF5722"])                      # blends
    bars = ax.barh(names, scores_list, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xlabel("RMSLE")
    ax.set_title("Model Comparison — Validation Set")
    ax.invert_yaxis()
    # 标注数值
    for bar, s in zip(bars, scores_list):
        ax.text(bar.get_width() + 0.0003, bar.get_y() + bar.get_height()/2,
                f"{s:.4f}", va="center", fontsize=9)
    save_and_log(fig, "models", "01_model_comparison.png")

    # 偏置修正搜索
    bias_vals = np.arange(0.95, 1.08, 0.005)
    bias_scores = [rmsle(y_val_real, np.expm1(np.clip(
        best_weights[0] * lgb_blend_log + best_weights[1] * xgb_blend_log +
        best_weights[2] * all_val_raw["cat"], 0, None)) * bf)
        for bf in bias_vals]
    plot_bias_factor = float(bias_vals[int(np.argmin(bias_scores))])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bias_vals, bias_scores, "o-", color="#FF5722", markersize=5, lw=1.5)
    ax.axvline(plot_bias_factor, color="green", ls="--", lw=1.2,
               label=f"Best factor={plot_bias_factor:.4f}")
    ax.set_xlabel("Bias Correction Factor")
    ax.set_ylabel("RMSLE")
    ax.set_title("Bias Correction Factor Search")
    ax.legend()
    save_and_log(fig, "validation", "07_bias_correction.png")

    print("  集成可视化完成")

safe_plot(plot_ensemble_analysis, "集成可视化")


# ════════════════════════════════════════════════════════════════
# 20. 全量训练 + submission_no_pseudo
# ════════════════════════════════════════════════════════════════

print("\n=== 20. 全量训练 ===")

# 释放验证阶段的大对象，给全量训练腾内存
# 注意: valid_part, all_val_raw 后续 section 21/22 仍需使用，不能删除
del X_tr, X_val, y_tr, train_part
gc.collect()
print("内存已清理")

X_full = train_data[feature_cols].replace([np.inf, -np.inf], np.nan)
y_full = train_data["sales_log"]
X_test_full = test_data[feature_cols].replace([np.inf, -np.inf], np.nan)

for df in [X_full, X_test_full]:
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")

final_models = {}

for seed in [42, 123, 456]:
    bi = lgb_models[seed].best_iteration or 1200
    final_models[f"lgb_{seed}"] = lgb.train(
        params={**lgb_base_params, "random_state": seed},
        train_set=lgb.Dataset(X_full, label=y_full),
        num_boost_round=int(min(bi * 1.2, 3000)))

for i, (depth, lr, ss, cs) in enumerate(xgb_configs):
    bi = xgb_models[(depth, lr)].best_iteration or 1200
    final_xgb_params = {
        "objective": "reg:squarederror", "tree_method": "hist",
        "learning_rate": lr, "max_depth": depth,
        "min_child_weight": 10, "subsample": ss,
        "colsample_bytree": cs, "gamma": 0.1,
        "reg_alpha": 0.5, "reg_lambda": 1.0,
        "random_state": 42 + i, "verbosity": 0, "nthread": -1,
    }
    if USE_GPU:
        final_xgb_params.update({"device": "cuda"})
    final_models[f"xgb_d{depth}"] = xgb.train(
        params=final_xgb_params,
        dtrain=xgb.DMatrix(X_full, label=y_full),
        num_boost_round=int(min(bi * 1.2, 3000)))

bi = cat_model.best_iteration_ or 1000
final_models["cat"] = CatBoostRegressor(
    **{**cat_params, "iterations": int(min(bi, 2000)),
       "early_stopping_rounds": None, "use_best_model": False})
final_models["cat"].fit(Pool(X_full.values, label=y_full.values), verbose=200)


# ════════════════════════════════════════════════════════════════
# 21. Round 1 预测 + 伪标签 + 最终预测
# ════════════════════════════════════════════════════════════════

print("\n=== 21. 预测 ===")

all_test_raw = {}
for seed in [42, 123, 456]:
    all_test_raw[f"lgb_{seed}"] = final_models[f"lgb_{seed}"].predict(X_test_full)
for (depth, lr, _, _) in xgb_configs:
    all_test_raw[f"xgb_d{depth}"] = final_models[f"xgb_d{depth}"].predict(xgb.DMatrix(X_test_full))
all_test_raw["cat"] = final_models["cat"].predict(X_test_full.values)

lgb_test = sum(lgb_w[k] * all_test_raw[f"lgb_{k}"] for k in lgb_w)
xgb_test = sum(xgb_w[k] * all_test_raw[k] for k in xgb_w)

final_no_pseudo_log = (best_weights[0] * lgb_test +
                       best_weights[1] * xgb_test +
                       best_weights[2] * all_test_raw["cat"])
final_no_pseudo = np.expm1(np.clip(final_no_pseudo_log, 0, None)).clip(0, None)

sub_no_pseudo = pd.DataFrame({
    "id": test_data["id"].values, "sales": final_no_pseudo
}).sort_values("id").reset_index(drop=True)
sub_no_pseudo.to_csv(f"{OUT}/submission_no_pseudo.csv", index=False)
print(f"已生成: {OUT}/submission_no_pseudo.csv")

# Pseudo-labeling（V7 修正: 伪标签后重新搜索 ensemble 权重）
print("\n=== Pseudo-labeling ===")
pseudo_pred = final_no_pseudo.copy()
pseudo_mask = ((pseudo_pred > 1.0) &
               (pseudo_pred < test_data["sf_mean"].fillna(global_mean).values * 3))
print(f"Pseudo samples: {pseudo_mask.sum():,} / {len(test_data):,}")
if USE_PSEUDO and pseudo_mask.sum() > 2000:
    print("  构建伪标签训练集...")
    gc.collect()
    X_pl = np.concatenate([X_full.values, X_test_full.values[pseudo_mask]])
    y_pl = np.concatenate([y_full.values, np.log1p(pseudo_pred[pseudo_mask])])
    del X_full, y_full
    gc.collect()
    print(f"  X_pl: {X_pl.shape}, X_full/y_full 已释放")
    for seed in [42, 123, 456]:
        print(f"  伪标签 LGB seed={seed} 训练中...")
        bi = lgb_models[seed].best_iteration or 1200
        final_models[f"lgb_{seed}"] = lgb.train(
            params={**lgb_base_params, "random_state": seed},
            train_set=lgb.Dataset(X_pl, label=y_pl),
            num_boost_round=int(min(bi * 1.2, 3000)))
    for seed in [42, 123, 456]:
        all_test_raw[f"lgb_{seed}"] = final_models[f"lgb_{seed}"].predict(X_test_full)

    # V7 修正: 伪标签改变了 LGB 预测分布，重新搜索三路融合权重
    # 用验证集的 LGB predictions（需要 pseudo-retrained LGB 在验证集上的预测作为近似）
    # 这里用简化策略: LGB 权重打 9 折，差额分配给 XGB 和 Cat
    old_lgb_w = best_weights[0]
    if old_lgb_w > 0.15:
        discount = 0.10  # LGB weight 降 10%
        new_lgb = old_lgb_w * (1 - discount)
        surplus = old_lgb_w * discount
        new_xgb = best_weights[1] + surplus * 0.6
        new_cat = best_weights[2] + surplus * 0.4
        best_weights = (new_lgb, new_xgb, new_cat)
        print(f"伪标签后权重调整: LGB {old_lgb_w:.3f}→{new_lgb:.3f}, "
              f"XGB {best_weights[1]:.3f}→{new_xgb:.3f}, "
              f"CAT {best_weights[2]:.3f}→{new_cat:.3f}")
else:
    print("Pseudo-labeling skipped (USE_PSEUDO=False or not enough samples)")

lgb_test = sum(lgb_w[k] * all_test_raw[f"lgb_{k}"] for k in lgb_w)
xgb_test = sum(xgb_w[k] * all_test_raw[k] for k in xgb_w)
final_log = (best_weights[0] * lgb_test +
             best_weights[1] * xgb_test +
             best_weights[2] * all_test_raw["cat"])

# V7 新增: 偏置修正（对数→原始空间的 exp 变换有负向偏置）
# 在验证集上搜索最优缩放因子
_lgb_val = sum(lgb_w[k] * all_val_raw[f"lgb_{k}"] for k in lgb_w)
_xgb_val = sum(xgb_w[k] * all_val_raw[k] for k in xgb_w)
_val_log = (best_weights[0] * _lgb_val +
            best_weights[1] * _xgb_val +
            best_weights[2] * all_val_raw["cat"])
bias_best, bias_factor = 999, 1.0
for bf in np.arange(0.98, 1.06, 0.005):
    s = rmsle(y_val_real, np.expm1(np.clip(_val_log, 0, None)) * bf)
    if s < bias_best:
        bias_best, bias_factor = s, float(bf)
print(f"偏置修正因子: {bias_factor:.4f} (val RMSLE: {bias_best:.6f})")

final = np.expm1(np.clip(final_log, 0, None)).clip(0, None) * bias_factor
final = np.clip(final, 0, None)

# V7 新增: 关店强制归零（最近4周销量为0的 store-family 预测直接灌0）
test_w_sales = test_data[["id", "store_nbr", "family"]].merge(
    sf_recent_for_zero, on=["store_nbr", "family"], how="left")
test_w_sales["recent_sum"] = test_w_sales["recent_sum"].fillna(0)
zero_mask_final = test_w_sales["recent_sum"] <= 0
final[zero_mask_final.values] = 0
n_zero = zero_mask_final.sum()
print(f"关店强制归零: {n_zero:,} / {len(test_data):,} rows")

submission = pd.DataFrame({
    "id": test_data["id"].values, "sales": final
}).sort_values("id").reset_index(drop=True)
submission.to_csv(f"{OUT}/submission.csv", index=False)
print(f"已生成: {OUT}/submission.csv")


# ════════════════════════════════════════════════════════════════
# 22. 后处理（内联版，直接输出 family_adaptive）
#    基于 V5 验证有效的 family 自适应融合
#    在 V7 的更强基模型上叠加，预期额外提 0.001-0.004
# ════════════════════════════════════════════════════════════════

print("\n=== 22. 后处理: family 自适应融合 ===")

# --- 22.1 季节性预测函数 ---
def make_seasonal_pred(target_df, history_df):
    """
    对每个 store-family × date，用历史同星期销量的加权平均做季节性预测
    """
    history = history_df.copy()
    target = target_df.copy()
    history["family"] = history["family"].astype(str)
    target["family"] = target["family"].astype(str)
    history["sales"] = history["sales"].clip(lower=0)
    history["sales_log"] = np.log1p(history["sales"])

    seas = target[["date", "store_nbr", "family", "onpromotion"]].copy()

    lags = [7, 14, 21, 28, 35, 42, 49, 56, 364, 371]
    lag_w = {7: 0.18, 14: 0.16, 21: 0.13, 28: 0.11, 35: 0.09,
             42: 0.07, 49: 0.05, 56: 0.04, 364: 0.12, 371: 0.05}

    for lag in lags:
        tmp = history[["date", "store_nbr", "family", "sales_log"]].copy()
        tmp["date"] = tmp["date"] + pd.Timedelta(days=lag)
        tmp = tmp.rename(columns={"sales_log": f"ll_{lag}"})
        seas = seas.merge(tmp, on=["date", "store_nbr", "family"], how="left")

    w_sum = np.zeros(len(seas))
    w_val = np.zeros(len(seas))
    for lag, w in lag_w.items():
        v = seas[f"ll_{lag}"].values
        m = ~np.isnan(v)
        w_val[m] += v[m] * w
        w_sum[m] += w
    seas["seasonal_log"] = np.where(w_sum > 0, w_val / np.maximum(w_sum, 1e-8), np.nan)

    # 兜底
    recent = history[history["date"] >= "2016-01-01"]
    sf_m = recent.groupby(["store_nbr", "family"])["sales_log"].mean()
    fm_m = recent.groupby("family")["sales_log"].mean()
    gm = recent["sales_log"].mean()

    seas = seas.merge(sf_m.rename("sf_l"), on=["store_nbr", "family"], how="left")
    seas = seas.merge(fm_m.rename("fm_l"), on="family", how="left")
    seas["seasonal_log"] = (seas["seasonal_log"].fillna(seas["sf_l"])
                            .fillna(seas["fm_l"]).fillna(gm))
    seas["seasonal_pred"] = np.expm1(seas["seasonal_log"]).clip(lower=0)

    # 促销修正
    ph = history[history["date"] >= "2017-01-01"]
    pm = ph[ph["onpromotion"] > 0].groupby("family")["sales"].mean()
    nm = ph[ph["onpromotion"] == 0].groupby("family")["sales"].mean()
    pr = (pm / nm).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(1.0, 1.5)
    seas["pr"] = seas["family"].map(pr.to_dict()).fillna(1.0)
    promo_m = seas["onpromotion"] > 0
    seas.loc[promo_m, "seasonal_pred"] *= (
        1.0 + 0.05 * np.log1p(seas.loc[promo_m, "onpromotion"]))
    seas["seasonal_pred"] = seas["seasonal_pred"].clip(lower=0)

    # 零销量标记
    sf_s = recent.groupby(["store_nbr", "family"])["sales"].sum()
    seas = seas.merge(sf_s.rename("sf_sum"), on=["store_nbr", "family"], how="left")
    seas["sf_sum"] = seas["sf_sum"].fillna(0)
    seas["is_zero"] = (seas["sf_sum"] <= 1).astype("int8")

    return seas["seasonal_pred"].values, seas["is_zero"].values


# --- 22.2 验证集模型预测 ---
#    （复用 section 19 的 all_val_raw / lgb_w / xgb_w / best_weights）
lgb_val = sum(lgb_w[k] * all_val_raw[f"lgb_{k}"] for k in lgb_w)
xgb_val = sum(xgb_w[k] * all_val_raw[k] for k in xgb_w)
model_val_log = (best_weights[0] * lgb_val +
                 best_weights[1] * xgb_val +
                 best_weights[2] * all_val_raw["cat"])
model_val_pred = np.expm1(np.clip(model_val_log, 0, None)).clip(0, None)
print(f"验证集模型 RMSLE: {rmsle(y_val_real, model_val_pred):.6f}")

# --- 22.3 验证集季节性预测 ---
v_hist = train[train["date"] < valid_part["date"].min()].copy()
v_hist["sales"] = v_hist["sales"].clip(upper=v_hist["sales"].quantile(0.995))
v_seas, v_zero = make_seasonal_pred(valid_part, v_hist)
print(f"验证集季节 RMSLE: {rmsle(y_val_real, v_seas):.6f}")

# --- 22.4 全局最佳权重 ---
gb_w, gb_s = 0.90, 999
for w in np.arange(0.70, 1.001, 0.005):
    s = rmsle(y_val_real, w * model_val_pred + (1 - w) * v_seas)
    if s < gb_s:
        gb_s, gb_w = s, float(w)
print(f"全局最佳权重: {gb_w:.3f}, RMSLE: {gb_s:.6f}")

# --- 22.5 Family 自适应权重 ---
print("搜索 family 自适应权重...")
family_w = []
for fam, grp in valid_part.groupby("family"):
    mask = valid_part["family"] == fam
    y = grp["sales"].values
    mp = model_val_pred[mask.values]
    sp = v_seas[mask.values]
    zr = v_zero[mask.values]
    n = len(grp)

    if n < 200:
        fw = gb_w
    else:
        step = 0.005 if n > 1000 else 0.01
        fw, fb = gb_w, 999
        for w in np.arange(0.70, 1.001, step):
            pred = w * mp + (1 - w) * sp
            pred[zr == 1] = mp[zr == 1]  # 零销量组合直接用模型
            s = rmsle(y, pred)
            if s < fb:
                fb, fw = s, float(w)
    fw = float(np.clip(fw, 0.75, 0.98))
    family_w.append({"family": fam, "model_weight": fw})

family_w = pd.DataFrame(family_w)
print(f"  family 权重: min={family_w['model_weight'].min():.3f}, "
      f"med={family_w['model_weight'].median():.3f}, "
      f"max={family_w['model_weight'].max():.3f}")

# --- 22.6 验证集评估 ---
v_fw = valid_part[["family"]].merge(family_w, on="family", how="left")
v_fw["model_weight"] = v_fw["model_weight"].fillna(gb_w)
v_blend = (v_fw["model_weight"].values * model_val_pred +
           (1 - v_fw["model_weight"].values) * v_seas)
v_blend[v_zero == 1] = model_val_pred[v_zero == 1]
print(f"Family 自适应 RMSLE: {rmsle(y_val_real, v_blend):.6f}")

# --- 22.7 测试集应用 ---
print("生成测试集 seasonal 预测...")
# 对齐: 季节性预测历史也做 99.5% clip（和模型训练保持一致）
train_clipped = train.copy()
train_clipped["sales"] = train_clipped["sales"].clip(upper=train_clipped["sales"].quantile(0.995))
t_seas, t_zero = make_seasonal_pred(test_data, train_clipped)
del train_clipped

# 测试集模型预测
lgb_test = sum(lgb_w[k] * all_test_raw[f"lgb_{k}"] for k in lgb_w)
xgb_test = sum(xgb_w[k] * all_test_raw[k] for k in xgb_w)
model_test_log = (best_weights[0] * lgb_test +
                  best_weights[1] * xgb_test +
                  best_weights[2] * all_test_raw["cat"])
model_test = np.expm1(np.clip(model_test_log, 0, None)).clip(0, None)

# Family 融合
t_fw = test_data[["family"]].merge(family_w, on="family", how="left")
t_fw["model_weight"] = t_fw["model_weight"].fillna(gb_w)
t_blend = (t_fw["model_weight"].values * model_test +
           (1 - t_fw["model_weight"].values) * t_seas)
t_blend[t_zero == 1] = model_test[t_zero == 1]

# 全局融合
t_global = gb_w * model_test + (1 - gb_w) * t_seas
t_global[t_zero == 1] = model_test[t_zero == 1]

# V7 修正: 偏置修正 + 关店强制归零
# 搜索最优缩放因子
bf_best, bf_factor = 999, 1.0
for factor in np.arange(0.98, 1.06, 0.005):
    s = rmsle(y_val_real, v_blend * factor)
    if s < bf_best:
        bf_best, bf_factor = s, float(factor)
print(f"后处理偏置修正因子: {bf_factor:.4f}")

# 关店强制归零（复用提前提取的信息，data 已释放）
t_merge = test_data[["id", "store_nbr", "family"]].merge(
    sf_recent_for_zero, on=["store_nbr", "family"], how="left")
t_merge["recent_sum"] = t_merge["recent_sum"].fillna(0)
zero_mask = t_merge["recent_sum"].values <= 0

# 输出（应用偏置修正 + 关店归零）
for name, pred, fname in [
    ("global", t_global, "submission_blend_global.csv"),
    ("family", t_blend,  "submission_family_adaptive.csv"),
]:
    pred = pred * bf_factor           # 偏置修正
    pred = np.clip(pred, 0, None)
    pred[zero_mask] = 0               # 关店归零
    sub = pd.DataFrame({
        "id": test_data["id"].values, "sales": pred
    }).sort_values("id").reset_index(drop=True)
    sub.to_csv(f"{OUT}/{fname}", index=False)
    print(f"  {OUT}/{fname} (均值={sub['sales'].mean():.2f})")


# ════════════════════════════════════════════════════════════════
# 22.8 验证集可视化分析
# ════════════════════════════════════════════════════════════════

print("\n--- 22.8 验证集可视化 ---")


def plot_validation_analysis():
    """验证集诊断图表"""

    # 22.8.1 预测值 vs 真实值散点图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Model only
    ax = axes[0]
    ax.scatter(y_val_real, model_val_pred, alpha=0.15, s=2, c="#2196F3", edgecolors="none")
    ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], "r--", lw=1.2, alpha=0.7, label="y=x")
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title(f"Model Only (RMSLE={rmsle(y_val_real, model_val_pred):.4f})")
    ax.legend()

    # Seasonal only
    ax = axes[1]
    ax.scatter(y_val_real, v_seas, alpha=0.15, s=2, c="#4CAF50", edgecolors="none")
    ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], "r--", lw=1.2, alpha=0.7, label="y=x")
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title(f"Seasonal Only (RMSLE={rmsle(y_val_real, v_seas):.4f})")
    ax.legend()

    # Family adaptive
    ax = axes[2]
    ax.scatter(y_val_real, v_blend * bf_factor, alpha=0.15, s=2, c="#FF9800", edgecolors="none")
    ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], "r--", lw=1.2, alpha=0.7, label="y=x")
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title(f"Family Adaptive (RMSLE={rmsle(y_val_real, v_blend * bf_factor):.4f})")
    ax.legend()

    fig.suptitle("Predicted vs Actual — Validation Set", fontsize=14, y=1.01)
    save_and_log(fig, "validation", "01_pred_vs_actual.png")

    # 22.8.2 残差分布
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    preds = [
        ("Model Only", model_val_pred, "#2196F3"),
        ("Seasonal Only", v_seas, "#4CAF50"),
        ("Family Adaptive", v_blend * bf_factor, "#FF9800"),
    ]
    for ax, (name, pred, color) in zip(axes, preds):
        residuals = np.log1p(np.clip(pred, 0, None)) - np.log1p(y_val_real)
        ax.hist(residuals, bins=80, color=color, alpha=0.7, edgecolor="white", lw=0.3)
        ax.axvline(0, color="red", ls="--", lw=1.2)
        ax.axvline(np.mean(residuals), color="black", ls="-", lw=1.0,
                   label=f"Mean={np.mean(residuals):.4f}")
        ax.set_xlabel("Log Residual (log(1+pred) - log(1+actual))")
        ax.set_ylabel("Count")
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.suptitle("Residual Distribution — Validation Set", fontsize=14, y=1.01)
    save_and_log(fig, "validation", "02_residual_distribution.png")

    # 22.8.3 按 family 的 RMSLE
    fam_scores = []
    for fam, grp in valid_part.groupby("family"):
        mask = valid_part["family"] == fam
        y = grp["sales"].values
        m_rmsle = rmsle(y, model_val_pred[mask.values])
        f_rmsle = rmsle(y, (v_blend * bf_factor)[mask.values])
        fam_scores.append({
            "family": fam, "model_rmsle": m_rmsle,
            "family_rmsle": f_rmsle, "n": len(grp),
            "improvement": m_rmsle - f_rmsle,
        })
    fam_df = pd.DataFrame(fam_scores).sort_values("family_rmsle", ascending=True)

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(fam_df))
    w = 0.35
    ax.bar(x - w/2, fam_df["model_rmsle"], w, label="Model Only", color="#2196F3", alpha=0.8)
    ax.bar(x + w/2, fam_df["family_rmsle"], w, label="Family Adaptive", color="#FF9800", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(fam_df["family"], rotation=90, fontsize=7)
    ax.set_ylabel("RMSLE")
    ax.set_title("RMSLE by Family — Model vs Family Adaptive")
    ax.legend()
    ax.axhline(y=gb_s, color="green", ls="--", lw=0.8, alpha=0.5, label=f"Global={gb_s:.4f}")
    save_and_log(fig, "validation", "03_rmsle_by_family.png")

    # 22.8.4 Family 自适应权重分布
    fig, ax = plt.subplots(figsize=(12, 5))
    fw_sorted = family_w.sort_values("model_weight")
    colors = ["#2196F3" if w < gb_w else "#FF9800" for w in fw_sorted["model_weight"]]
    ax.bar(range(len(fw_sorted)), fw_sorted["model_weight"], color=colors, alpha=0.8)
    ax.axhline(y=gb_w, color="green", ls="--", lw=1.5, label=f"Global weight = {gb_w:.3f}")
    ax.set_xticks(range(len(fw_sorted)))
    ax.set_xticklabels(fw_sorted["family"], rotation=90, fontsize=7)
    ax.set_ylabel("Model Weight")
    ax.set_title("Family Adaptive Weights (blue=more model, orange=more seasonal)")
    ax.legend()
    ax.set_ylim(0.7, 1.0)
    save_and_log(fig, "validation", "04_family_weights.png")

    # 22.8.5 残差时序（按日期聚合）
    vp_with_pred = valid_part.copy()
    vp_with_pred["model_pred"] = model_val_pred
    vp_with_pred["family_pred"] = v_blend * bf_factor
    vp_with_pred["model_resid"] = np.log1p(np.clip(model_val_pred, 0, None)) - np.log1p(y_val_real)
    vp_with_pred["family_resid"] = np.log1p(np.clip(v_blend * bf_factor, 0, None)) - np.log1p(y_val_real)

    daily_resid = vp_with_pred.groupby("date")[["model_resid", "family_resid"]].mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_resid.index, daily_resid["model_resid"], color="#2196F3", lw=1.2, alpha=0.8, label="Model Only")
    ax.plot(daily_resid.index, daily_resid["family_resid"], color="#FF9800", lw=1.2, alpha=0.8, label="Family Adaptive")
    ax.axhline(0, color="red", ls="--", lw=0.8)
    ax.fill_between(daily_resid.index, 0, daily_resid["family_resid"],
                    alpha=0.15, color="#FF9800")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Log Residual")
    ax.set_title("Residual Over Time — Validation Period")
    ax.legend()
    fig.autofmt_xdate()
    save_and_log(fig, "validation", "05_residual_over_time.png")

    # 22.8.6 集成权重搜索热力图
    # (记录 ensemble 搜索空间)
    # 这个在 section 19 之后做，此处跳过
    print("  验证集可视化完成")


safe_plot(plot_validation_analysis, "验证集可视化")


print("\n" + "=" * 60)
print("V7 完成")
print("=" * 60)
print(f"特征数: {len(feature_cols)}")
print(f"傅里叶频率: [3.5, 7, 30, 365] × 多阶")
print(f"验证窗口: 31天")
print(f"后处理: family 自适应 + 全局融合（内联）")
print(f"\n提交文件:")
print(f"  1. {OUT}/submission_no_pseudo.csv")
print(f"  2. {OUT}/submission.csv")
print(f"  3. {OUT}/submission_blend_global.csv")
print(f"  4. {OUT}/submission_family_adaptive.csv  ← 预期最强")
print("=" * 60)


# ════════════════════════════════════════════════════════════════
# 23. 实验报告打包
# ════════════════════════════════════════════════════════════════

print("\n=== 23. 生成实验报告 ===")

# --- 23.1 汇总文本报告 ---
report_lines = []
report_lines.append("=" * 70)
report_lines.append("Store Sales V7 — 实验报告")
report_lines.append("=" * 70)
report_lines.append("")
report_lines.append(f"特征数量: {len(feature_cols)}")
report_lines.append(f"傅里叶频率: [3.5, 7, 30, 365] × 多阶")
report_lines.append(f"验证窗口: 31 天 (2017-07-16 ~ 2017-08-15)")
report_lines.append(f"训练范围: 2014-01-01 ~ 2017-07-15")
report_lines.append(f"Optuna trials: {N_TRIALS}")
report_lines.append("")
report_lines.append("--- 模型架构 ---")
report_lines.append("  LGB × 3 (seed 42/123/456)")
report_lines.append("  XGB × 3 (depth 9/7/11)")
report_lines.append("  CatBoost × 1")
report_lines.append(f"  LGB blend weights: {lgb_w}")
report_lines.append(f"  XGB blend weights: {xgb_w}")
report_lines.append("")
report_lines.append("--- 验证集结果 ---")
report_lines.append(f"  Baseline RMSLE:           {rmsle(y_val_real, b_pred):.6f}")
for name, pred_log in all_val_raw.items():
    report_lines.append(f"  {name:<20} RMSLE: {rmsle(y_val_real, np.expm1(np.clip(pred_log, 0, None))):.6f}")
report_lines.append(f"  LGB blend RMSLE:          {lgb_best:.6f}")
report_lines.append(f"  XGB blend RMSLE:          {xgb_best:.6f}")
report_lines.append(f"  Final ensemble RMSLE:     {best_ensemble:.6f}")
report_lines.append(f"  Final weights: LGB={best_weights[0]:.2f} XGB={best_weights[1]:.2f} CAT={best_weights[2]:.2f}")
report_lines.append(f"  Bias correction factor:   {bias_factor:.4f}")
report_lines.append("")
report_lines.append("--- 后处理结果 ---")
report_lines.append(f"  模型验证 RMSLE:           {rmsle(y_val_real, model_val_pred):.6f}")
report_lines.append(f"  季节验证 RMSLE:           {rmsle(y_val_real, v_seas):.6f}")
report_lines.append(f"  全局融合 RMSLE:           {gb_s:.6f} (model_w={gb_w:.3f})")
report_lines.append(f"  Family 自适应 RMSLE:      {rmsle(y_val_real, v_blend):.6f}")
report_lines.append(f"  后处理偏置修正因子:        {bf_factor:.4f}")
report_lines.append("")
report_lines.append("--- 关键创新点 ---")
report_lines.append("  1. Periodogram 驱动傅里叶频率 [3.5, 7, 30, 365]")
report_lines.append("  2. 趋势特征 (Trend + Trend² + Family×Trend + Cluster×Trend)")
report_lines.append("  3. Rank 编码替代复杂交互")
report_lines.append("  4. STL 风格分解 (trend/seasonal/resid/strength/chg)")
report_lines.append("  5. 异常值 clip (per store-family 99.5%)")
report_lines.append("  6. 发薪日特征 + 关店检测 + 节假日交互")
report_lines.append("  7. 31 天扩展验证窗口")
report_lines.append("  8. Family 自适应后处理融合")
report_lines.append("  9. 偏置修正 + 关店强制归零")
report_lines.append("=" * 70)

report_path = os.path.join(REPORT_DIR, "experiment_summary.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"[SAVED] experiment_summary.txt")

# --- 23.2 特征列表 ---
with open(os.path.join(REPORT_DIR, "feature_list.txt"), "w", encoding="utf-8") as f:
    f.write(f"Total features: {len(feature_cols)}\n\n")
    for i, c in enumerate(feature_cols):
        f.write(f"  {i:3d}. {c}\n")
print(f"[SAVED] feature_list.txt")

# --- 23.3 打包 ZIP ---
zip_path = os.path.join(OUT, "experiment_report.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(REPORT_DIR):
        for fname in files:
            full = os.path.join(root, fname)
            arcname = os.path.relpath(full, REPORT_DIR)
            zf.write(full, arcname)
    # 也打包提交文件
    for sub_file in [
        "submission_no_pseudo.csv", "submission.csv",
        "submission_blend_global.csv", "submission_family_adaptive.csv"
    ]:
        sp = os.path.join(OUT, sub_file)
        if os.path.exists(sp):
            zf.write(sp, os.path.basename(sp))

zip_size_mb = os.path.getsize(zip_path) / 1024**2
print(f"\n{'=' * 60}")
print(f"实验报告已打包: {zip_path}")
print(f"文件大小: {zip_size_mb:.1f} MB")
print(f"包含: {len(os.listdir(REPORT_DIR))} 个子目录 + 汇总文件")
print(f"可直接在 Kaggle Output 面板下载")
print(f"{'=' * 60}")
