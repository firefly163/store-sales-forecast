# ============================================================
# Store Sales - Time Series Forecasting 完整稳定版
# LightGBM + 多源数据融合 + 时间序列滞后特征 + 移动平均特征
# ============================================================

import os
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from lightgbm import LGBMRegressor


# ============================================================
# 0. 基础设置
# ============================================================

DATA_PATH = "/kaggle/input/competitions/store-sales-time-series-forecasting"
OUTPUT_PATH = "/kaggle/working"

print("数据目录：", DATA_PATH)
print(os.listdir(DATA_PATH))


# ============================================================
# 1. 读取数据
# ============================================================

train = pd.read_csv(
    f"{DATA_PATH}/train.csv",
    parse_dates=["date"],
    dtype={
        "store_nbr": "int16",
        "family": "category",
        "sales": "float32",
        "onpromotion": "int32"
    }
)

test = pd.read_csv(
    f"{DATA_PATH}/test.csv",
    parse_dates=["date"],
    dtype={
        "store_nbr": "int16",
        "family": "category",
        "onpromotion": "int32"
    }
)

stores = pd.read_csv(f"{DATA_PATH}/stores.csv")
oil = pd.read_csv(f"{DATA_PATH}/oil.csv", parse_dates=["date"])
holidays = pd.read_csv(f"{DATA_PATH}/holidays_events.csv", parse_dates=["date"])
transactions = pd.read_csv(f"{DATA_PATH}/transactions.csv", parse_dates=["date"])
sample_submission = pd.read_csv(f"{DATA_PATH}/sample_submission.csv")

print("train:", train.shape)
print("test:", test.shape)
print("stores:", stores.shape)
print("oil:", oil.shape)
print("holidays:", holidays.shape)
print("transactions:", transactions.shape)


# ============================================================
# 2. 定义评价函数 RMSLE
# ============================================================

def rmsle(y_true, y_pred):
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


# ============================================================
# 3. 合并 train 和 test，统一做特征工程
# ============================================================

train["is_train"] = 1
test["is_train"] = 0
test["sales"] = np.nan

data = pd.concat([train, test], axis=0, ignore_index=True)

print("合并后数据：", data.shape)


# ============================================================
# 4. 合并门店信息
# ============================================================

stores = stores.rename(columns={"type": "store_type"})
data = data.merge(stores, on="store_nbr", how="left")


# ============================================================
# 5. 油价特征
# ============================================================

full_dates = pd.DataFrame({
    "date": pd.date_range(data["date"].min(), data["date"].max(), freq="D")
})

oil = full_dates.merge(oil, on="date", how="left")
oil["dcoilwtico"] = oil["dcoilwtico"].interpolate()
oil["dcoilwtico"] = oil["dcoilwtico"].bfill()
oil["dcoilwtico"] = oil["dcoilwtico"].ffill()

oil["oil_diff_1"] = oil["dcoilwtico"].diff().fillna(0)
oil["oil_rolling_mean_7"] = oil["dcoilwtico"].rolling(7, min_periods=1).mean()
oil["oil_rolling_mean_14"] = oil["dcoilwtico"].rolling(14, min_periods=1).mean()

data = data.merge(oil, on="date", how="left")


# ============================================================
# 6. 节假日特征
# ============================================================

holidays_valid = holidays[
    holidays["transferred"].astype(str).str.lower() != "true"
].copy()

# 补班日
work_days = holidays_valid[
    holidays_valid["type"] == "Work Day"
][["date"]].drop_duplicates()
work_days["is_work_day"] = 1

data = data.merge(work_days, on="date", how="left")
data["is_work_day"] = data["is_work_day"].fillna(0)

# 全国节假日
national_holidays = holidays_valid[
    (holidays_valid["locale"] == "National") &
    (holidays_valid["type"] != "Work Day")
][["date"]].drop_duplicates()
national_holidays["is_national_holiday"] = 1

data = data.merge(national_holidays, on="date", how="left")
data["is_national_holiday"] = data["is_national_holiday"].fillna(0)

# 地区节假日
regional_holidays = holidays_valid[
    (holidays_valid["locale"] == "Regional") &
    (holidays_valid["type"] != "Work Day")
][["date", "locale_name"]].drop_duplicates()

regional_holidays = regional_holidays.rename(columns={"locale_name": "state"})
regional_holidays["is_regional_holiday"] = 1

data = data.merge(regional_holidays, on=["date", "state"], how="left")
data["is_regional_holiday"] = data["is_regional_holiday"].fillna(0)

# 城市节假日
local_holidays = holidays_valid[
    (holidays_valid["locale"] == "Local") &
    (holidays_valid["type"] != "Work Day")
][["date", "locale_name"]].drop_duplicates()

local_holidays = local_holidays.rename(columns={"locale_name": "city"})
local_holidays["is_local_holiday"] = 1

data = data.merge(local_holidays, on=["date", "city"], how="left")
data["is_local_holiday"] = data["is_local_holiday"].fillna(0)

# 综合节假日
data["is_holiday"] = (
    data["is_national_holiday"] +
    data["is_regional_holiday"] +
    data["is_local_holiday"]
)

data["is_holiday"] = (data["is_holiday"] > 0).astype("int8")

# 如果是补班日，则不算普通节假日
data.loc[data["is_work_day"] == 1, "is_holiday"] = 0


# ============================================================
# 7. 日期特征
# ============================================================

data["year"] = data["date"].dt.year.astype("int16")
data["month"] = data["date"].dt.month.astype("int8")
data["day"] = data["date"].dt.day.astype("int8")
data["dayofweek"] = data["date"].dt.dayofweek.astype("int8")
data["weekofyear"] = data["date"].dt.isocalendar().week.astype("int16")
data["quarter"] = data["date"].dt.quarter.astype("int8")
data["dayofyear"] = data["date"].dt.dayofyear.astype("int16")

data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype("int8")
data["is_month_start"] = data["date"].dt.is_month_start.astype("int8")
data["is_month_end"] = data["date"].dt.is_month_end.astype("int8")
data["is_quarter_start"] = data["date"].dt.is_quarter_start.astype("int8")
data["is_quarter_end"] = data["date"].dt.is_quarter_end.astype("int8")

data["time_idx"] = (data["date"] - data["date"].min()).dt.days.astype("int32")


# ============================================================
# 8. 促销特征
# ============================================================

data["onpromotion"] = data["onpromotion"].fillna(0)
data["onpromotion_log"] = np.log1p(data["onpromotion"]).astype("float32")
data["has_promotion"] = (data["onpromotion"] > 0).astype("int8")


# ============================================================
# 9. 交易量统计特征
# ============================================================

transactions["store_nbr"] = transactions["store_nbr"].astype("int16")
transactions["month"] = transactions["date"].dt.month.astype("int8")
transactions["dayofweek"] = transactions["date"].dt.dayofweek.astype("int8")

txn_stats = transactions.groupby(
    ["store_nbr", "month", "dayofweek"],
    as_index=False
)["transactions"].mean()

txn_stats = txn_stats.rename(columns={
    "transactions": "transactions_mean_store_month_dow"
})

data = data.merge(
    txn_stats,
    on=["store_nbr", "month", "dayofweek"],
    how="left"
)

data["transactions_mean_store_month_dow"] = data[
    "transactions_mean_store_month_dow"
].fillna(data["transactions_mean_store_month_dow"].median())


# ============================================================
# 10. 时间序列滞后特征和移动统计特征
# ============================================================

print("开始构造时间序列特征...")

data["sales"] = data["sales"].astype("float32")
data["sales_log"] = np.log1p(data["sales"].clip(lower=0))

data = data.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

group_cols = ["store_nbr", "family"]

# 测试集为未来 16 天，因此使用 lag >= 16，避免用到测试集未知销量
lag_list = [16, 17, 18, 21, 28, 56, 112, 364]

for lag in lag_list:
    col = f"sales_lag_{lag}"
    data[col] = data.groupby(group_cols, observed=True)["sales_log"].shift(lag)
    data[col] = data[col].fillna(0).astype("float32")

# 移动平均和移动标准差，全部基于 shift(16) 之后的历史销量
rolling_windows = [7, 14, 28, 56, 112]

for window in rolling_windows:
    mean_col = f"sales_roll_mean_{window}_lag16"
    std_col = f"sales_roll_std_{window}_lag16"

    data[mean_col] = data.groupby(group_cols, observed=True)["sales_log"].transform(
        lambda x: x.shift(16).rolling(window, min_periods=1).mean()
    )

    data[std_col] = data.groupby(group_cols, observed=True)["sales_log"].transform(
        lambda x: x.shift(16).rolling(window, min_periods=2).std()
    )

    data[mean_col] = data[mean_col].fillna(0).astype("float32")
    data[std_col] = data[std_col].fillna(0).astype("float32")

# 促销滞后特征
for lag in [16, 28, 56]:
    col = f"promo_lag_{lag}"
    data[col] = data.groupby(group_cols, observed=True)["onpromotion"].shift(lag)
    data[col] = data[col].fillna(0).astype("float32")

print("时间序列特征构造完成")


# ============================================================
# 11. 类别变量编码
# 注意：不要删除原始 family 列，后面 baseline 和分析还要用
# ============================================================

cat_cols = ["family", "city", "state", "store_type"]

for col in cat_cols:
    le = LabelEncoder()
    data[col + "_enc"] = le.fit_transform(data[col].astype(str))

print("类别编码完成")


# ============================================================
# 12. Baseline 历史均值模型
# ============================================================

print("计算 Baseline...")

raw_base = train[["date", "store_nbr", "family", "sales"]].copy()
raw_base["family"] = raw_base["family"].astype(str)

valid_start_date = pd.Timestamp("2017-07-31")

base_train = raw_base[raw_base["date"] < valid_start_date].copy()
base_valid = raw_base[raw_base["date"] >= valid_start_date].copy()

base_means = base_train.groupby(
    ["store_nbr", "family"],
    as_index=False
)["sales"].mean()

base_means = base_means.rename(columns={"sales": "base_pred"})

base_valid = base_valid.merge(
    base_means,
    on=["store_nbr", "family"],
    how="left"
)

global_mean = base_train["sales"].mean()
base_valid["base_pred"] = base_valid["base_pred"].fillna(global_mean)

baseline_score = rmsle(base_valid["sales"].values, base_valid["base_pred"].values)

print("Baseline RMSLE:", baseline_score)


# ============================================================
# 13. 构造模型训练特征
# ============================================================

lag_roll_cols = [
    col for col in data.columns
    if col.startswith("sales_lag_")
    or col.startswith("sales_roll_")
    or col.startswith("promo_lag_")
]

feature_cols = [
    "store_nbr",
    "family_enc",
    "onpromotion",
    "onpromotion_log",
    "has_promotion",

    "city_enc",
    "state_enc",
    "store_type_enc",
    "cluster",

    "dcoilwtico",
    "oil_diff_1",
    "oil_rolling_mean_7",
    "oil_rolling_mean_14",

    "transactions_mean_store_month_dow",

    "year",
    "month",
    "day",
    "dayofweek",
    "weekofyear",
    "quarter",
    "dayofyear",
    "time_idx",

    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",

    "is_work_day",
    "is_national_holiday",
    "is_regional_holiday",
    "is_local_holiday",
    "is_holiday"
] + lag_roll_cols

print("特征数量:", len(feature_cols))
print("前 20 个特征:", feature_cols[:20])


# ============================================================
# 14. 划分训练集、验证集、测试集
# ============================================================

train_data = data[data["is_train"] == 1].copy()
test_data = data[data["is_train"] == 0].copy()

# 为了训练速度和数据时效性，使用 2016 年以后的数据
train_data = train_data[train_data["date"] >= "2016-01-01"].copy()

valid_start_date = pd.Timestamp("2017-07-31")

train_part = train_data[train_data["date"] < valid_start_date].copy()
valid_part = train_data[train_data["date"] >= valid_start_date].copy()

X_train = train_part[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y_train = train_part["sales_log"]

X_valid = valid_part[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y_valid_log = valid_part["sales_log"]
y_valid_real = valid_part["sales"].values

X_test = test_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

# 降低内存占用
for col in X_train.columns:
    if X_train[col].dtype == "float64":
        X_train[col] = X_train[col].astype("float32")
        X_valid[col] = X_valid[col].astype("float32")
        X_test[col] = X_test[col].astype("float32")

print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("X_test:", X_test.shape)

gc.collect()


# ============================================================
# 15. LightGBM 模型训练
# ============================================================

print("开始训练 LightGBM...")

model = LGBMRegressor(
    objective="regression",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=128,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.3,
    random_state=42,
    n_jobs=-1,
    force_col_wise=True,
    verbosity=-1
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid_log)],
    eval_metric="rmse",
    callbacks=[
        lgb.early_stopping(stopping_rounds=150),
        lgb.log_evaluation(period=100)
    ]
)

best_iter = model.best_iteration_

if best_iter is None or best_iter <= 0:
    best_iter = 1500

print("best_iteration:", best_iter)


# ============================================================
# 16. 验证集评估
# ============================================================

valid_pred_log = model.predict(X_valid, num_iteration=best_iter)
valid_pred = np.expm1(valid_pred_log)
valid_pred = np.clip(valid_pred, 0, None)

lgb_score = rmsle(y_valid_real, valid_pred)

print("\n" + "=" * 50)
print("模型表现")
print("=" * 50)
print(f"Baseline RMSLE : {baseline_score:.6f}")
print(f"LightGBM RMSLE : {lgb_score:.6f}")

improve = (baseline_score - lgb_score) / baseline_score * 100
print(f"相对 Baseline 提升：{improve:.2f}%")
print("=" * 50)


# ============================================================
# 17. 保存特征重要性
# ============================================================

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv(f"{OUTPUT_PATH}/feature_importance.csv", index=False)

plt.figure(figsize=(10, 8))
top_imp = importance.head(25).sort_values("importance")
plt.barh(top_imp["feature"], top_imp["importance"])
plt.title("LightGBM Top 25 Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

print("特征重要性已保存：")
print(f"{OUTPUT_PATH}/feature_importance.csv")
print(f"{OUTPUT_PATH}/feature_importance.png")


# ============================================================
# 18. 使用全部训练数据重新训练最终模型
# ============================================================

print("使用全部训练数据重新训练最终模型...")

X_full = train_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y_full = train_data["sales_log"]

for col in X_full.columns:
    if X_full[col].dtype == "float64":
        X_full[col] = X_full[col].astype("float32")

final_model = LGBMRegressor(
    objective="regression",
    n_estimators=best_iter,
    learning_rate=0.03,
    num_leaves=128,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.3,
    random_state=42,
    n_jobs=-1,
    force_col_wise=True,
    verbosity=-1
)

final_model.fit(X_full, y_full)


# ============================================================
# 19. 预测测试集，生成提交文件
# ============================================================

print("预测测试集...")

test_pred_log = final_model.predict(X_test)
test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({
    "id": test_data["id"].values,
    "sales": test_pred
})

submission = submission.sort_values("id").reset_index(drop=True)

submission.to_csv(f"{OUTPUT_PATH}/submission.csv", index=False)

print("提交文件已生成：")
print(f"{OUTPUT_PATH}/submission.csv")

print(submission.head())
print(submission.tail())

print("\n全部完成！请下载 /kaggle/working/submission.csv 并提交到 Kaggle。")