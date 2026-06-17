"""
配置文件 — Store Sales Time Series Forecasting
==============================================
Kaggle经典赛题：预测厄瓜多尔 Favorita 连锁店的商品销量
"""

import os
from pathlib import Path

# ============ 项目路径 ============
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "figures"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"

for d in [DATA_DIR, MODELS_DIR, FIGURES_DIR, SUBMISSIONS_DIR]:
    d.mkdir(exist_ok=True, parents=True)

# ============ 数据文件 ============
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
STORES_CSV = "stores.csv"
OIL_CSV = "oil.csv"
HOLIDAYS_CSV = "holidays_events.csv"
TRANSACTIONS_CSV = "transactions.csv"
SAMPLE_SUBMISSION = "sample_submission.csv"

# ============ 时间范围 ============
# 训练数据: 2013-01-01 ~ 2017-08-15
# 测试数据: 2017-08-16 ~ 2017-08-31
TRAIN_START = "2013-01-01"
TRAIN_END = "2017-08-15"
TEST_START = "2017-08-16"
TEST_END = "2017-08-31"

# ============ 特征工程参数 ============
# 滞后特征窗口（天）
LAG_DAYS = [1, 7, 14, 21, 28, 60, 90]

# 滚动窗口统计（天）
ROLLING_WINDOWS = [7, 14, 28, 60]

# 指数加权移动平均窗口
EWM_WINDOWS = [7, 14, 28]

# 日期特征
DATE_FEATURES = [
    "year", "month", "day", "dayofweek",
    "dayofyear", "weekofyear", "quarter",
    "is_weekend", "is_month_start", "is_month_end",
    "days_to_end_of_month", "days_from_start_of_month"
]

# ============ 模型参数 ============
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "num_leaves": 256,
    "max_depth": 10,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l1": 0.5,
    "lambda_l2": 0.5,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "gamma": 0.1,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1,
}

# ============ 训练参数 ============
VAL_SPLIT_DATE = "2017-07-01"  # 最后1.5个月作为验证集
CV_FOLDS = 3                    # 时间序列交叉验证折数
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 200

# ============ 预测 ============
# 商店和商品族列表（从数据中动态获取）
# 共54家店 × 33个商品族 = 1782个组合
N_STORES = 54
N_FAMILIES = 33

# 随机种子
SEED = 42
