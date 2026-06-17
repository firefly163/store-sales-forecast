"""
特征工程模块
=============
时间序列预测的核心在于特征工程。本模块从原始数据中提取丰富的特征。

特征分类：
1. 日期特征 — 年/月/日/周/季度等时间属性
2. 滞后特征 — 过去N天的销售额 (lag features)
3. 滚动窗口特征 — 过去N天的均值/标准差/最大/最小值
4. 指数加权移动平均 (EWM) — 对近期数据赋予更高权重
5. 促销特征 — 促销商品数、促销强度
6. 节假日特征 — 节假日类型、节前/节后窗口
7. 商店/商品族特征 — 编码、交互特征
8. 外部特征 — 油价、交易量
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder

from config import (
    LAG_DAYS, ROLLING_WINDOWS, EWM_WINDOWS, DATE_FEATURES,
    SEED
)
from utils import print_section


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从date列提取丰富的日期特征

    这些特征帮助模型理解：
    - 周期性模式（周几、月中哪几天等）
    - 季节性模式（月份、季度）
    - 趋势（年份）
    """
    df = df.copy()
    d = df["date"]

    df["year"] = d.dt.year.astype("int16")
    df["month"] = d.dt.month.astype("int8")
    df["day"] = d.dt.day.astype("int8")
    df["dayofweek"] = d.dt.dayofweek.astype("int8")
    df["dayofyear"] = d.dt.dayofyear.astype("int16")
    df["weekofyear"] = d.dt.isocalendar().week.astype("int8")
    df["quarter"] = d.dt.quarter.astype("int8")
    df["is_weekend"] = (d.dt.dayofweek >= 5).astype("int8")
    df["is_month_start"] = d.dt.is_month_start.astype("int8")
    df["is_month_end"] = d.dt.is_month_end.astype("int8")
    df["days_to_end_of_month"] = (d.dt.days_in_month - d.dt.day).astype("int8")
    df["days_from_start_of_month"] = (d.dt.day - 1).astype("int8")

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    对类别特征进行标签编码

    将 store_nbr, family, city, state, type 等转换为数值
    使用 LabelEncoder 而非 OneHot 以保持维度可控
    """
    df = df.copy()
    categoricals = ["family", "city", "state", "type", "holiday_type"]

    for col in categoricals:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            df.drop(columns=[col], inplace=True)

    return df


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    group_cols: List[str] = ["store_nbr", "family"],
    lag_days: List[int] = None,
) -> pd.DataFrame:
    """
    创建滞后特征 (Lag Features)

    滞后特征是时间序列预测最重要的特征之一。
    对于每个 (store_nbr, family) 组合，创建过去第t天的销售额特征。

    例如：
    - lag_1: 昨天的销售额
    - lag_7: 一周前的销售额
    - lag_28: 四周前的销售额

    注意：需要先按日期排序，然后按组合分组创建滞后特征
    """
    if lag_days is None:
        lag_days = LAG_DAYS

    df = df.sort_values(["date"] + group_cols).reset_index(drop=True)

    for lag in lag_days:
        df[f"lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag)

    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    group_cols: List[str] = ["store_nbr", "family"],
    windows: List[int] = None,
) -> pd.DataFrame:
    """
    创建滚动窗口统计特征

    对每个 (store_nbr, family) 组合，计算过去N天的统计量：
    - rolling_mean_N: 过去N天均值（反映近期平均水平）
    - rolling_std_N: 过去N天标准差（反映波动性）
    - rolling_min_N/N_max_N: 过去N天极值
    - rolling_median_N: 过去N天中位数（抗异常值）

    滚动窗口特征比单点滞后特征更稳定，能更好地捕捉趋势。
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    df = df.sort_values(["date"] + group_cols).reset_index(drop=True)

    for w in windows:
        grp = df.groupby(group_cols)[target_col]
        df[f"rolling_mean_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).std()
        )
        df[f"rolling_min_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).min()
        )
        df[f"rolling_max_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).max()
        )
        df[f"rolling_median_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).median()
        )

    return df


def create_ewm_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    group_cols: List[str] = ["store_nbr", "family"],
    windows: List[int] = None,
) -> pd.DataFrame:
    """
    创建指数加权移动平均特征 (EWM)

    相比普通滚动均值，EWM对近期数据赋予更高的权重，
    对趋势变化的响应更快。

    ewm_mean_N: 以span=N的指数加权均值
    ewm_std_N: 以span=N的指数加权标准差
    """
    if windows is None:
        windows = EWM_WINDOWS

    df = df.sort_values(["date"] + group_cols).reset_index(drop=True)

    for w in windows:
        grp = df.groupby(group_cols)[target_col]
        df[f"ewm_mean_{w}"] = grp.transform(
            lambda x: x.shift(1).ewm(span=w, adjust=False).mean()
        )
        df[f"ewm_std_{w}"] = grp.transform(
            lambda x: x.shift(1).ewm(span=w, adjust=False).std()
        )

    return df


def create_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建促销相关特征

    促销对销售额有显著影响（见EDA分析），需要提取：
    - 是否在促销
    - 促销商品数量
    - 过去的促销强度滚动统计
    """
    df = df.copy()

    # 是否有促销
    df["has_promotion"] = (df["onpromotion"] > 0).astype("int8")

    # 促销商品数的滚动均值（反映该组合近期是否经常促销）
    df = df.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)
    for w in [7, 14, 28]:
        grp = df.groupby(["store_nbr", "family"])["onpromotion"]
        df[f"promo_roll_mean_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )

    return df


def create_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建节假日相关特征

    - 是否为节假日
    - 节假日类型编码
    - 节假日前/后N天窗口标记
    - 距最近节假日的天数
    """
    df = df.copy()

    # is_holiday 已经存在
    # 节假日前后窗口（+/-3天）
    # 获取所有节假日日期
    holiday_dates = df[df["is_holiday"] == 1]["date"].unique()

    # 标记节假日前后的日期
    df["near_holiday"] = 0
    for h_date in holiday_dates:
        for offset in [-3, -2, -1, 0, 1, 2, 3]:
            target_date = pd.Timestamp(h_date) + pd.Timedelta(days=offset)
            df.loc[df["date"] == target_date, "near_holiday"] = 1

    df["near_holiday"] = df["near_holiday"].astype("int8")

    return df


def create_store_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建商店-商品族交互特征

    不同商店的不同商品族有不同的销售模式：
    - 该组合的历史平均销售额
    - 该商品族在所有店的平均销售额
    - 该店所有商品族的平均销售额
    """
    df = df.copy()

    # 计算每个 (store, family) 组合的历史平均销售额
    sf_mean = df.groupby(["store_nbr", "family"])["sales"].transform("mean")
    df["store_family_avg"] = sf_mean

    # 该商品族在所有商店的平均销售额
    family_mean = df.groupby("family")["sales"].transform("mean")
    df["family_avg"] = family_mean

    # 该商店所有商品族的平均销售额
    store_mean = df.groupby("store_nbr")["sales"].transform("mean")
    df["store_avg"] = store_mean

    # 该组合偏离全局的程度
    df["sf_vs_family_ratio"] = sf_mean / (family_mean + 1)

    return df


def create_oil_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建油价相关特征

    - 油价变化率（反映经济波动对消费的影响）
    - 油价滚动统计
    """
    df = df.sort_values("date").reset_index(drop=True)

    # 油价日变化率
    df["oil_change"] = df.groupby("date")["dcoilwtico"].transform("first").pct_change()
    df["oil_change"] = df["oil_change"].fillna(0)

    # 油价7天和30天移动平均
    oil_daily = df.groupby("date")["dcoilwtico"].first()
    df["oil_ma7"] = df["date"].map(
        oil_daily.rolling(7).mean().to_dict()
    )
    df["oil_ma30"] = df["date"].map(
        oil_daily.rolling(30).mean().to_dict()
    )

    return df


def create_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建交易量相关特征

    transactions 是每家店每日的总交易笔数，与销售额高度相关
    """
    df = df.copy()

    # 交易量的滚动统计
    df = df.sort_values(["date", "store_nbr"]).reset_index(drop=True)
    for w in [7, 14, 28]:
        grp = df.groupby("store_nbr")["transactions"]
        df[f"trans_roll_mean_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )

    return df


def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "sales",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    完整的特征工程流程

    将原始训练/测试数据转换为可用于模型训练的特征矩阵。

    Args:
        train_df: 包含原始列的训练DataFrame
        test_df: 包含原始列的测试DataFrame
        target_col: 目标变量名称

    Returns:
        X_train: 训练特征矩阵
        X_test: 测试特征矩阵
        feature_cols: 最终使用的特征列名列表
    """
    print_section("特征工程")

    # ---- Step 1: 日期特征 ----
    print("[1/9] 创建日期特征...")
    train_df = add_date_features(train_df)
    test_df = add_date_features(test_df)

    # ---- Step 2: 滞后特征 ----
    print(f"[2/9] 创建滞后特征 (lag={LAG_DAYS})...")
    train_df = create_lag_features(train_df, target_col)

    # 测试集的滞后特征需要特殊处理：
    # 测试集日期从2017-08-16开始，lag_1需要2017-08-15的数据
    # 因此需要拼接训练集最后N天的数据
    last_n_days = max(LAG_DAYS + ROLLING_WINDOWS + EWM_WINDOWS) + 7
    last_train_date = train_df["date"].max()
    cutoff_date = last_train_date - pd.Timedelta(days=last_n_days)
    recent_train = train_df[train_df["date"] >= cutoff_date].copy()

    # 合并训练集尾部 + 测试集，一起创建滞后特征
    combined = pd.concat([recent_train, test_df], axis=0, ignore_index=True)
    combined = combined.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)
    combined = create_lag_features(combined, target_col)
    # 只取测试集部分
    test_df = combined[combined["date"] >= test_df["date"].min()].copy()

    # ---- Step 3: 滚动窗口特征 ----
    print(f"[3/9] 创建滚动窗口特征 (windows={ROLLING_WINDOWS})...")
    train_df = create_rolling_features(train_df, target_col)

    # 测试集：在合并数据上创建滚动特征
    combined2 = pd.concat([recent_train, test_df[["date", "store_nbr", "family"]]],
                          axis=0, ignore_index=True)
    # 重新合并完整列
    combined2 = combined2.merge(
        combined[["date", "store_nbr", "family"] + [c for c in combined.columns
                   if c.startswith("lag_")]],
        on=["date", "store_nbr", "family"], how="left"
    )
    combined2 = create_rolling_features(combined2, target_col)
    test_df = combined2[combined2["date"] >= test_df["date"].min()].copy()
    # 确保列一致
    for col in train_df.columns:
        if col.startswith("rolling_") and col not in test_df.columns:
            test_df[col] = 0

    # ---- Step 4: EWM特征 ----
    print(f"[4/9] 创建EWM特征 (windows={EWM_WINDOWS})...")
    train_df = create_ewm_features(train_df, target_col)

    # ---- Step 5: 促销特征 ----
    print("[5/9] 创建促销特征...")
    train_df = create_promo_features(train_df)
    test_df = create_promo_features(test_df)

    # ---- Step 6: 节假日特征 ----
    print("[6/9] 创建节假日特征...")
    train_df = create_holiday_features(train_df)
    test_df = create_holiday_features(test_df)

    # ---- Step 7: 商店-商品族交互特征 ----
    print("[7/9] 创建商店-商品族交互特征...")
    train_df = create_store_family_features(train_df)
    # 测试集使用训练集的统计量
    sf_mean_map = train_df.groupby(["store_nbr", "family"])["sales"].transform("mean")
    train_df["store_family_avg"] = sf_mean_map
    sf_map = train_df.groupby(["store_nbr", "family"])["store_family_avg"].first().to_dict()
    test_df["store_family_avg"] = test_df.apply(
        lambda r: sf_map.get((r["store_nbr"], r["family"]), 0), axis=1
    )
    family_mean_map = train_df.groupby("family")["sales"].mean().to_dict()
    test_df["family_avg"] = test_df["family"].map(family_mean_map)
    store_mean_map = train_df.groupby("store_nbr")["sales"].mean().to_dict()
    test_df["store_avg"] = test_df["store_nbr"].map(store_mean_map)
    test_df["sf_vs_family_ratio"] = test_df["store_family_avg"] / (test_df["family_avg"] + 1)

    # ---- Step 8: 外部特征 ----
    print("[8/9] 创建外部特征 (油价/交易量)...")
    train_df = create_oil_features(train_df)
    test_df = create_oil_features(test_df)
    train_df = create_transaction_features(train_df)
    test_df = create_transaction_features(test_df)

    # ---- Step 9: 类别编码和特征选择 ----
    print("[9/9] 编码类别特征并选择最终特征集...")
    train_df = encode_categoricals(train_df)
    test_df = encode_categoricals(test_df)

    # 确定最终使用的特征列
    exclude_cols = [
        "date", "sales", "id", "description", "transferred",
        "locale_name", "holiday_type_regional"
    ]
    feature_cols = [c for c in train_df.columns
                    if c not in exclude_cols
                    and train_df[c].dtype in ["int8", "int16", "int32", "int64",
                                               "float16", "float32", "float64"]]

    print(f"\n最终特征数: {len(feature_cols)}")
    print(f"训练集大小: {train_df[feature_cols].shape}")
    print(f"测试集大小: {test_df[feature_cols].shape}")

    return train_df, test_df, feature_cols
