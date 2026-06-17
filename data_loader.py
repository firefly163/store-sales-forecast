"""
数据加载与合并模块
===================
本赛题数据包含6个CSV文件，需要将它们合并为一个完整的训练/测试数据集。

数据文件说明：
- train.csv        : 训练数据，2013-01-01 ~ 2017-08-15的日销售记录
- test.csv         : 测试数据，2017-08-16 ~ 2017-08-31
- stores.csv       : 54家门店的元数据（城市、州、类型、集群）
- oil.csv          : 厄瓜多尔每日油价
- holidays_events.csv : 节假日和特殊事件
- transactions.csv : 每家店每日的交易笔数（辅助特征）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from config import (
    DATA_DIR, TRAIN_CSV, TEST_CSV, STORES_CSV,
    OIL_CSV, HOLIDAYS_CSV, TRANSACTIONS_CSV, SAMPLE_SUBMISSION
)
from utils import reduce_memory, print_section


def load_raw_data() -> dict:
    """
    加载所有原始CSV文件

    Returns:
        dict: 包含所有DataFrame的字典
    """
    print_section("加载原始数据")

    raw = {}

    # 1. 训练数据 (最大的文件，约1.2亿行)
    print("加载训练数据 (train.csv)...")
    raw["train"] = pd.read_csv(
        DATA_DIR / TRAIN_CSV,
        parse_dates=["date"],
        dtype={
            "store_nbr": "int8",
            "family": "category",
            "sales": "float32",
            "onpromotion": "int32",
        }
    )
    print(f"  训练集: {raw['train'].shape[0]:,} 行 × {raw['train'].shape[1]} 列")
    print(f"  日期范围: {raw['train']['date'].min()} ~ {raw['train']['date'].max()}")
    print(f"  商店数: {raw['train']['store_nbr'].nunique()}, 商品族: {raw['train']['family'].nunique()}")

    # 2. 测试数据
    print("加载测试数据 (test.csv)...")
    raw["test"] = pd.read_csv(
        DATA_DIR / TEST_CSV,
        parse_dates=["date"],
        dtype={
            "store_nbr": "int8",
            "family": "category",
            "onpromotion": "int32",
        }
    )
    print(f"  测试集: {raw['test'].shape[0]:,} 行 × {raw['test'].shape[1]} 列")
    print(f"  日期范围: {raw['test']['date'].min()} ~ {raw['test']['date'].max()}")

    # 3. 商店元数据
    print("加载商店数据 (stores.csv)...")
    raw["stores"] = pd.read_csv(DATA_DIR / STORES_CSV)
    raw["stores"]["store_nbr"] = raw["stores"]["store_nbr"].astype("int8")
    print(f"  商店数: {len(raw['stores'])}")
    print(f"  城市数: {raw['stores']['city'].nunique()}, "
          f"州数: {raw['stores']['state'].nunique()}, "
          f"商店类型: {raw['stores']['type'].nunique()}")

    # 4. 油价数据
    print("加载油价数据 (oil.csv)...")
    raw["oil"] = pd.read_csv(DATA_DIR / OIL_CSV, parse_dates=["date"])
    raw["oil"]["dcoilwtico"] = raw["oil"]["dcoilwtico"].astype("float32")
    print(f"  油价记录: {len(raw['oil']):,} 行")
    print(f"  日期范围: {raw['oil']['date'].min()} ~ {raw['oil']['date'].max()}")
    print(f"  缺失值: {raw['oil']['dcoilwtico'].isnull().sum()}")

    # 5. 节假日数据
    print("加载节假日数据 (holidays_events.csv)...")
    raw["holidays"] = pd.read_csv(DATA_DIR / HOLIDAYS_CSV, parse_dates=["date"])
    print(f"  节假日记录: {len(raw['holidays']):,} 行")
    print(f"  类型: {raw['holidays']['type'].unique().tolist()}")
    print(f"  地区: {raw['holidays']['locale'].unique().tolist()}")

    # 6. 交易数据
    print("加载交易数据 (transactions.csv)...")
    raw["transactions"] = pd.read_csv(
        DATA_DIR / TRANSACTIONS_CSV, parse_dates=["date"]
    )
    raw["transactions"]["store_nbr"] = raw["transactions"]["store_nbr"].astype("int8")
    print(f"  交易记录: {len(raw['transactions']):,} 行")

    return raw


def merge_datasets(raw: dict) -> pd.DataFrame:
    """
    合并所有数据源为统一的训练DataFrame

    合并策略:
    1. train与stores合并 (通过store_nbr)
    2. 与oil合并 (通过date)
    3. 与holidays合并 (通过date + locale条件)
    4. 与transactions合并 (通过date + store_nbr)

    Returns:
        合并后的完整DataFrame
    """
    print_section("合并数据集")

    df = raw["train"].copy()

    # ---- Step 1: 合并商店信息 ----
    print("1. 合并商店元数据...")
    df = df.merge(raw["stores"], on="store_nbr", how="left")
    print(f"   合并后: {df.shape[0]:,} 行 × {df.shape[1]} 列")

    # ---- Step 2: 合并油价 ----
    print("2. 合并油价数据...")
    # 油价数据有缺失值（周末和节假日不公布），需要前向填充
    oil = raw["oil"].copy()
    oil["dcoilwtico"] = oil["dcoilwtico"].fillna(method="ffill")
    df = df.merge(oil, on="date", how="left")
    print(f"   合并后: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    print(f"   油价缺失: {df['dcoilwtico'].isnull().sum():,} 行")

    # ---- Step 3: 合并节假日信息 ----
    print("3. 合并节假日数据...")
    holidays = raw["holidays"].copy()

    # 国家层面的节假日 → 对所有店有效
    national = holidays[holidays["locale"] == "National"].copy()
    national = national.rename(columns={"type": "holiday_type"})
    df = df.merge(
        national[["date", "holiday_type", "description", "transferred"]],
        on="date", how="left", suffixes=("", "_national")
    )

    # 地区层面的节假日 → 只对对应城市的店有效
    regional = holidays[holidays["locale"] == "Regional"].copy()
    regional = regional.rename(columns={"type": "holiday_type_regional"})

    # 需要根据 locale_name 与 store 的 state 或 city 匹配
    # 先合并所有regional节假日，然后筛选
    df = df.merge(
        regional[["date", "holiday_type_regional", "description", "locale_name", "transferred"]],
        on="date", how="left", suffixes=("", "_regional")
    )

    # 如果regional节日的地点与店铺所在州/城市匹配，则保留
    df["is_regional_match"] = (
        (df["locale_name"] == df["state"]) |
        (df["locale_name"] == df["city"])
    )
    df.loc[~df["is_regional_match"], "holiday_type_regional"] = np.nan
    df.drop(columns=["locale_name", "is_regional_match"], inplace=True)

    # 合并节日类型
    df["holiday_type"] = df["holiday_type"].fillna(df["holiday_type_regional"])
    df.drop(columns=["holiday_type_regional"], inplace=True)
    df["holiday_type"] = df["holiday_type"].fillna("None")
    df["is_holiday"] = (df["holiday_type"] != "None").astype("int8")

    print(f"   节假日分布:\n{df['holiday_type'].value_counts().to_string()}")

    # ---- Step 4: 合并交易数据 ----
    print("4. 合并交易数据...")
    df = df.merge(raw["transactions"], on=["date", "store_nbr"], how="left")
    # 用当天的中位数填充缺失的交易数
    df["transactions"] = df.groupby("date")["transactions"].transform(
        lambda x: x.fillna(x.median())
    )
    # 仍有缺失的用全局中位数
    df["transactions"] = df["transactions"].fillna(df["transactions"].median())
    print(f"   合并后: {df.shape[0]:,} 行 × {df.shape[1]} 列")

    # ---- Step 5: 数据类型优化 ----
    print("5. 数据类型优化...")
    for col in ["city", "state", "type", "cluster"]:
        df[col] = df[col].astype("category")
    df["family"] = df["family"].astype("category")
    df["holiday_type"] = df["holiday_type"].astype("category")

    print(f"\n最终数据集: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    print(f"商店×商品族×日期 组合数: {df.groupby(['store_nbr', 'family', 'date']).ngroups:,}")

    return df


def prepare_test_features(raw: dict) -> pd.DataFrame:
    """
    为测试集准备与训练集相同的特征结构

    Returns:
        处理后的测试DataFrame
    """
    print_section("准备测试集特征")

    df = raw["test"].copy()

    # 合并商店信息
    df = df.merge(raw["stores"], on="store_nbr", how="left")

    # 合并油价
    oil = raw["oil"].copy()
    oil["dcoilwtico"] = oil["dcoilwtico"].fillna(method="ffill")
    df = df.merge(oil, on="date", how="left")

    # 合并节假日（同训练集逻辑）
    holidays = raw["holidays"].copy()
    national = holidays[holidays["locale"] == "National"].copy()
    national = national.rename(columns={"type": "holiday_type"})
    df = df.merge(national[["date", "holiday_type"]], on="date", how="left")

    regional = holidays[holidays["locale"] == "Regional"].copy()
    regional = regional.rename(columns={"type": "holiday_type_regional"})
    df = df.merge(
        regional[["date", "holiday_type_regional", "locale_name"]],
        on="date", how="left"
    )
    df["is_regional_match"] = (
        (df["locale_name"] == df["state"]) |
        (df["locale_name"] == df["city"])
    )
    df.loc[~df["is_regional_match"], "holiday_type_regional"] = np.nan
    df.drop(columns=["locale_name", "is_regional_match"], inplace=True)

    df["holiday_type"] = df["holiday_type"].fillna(df["holiday_type_regional"])
    df.drop(columns=["holiday_type_regional"], inplace=True)
    df["holiday_type"] = df["holiday_type"].fillna("None")
    df["is_holiday"] = (df["holiday_type"] != "None").astype("int8")

    # 合并交易数据
    df = df.merge(raw["transactions"], on=["date", "store_nbr"], how="left")
    df["transactions"] = df.groupby("date")["transactions"].transform(
        lambda x: x.fillna(x.median())
    )
    df["transactions"] = df["transactions"].fillna(df["transactions"].median())

    # 数据类型优化
    for col in ["city", "state", "type", "cluster"]:
        df[col] = df[col].astype("category")
    df["family"] = df["family"].astype("category")
    df["holiday_type"] = df["holiday_type"].astype("category")

    print(f"测试集特征: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    return df


def load_and_merge() -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    完整的加载和合并流程

    Returns:
        train_df: 合并后的训练DataFrame
        test_df: 合并后的测试DataFrame
        raw: 原始数据字典
    """
    raw = load_raw_data()
    train_df = merge_datasets(raw)
    test_df = prepare_test_features(raw)

    return train_df, test_df, raw
