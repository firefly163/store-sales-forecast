"""
工具函数 — 评分指标、数据压缩、日志等
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Logarithmic Error (RMSLE)
    这是本赛题的官方评估指标

    公式: RMSLE = sqrt(1/n * Σ(log(1+ŷ) - log(1+y))²)

    RMSLE 相比 RMSE 的优势：
    1. 对大值不那么敏感（对数变换压缩了尺度）
    2. 对低销量商品的预测误差不过分惩罚
    3. 符合零售业务中"相对误差比绝对误差更重要"的特点
    """
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def rmsle_score(y_true, y_pred):
    """sklearn-compatible scorer (返回负值，因为sklearn要求越大越好)"""
    return -rmsle(y_true, y_pred)


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    降低 DataFrame 内存占用
    通过将数据类型转换为更小的 numpy dtype
    对于大数据集（如本赛题的train.csv有1.2亿行），这个函数非常重要
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"原始内存占用: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type != "datetime64[ns]":
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"压缩后内存占用: {end_mem:.2f} MB (减少 {100 * (start_mem - end_mem) / start_mem:.1f}%)")
    return df


def print_section(title: str, width: int = 70):
    """打印格式化的章节标题"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def create_submission_csv(
    ids: np.ndarray,
    predictions: np.ndarray,
    save_path: Path,
    model_name: str = "model",
) -> Path:
    """
    创建符合Kaggle提交格式的CSV文件

    submission.csv 格式:
        id,sales
        3000888,7.0
        3000889,10.0
        ...
    """
    submission = pd.DataFrame({
        "id": ids,
        "sales": predictions
    })
    # 确保sales非负
    submission["sales"] = submission["sales"].clip(lower=0)

    filepath = save_path / f"submission_{model_name}.csv"
    submission.to_csv(filepath, index=False)
    print(f"提交文件已保存: {filepath}")
    print(f"预测样本数: {len(submission):,}")
    print(f"预测范围: [{submission['sales'].min():.2f}, {submission['sales'].max():.2f}]")
    return filepath
