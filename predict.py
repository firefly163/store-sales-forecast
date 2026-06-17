"""
预测与提交模块
===============
使用训练好的模型对测试集进行预测，并生成Kaggle提交文件。
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from typing import Dict

from config import SUBMISSIONS_DIR, MODELS_DIR
from train import load_lgbm, load_xgb, EnsembleModel
from utils import rmsle, print_section, create_submission_csv


def predict_lgbm(
    model,
    test_df: pd.DataFrame,
    feature_cols: list,
) -> np.ndarray:
    """LightGBM 预测"""
    X_test = test_df[feature_cols].fillna(0).values
    preds = model.predict(X_test)
    return np.clip(preds, 0, None)


def predict_xgboost(
    model,
    test_df: pd.DataFrame,
    feature_cols: list,
) -> np.ndarray:
    """XGBoost 预测"""
    X_test = test_df[feature_cols].fillna(0).values
    dtest = xgb.DMatrix(X_test)
    preds = model.predict(dtest)
    return np.clip(preds, 0, None)


def predict_ensemble(
    lgb_model,
    xgb_model,
    test_df: pd.DataFrame,
    feature_cols: list,
    lgb_weight: float = 0.6,
    xgb_weight: float = 0.4,
) -> np.ndarray:
    """集成预测"""
    ensemble = EnsembleModel()
    preds = ensemble.predict_lgb_xgb(
        test_df[feature_cols].fillna(0),
        lgb_model,
        xgb_model,
        lgb_weight,
        xgb_weight,
    )
    return np.clip(preds, 0, None)


def generate_all_submissions(
    test_df: pd.DataFrame,
    feature_cols: list,
    ids: np.ndarray = None,
) -> Dict[str, Path]:
    """
    生成所有模型的提交文件

    Returns:
        模型名 -> 提交文件路径 的映射
    """
    print_section("生成提交文件")

    submissions = {}

    # 获取ID列
    if ids is None:
        if "id" in test_df.columns:
            ids = test_df["id"].values
        else:
            ids = np.arange(len(test_df))

    # 1. Baseline（简单均值）
    print("\n[1/4] Baseline 预测...")
    from train import BaselineModel
    baseline = BaselineModel.load()
    baseline_preds = baseline.predict(test_df)
    path = create_submission_csv(ids, baseline_preds, SUBMISSIONS_DIR, "baseline")
    submissions["baseline"] = path

    # 2. LightGBM
    print("\n[2/4] LightGBM 预测...")
    try:
        lgb_model = load_lgbm()
        lgb_preds = predict_lgbm(lgb_model, test_df, feature_cols)
        path = create_submission_csv(ids, lgb_preds, SUBMISSIONS_DIR, "lightgbm")
        submissions["lightgbm"] = path
    except FileNotFoundError:
        print("  LightGBM 模型文件未找到，跳过")

    # 3. XGBoost
    print("\n[3/4] XGBoost 预测...")
    try:
        xgb_model = load_xgb()
        xgb_preds = predict_xgboost(xgb_model, test_df, feature_cols)
        path = create_submission_csv(ids, xgb_preds, SUBMISSIONS_DIR, "xgboost")
        submissions["xgboost"] = path
    except FileNotFoundError:
        print("  XGBoost 模型文件未找到，跳过")

    # 4. Ensemble
    print("\n[4/4] Ensemble 预测...")
    try:
        lgb_model = load_lgbm()
        xgb_model = load_xgb()
        ensemble_preds = predict_ensemble(
            lgb_model, xgb_model, test_df, feature_cols
        )
        path = create_submission_csv(ids, ensemble_preds, SUBMISSIONS_DIR, "ensemble")
        submissions["ensemble"] = path
    except FileNotFoundError:
        print("  模型文件未找到，跳过集成预测")

    print_section("提交文件汇总")
    for name, path in submissions.items():
        print(f"  {name}: {path}")

    return submissions
