"""
模型训练模块
=============
包含三种模型方案：
1. Baseline — 简单均值预测（用于建立性能下限）
2. LightGBM — 梯度提升树，时间序列预测的主流选择
3. XGBoost — 另一个强力的集成树模型
4. Ensemble — 加权集成，进一步提升分数

训练策略：时间序列交叉验证
- 验证集: 2017-07-01 ~ 2017-08-15（最后1.5个月）
- 训练集: 2017-07-01 之前的数据
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Dict, Optional
from datetime import datetime

import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from config import (
    MODELS_DIR, SUBMISSIONS_DIR, SEED,
    LGBM_PARAMS, XGB_PARAMS,
    VAL_SPLIT_DATE, EARLY_STOPPING_ROUNDS, VERBOSE_EVAL, CV_FOLDS,
    TRAIN_END, TEST_START
)
from utils import rmsle, print_section


# ============================================================
# Baseline: 简单均值预测
# ============================================================
class BaselineModel:
    """
    基线模型：用历史均值作为预测

    这不是一个真正的机器学习模型，而是一个简单的统计基准。
    对每个 (store_nbr, family) 组合，用其历史平均销售额作为未来预测。

    目的：
    1. 建立预测性能的下限
    2. 任何机器学习模型都应该至少比这个基线好
    3. 帮助理解数据的结构
    """

    def __init__(self):
        self.means_ = None  # 每个组合的历史均值

    def fit(self, df: pd.DataFrame):
        """计算每个 (store_nbr, family) 组合的平均销售额"""
        self.means_ = df.groupby(["store_nbr", "family"])["sales"].mean()
        self.means_ = self.means_.to_dict()
        # 添加全局均值作为兜底
        self.global_mean_ = df["sales"].mean()
        print(f"Baseline: 全局均值={self.global_mean_:.2f}, "
              f"组合数={len(self.means_)}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """对每条记录预测其对应组合的历史均值"""
        preds = []
        for _, row in df.iterrows():
            key = (row["store_nbr"], row["family"])
            preds.append(self.means_.get(key, self.global_mean_))
        return np.array(preds)

    def save(self, path: Path = None):
        if path is None:
            path = MODELS_DIR / "baseline.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path = None):
        if path is None:
            path = MODELS_DIR / "baseline.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)


def train_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
    """训练并评估基线模型"""
    print_section("Baseline: 历史均值预测")
    model = BaselineModel()
    model.fit(train_df)
    preds = model.predict(val_df)
    score = rmsle(val_df["sales"].values, preds)
    model.save()
    print(f"Baseline RMSLE: {score:.6f}")
    return {"model": model, "rmsle": score, "predictions": preds}


# ============================================================
# LightGBM 模型
# ============================================================
def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Dict = None,
    use_early_stopping: bool = True,
) -> Tuple[lgb.Booster, Dict]:
    """
    LightGBM 训练

    LightGBM 是基于梯度提升的决策树模型，特点：
    - 训练速度快（基于直方图的算法）
    - 内存占用小
    - 对类别特征有原生支持
    - 在Kaggle表格数据竞赛中表现优异

    本赛题选择 LightGBM 的原因：
    1. 数据量大（百万级），LightGBM比XGBoost更快
    2. 特征维度高，LightGBM的leaf-wise生长策略更高效
    3. 有大量类别特征，LightGBM原生支持
    """
    print_section("LightGBM 训练")

    if params is None:
        params = LGBM_PARAMS.copy()

    # 创建数据集
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    # 训练
    print(f"训练样本: {len(X_train):,}, 验证样本: {len(X_val):,}")
    print(f"特征数: {X_train.shape[1]}")

    callbacks = []
    if use_early_stopping:
        callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS))
        callbacks.append(lgb.log_evaluation(VERBOSE_EVAL))

    model = lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        num_boost_round=params.get("n_estimators", 3000),
        callbacks=callbacks,
    )

    # 评估
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_score = rmsle(y_train, train_pred)
    val_score = rmsle(y_val, val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, val_pred))

    results = {
        "train_rmsle": train_score,
        "val_rmsle": val_score,
        "val_rmse": rmse_val,
        "best_iteration": model.best_iteration,
        "feature_importance": model.feature_importance(importance_type="gain"),
        "feature_names": model.feature_name(),
    }

    print(f"训练 RMSLE: {train_score:.6f}")
    print(f"验证 RMSLE: {val_score:.6f}")
    print(f"最佳迭代: {model.best_iteration}")

    return model, results


def save_lgbm(model: lgb.Booster, name: str = "lgbm"):
    """保存 LightGBM 模型"""
    path = MODELS_DIR / f"{name}.txt"
    model.save_model(str(path))
    print(f"模型已保存: {path}")


def load_lgbm(name: str = "lgbm") -> lgb.Booster:
    """加载 LightGBM 模型"""
    path = MODELS_DIR / f"{name}.txt"
    return lgb.Booster(model_file=str(path))


# ============================================================
# XGBoost 模型
# ============================================================
def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Dict = None,
) -> Tuple[xgb.Booster, Dict]:
    """
    XGBoost 训练

    XGBoost 是另一个顶级梯度提升框架，与 LightGBM 互补：
    - level-wise 生长策略，更不容易过拟合
    - 对噪声数据更鲁棒
    - 与 LightGBM 集成可以互补提升

    LightGBM vs XGBoost 在本赛题的对比：
    - LightGBM: 更快，对大数据集更友好
    - XGBoost: 更稳定，对异常值更鲁棒
    - 集成两者通常能获得更好结果（见 Ensemble 部分）
    """
    print_section("XGBoost 训练")

    if params is None:
        params = XGB_PARAMS.copy()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    print(f"训练样本: {len(X_train):,}, 验证样本: {len(X_val):,}")
    print(f"特征数: {X_train.shape[1]}")

    evals = [(dtrain, "train"), (dval, "valid")]
    evals_result = {}

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=params.get("n_estimators", 2000),
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=VERBOSE_EVAL,
    )

    # 评估
    train_pred = model.predict(dtrain)
    val_pred = model.predict(dval)
    train_score = rmsle(y_train, train_pred)
    val_score = rmsle(y_val, val_pred)

    results = {
        "train_rmsle": train_score,
        "val_rmsle": val_score,
        "best_iteration": model.best_iteration,
        "evals_result": evals_result,
    }

    print(f"训练 RMSLE: {train_score:.6f}")
    print(f"验证 RMSLE: {val_score:.6f}")
    print(f"最佳迭代: {model.best_iteration}")

    return model, results


def save_xgb(model: xgb.Booster, name: str = "xgboost"):
    """保存 XGBoost 模型"""
    path = MODELS_DIR / f"{name}.json"
    model.save_model(str(path))
    print(f"模型已保存: {path}")


def load_xgb(name: str = "xgboost") -> xgb.Booster:
    """加载 XGBoost 模型"""
    path = MODELS_DIR / f"{name}.json"
    model = xgb.Booster()
    model.load_model(str(path))
    return model


# ============================================================
# 集成模型
# ============================================================
class EnsembleModel:
    """
    加权集成模型

    将多个模型的预测结果加权平均，通常能获得比任一单模型更好的结果。

    原理：
    1. 不同模型有不同的偏置-方差特性
    2. 加权平均可以降低方差（类似于bagging的思想）
    3. 当模型误差不相关时，集成效果最好

    集成策略：
    - 简单加权平均：pred = w1*pred1 + w2*pred2 + ...
    - 权重通过验证集上的RMSLE表现来确定
    """

    def __init__(self):
        self.models = []
        self.weights = []

    def add_model(self, model, weight: float, name: str, predict_fn=None):
        """添加模型及其权重"""
        self.models.append({
            "model": model,
            "weight": weight,
            "name": name,
            "predict_fn": predict_fn,
        })

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """加权预测"""
        if len(self.models) == 0:
            raise ValueError("没有添加任何模型")

        total_weight = sum(m["weight"] for m in self.models)
        preds = np.zeros(len(X))

        for m in self.models:
            w = m["weight"] / total_weight
            if m["predict_fn"] is not None:
                p = m["predict_fn"](m["model"], X)
            elif hasattr(m["model"], "predict"):
                p = m["model"].predict(X)
            else:
                raise ValueError(f"模型 {m['name']} 没有predict方法")

            preds += w * p

        return np.clip(preds, 0, None)

    def predict_lgb_xgb(
        self, X: pd.DataFrame,
        lgb_model: lgb.Booster,
        xgb_model: xgb.Booster,
        lgb_weight: float = 0.6,
        xgb_weight: float = 0.4,
    ) -> np.ndarray:
        """LightGBM + XGBoost 加权预测（便捷方法）"""
        dxgb = xgb.DMatrix(X)
        lgb_pred = lgb_model.predict(X)
        xgb_pred = xgb_model.predict(dxgb)
        total_w = lgb_weight + xgb_weight
        preds = (lgb_weight * lgb_pred + xgb_weight * xgb_pred) / total_w
        return np.clip(preds, 0, None)


def train_ensemble(
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    lgb_model: lgb.Booster,
    xgb_model: xgb.Booster,
) -> EnsembleModel:
    """
    训练集成模型（通过验证集调优权重）
    """
    print_section("集成模型: LightGBM + XGBoost")

    # 尝试不同权重组合
    best_score = float("inf")
    best_w = (0.5, 0.5)

    ensemble = EnsembleModel()

    for w1 in np.arange(0.0, 1.01, 0.1):
        w2 = 1.0 - w1
        preds = ensemble.predict_lgb_xgb(X_val, lgb_model, xgb_model, w1, w2)
        score = rmsle(y_val, preds)
        print(f"  LGBM={w1:.1f}, XGB={w2:.1f} → RMSLE={score:.6f}")

        if score < best_score:
            best_score = score
            best_w = (w1, w2)

    print(f"\n最佳权重: LGBM={best_w[0]:.1f}, XGB={best_w[1]:.1f}")
    print(f"集成验证 RMSLE: {best_score:.6f}")

    ensemble.add_model(lgb_model, best_w[0], "LightGBM")
    ensemble.add_model(xgb_model, best_w[1], "XGBoost")

    return ensemble


# ============================================================
# 完整训练流程
# ============================================================
def run_full_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "sales",
) -> Dict:
    """
    完整的训练流程：
    1. 基线模型
    2. LightGBM
    3. XGBoost
    4. 集成模型
    """
    print("=" * 70)
    print("  开始完整训练流程")
    print(f"  训练集: {train_df.shape[0]:,} 行, 验证集: {val_df.shape[0]:,} 行")
    print(f"  特征数: {len(feature_cols)}")
    print("=" * 70)

    # 准备数据
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df[target_col].values

    results = {}

    # 1. Baseline
    print("\n>>> 阶段1: 基线模型 <<<")
    baseline_df = train_df.copy()
    # 基线只用 date < VAL_SPLIT_DATE 的数据
    baseline_train = baseline_df[baseline_df["date"] < VAL_SPLIT_DATE]
    baseline_val = val_df.copy()
    base_result = train_baseline(baseline_train, baseline_val)
    results["baseline"] = base_result

    # 2. LightGBM
    print("\n>>> 阶段2: LightGBM <<<")
    lgb_model, lgb_results = train_lightgbm(X_train, y_train, X_val, y_val)
    save_lgbm(lgb_model)
    results["lightgbm"] = lgb_results

    # 3. XGBoost
    print("\n>>> 阶段3: XGBoost <<<")
    xgb_model, xgb_results = train_xgboost(X_train, y_train, X_val, y_val)
    save_xgb(xgb_model)
    results["xgboost"] = xgb_results

    # 4. Ensemble
    print("\n>>> 阶段4: 集成模型 <<<")
    ensemble = train_ensemble(X_val, y_val, lgb_model, xgb_model)
    results["ensemble"] = ensemble

    # 5. 总结
    print_section("训练结果总结")
    print(f"{'模型':<20} {'验证RMSLE':>12}")
    print("-" * 35)
    print(f"{'Baseline (均值)':<20} {base_result['rmsle']:>12.6f}")
    print(f"{'LightGBM':<20} {lgb_results['val_rmsle']:>12.6f}")
    print(f"{'XGBoost':<20} {xgb_results['val_rmsle']:>12.6f}")

    # 计算集成在验证集上的分数
    ensemble_preds = ensemble.predict_lgb_xgb(X_val, lgb_model, xgb_model,
                                               lgb_weight=0.6, xgb_weight=0.4)
    ensemble_score = rmsle(y_val, ensemble_preds)
    print(f"{'Ensemble':<20} {ensemble_score:>12.6f}")

    improvement = (base_result["rmsle"] - ensemble_score) / base_result["rmsle"] * 100
    print(f"\n相比基线提升: {improvement:.1f}%")

    return results
