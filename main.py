"""
Store Sales - Time Series Forecasting
=====================================
Kaggle 比赛: 预测厄瓜多尔 Favorita 连锁店各商品族的日销售额

运行方式:
    python main.py                    # 完整流程（数据加载→特征工程→训练→预测）
    python main.py --skip-eda         # 跳过EDA（数据量大时推荐）
    python main.py --phase train      # 仅训练
    python main.py --phase predict    # 仅预测（需要已有训练好的模型）
    python main.py --quick            # 快速模式（采样训练，用于调试）

作者: 课程小组
日期: 2026年6月
"""

import argparse
import sys
import time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, MODELS_DIR, FIGURES_DIR, SUBMISSIONS_DIR,
    VAL_SPLIT_DATE, SEED
)
from data_loader import load_and_merge
from feature_engineering import build_features
from train import run_full_training
from predict import generate_all_submissions
from utils import print_section


def parse_args():
    parser = argparse.ArgumentParser(description="Store Sales Time Series Forecasting")

    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "eda", "features", "train", "predict"],
                        help="运行阶段 (default: all)")

    parser.add_argument("--skip-eda", action="store_true",
                        help="跳过EDA（数据量大时可节省时间）")

    parser.add_argument("--quick", action="store_true",
                        help="快速调试模式（采样10%数据）")

    parser.add_argument("--n-samples", type=int, default=0,
                        help="采样行数（0=全部数据）")

    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    print("=" * 70)
    print("  Store Sales — 时间序列销售预测")
    print("  Kaggle: store-sales-time-series-forecasting")
    print("=" * 70)

    # ============================================================
    # Phase 1: 数据加载
    # ============================================================
    if args.phase in ["all", "eda", "features"]:
        print_section("PHASE 1: 数据加载与合并")
        train_df, test_df, raw_data = load_and_merge()

        # 快速模式：采样
        if args.quick or args.n_samples > 0:
            n = args.n_samples if args.n_samples > 0 else 500000
            print(f"\n⚡ 快速模式: 采样 {n:,} 行")
            train_df = train_df.sample(n=n, random_state=SEED).sort_values(
                ["date", "store_nbr", "family"]
            ).reset_index(drop=True)
            print(f"  采样后训练集: {train_df.shape[0]:,} 行")

    # ============================================================
    # Phase 2: EDA（可选）
    # ============================================================
    if args.phase in ["all", "eda"] and not args.skip_eda:
        print_section("PHASE 2: 探索性数据分析")
        try:
            from eda import run_full_eda
            run_full_eda(train_df, raw_data)
        except Exception as e:
            print(f"EDA出错（跳过）: {e}")

    # ============================================================
    # Phase 3: 特征工程
    # ============================================================
    if args.phase in ["all", "features"]:
        print_section("PHASE 3: 特征工程")
        train_df, test_df, feature_cols = build_features(train_df, test_df)

        # 保存特征列名，供后续阶段使用
        import json
        with open(MODELS_DIR / "feature_cols.json", "w") as f:
            json.dump(feature_cols, f)
        print(f"特征列名已保存至: {MODELS_DIR / 'feature_cols.json'}")

    # ============================================================
    # Phase 4: 模型训练
    # ============================================================
    if args.phase in ["all", "train"]:
        print_section("PHASE 4: 模型训练")

        # 划分训练/验证集
        val_mask = train_df["date"] >= VAL_SPLIT_DATE
        train_data = train_df[~val_mask].copy()
        val_data = train_df[val_mask].copy()
        print(f"训练集: {train_data.shape[0]:,} 行 (2013-01 ~ 2017-06)")
        print(f"验证集: {val_data.shape[0]:,} 行 (2017-07 ~ 2017-08)")

        # 运行训练流程
        results = run_full_training(train_data, val_data, feature_cols, "sales")

        # 保存结果摘要
        import json
        summary = {
            "baseline_rmsle": float(results["baseline"]["rmsle"]),
            "lightgbm_rmsle": float(results["lightgbm"]["val_rmsle"]),
            "xgboost_rmsle": float(results["xgboost"]["val_rmsle"]),
        }
        with open(MODELS_DIR / "results_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # ============================================================
    # Phase 5: 预测与提交
    # ============================================================
    if args.phase in ["all", "predict"]:
        print_section("PHASE 5: 预测与生成提交文件")

        # 加载特征列名
        import json
        with open(MODELS_DIR / "feature_cols.json", "r") as f:
            feature_cols = json.load(f)

        # 生成所有提交文件
        submissions = generate_all_submissions(test_df, feature_cols)

    # ============================================================
    # 完成
    # ============================================================
    elapsed = time.time() - start_time
    print_section(f"全部完成! 耗时: {elapsed/60:.1f} 分钟")
    print(f"\n提交文件目录: {SUBMISSIONS_DIR}")
    print(f"模型文件目录: {MODELS_DIR}")
    print(f"图表文件目录: {FIGURES_DIR}")
    print(f"\n下一步:")
    print(f"  1. 在Kaggle上传 {SUBMISSIONS_DIR}/submission_ensemble.csv")
    print(f"  2. 查看Public Leaderboard排名")
    print(f"  3. 根据排名调整特征工程和模型参数")
    print(f"  4. 重复训练-预测-提交循环以改进分数")


if __name__ == "__main__":
    main()
