# Store Sales V8 — 基于 Darts 的全局时间序列预测方案

Kaggle [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) 竞赛方案。

**Leaderboard 得分: 0.03792**

## 架构总览

```
per-family Darts 全局模型
  ├── LightGBM × 4 个滞后阶数 (63, 7, 365, 730)
  │     ├── 全历史训练
  │     └── 2015 年起训练
  ├── XGBoost × 4 个滞后阶数 (63, 7, 365, 730)
  │     ├── 全历史训练
  │     └── 2015 年起训练
  └── 融合: avg(avg(LGBM_full, LGBM_2015), avg(XGB_full, XGB_2015))
        ├── 可选的 linear / log-space 权重网格搜索（基于 16 天回测）
        └── 零销量店铺强制归零
```

## 核心特性

- **Darts 全局模型** — 每个商品品类一个模型，内含 54 个门店序列
- **双窗口平均融合** — 全历史 + 2015 年后两个训练窗口取平均
- **组件权重搜索** — 在 linear 和 log 空间穷举搜索最优融合权重
- **GPU 训练支持** — LightGBM（OpenCL）+ XGBoost（CUDA）自动检测
- **自动诊断** — EDA 图表、验证集诊断、特征分析、实验报告 zip 自动打包

## 快速开始

```bash
# 安装依赖
pip install -U darts[notorch] lightgbm xgboost matplotlib seaborn tqdm scikit-learn

# 环境变量（可选）
export USE_LGB_GPU=1        # LightGBM GPU 加速
export USE_XGB_GPU=1        # XGBoost GPU 加速
export RUN_BLEND_OPT=1      # 开启权重网格搜索
export USE_OPT_WEIGHTS=1    # 使用搜索到的最优权重

# 运行
python store_sales_v8.py
```

## 平台支持

| 平台 | GPU | 权重搜索 | 预计耗时 |
|------|-----|---------|---------|
| Kaggle T4 | ✓ | ✓ | ~50 分钟 |
| AutoDL 3080 | ✓ | ✓ | ~30 分钟 |
| AutoDL CPU | ✗ | ✗ | ~2.5 小时 |

## 输出文件

```
{output_dir}/
├── submission_v8_clean.csv           # 主提交文件
├── submission_v8_local_opt.csv       # 权重优化版（需开启 RUN_BLEND_OPT）
├── v8_local_blend_weights.csv        # 权重搜索结果表
├── v8_clean_stats.csv                # 各组件预测统计
└── experiment_report.zip             # 全部图表 + 实验汇总（可直接下载）
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `store_sales_v8.py` | **主方案** — Darts 全局模型 + 多模型集成 |
| `kaggle_notebook_v7.py` | V7 基线 — LGB+XGB+CatBoost + 傅里叶 + STL 分解 |
| `store_sales_darts_lgb_038.py` | 早期 Darts 实验版本 |
| `store_sales_h_blend_03795.py` | 独立权重搜索实验 |
| `feature_engineering.py` | 特征工程模块（滞后、滚动、傅里叶） |
| `eda.py` | 探索性数据分析 |
| `train.py` / `predict.py` | 训练与预测模块 |
| `data_loader.py` | 数据加载工具 |
| `config.py` | 全局配置常量 |
| `make_report_assets.py` | 实验报告图表生成 |
| `build_store_sales_report_docx.py` | DOCX 格式实验报告生成器 |

## 数据

本方案使用 Kaggle Store Sales 竞赛数据，包含：
- `train.csv` / `test.csv` — 训练与测试集
- `stores.csv` — 门店元信息（城市、州、类型、集群）
- `oil.csv` — 每日原油价格
- `transactions.csv` — 每日门店交易笔数
- `holidays_events.csv` — 节假日与事件日历

原始数据请从 [Kaggle 竞赛页面](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) 下载，放入项目根目录或通过 `STORE_SALES_DATA` 环境变量指定路径。

## 引用说明

本方案参考了 Kaggle 公开高分 notebook 中"全局时间序列建模 + 多窗口平均融合"的核心思想（Darts global models, per-family training, multi-lag ensemble, full-history / 2015+ dual-window averaging）。

在此基础上完成的独立工作：
- AutoDL / Kaggle 双平台适配与 GPU 训练自动检测
- LightGBM + XGBoost 双模型组件融合框架
- Linear / Log-space 双空间权重网格搜索
- 16 天回测验证 + 零销量店铺强制归零稳健性校验
- 全流程可视化诊断与实验报告自动打包

## 开源协议

MIT
