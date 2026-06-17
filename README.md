# Store Sales V8 — Darts Global Forecasting Pipeline

Kaggle [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition solution.

**Leaderboard score: 0.03792**

## Architecture

```
per-family Darts global models
  ├── LightGBM × 4 lags (63, 7, 365, 730)
  │     ├── full-history training
  │     └── 2015+ training
  ├── XGBoost × 4 lags (63, 7, 365, 730)
  │     ├── full-history training
  │     └── 2015+ training
  └── Blend: avg(avg(LGBM_full, LGBM_2015), avg(XGB_full, XGB_2015))
        ├── optional linear/log-space weight grid search (16-day backtest)
        └── zero-sales store gating
```

## Features

- **Darts global models** — one model per product family, 54 store series each
- **Dual-window averaging** — full history + 2015+ windows
- **Component blend optimizer** — exhaustive weight search in linear & log space
- **GPU support** — LightGBM GPU (CUDA) + XGBoost GPU (hist)
- **Auto diagnostics** — EDA charts, validation plots, feature analysis, auto-packaged report zip

## Quick Start

```bash
# Install
pip install -U darts[notorch] lightgbm xgboost matplotlib seaborn tqdm scikit-learn

# Set env (optional)
export USE_LGB_GPU=1        # CUDA GPU training
export RUN_BLEND_OPT=1      # enable weight search
export USE_OPT_WEIGHTS=1    # use best weights from search

# Run
python store_sales_v8.py
```

## Platform Support

| Platform | GPU | Blend Opt | Time |
|----------|-----|-----------|------|
| Kaggle T4 | ✓ | ✓ | ~50 min |
| AutoDL 3080 | ✓ | ✓ | ~30 min |
| AutoDL CPU | ✗ | ✗ | ~2.5 hr |

## Output

```
{output_dir}/
├── submission_v8_clean.csv           # main submission
├── submission_v8_local_opt.csv       # blend-optimized (if RUN_BLEND_OPT=1)
├── v8_local_blend_weights.csv        # weight search results
├── v8_clean_stats.csv                # per-component statistics
└── experiment_report.zip             # all charts + summary (downloadable)
```

## File Structure

| File | Description |
|------|-------------|
| `store_sales_v8.py` | Main pipeline (Darts global models + ensemble) |
| `kaggle_notebook_v7.py` | V7 baseline (LGB+XGB+CatBoost + Fourier + STL) |
| `store_sales_darts_lgb_038.py` | Early Darts experiment |
| `store_sales_h_blend_03795.py` | Blend optimizer standalone |
| `feature_engineering.py` | Feature generation (lags, rolling, Fourier) |
| `eda.py` | Exploratory data analysis |
| `train.py` / `predict.py` | Training/prediction modules |
| `data_loader.py` | Data loading utilities |
| `config.py` | Configuration constants |
| `make_report_assets.py` | Report figure generation |
| `build_store_sales_report_docx.py` | DOCX report builder |

## Attribution

This work references the global time-series modeling approach from public high-score Kaggle notebooks, with independent work on:
- AutoDL / Kaggle dual-platform GPU adaptation
- LightGBM + XGBoost dual-model component fusion
- Linear/log-space dual-space weight grid search
- 16-day backtest validation with zero-sales store gating
- Full-pipeline visualization diagnostics and automated report packaging


## License

MIT
