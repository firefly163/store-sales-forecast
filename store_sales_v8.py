"""
Store Sales V8 — Darts-based global forecasting pipeline.
Architecture: per-family global models with multi-lag ensemble.

Attribution:
  本方案参考了 Kaggle 公开高分 notebook 中"全局时间序列建模 + 多窗口平均融合"
  的核心思想（Darts global models, per-family training, multi-lag ensemble,
  full-history / 2015+ dual-window averaging）。
  在此基础上完成了以下独立工作：
    - AutoDL / Kaggle 双平台适配与 GPU 训练配置
    - LightGBM + XGBoost 双模型组件融合
    - Linear / Log-space 双空间权重网格搜索
    - 16 天回测验证 + 零销量店面强制归零稳健性校验
    - 全流程可视化诊断与实验报告自动打包
"""

from __future__ import annotations

import gc
import os
import time
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
    "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
})
sns.set_style("whitegrid")
sns.set_palette("Set2")

try:
    from darts import TimeSeries
    from darts.dataprocessing import Pipeline
    from darts.dataprocessing.transformers import (
        InvertibleMapper,
        Scaler,
        StaticCovariatesTransformer,
    )
    from darts.dataprocessing.transformers.missing_values_filler import (
        MissingValuesFiller,
    )
    from darts.metrics import rmsle
    from darts.models import LightGBMModel, XGBModel
    from darts.models.filtering.moving_average_filter import MovingAverageFilter
except Exception as exc:
    raise SystemExit(
        "Missing Darts dependencies. Install first:\n"
        "  pip install -U darts lightgbm xgboost\n"
        f"\nImport error: {exc}"
    )


# ══════════════════════════════════════════════════════
# Runtime configuration
# ══════════════════════════════════════════════════════

_OUTPUT_FILENAME = os.environ.get("DARTS_OUTPUT_NAME", "submission_v8_clean.csv")
_SEED = 0
_PREDICTION_WINDOW = 16
_ZERO_GATE_WINDOW = int(os.environ.get("ZERO_FC_WINDOW", "21"))
_TRUNCATE_BEFORE = "2015-01-01"

# GPU toggles — CPU by default for broader compatibility
_FLAG_LGB_GPU = os.environ.get("USE_LGB_GPU", "0").strip().lower() in {"1", "true", "yes"}
_FLAG_XGB_GPU = os.environ.get("USE_XGB_GPU", "0").strip().lower() in {"1", "true", "yes"}
_FLAG_RUN_VAL = os.environ.get("RUN_VALIDATION", "0").strip().lower() in {"1", "true", "yes"}
_FLAG_BLEND_OPT = os.environ.get("RUN_BLEND_OPT", "0").strip().lower() in {"1", "true", "yes"}
_FLAG_USE_OPT_W = os.environ.get("USE_OPT_WEIGHTS", "0").strip().lower() in {"1", "true", "yes"}
_FLAG_SAVE_PARTS = os.environ.get("SAVE_COMPONENTS", "1").strip().lower() not in {"0", "false", "no"}
_BLEND_SPACE = os.environ.get("COMPONENT_BLEND_SPACE", "linear").strip().lower()
_MANUAL_WEIGHTS_RAW = os.environ.get("COMPONENT_WEIGHTS", "").strip()

# XGBoost hyper-parameters
_XGB_TREES = int(os.environ.get("XGB_N_ESTIMATORS", "100"))
_XGB_LR = float(os.environ.get("XGB_LEARNING_RATE", "0.1"))
_XGB_DEPTH = int(os.environ.get("XGB_MAX_DEPTH", "6"))
_XGB_SUBSAMPLE = float(os.environ.get("XGB_SUBSAMPLE", "0.8"))
_XGB_COL_SAMPLE = float(os.environ.get("XGB_COLSAMPLE_BYTREE", "0.8"))
_XGB_MIN_CHILD = os.environ.get("XGB_MIN_CHILD_WEIGHT", "").strip()
_XGB_ALPHA = os.environ.get("XGB_REG_ALPHA", "").strip()
_XGB_LAMBDA = os.environ.get("XGB_REG_LAMBDA", "").strip()


def _parse_int_list(env_var: str, fallback: list[int]) -> list[int]:
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return fallback
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"{env_var} must contain at least one integer")
    return vals


def _parse_float_tuple(text: str, n_expected: int, label: str) -> list[float]:
    parts = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) != n_expected:
        raise ValueError(f"{label}: expected {n_expected} floats, got {len(parts)}")
    if not np.isclose(sum(parts), 1.0):
        raise ValueError(f"{label}: weights must sum to 1.0, got {sum(parts)}")
    return parts


_LAG_LIST = _parse_int_list(
    "DARTS_LAGS",
    [int(os.environ.get("BASE_LAG", "63")), 7, 365, 730],
)


def trace(message: str) -> None:
    """Lightweight structured logging."""
    print(message, flush=True)


# ══════════════════════════════════════════════════════
# File-system discovery
# ══════════════════════════════════════════════════════

def locate_dataset() -> Path:
    env = os.environ.get("STORE_SALES_DATA") or os.environ.get("DATA_DIR")
    search = []
    if env:
        search.append(Path(env))
    search.extend([
        Path("/kaggle/input/store-sales-time-series-forecasting"),
        Path("/kaggle/input/store-sales"),
        Path("/root/datasets"),
        Path("/root/autodl-tmp/store-sales-time-series-forecasting"),
        Path("/root/autodl-tmp/store-sales"),
        Path.cwd(),
    ])
    mandatory = {"train.csv", "test.csv", "stores.csv", "oil.csv", "transactions.csv", "holidays_events.csv"}
    for p in search:
        if p.exists() and mandatory.issubset({f.name for f in p.iterdir() if f.is_file()}):
            return p
    for base in [Path("/kaggle/input"), Path("/root/autodl-tmp"), Path("/root")]:
        if not base.exists():
            continue
        for dirpath, _, filenames in os.walk(base):
            if mandatory.issubset(set(filenames)):
                return Path(dirpath)
    raise FileNotFoundError(
        "Cannot locate competition data. Set STORE_SALES_DATA env var to the "
        "folder with train/test/stores/oil/transactions/holidays_events CSV files."
    )


def resolve_output_path() -> Path:
    env = os.environ.get("STORE_SALES_OUT") or os.environ.get("OUTPUT_DIR")
    if env:
        out = Path(env)
    elif Path("/kaggle/working").exists():
        out = Path("/kaggle/working")
    elif Path("/root").exists():
        out = Path("/root/output")
    else:
        out = Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _check_unique_ids(sub_df: pd.DataFrame, stage: str) -> None:
    duplicate_mask = sub_df["id"].notna() & sub_df["id"].duplicated(keep=False)
    if duplicate_mask.any():
        ctx_cols = [
            c for c in ["id", "date", "store_nbr", "family", "sales"]
            if c in sub_df.columns
        ]
        sample = sub_df.loc[duplicate_mask, ctx_cols].head(20)
        raise AssertionError(f"{stage}: duplicate ids found\n{sample}")


# ══════════════════════════════════════════════════════
# Data ingestion & preprocessing
# ══════════════════════════════════════════════════════

def ingest_source_files(folder: Path):
    trace(f"Reading data from: {folder}")
    tr = pd.read_csv(folder / "train.csv", parse_dates=["date"])
    te = pd.read_csv(folder / "test.csv", parse_dates=["date"])
    crude = pd.read_csv(folder / "oil.csv", parse_dates=["date"]).rename(columns={"dcoilwtico": "oil"})
    stores_df = pd.read_csv(folder / "stores.csv")
    txn = pd.read_csv(folder / "transactions.csv", parse_dates=["date"])
    evt = pd.read_csv(folder / "holidays_events.csv", parse_dates=["date"])
    trace(f"train={tr.shape}, test={te.shape}, stores={stores_df.shape}")
    return tr, te, crude, stores_df, txn, evt


def _clean_holiday_desc(text: str, stores_df: pd.DataFrame) -> str:
    if "futbol" in text:
        return "futbol"
    garbage = list(set(stores_df.city.str.lower()) | set(stores_df.state.str.lower()))
    for w in garbage:
        text = text.replace(w, "")
    return text


def assemble_modeling_table(train_raw, test_raw, crude_df, stores_df, txn_df, evt_df):
    trace("\n[1/6] Building unified calendar + features table ...")

    tr_start = train_raw.date.min().date()
    tr_end = train_raw.date.max().date()
    te_end = test_raw.date.max().date()

    gap_dates = pd.date_range(tr_start, tr_end).difference(train_raw.date.unique())
    gap_strs = gap_dates.strftime("%Y-%m-%d").tolist()

    full_idx = pd.MultiIndex.from_product(
        [pd.date_range(tr_start, tr_end), train_raw.store_nbr.unique(), train_raw.family.unique()],
        names=["date", "store_nbr", "family"],
    )
    train_raw = train_raw.set_index(["date", "store_nbr", "family"]).reindex(full_idx).reset_index()
    train_raw[["sales", "onpromotion"]] = train_raw[["sales", "onpromotion"]].fillna(0.0)
    train_raw["id"] = train_raw["id"].interpolate(method="linear")

    # Oil: forward/backward fill across full date range
    crude_df = crude_df.merge(
        pd.DataFrame({"date": pd.date_range(tr_start, te_end)}),
        on="date", how="outer",
    ).sort_values("date", ignore_index=True)
    crude_df["oil"] = crude_df["oil"].interpolate(method="linear", limit_direction="both")

    # Transactions: zero-fill on zero-sales days, then interpolate per store
    store_daily = train_raw.groupby(["date", "store_nbr"], observed=True)["sales"].sum().reset_index()
    txn_df = txn_df.merge(store_daily, on=["date", "store_nbr"], how="outer")
    txn_df = txn_df.sort_values(["date", "store_nbr"], ignore_index=True)
    txn_df.loc[txn_df["sales"].eq(0), "transactions"] = 0.0
    txn_df = txn_df.drop(columns=["sales"])
    txn_df["transactions"] = txn_df.groupby("store_nbr", group_keys=False)["transactions"].apply(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )

    # Holidays: clean descriptions, split work days from real holidays, one-hot encode
    evt_df = evt_df.copy()
    evt_df["description"] = (
        evt_df.apply(
            lambda r: str(r.description).lower().replace(str(r.locale_name).lower(), ""), axis=1
        )
        .apply(lambda s: _clean_holiday_desc(s, stores_df))
        .replace(r"[+-]\d+|\b(de|del|traslado|recupero|puente|-)\b", "", regex=True)
        .replace(r"\s+|-", " ", regex=True)
        .str.strip()
    )
    evt_df = evt_df[evt_df.transferred.eq(False)].copy()

    wd = evt_df[evt_df.type.eq("Work Day")]
    wd = wd[["date", "type"]].rename(columns={"type": "work_day"})
    wd = wd.drop_duplicates("date", keep="first").reset_index(drop=True)
    wd["work_day"] = wd["work_day"].notna().astype(int)
    evt_df = evt_df[evt_df.type != "Work Day"].reset_index(drop=True)

    nat = evt_df[evt_df.locale.eq("National")]
    nat = nat[["date", "description"]].drop_duplicates()
    nat = pd.get_dummies(nat, columns=["description"], prefix="nat")
    nat = nat.groupby("date", as_index=False).sum()
    nat = nat.rename(columns={"nat_primer grito independencia": "nat_primer grito"})

    _holiday_keys = [
        "nat_terremoto", "nat_navidad", "nat_dia la madre",
        "nat_dia trabajo", "nat_primer dia ano", "nat_futbol", "nat_dia difuntos",
    ]
    for hk in _holiday_keys:
        if hk not in nat.columns:
            nat[hk] = 0
    nat_compact = nat[["date", *_holiday_keys]]

    # Merge everything
    combined = pd.concat([train_raw, test_raw], axis=0, ignore_index=True)
    combined = combined.merge(txn_df, on=["date", "store_nbr"], how="left")
    _check_unique_ids(combined, "transactions feature merge")
    combined = combined.merge(crude_df, on="date", how="left")
    _check_unique_ids(combined, "oil feature merge")
    combined = combined.merge(stores_df, on="store_nbr", how="left")
    _check_unique_ids(combined, "store metadata merge")
    combined = combined.merge(wd, on="date", how="left")
    _check_unique_ids(combined, "work day holiday merge")
    combined = combined.merge(nat_compact, on="date", how="left")
    _check_unique_ids(combined, "national holiday merge")
    combined = combined.sort_values(["date", "store_nbr", "family"], ignore_index=True)

    combined[["work_day", *_holiday_keys]] = combined[["work_day", *_holiday_keys]].fillna(0)
    combined["transactions"] = combined["transactions"].fillna(0.0)

    # Calendar features
    combined["day"] = combined.date.dt.day
    combined["month"] = combined.date.dt.month
    combined["year"] = combined.date.dt.year
    combined["day_of_week"] = combined.date.dt.dayofweek
    combined["day_of_year"] = combined.date.dt.dayofyear
    combined["week_of_year"] = combined.date.dt.isocalendar().week.astype(int)
    combined["date_index"] = combined.date.factorize()[0]

    # Zero-sales imputation markers
    zero_dates = pd.to_datetime(
        gap_strs + [f"{y}-01-01" for y in range(2013, 2018)]
    )
    z_mask = (
        combined.date.isin(zero_dates)
        & combined.sales.eq(0)
        & combined.onpromotion.eq(0)
    )
    combined.loc[z_mask, ["sales", "onpromotion"]] = np.nan

    # String-encode categoricals for Darts static covariates
    combined["store_nbr"] = combined["store_nbr"].apply(lambda x: f"store_nbr_{x}")
    combined["cluster"] = combined["cluster"].apply(lambda x: f"cluster_{x}")
    combined["type"] = combined["type"].apply(lambda x: f"type_{x}")
    combined["city"] = combined["city"].apply(lambda x: f"city_{str(x).lower()}")
    combined["state"] = combined["state"].apply(lambda x: f"state_{str(x).lower()}")

    trace(f"assemble_modeling_table → {combined.shape}, families={combined.family.nunique()}")
    return combined, _holiday_keys, pd.Timestamp(tr_end), test_raw.copy()


# ══════════════════════════════════════════════════════
# Darts transformation pipeline
# ══════════════════════════════════════════════════════

def build_transform_pipeline(with_static=False, with_log=False):
    operations = [MissingValuesFiller(n_jobs=-1)]
    if with_static:
        operations.append(
            StaticCovariatesTransformer(
                transformer_cat=OneHotEncoder(handle_unknown="ignore"),
                n_jobs=-1,
            )
        )
    if with_log:
        operations.append(InvertibleMapper(fn=np.log1p, inverse_fn=np.expm1, n_jobs=-1))
    operations.append(Scaler())
    return Pipeline(operations)


def extract_target_sequences(table: pd.DataFrame, cutoff: pd.Timestamp, static_features):
    target_map = {}
    pipe_map = {}
    ident_map = {}

    for cat in tqdm(table.family.unique(), desc="Target sequences"):
        sub = table[(table.family.eq(cat)) & (table.date.le(cutoff))]
        pipe = build_transform_pipeline(with_static=True, with_log=True)

        seqs = TimeSeries.from_group_dataframe(
            df=sub,
            time_col="date",
            value_cols="sales",
            group_cols="store_nbr",
            static_cols=static_features,
        )
        seq_ids = [
            {"store_nbr": s.static_covariates["store_nbr"].iloc[0], "family": cat}
            for s in seqs
        ]
        seqs = pipe.fit_transform(seqs)

        target_map[cat] = [s.astype(np.float32) for s in seqs]
        pipe_map[cat] = pipe[2:]
        ident_map[cat] = seq_ids

    return target_map, pipe_map, ident_map


def extract_covariate_sequences(
    table: pd.DataFrame,
    cutoff: pd.Timestamp,
    past_features,
    future_features,
    past_ma_features=None,
    future_ma_features=None,
    past_ma_windows=(7, 28),
    future_ma_windows=(7, 28),
):
    past_map = {}
    future_map = {}
    base_pipe = build_transform_pipeline()

    for cat in tqdm(table.family.unique(), desc="Covariate sequences"):
        sub = table[table.family.eq(cat)]

        # Past covariates (history only)
        p_covs = TimeSeries.from_group_dataframe(
            df=sub[sub.date.le(cutoff)],
            time_col="date",
            value_cols=past_features,
            group_cols="store_nbr",
        )
        p_covs = [p.with_static_covariates(None) for p in p_covs]
        p_covs = base_pipe.fit_transform(p_covs)

        if past_ma_features is not None:
            for w in past_ma_windows:
                ma = MovingAverageFilter(window=w)
                old = [f"rolling_mean_{w}_{c}" for c in past_ma_features]
                new = [f"{c}_ma{w}" for c in past_ma_features]
                smooth = [
                    ma.filter(p[past_ma_features]).with_columns_renamed(old, new)
                    for p in p_covs
                ]
                p_covs = [pi.stack(si) for pi, si in zip(p_covs, smooth)]

        # Future covariates (full range)
        f_covs = TimeSeries.from_group_dataframe(
            df=sub,
            time_col="date",
            value_cols=future_features,
            group_cols="store_nbr",
        )
        f_covs = [f.with_static_covariates(None) for f in f_covs]
        f_covs = base_pipe.fit_transform(f_covs)

        if future_ma_features is not None:
            for w in future_ma_windows:
                ma = MovingAverageFilter(window=w)
                old = [f"rolling_mean_{w}_{c}" for c in future_ma_features]
                new = [f"{c}_ma{w}" for c in future_ma_features]
                smooth = [
                    ma.filter(f[future_ma_features]).with_columns_renamed(old, new)
                    for f in f_covs
                ]
                f_covs = [fi.stack(si) for fi, si in zip(f_covs, smooth)]

        past_map[cat] = [p.astype(np.float32) for p in p_covs]
        future_map[cat] = [f.astype(np.float32) for f in f_covs]

    return past_map, future_map


# ══════════════════════════════════════════════════════
# Global forecast engine
# ══════════════════════════════════════════════════════

class GlobalForecastEngine:
    """Trains and ensembles Darts global models per product family."""

    def __init__(
        self,
        target_map,
        pipe_map,
        ident_map,
        past_map,
        future_map,
        horizon,
        n_folds,
        zero_window,
        keep_static=None,
        keep_past=None,
        keep_future=None,
    ):
        self.target_map = target_map.copy()
        self.pipe_map = pipe_map.copy()
        self.ident_map = ident_map.copy()
        self.past_map = past_map.copy()
        self.future_map = future_map.copy()
        self.horizon = horizon
        self.n_folds = n_folds
        self.zero_window = zero_window
        self.keep_static = keep_static
        self.keep_past = keep_past
        self.keep_future = keep_future
        self._apply_filters()

    def _apply_filters(self):
        for cat in tqdm(self.target_map.keys(), desc="Filter setup"):
            if self.keep_static != "keep_all":
                if self.keep_static is not None:
                    seqs = self.target_map[cat]
                    masked = [
                        c for c in seqs[0].static_covariates.columns
                        if c.startswith(tuple(self.keep_static))
                    ]
                    df_list = [s.static_covariates[masked] for s in seqs]
                    self.target_map[cat] = [
                        s.with_static_covariates(d) for s, d in zip(seqs, df_list)
                    ]
                else:
                    self.target_map[cat] = [
                        s.with_static_covariates(None) for s in self.target_map[cat]
                    ]

            if self.keep_past != "keep_all":
                self.past_map[cat] = (
                    [p[self.keep_past] for p in self.past_map[cat]]
                    if self.keep_past is not None else None
                )

            if self.keep_future != "keep_all":
                self.future_map[cat] = (
                    [f[self.keep_future] for f in self.future_map[cat]]
                    if self.keep_future is not None else None
                )

    @staticmethod
    def _clamp(arr):
        return np.clip(arr, a_min=0.0, a_max=None)

    def _split_train_val(self, seqs, tail_len):
        train = [s[:-tail_len] for s in seqs]
        val_end = -tail_len + self.horizon
        if val_end >= 0:
            val_end = None
        valid = [s[-tail_len:val_end] for s in seqs]
        return train, valid

    def _instantiate_models(self, model_keys, configs):
        registry = {"lgbm": LightGBMModel, "xgb": XGBModel}
        if len(model_keys) != len(configs):
            raise ValueError("model_keys and configs must have same length")
        resolved = []
        for key, cfg in zip(model_keys, configs):
            c = dict(cfg)
            if key == "lgbm":
                c.setdefault("n_jobs", -1)
                if _FLAG_LGB_GPU:
                    c.setdefault("device", "gpu")
            if key == "xgb":
                c = {"tree_method": "hist", **c}
                if _FLAG_XGB_GPU:
                    c.setdefault("device", "cuda")
            resolved.append(c)
        return [registry[key](**resolved[i]) for i, key in enumerate(model_keys)]

    def _produce_forecasts(self, models, train_seqs, pipe, p_covs, f_covs, truncate_before):
        if truncate_before is not None:
            boundary = pd.Timestamp(truncate_before) - pd.Timedelta(days=1)
            train_seqs = [s.drop_before(boundary) for s in train_seqs]

        fit_kwargs = {"series": train_seqs, "past_covariates": p_covs, "future_covariates": f_covs}
        n_stores = len(train_seqs)

        zero_template = TimeSeries.from_dataframe(
            df=pd.DataFrame({
                "date": pd.date_range(train_seqs[0].end_time(), periods=self.horizon + 1)[1:],
                "sales": np.zeros(self.horizon),
            }),
            time_col="date", value_cols="sales",
        )

        per_model_preds = []
        running_ensemble = [0.0 for _ in range(n_stores)]

        for idx, mdl in enumerate(models, start=1):
            trace(f"    fitting member {idx}/{len(models)}: {mdl.__class__.__name__}")
            mdl.fit(**fit_kwargs)
            raw_pred = mdl.predict(n=self.horizon, **fit_kwargs)
            restored = pipe.inverse_transform(raw_pred)

            for j in range(n_stores):
                if train_seqs[j][-self.zero_window:].values().sum() == 0:
                    restored[j] = zero_template

            cleaned = [p.map(self._clamp) for p in restored]
            per_model_preds.append(cleaned)
            scale = 1.0 / len(models)
            for j in range(n_stores):
                running_ensemble[j] += cleaned[j] * scale

            gc.collect()

        return per_model_preds, running_ensemble

    @staticmethod
    def _score(actual, pred):
        try:
            return rmsle(actual, pred, series_reduction=np.mean)
        except TypeError:
            return rmsle(actual, pred, inter_reduction=np.mean)

    @staticmethod
    def _ts_to_df(series):
        if hasattr(series, "pd_dataframe"):
            return series.pd_dataframe()
        return series.to_dataframe()

    def run_validation(self, model_keys, configs, truncate_before=None):
        model_scores = []
        ens_scores = []

        for cat in tqdm(self.target_map, desc="Validation"):
            seqs = self.target_map[cat]
            pipe = self.pipe_map[cat]
            p_covs = self.past_map[cat]
            f_covs = self.future_map[cat]

            fold_metrics = []
            ens_acc = 0.0
            for fold_i in range(self.n_folds):
                tail = (self.n_folds - fold_i) * self.horizon
                train, valid = self._split_train_val(seqs, tail)
                valid = pipe.inverse_transform(valid)
                models = self._instantiate_models(model_keys, configs)
                pred_list, ens = self._produce_forecasts(
                    models, train, pipe, p_covs, f_covs, truncate_before
                )
                fold_metrics.append([self._score(valid, p) / self.n_folds for p in pred_list])
                if len(models) > 1:
                    ens_acc += self._score(valid, ens) / self.n_folds

            fold_metrics = np.sum(fold_metrics, axis=0)
            model_scores.append(fold_metrics)
            ens_scores.append(ens_acc)
            parts = " - ".join(f"{k}: {v:.5f}" for k, v in zip(model_keys, fold_metrics))
            extra = f" - ens: {ens_acc:.5f}" if len(model_keys) > 1 else ""
            trace(f"{cat}: {parts}{extra}")

        avg_models = np.mean(model_scores, axis=0)
        parts = " - ".join(f"{k}: {v:.5f}" for k, v in zip(model_keys, avg_models))
        extra = f" - ens: {np.mean(ens_scores):.5f}" if len(model_keys) > 1 else ""
        trace(f"Average RMSLE | {parts}{extra}")

    def predict_test(self, model_keys, configs, truncate_before=None):
        all_frames = []
        for cat in tqdm(self.target_map.keys(), desc="Test forecasts"):
            trace(f"  family={cat}, truncate_before={truncate_before}")
            seqs = self.target_map[cat]
            pipe = self.pipe_map[cat]
            ids = self.ident_map[cat]
            p_covs = self.past_map[cat]
            f_covs = self.future_map[cat]

            models = self._instantiate_models(model_keys, configs)
            _, ens = self._produce_forecasts(models, seqs, pipe, p_covs, f_covs, truncate_before)
            df_list = [self._ts_to_df(p).assign(**i) for p, i in zip(ens, ids)]
            all_frames.append(pd.concat(df_list, axis=0))
            gc.collect()

        result = pd.concat(all_frames, axis=0)
        return result.rename_axis(None, axis=1).reset_index(names="date")

    def predict_validation(self, model_keys, configs, val_len=16, truncate_before=None):
        pred_frames = []
        actual_frames = []

        for cat in tqdm(self.target_map.keys(), desc="Validation forecasts"):
            trace(f"  validation family={cat}, truncate_before={truncate_before}")
            seqs = self.target_map[cat]
            pipe = self.pipe_map[cat]
            ids = self.ident_map[cat]
            p_covs = self.past_map[cat]
            f_covs = self.future_map[cat]

            train, valid = self._split_train_val(seqs, val_len)
            valid = pipe.inverse_transform(valid)

            models = self._instantiate_models(model_keys, configs)
            _, ens = self._produce_forecasts(models, train, pipe, p_covs, f_covs, truncate_before)

            pred_frames.append(
                pd.concat([self._ts_to_df(p).assign(**i) for p, i in zip(ens, ids)], axis=0)
            )
            actual_frames.append(
                pd.concat(
                    [self._ts_to_df(v).rename(columns={"sales": "actual"}).assign(**i)
                     for v, i in zip(valid, ids)], axis=0
                )
            )
            gc.collect()

        preds = pd.concat(pred_frames, axis=0).rename_axis(None, axis=1).reset_index(names="date")
        acts = pd.concat(actual_frames, axis=0).rename_axis(None, axis=1).reset_index(names="date")
        return preds, acts


# ══════════════════════════════════════════════════════
# Model configuration builders
# ══════════════════════════════════════════════════════

def configure_lightgbm_ensemble(use_past=True, use_future=True):
    base = {
        "random_state": _SEED,
        "lags": _LAG_LIST[0],
        "lags_past_covariates": list(range(-16, -23, -1)) if use_past else None,
        "lags_future_covariates": (14, 1) if use_future else None,
        "output_chunk_length": 1,
    }
    keys = ["lgbm"] * len(_LAG_LIST)
    cfgs = [dict(base, lags=lag) for lag in _LAG_LIST]
    return keys, cfgs


def configure_xgboost_ensemble(use_past=True, use_future=True):
    base = {
        "random_state": _SEED,
        "lags": _LAG_LIST[0],
        "lags_past_covariates": list(range(-16, -23, -1)) if use_past else None,
        "lags_future_covariates": (14, 1) if use_future else None,
        "output_chunk_length": 1,
        "n_estimators": _XGB_TREES,
        "learning_rate": _XGB_LR,
        "max_depth": _XGB_DEPTH,
        "subsample": _XGB_SUBSAMPLE,
        "colsample_bytree": _XGB_COL_SAMPLE,
        "tree_method": "hist",
    }
    for k, v in [("min_child_weight", _XGB_MIN_CHILD), ("reg_alpha", _XGB_ALPHA), ("reg_lambda", _XGB_LAMBDA)]:
        if v:
            base[k] = float(v)
    keys = ["xgb"] * len(_LAG_LIST)
    cfgs = [dict(base, lags=lag) for lag in _LAG_LIST]
    return keys, cfgs


# ══════════════════════════════════════════════════════
# Ensemble blending utilities
# ══════════════════════════════════════════════════════

def blend_linear(*frames: pd.DataFrame, weights=None) -> pd.DataFrame:
    if not frames:
        raise ValueError("need at least one frame")
    ref = frames[0][["date", "store_nbr", "family"]].copy()
    if weights is None:
        weights = [1.0 / len(frames)] * len(frames)
    if len(weights) != len(frames):
        raise ValueError("weight count mismatch")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError(f"weights must sum to 1.0, got {sum(weights)}")
    key = ["date", "store_nbr", "family"]
    acc = np.zeros(len(ref), dtype=np.float64)
    for i, (f, w) in enumerate(zip(frames, weights), 1):
        if not ref[key].equals(f[key]):
            raise ValueError(f"frame {i} not aligned")
        acc += w * f["sales"].to_numpy(dtype=np.float64)
    ref["sales"] = np.maximum(acc, 0.0)
    return ref


def blend_log_space(*frames: pd.DataFrame, weights=None) -> pd.DataFrame:
    if not frames:
        raise ValueError("need at least one frame")
    ref = frames[0][["date", "store_nbr", "family"]].copy()
    if weights is None:
        weights = [1.0 / len(frames)] * len(frames)
    if len(weights) != len(frames):
        raise ValueError("weight count mismatch")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError(f"weights must sum to 1.0, got {sum(weights)}")
    key = ["date", "store_nbr", "family"]
    acc = np.zeros(len(ref), dtype=np.float64)
    for i, (f, w) in enumerate(zip(frames, weights), 1):
        if not ref[key].equals(f[key]):
            raise ValueError(f"frame {i} not aligned")
        acc += w * np.log1p(np.maximum(f["sales"].to_numpy(dtype=np.float64), 0.0))
    ref["sales"] = np.maximum(np.expm1(acc), 0.0)
    return ref


def format_kaggle_output(raw_test: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    preds = preds.copy()
    preds["store_nbr"] = preds["store_nbr"].replace("store_nbr_", "", regex=True).astype(int)
    sub = raw_test.merge(preds, on=["date", "store_nbr", "family"], how="left")[["id", "sales"]]
    sub["sales"] = sub["sales"].fillna(0).clip(lower=0)
    sub["id"] = sub["id"].astype(int)
    _check_unique_ids(sub, "final submission")
    return sub


def compute_rmsle(y_act: np.ndarray, y_hat: np.ndarray) -> float:
    y_act = np.maximum(y_act.astype(np.float64), 0.0)
    y_hat = np.maximum(y_hat.astype(np.float64), 0.0)
    return float(np.sqrt(np.mean((np.log1p(y_hat) - np.log1p(y_act)) ** 2)))


def _align_components(actual: pd.DataFrame, parts: dict[str, pd.DataFrame]):
    key = ["date", "store_nbr", "family"]
    act_sort = actual.sort_values(key).reset_index(drop=True)
    y_act = act_sort["actual"].to_numpy(dtype=np.float64)
    out = {}
    for name, frame in parts.items():
        f_sort = frame.sort_values(key).reset_index(drop=True)
        if not act_sort[key].equals(f_sort[key]):
            raise ValueError(f"misaligned component: {name}")
        out[name] = np.maximum(f_sort["sales"].to_numpy(dtype=np.float64), 0.0)
    return y_act, out


def _walk_simplex(n_parts: int, step: float):
    total_units = int(round(1.0 / step))
    if not np.isclose(total_units * step, 1.0):
        raise ValueError("BLEND_GRID_STEP must divide 1.0 evenly")

    def _recurse(prefix, remain, slots):
        if slots == 1:
            yield prefix + [remain]
            return
        for v in range(remain + 1):
            yield from _recurse(prefix + [v], remain - v, slots - 1)

    for combo in _recurse([], total_units, n_parts):
        yield np.array(combo, dtype=np.float64) * step


def optimize_component_weights(
    actual: pd.DataFrame,
    parts: dict[str, pd.DataFrame],
    step: float = 0.05,
    top_k: int = 30,
) -> pd.DataFrame:
    names = list(parts)
    y_act, arr_map = _align_components(actual, parts)
    mat = np.vstack([arr_map[n] for n in names])
    log_mat = np.log1p(mat)

    records = []
    for w in _walk_simplex(len(names), step):
        if np.count_nonzero(w) == 0:
            continue
        records.append({
            "space": "linear",
            "rmsle": compute_rmsle(y_act, np.dot(w, mat)),
            "component_order": "|".join(names),
            "weights": "|".join(f"{x:.6f}" for x in w),
        })
        records.append({
            "space": "log",
            "rmsle": compute_rmsle(y_act, np.expm1(np.dot(w, log_mat))),
            "component_order": "|".join(names),
            "weights": "|".join(f"{x:.6f}" for x in w),
        })

    result = pd.DataFrame(records).sort_values("rmsle", ignore_index=True)
    return result.head(top_k)


def apply_weight_solution(parts: dict[str, pd.DataFrame], row: pd.Series) -> pd.DataFrame:
    names = str(row["component_order"]).split("|")
    w = [float(x) for x in str(row["weights"]).split("|")]
    frames = [parts[n] for n in names]
    if row["space"] == "log":
        return blend_log_space(*frames, weights=w)
    return blend_linear(*frames, weights=w)


def apply_manual_weights(
    parts: dict[str, pd.DataFrame],
    raw_weights: str,
    space: str,
) -> pd.DataFrame:
    default_order = ["lgb_full", "lgb_2015", "xgb_full", "xgb_2015"]
    w = _parse_float_tuple(raw_weights, len(default_order), "COMPONENT_WEIGHTS")
    frames = [parts[n] for n in default_order]
    if space == "log":
        return blend_log_space(*frames, weights=w)
    if space == "linear":
        return blend_linear(*frames, weights=w)
    raise ValueError("COMPONENT_BLEND_SPACE must be 'linear' or 'log'")


def persist_component_submission(
    out_dir: Path,
    raw_test: pd.DataFrame,
    label: str,
    preds: pd.DataFrame,
    log_rows: list[dict[str, object]],
) -> pd.DataFrame:
    sub = format_kaggle_output(raw_test, preds)
    path = out_dir / label
    sub.to_csv(path, index=False)
    log_rows.append({
        "file": label,
        "rows": len(sub),
        "zero_rows": int((sub["sales"] == 0).sum()),
        "sales_min": float(sub["sales"].min()),
        "sales_mean": float(sub["sales"].mean()),
        "sales_median": float(sub["sales"].median()),
        "sales_max": float(sub["sales"].max()),
    })
    trace(f"Saved: {path}")
    return sub


def run_weight_search(
    engine: GlobalForecastEngine,
    lgb_keys, lgb_cfgs,
    xgb_keys, xgb_cfgs,
    out_dir: Path,
):
    trace("\n[blend-opt] Generating validation hold-out predictions ...")
    lgb_full_val, act = engine.predict_validation(
        model_keys=lgb_keys, configs=lgb_cfgs,
        val_len=_PREDICTION_WINDOW, truncate_before=None,
    )
    lgb_2015_val, act2 = engine.predict_validation(
        model_keys=lgb_keys, configs=lgb_cfgs,
        val_len=_PREDICTION_WINDOW, truncate_before=_TRUNCATE_BEFORE,
    )
    xgb_full_val, act3 = engine.predict_validation(
        model_keys=xgb_keys, configs=xgb_cfgs,
        val_len=_PREDICTION_WINDOW, truncate_before=None,
    )
    xgb_2015_val, act4 = engine.predict_validation(
        model_keys=xgb_keys, configs=xgb_cfgs,
        val_len=_PREDICTION_WINDOW, truncate_before=_TRUNCATE_BEFORE,
    )

    align_key = ["date", "store_nbr", "family", "actual"]
    if not act[align_key].equals(act2[align_key]):
        raise ValueError("LGBM actuals mismatch (full vs 2015)")
    if not act[align_key].equals(act3[align_key]):
        raise ValueError("actuals mismatch (LGBM vs XGB)")
    if not act[align_key].equals(act4[align_key]):
        raise ValueError("XGB actuals mismatch (full vs 2015)")

    parts = {"lgb_full": lgb_full_val, "lgb_2015": lgb_2015_val,
             "xgb_full": xgb_full_val, "xgb_2015": xgb_2015_val}
    step = float(os.environ.get("BLEND_GRID_STEP", "0.05"))
    results = optimize_component_weights(act, parts, step=step, top_k=50)
    out_csv = out_dir / "v8_local_blend_weights.csv"
    results.to_csv(out_csv, index=False)
    trace(f"[blend-opt] Saved: {out_csv}")
    trace("[blend-opt] Best configurations:")
    trace(results.head(10).to_string(index=False))
    return results, act, parts


# ══════════════════════════════════════════════════════
# Visualization & reporting
# ══════════════════════════════════════════════════════

def _ensure_report_dirs(base: Path):
    for d in ["eda", "features", "models", "validation", "predictions"]:
        (base / d).mkdir(parents=True, exist_ok=True)


def _save_figure(fig, base: Path, sub: str, fname: str):
    path = base / sub / fname
    fig.savefig(str(path), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    trace(f"  [FIG] {sub}/{fname}")


def _plot_safe(fn, label: str = "chart"):
    try:
        fn()
    except Exception as e:
        trace(f"  [WARN] {label} failed: {e}")


def render_exploratory_charts(table, base: Path):
    train_slice = table[table["sales"].notna()].copy()
    top_cats = (
        train_slice.groupby("family")["sales"].mean()
        .sort_values(ascending=False).head(6).index.tolist()
    )
    dow_labels = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

    # 1 — Distribution
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 5))
    clipped = train_slice["sales"].clip(upper=train_slice["sales"].quantile(0.999))
    a0.hist(clipped, bins=100, color="#2196F3", alpha=0.7, edgecolor="white", lw=0.3)
    a0.axvline(train_slice["sales"].mean(), color="red", ls="--", lw=1.2,
               label=f"Mean={train_slice['sales'].mean():.1f}")
    a0.set_xlabel("Sales"); a0.set_title("Sales Distribution (99.9% clip)"); a0.legend()
    logv = np.log1p(train_slice["sales"].clip(lower=0))
    a1.hist(logv, bins=100, color="#4CAF50", alpha=0.7, edgecolor="white", lw=0.3)
    a1.axvline(logv.mean(), color="red", ls="--", lw=1.2, label=f"Mean={logv.mean():.2f}")
    a1.set_xlabel("log(1+sales)"); a1.set_title("Log-Sales Distribution"); a1.legend()
    fig.suptitle("Sales Distribution Analysis", fontsize=14, y=1.01)
    _save_figure(fig, base, "eda", "01_sales_distribution.png")

    # 2 — Time series
    daily = train_slice.groupby(["date", "family"])["sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(16, 6))
    for cat in top_cats:
        s = daily[daily["family"] == cat].set_index("date")
        ax.plot(s.index, s["sales"].rolling(7, center=True, min_periods=1).mean(),
                lw=1.0, alpha=0.8, label=cat)
    ax.axvline(pd.Timestamp("2016-04-16"), color="red", ls="--", lw=1.5, alpha=0.7,
               label="Earthquake 2016-04-16")
    ax.set_xlabel("Date"); ax.set_ylabel("Daily Sales (7d MA)")
    ax.set_title("Top-6 Product Families — Sales Over Time")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    _save_figure(fig, base, "eda", "02_sales_timeseries.png")

    # 3 — Heatmap
    heat = train_slice.pivot_table(values="sales", index="store_nbr", columns="family", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(heat, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Avg Sales"},
                xticklabels=True, yticklabels=True, linewidths=0, rasterized=True)
    ax.set_xlabel("Family"); ax.set_ylabel("Store")
    ax.set_title("Average Sales — Store × Family Heatmap")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    _save_figure(fig, base, "eda", "03_store_family_heatmap.png")

    # 4 — Weekly pattern
    dow_data = train_slice.groupby(["day_of_week", "family"])["sales"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    for cat in top_cats:
        s = dow_data[dow_data["family"] == cat]
        ax.plot(s["day_of_week"], s["sales"], "o-", lw=1.5, ms=5, label=cat)
    ax.set_xticks(range(7)); ax.set_xticklabels([dow_labels[i] for i in range(7)])
    ax.set_xlabel("Day of Week"); ax.set_ylabel("Average Sales")
    ax.set_title("Weekly Sales Pattern — Top Families"); ax.legend(fontsize=8)
    _save_figure(fig, base, "eda", "04_weekly_pattern.png")

    # 5 — Monthly trend
    monthly = train_slice.groupby(["year", "month"])["sales"].mean().reset_index()
    monthly["ym"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(len(monthly)), monthly["sales"], "o-", color="#2196F3", lw=1.5, ms=5)
    ax.set_xticks(range(0, len(monthly), 3))
    ax.set_xticklabels(monthly["ym"].iloc[::3], rotation=45, fontsize=8)
    ax.set_xlabel("Year-Month"); ax.set_ylabel("Average Sales")
    ax.set_title("Monthly Average Sales Trend")
    eq_pos = len(monthly[monthly["ym"] < "2016-04"])
    ax.axvline(x=eq_pos, color="red", ls="--", lw=1.2, alpha=0.7, label="Earthquake 2016-04")
    ax.legend()
    _save_figure(fig, base, "eda", "05_monthly_trend.png")

    # 6 — Oil vs Sales
    agg = train_slice.groupby("date").agg(
        sales=("sales", "mean"), oil=("oil", "first")).reset_index()
    fig, ax1 = plt.subplots(figsize=(16, 5))
    ax1.plot(agg["date"], agg["sales"].rolling(30, min_periods=1).mean(),
             color="#2196F3", lw=1.5, label="Avg Sales (30d MA)")
    ax1.set_ylabel("Average Sales", color="#2196F3"); ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax2 = ax1.twinx()
    ax2.plot(agg["date"], agg["oil"].rolling(30, min_periods=1).mean(),
             color="#FF5722", lw=1.5, alpha=0.8, label="Oil Price (30d MA)")
    ax2.set_ylabel("Oil Price", color="#FF5722"); ax2.tick_params(axis="y", labelcolor="#FF5722")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax1.set_xlabel("Date"); fig.autofmt_xdate()
    fig.suptitle("Oil Price vs Sales (30-day Moving Average)", fontsize=14, y=1.01)
    _save_figure(fig, base, "eda", "06_oil_vs_sales.png")

    # 7 — Holiday effect
    nat_cols = [c for c in table.columns if c.startswith("nat_")]
    if nat_cols:
        h_dates = sorted(set().union(*[set(table.loc[table[c] == 1, "date"].unique()) for c in nat_cols]))
        if h_dates:
            is_h = train_slice["date"].isin(h_dates)
            fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 5))
            cap = train_slice["sales"].quantile(0.99)
            bp = a0.boxplot(
                [train_slice.loc[is_h, "sales"].clip(upper=cap),
                 train_slice.loc[~is_h, "sales"].clip(upper=cap)],
                labels=["Holiday", "Non-Holiday"], patch_artist=True, widths=0.5,
            )
            bp["boxes"][0].set_facecolor("#FF5722"); bp["boxes"][1].set_facecolor("#4CAF50")
            a0.set_ylabel("Sales"); a0.set_title("Sales: Holiday vs Non-Holiday")
            h_dow = train_slice.loc[is_h].groupby("day_of_week")["sales"].mean()
            a1.bar(h_dow.index, h_dow.values, color="#FF9800", alpha=0.8, edgecolor="white", lw=0.3)
            a1.set_xticks(range(7)); a1.set_xticklabels([dow_labels[i] for i in range(7)])
            a1.set_xlabel("Day of Week"); a1.set_ylabel("Average Sales")
            a1.set_title("Holiday Sales by Day of Week")
            fig.suptitle("Holiday Effect Analysis", fontsize=14, y=1.01)
            _save_figure(fig, base, "eda", "07_holiday_effect.png")

    trace("EDA charts done (7)")


def render_target_samples(target_map, base: Path):
    cats = list(target_map.keys())
    sample = cats[:min(3, len(cats))]
    fig, axes = plt.subplots(len(sample), 1, figsize=(14, 3.5 * len(sample)))
    if len(sample) == 1:
        axes = [axes]
    for ax, cat in zip(axes, sample):
        for i, ts in enumerate(target_map[cat][:min(5, len(target_map[cat]))]):
            ax.plot(ts.time_index, ts.values().flatten(), lw=0.6, alpha=0.8,
                    label=f"store {ts.static_covariates.iloc[0,0]}" if ts.has_static_covariates else f"series {i}")
        ax.set_title(f"Family: {cat} — Target (log-scaled)")
        ax.set_ylabel("log(1+sales) scaled"); ax.legend(fontsize=6, ncol=2)
    fig.autofmt_xdate()
    fig.suptitle("Target Series Examples (after pipeline)", fontsize=14, y=1.01)
    _save_figure(fig, base, "features", "01_target_series.png")
    trace("Target sample charts done")


def render_covariate_samples(future_map, base: Path):
    cats = list(future_map.keys())
    seqs = future_map[cats[0]]
    first = seqs[0]
    cols = first.columns.tolist()
    show = cols[:min(8, len(cols))]
    fig, axes = plt.subplots(len(show), 1, figsize=(14, 2 * len(show)), sharex=True)
    for ax, c in zip(axes, show):
        ax.plot(first.time_index, first[c].values().flatten(), lw=0.6, color="#2196F3")
        ax.set_ylabel(c, fontsize=8)
    axes[-1].set_xlabel("Date"); fig.autofmt_xdate()
    fig.suptitle(f"Future Covariates — Family: {cats[0]}", fontsize=14, y=1.01)
    _save_figure(fig, base, "features", "02_covariates.png")
    trace("Covariate sample charts done")


def render_validation_diagnostics(actual, parts, base: Path):
    key = ["date", "store_nbr", "family"]
    act_sort = actual.sort_values(key).reset_index(drop=True)
    y_act = act_sort["actual"].to_numpy(dtype=np.float64)

    n = len(parts)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([axes])
    for ax, (name, frame) in zip(axes.flatten(), parts.items()):
        f_sort = frame.sort_values(key).reset_index(drop=True)
        y_hat = np.maximum(f_sort["sales"].to_numpy(dtype=np.float64), 0.0)
        s = float(np.sqrt(np.mean((np.log1p(y_hat) - np.log1p(y_act)) ** 2)))
        ax.scatter(y_act[::50], y_hat[::50], alpha=0.15, s=2, color="#2196F3", edgecolors="none")
        mx = max(y_act.max(), y_hat.max()) * 1.05
        ax.plot([0, mx], [0, mx], "r--", lw=1.0, alpha=0.5)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title(f"{name} (RMSLE={s:.4f})")
    for ax in axes.flatten()[n:]:
        ax.set_visible(False)
    fig.suptitle("Predicted vs Actual — Validation Set", fontsize=14, y=1.01)
    _save_figure(fig, base, "validation", "01_pred_vs_actual.png")

    # Residuals
    fig, axes = plt.subplots(1, min(4, n), figsize=(4.5 * min(4, n), 4))
    if n == 1:
        axes = [axes]
    for ax, (name, frame) in zip(axes, list(parts.items())[:4]):
        f_sort = frame.sort_values(key).reset_index(drop=True)
        y_hat = np.maximum(f_sort["sales"].to_numpy(dtype=np.float64), 0.0)
        res = np.log1p(y_hat) - np.log1p(y_act)
        ax.hist(res, bins=80, color="#2196F3", alpha=0.7, edgecolor="white", lw=0.3)
        ax.axvline(0, color="red", ls="--", lw=1.0)
        ax.axvline(np.mean(res), color="black", ls="-", lw=1.0, label=f"Mean={np.mean(res):.4f}")
        ax.set_xlabel("log residual"); ax.set_title(name); ax.legend(fontsize=7)
    fig.suptitle("Residual Distribution — Validation Set", fontsize=14, y=1.01)
    _save_figure(fig, base, "validation", "02_residual_distribution.png")

    # Component comparison
    scores = {}
    for name, frame in parts.items():
        f_sort = frame.sort_values(key).reset_index(drop=True)
        y_hat = np.maximum(f_sort["sales"].to_numpy(dtype=np.float64), 0.0)
        scores[name] = float(np.sqrt(np.mean((np.log1p(y_hat) - np.log1p(y_act)) ** 2)))
    fig, ax = plt.subplots(figsize=(10, 5))
    order = sorted(scores.items(), key=lambda x: x[1])
    names_s = [x[0] for x in order]; vals_s = [x[1] for x in order]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names_s)))
    bars = ax.barh(names_s, vals_s, color=colors, alpha=0.85, edgecolor="white")
    for b, v in zip(bars, vals_s):
        ax.text(b.get_width() + 0.0003, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlabel("RMSLE"); ax.set_title("Component RMSLE Comparison"); ax.invert_yaxis()
    _save_figure(fig, base, "validation", "03_component_comparison.png")
    trace("Validation diagnostics done")


def render_prediction_summary(comp_preds, final_preds, base: Path):
    vals = {}
    for k, df in comp_preds.items():
        vals[k] = np.maximum(df["sales"].to_numpy(dtype=np.float64), 0.0)
    vals["final"] = np.maximum(final_preds["sales"].to_numpy(dtype=np.float64), 0.0)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 5))
    plot_keys = [k for k in vals if k != "final" and not k.startswith(("lgbm_", "xgb_"))]
    for k in plot_keys[:6]:
        v = vals[k][vals[k] > 0]
        a0.hist(np.log1p(v), bins=60, alpha=0.4, label=k, density=True)
    a0.set_xlabel("log(1+sales)"); a0.set_title("Component Prediction Distributions"); a0.legend(fontsize=7)

    means = {k: v.mean() for k, v in vals.items()}
    order = sorted(means.items(), key=lambda x: x[1])
    a1.barh([x[0] for x in order], [x[1] for x in order], color="#FF9800", alpha=0.8, edgecolor="white")
    a1.set_xlabel("Mean Predicted Sales"); a1.set_title("Mean Prediction by Component")
    fig.suptitle("Prediction Analysis", fontsize=14, y=1.01)
    _save_figure(fig, base, "predictions", "01_prediction_distribution.png")
    trace("Prediction summary charts done")


def render_weight_landscape(results_df: pd.DataFrame, base: Path):
    if results_df is None or results_df.empty:
        return
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, sp in zip([a0, a1], ["linear", "log"]):
        sub = results_df[results_df["space"] == sp].head(15)
        if sub.empty:
            ax.text(0.5, 0.5, f"No {sp} results", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{sp} space"); continue
        ax.barh(range(len(sub)), sub["rmsle"], color="#2196F3", alpha=0.8, edgecolor="white")
        ax.set_title(f"{sp} space — Best RMSLE={sub.iloc[0]['rmsle']:.5f}")
        ax.set_xlabel("RMSLE"); ax.invert_yaxis()
    fig.suptitle("Blend Weight Search Results", fontsize=14, y=1.01)
    _save_figure(fig, base, "models", "01_blend_search.png")
    trace("Weight landscape chart done")


def bundle_output_archive(report_dir: Path, out_dir: Path, sub_paths):
    lines = [
        "=" * 70,
        "Store Sales V8 — Global Forecasting Pipeline — Experiment Report",
        "=" * 70,
        "",
        "Architecture: Darts global models, per product family",
        f"  LGBM lags: {_LAG_LIST}",
        f"  XGB  lags: {_LAG_LIST}",
        f"  Forecast window: {_PREDICTION_WINDOW} days",
        f"  Zero-fc guard window: {_ZERO_GATE_WINDOW} days",
        f"  Train-from threshold: {_TRUNCATE_BEFORE}",
        "",
        "Component decomposition:",
        "  lgbm_full  — LightGBM on complete history",
        "  lgbm_2015  — LightGBM on 2015 onwards",
        "  xgb_full   — XGBoost on complete history",
        "  xgb_2015   — XGBoost on 2015 onwards",
        "  Final  = avg(avg(lgbm_full, lgbm_2015), avg(xgb_full, xgb_2015))",
        "",
        "Past covariates: transactions (store-level daily)",
        "Future covariates: oil, onpromotion, calendar features + national holidays",
        f"XGBoost params: trees={_XGB_TREES}, lr={_XGB_LR}, depth={_XGB_DEPTH}, "
        f"subsample={_XGB_SUBSAMPLE}",
        "=" * 70,
    ]
    (report_dir / "experiment_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    trace(f"[REPORT] {report_dir / 'experiment_summary.txt'}")

    zip_path = out_dir / "experiment_report.zip"
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(str(report_dir)):
            for fn in files:
                full = os.path.join(root, fn)
                zf.write(full, os.path.relpath(full, str(report_dir)))
        for p in sub_paths:
            if os.path.exists(str(p)):
                zf.write(str(p), os.path.basename(str(p)))
    mb = os.path.getsize(str(zip_path)) / 1024**2
    trace(f"\n{'='*60}")
    trace(f"Report archive: {zip_path}  ({mb:.1f} MB)")
    trace(f"Download from the output panel.")
    trace(f"{'='*60}")


# ══════════════════════════════════════════════════════
# Main orchestration
# ══════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()
    out = resolve_output_path()
    src = locate_dataset()

    trace("=" * 72)
    trace("Store Sales — Darts Global Forecast Pipeline")
    trace(f"Output: {out}  |  Data: {src}")
    trace(f"OUTPUT_NAME={_OUTPUT_FILENAME}")
    trace(f"LGB_GPU={_FLAG_LGB_GPU}  XGB_GPU={_FLAG_XGB_GPU}  "
          f"RUN_VAL={_FLAG_RUN_VAL}  BLEND_OPT={_FLAG_BLEND_OPT}  "
          f"USE_OPT_W={_FLAG_USE_OPT_W}  SAVE_PARTS={_FLAG_SAVE_PARTS}")
    trace(f"ZERO_WINDOW={_ZERO_GATE_WINDOW}  LAGS={_LAG_LIST}  "
          f"BLEND_SPACE={_BLEND_SPACE}  MANUAL_W={_MANUAL_WEIGHTS_RAW or '(avg)'}")
    trace(f"XGB: trees={_XGB_TREES} lr={_XGB_LR} depth={_XGB_DEPTH} "
          f"subsample={_XGB_SUBSAMPLE} colsample={_XGB_COL_SAMPLE} "
          f"min_child={_XGB_MIN_CHILD or '-'} alpha={_XGB_ALPHA or '-'} lambda={_XGB_LAMBDA or '-'}")
    trace("=" * 72)

    # Phase 1 — Load & prepare
    tr, te, crude, stores_df, txn, evt = ingest_source_files(src)
    table, holiday_keys, tr_end, raw_te = assemble_modeling_table(tr, te, crude, stores_df, txn, evt)

    report = out / "report"
    _ensure_report_dirs(report)
    trace(f"Report output: {report}")
    _plot_safe(lambda: render_exploratory_charts(table, report), "EDA")

    # Phase 2 — Target sequences
    trace("\n[2/6] Extracting target sequences ...")
    static_feats = ["city", "state", "type", "cluster"]
    tgt_map, pipe_map, id_map = extract_target_sequences(table, tr_end, static_feats)
    _plot_safe(lambda: render_target_samples(tgt_map, report), "target samples")

    # Phase 3 — Covariates
    trace("\n[3/6] Extracting covariate sequences ...")
    past_feats = ["transactions"]
    future_feats = [
        "oil", "onpromotion", "day", "month", "year",
        "day_of_week", "day_of_year", "week_of_year", "date_index",
        "work_day", *holiday_keys,
    ]
    p_map, f_map = extract_covariate_sequences(
        table, tr_end,
        past_features=past_feats, future_features=future_feats,
        past_ma_features=None, future_ma_features=["oil", "onpromotion"],
    )
    _plot_safe(lambda: render_covariate_samples(f_map, report), "covariate samples")

    # Phase 4 — Engine init
    trace("\n[4/6] Initializing forecast engine ...")
    engine = GlobalForecastEngine(
        target_map=tgt_map, pipe_map=pipe_map, ident_map=id_map,
        past_map=p_map, future_map=f_map,
        horizon=_PREDICTION_WINDOW, n_folds=1, zero_window=_ZERO_GATE_WINDOW,
        keep_static="keep_all", keep_past="keep_all", keep_future="keep_all",
    )

    lgb_keys, lgb_cfgs = configure_lightgbm_ensemble(use_past=True, use_future=True)
    xgb_keys, xgb_cfgs = configure_xgboost_ensemble(use_past=True, use_future=True)

    opt_df = None; val_act = None; val_parts = None
    if _FLAG_BLEND_OPT:
        opt_df, val_act, val_parts = run_weight_search(
            engine, lgb_keys, lgb_cfgs, xgb_keys, xgb_cfgs, out,
        )
        _plot_safe(lambda: render_weight_landscape(opt_df, report), "weight landscape")
        if val_act is not None and val_parts is not None:
            _plot_safe(lambda: render_validation_diagnostics(val_act, val_parts, report), "validation diag")

    if _FLAG_RUN_VAL:
        trace("\n[optional] 16-day backtest (2015+) ...")
        trace("[validation] LightGBM:")
        engine.run_validation(model_keys=lgb_keys, configs=lgb_cfgs, truncate_before=_TRUNCATE_BEFORE)
        trace("[validation] XGBoost:")
        engine.run_validation(model_keys=xgb_keys, configs=xgb_cfgs, truncate_before=_TRUNCATE_BEFORE)

    # Phase 5 — Generate predictions
    trace("\n[5/6] Generating forecasts ...")
    lgb_full = engine.predict_test(model_keys=lgb_keys, configs=lgb_cfgs, truncate_before=None)
    lgb_2015 = engine.predict_test(model_keys=lgb_keys, configs=lgb_cfgs, truncate_before=_TRUNCATE_BEFORE)
    xgb_full = engine.predict_test(model_keys=xgb_keys, configs=xgb_cfgs, truncate_before=None)
    xgb_2015 = engine.predict_test(model_keys=xgb_keys, configs=xgb_cfgs, truncate_before=_TRUNCATE_BEFORE)

    # Phase 6 — Blend & save
    trace("\n[6/6] Ensemble blending ...")
    lgb_ens = blend_linear(lgb_full, lgb_2015)
    xgb_ens = blend_linear(xgb_full, xgb_2015)
    final_preds = blend_linear(lgb_ens, xgb_ens)

    comp_map = {
        "lgbm_full": lgb_full, "lgbm_2015": lgb_2015, "lgbm_final": lgb_ens,
        "xgb_full": xgb_full, "xgb_2015": xgb_2015, "xgb_final": xgb_ens,
        "v8_base": final_preds,
    }

    log_rows = []
    if _FLAG_SAVE_PARTS:
        for name, preds in comp_map.items():
            persist_component_submission(
                out_dir=out, raw_test=raw_te, label=f"submission_{name}.csv",
                preds=preds, log_rows=log_rows,
            )

    final_parts = {"lgb_full": lgb_full, "lgb_2015": lgb_2015,
                   "xgb_full": xgb_full, "xgb_2015": xgb_2015}

    manual_sub = None
    if _MANUAL_WEIGHTS_RAW:
        manual_preds = apply_manual_weights(final_parts, _MANUAL_WEIGHTS_RAW, _BLEND_SPACE)
        manual_sub = persist_component_submission(
            out_dir=out, raw_test=raw_te, label="submission_v8_manual_blend.csv",
            preds=manual_preds, log_rows=log_rows,
        )

    opt_sub = None
    if (manual_sub is None and _FLAG_BLEND_OPT and _FLAG_USE_OPT_W
            and opt_df is not None and not opt_df.empty):
        opt_preds = apply_weight_solution(final_parts, opt_df.iloc[0])
        opt_sub = persist_component_submission(
            out_dir=out, raw_test=raw_te, label="submission_v8_local_opt.csv",
            preds=opt_preds, log_rows=log_rows,
        )

    if manual_sub is not None:
        final_sub = manual_sub
    elif opt_sub is not None:
        final_sub = opt_sub
    else:
        final_sub = format_kaggle_output(raw_te, final_preds)

    main_path = out / _OUTPUT_FILENAME
    default_path = out / "submission.csv"
    final_sub.to_csv(main_path, index=False)
    final_sub.to_csv(default_path, index=False)

    log_rows.append({
        "file": _OUTPUT_FILENAME,
        "rows": len(final_sub),
        "zero_rows": int((final_sub["sales"] == 0).sum()),
        "sales_min": float(final_sub["sales"].min()),
        "sales_mean": float(final_sub["sales"].mean()),
        "sales_median": float(final_sub["sales"].median()),
        "sales_max": float(final_sub["sales"].max()),
        "elapsed_minutes": round((time.time() - t0) / 60, 2),
    })
    stats_df = pd.DataFrame(log_rows)
    stats_csv = out / f"{Path(_OUTPUT_FILENAME).stem}_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    trace(f"Saved: {main_path}  |  {default_path}  |  {stats_csv}")
    trace(stats_df.to_string(index=False))

    # === Report packaging ===
    _plot_safe(
        lambda: render_prediction_summary(
            {k: v for k, v in comp_map.items()}, final_preds, report
        ), "prediction summary"
    )
    sub_paths = [main_path, default_path]
    if _FLAG_SAVE_PARTS:
        for name in comp_map:
            p = out / f"submission_{name}.csv"
            if p.exists():
                sub_paths.append(p)
    if manual_sub is not None:
        sub_paths.append(out / "submission_v8_manual_blend.csv")
    if opt_sub is not None:
        sub_paths.append(out / "submission_v8_local_opt.csv")
    _plot_safe(lambda: bundle_output_archive(report, out, sub_paths), "archive")

    trace("Done.")


if __name__ == "__main__":
    main()
