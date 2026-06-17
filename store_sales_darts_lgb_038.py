"""
Store Sales Darts + LightGBM 0.38 repro script.

Based on Chong Zhen Jie's public Kaggle notebook:
https://www.kaggle.com/code/chongzhenjie/ecuador-store-sales-global-forecasting-lightgbm

This script keeps the high-score notebook's core recipe and removes EDA/plots:
- 33 family-level global LightGBM models through Darts.
- 54 store series per family with static store covariates.
- transactions as past covariate.
- oil, onpromotion, date and selected holiday features as future covariates.
- 4 LightGBM members with different target lags: 63, 7, 365, 730.
- Average predictions from full-history training and 2015+ training.

Outputs:
- submission_darts_lgb_038.csv
- submission.csv
- submission_darts_lgb_038_stats.csv
"""

from __future__ import annotations

import gc
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm

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
    from darts.models import LightGBMModel
    from darts.models.filtering.moving_average_filter import MovingAverageFilter
except Exception as exc:  # pragma: no cover - runtime environment guidance
    raise SystemExit(
        "Missing Darts dependencies. Install on AutoDL/Kaggle first, for example:\n"
        "  pip install -U darts lightgbm\n"
        "If your mirror/environment still uses Unit8 extras, try:\n"
        '  pip install -U "u8darts[notorch]" lightgbm\n'
        f"\nOriginal import error: {exc}"
    )


COMPETITION = "store-sales-time-series-forecasting"
DARTS_RECIPE = os.environ.get("DARTS_RECIPE", "chong").strip().lower()
OUTPUT_NAME = os.environ.get(
    "DARTS_OUTPUT_NAME",
    "submission_darts_lgb_038.csv"
    if DARTS_RECIPE == "chong"
    else f"submission_darts_lgb_{DARTS_RECIPE}_038.csv",
)
RANDOM_STATE = 0
FORECAST_HORIZON = 16
ZERO_FC_WINDOW = 21
DROP_BEFORE = "2015-01-01"

# Keep default close to the public notebook. If your LightGBM build supports GPU,
# run with USE_LGB_GPU=1. Many pip LightGBM wheels are CPU-only, so CPU is default.
USE_LGB_GPU = os.environ.get("USE_LGB_GPU", "0").strip().lower() in {"1", "true", "yes"}
RUN_VALIDATION = os.environ.get("RUN_VALIDATION", "0").strip().lower() in {"1", "true", "yes"}


def log(msg: str) -> None:
    print(msg, flush=True)


def find_data_dir() -> Path:
    env_path = os.environ.get("STORE_SALES_DATA") or os.environ.get("DATA_DIR")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path("/kaggle/input/store-sales-time-series-forecasting"),
            Path("/kaggle/input/store-sales"),
            Path("/root/datasets"),
            Path("/root/autodl-tmp/store-sales-time-series-forecasting"),
            Path("/root/autodl-tmp/store-sales"),
            Path.cwd(),
        ]
    )

    required = {"train.csv", "test.csv", "stores.csv", "oil.csv", "transactions.csv", "holidays_events.csv"}
    for path in candidates:
        if path.exists() and required.issubset({p.name for p in path.iterdir() if p.is_file()}):
            return path

    for root in [Path("/kaggle/input"), Path("/root/autodl-tmp"), Path("/root")]:
        if not root.exists():
            continue
        for path, _, files in os.walk(root):
            if required.issubset(set(files)):
                return Path(path)

    raise FileNotFoundError(
        "Could not locate Store Sales data. Set STORE_SALES_DATA to the folder "
        "containing train.csv/test.csv/stores.csv/oil.csv/transactions.csv/holidays_events.csv."
    )


def find_output_dir() -> Path:
    env_path = os.environ.get("STORE_SALES_OUT") or os.environ.get("OUTPUT_DIR")
    if env_path:
        out = Path(env_path)
    elif Path("/kaggle/working").exists():
        out = Path("/kaggle/working")
    elif Path("/root").exists():
        out = Path("/root/output")
    else:
        out = Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    return out


def assert_unique_submission(submission: pd.DataFrame, stage: str) -> None:
    if submission["id"].duplicated().any():
        dup = submission.loc[submission["id"].duplicated(keep=False)].head(20)
        raise AssertionError(f"{stage}: duplicated submission ids\n{dup}")


def load_raw_data(data_dir: Path):
    log(f"Data path: {data_dir}")
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["date"])
    test = pd.read_csv(data_dir / "test.csv", parse_dates=["date"])
    oil = pd.read_csv(data_dir / "oil.csv", parse_dates=["date"]).rename(
        columns={"dcoilwtico": "oil"}
    )
    store = pd.read_csv(data_dir / "stores.csv")
    transaction = pd.read_csv(data_dir / "transactions.csv", parse_dates=["date"])
    holiday = pd.read_csv(data_dir / "holidays_events.csv", parse_dates=["date"])
    log(f"train={train.shape}, test={test.shape}, stores={store.shape}")
    return train, test, oil, store, transaction, holiday


def process_holiday_description(s: str, store: pd.DataFrame) -> str:
    if "futbol" in s:
        return "futbol"
    to_remove = list(set(store.city.str.lower()) | set(store.state.str.lower()))
    for word in to_remove:
        s = s.replace(word, "")
    return s


def prepare_data(train, test, oil, store, transaction, holiday):
    log("\n[1/6] Preparing calendar, oil, transactions and holidays...")

    train_start = train.date.min().date()
    train_end = train.date.max().date()
    test_end = test.date.max().date()

    missing_dates = pd.date_range(train_start, train_end).difference(train.date.unique())
    missing_date_list = missing_dates.strftime("%Y-%m-%d").tolist()

    multi_idx = pd.MultiIndex.from_product(
        [pd.date_range(train_start, train_end), train.store_nbr.unique(), train.family.unique()],
        names=["date", "store_nbr", "family"],
    )
    train = train.set_index(["date", "store_nbr", "family"]).reindex(multi_idx).reset_index()
    train[["sales", "onpromotion"]] = train[["sales", "onpromotion"]].fillna(0.0)
    train["id"] = train["id"].interpolate(method="linear")

    oil = oil.merge(
        pd.DataFrame({"date": pd.date_range(train_start, test_end)}),
        on="date",
        how="outer",
    ).sort_values("date", ignore_index=True)
    oil["oil"] = oil["oil"].interpolate(method="linear", limit_direction="both")

    store_sales = train.groupby(["date", "store_nbr"], observed=True)["sales"].sum().reset_index()
    transaction = transaction.merge(
        store_sales,
        on=["date", "store_nbr"],
        how="outer",
    ).sort_values(["date", "store_nbr"], ignore_index=True)
    transaction.loc[transaction["sales"].eq(0), "transactions"] = 0.0
    transaction = transaction.drop(columns=["sales"])
    transaction["transactions"] = transaction.groupby("store_nbr", group_keys=False)[
        "transactions"
    ].apply(lambda x: x.interpolate(method="linear", limit_direction="both"))

    holiday = holiday.copy()
    holiday["description"] = holiday.apply(
        lambda x: str(x.description).lower().replace(str(x.locale_name).lower(), ""),
        axis=1,
    ).apply(lambda s: process_holiday_description(s, store)).replace(
        r"[+-]\d+|\b(de|del|traslado|recupero|puente|-)\b", "", regex=True
    ).replace(
        r"\s+|-", " ", regex=True
    ).str.strip()

    holiday = holiday[holiday.transferred.eq(False)].copy()

    work_days = holiday[holiday.type.eq("Work Day")]
    work_days = work_days[["date", "type"]].rename(columns={"type": "work_day"})
    work_days = work_days.drop_duplicates("date", keep="first").reset_index(drop=True)
    work_days["work_day"] = work_days["work_day"].notna().astype(int)
    holiday = holiday[holiday.type != "Work Day"].reset_index(drop=True)

    national_holidays = holiday[holiday.locale.eq("National")]
    national_holidays = national_holidays[["date", "description"]].drop_duplicates()
    national_holidays = pd.get_dummies(national_holidays, columns=["description"], prefix="nat")
    national_holidays = national_holidays.groupby("date", as_index=False).sum()
    national_holidays = national_holidays.rename(
        columns={"nat_primer grito independencia": "nat_primer grito"}
    )

    selected_holidays = [
        "nat_terremoto",
        "nat_navidad",
        "nat_dia la madre",
        "nat_dia trabajo",
        "nat_primer dia ano",
        "nat_futbol",
        "nat_dia difuntos",
    ]
    for col in selected_holidays:
        if col not in national_holidays.columns:
            national_holidays[col] = 0
    keep_national_holidays = national_holidays[["date", *selected_holidays]]

    data = pd.concat([train, test], axis=0, ignore_index=True)
    data = data.merge(transaction, on=["date", "store_nbr"], how="left")
    data = data.merge(oil, on="date", how="left")
    data = data.merge(store, on="store_nbr", how="left")
    data = data.merge(work_days, on="date", how="left")
    data = data.merge(keep_national_holidays, on="date", how="left")
    data = data.sort_values(["date", "store_nbr", "family"], ignore_index=True)

    data[["work_day", *selected_holidays]] = data[["work_day", *selected_holidays]].fillna(0)
    data["transactions"] = data["transactions"].fillna(0.0)

    data["day"] = data.date.dt.day
    data["month"] = data.date.dt.month
    data["year"] = data.date.dt.year
    data["day_of_week"] = data.date.dt.dayofweek
    data["day_of_year"] = data.date.dt.dayofyear
    data["week_of_year"] = data.date.dt.isocalendar().week.astype(int)
    data["date_index"] = data.date.factorize()[0]

    zero_sales_dates = pd.to_datetime(
        missing_date_list + [f"{year}-01-01" for year in range(2013, 2018)]
    )
    zero_mask = (
        data.date.isin(zero_sales_dates)
        & data.sales.eq(0)
        & data.onpromotion.eq(0)
    )
    data.loc[zero_mask, ["sales", "onpromotion"]] = np.nan

    data["store_nbr"] = data["store_nbr"].apply(lambda x: f"store_nbr_{x}")
    data["cluster"] = data["cluster"].apply(lambda x: f"cluster_{x}")
    data["type"] = data["type"].apply(lambda x: f"type_{x}")
    data["city"] = data["city"].apply(lambda x: f"city_{str(x).lower()}")
    data["state"] = data["state"].apply(lambda x: f"state_{str(x).lower()}")

    log(f"prepared data={data.shape}, families={data.family.nunique()}")
    return data, selected_holidays, pd.Timestamp(train_end), test.copy()


def get_pipeline(static_covs_transform=False, log_transform=False):
    steps = [MissingValuesFiller(n_jobs=-1)]

    if static_covs_transform:
        steps.append(
            StaticCovariatesTransformer(
                transformer_cat=OneHotEncoder(handle_unknown="ignore"),
                n_jobs=-1,
            )
        )

    if log_transform:
        steps.append(InvertibleMapper(fn=np.log1p, inverse_fn=np.expm1, n_jobs=-1))

    steps.append(Scaler())
    return Pipeline(steps)


def get_target_series(data: pd.DataFrame, train_end: pd.Timestamp, static_cols):
    target_dict = {}
    pipe_dict = {}
    id_dict = {}

    for fam in tqdm(data.family.unique(), desc="Extracting target series"):
        df = data[(data.family.eq(fam)) & (data.date.le(train_end))]
        pipe = get_pipeline(static_covs_transform=True, log_transform=True)

        target = TimeSeries.from_group_dataframe(
            df=df,
            time_col="date",
            value_cols="sales",
            group_cols="store_nbr",
            static_cols=static_cols,
        )

        target_id = [
            {"store_nbr": t.static_covariates["store_nbr"].iloc[0], "family": fam}
            for t in target
        ]
        target = pipe.fit_transform(target)

        target_dict[fam] = [t.astype(np.float32) for t in target]
        pipe_dict[fam] = pipe[2:]
        id_dict[fam] = target_id

    return target_dict, pipe_dict, id_dict


def get_covariates(
    data: pd.DataFrame,
    train_end: pd.Timestamp,
    past_cols,
    future_cols,
    past_ma_cols=None,
    future_ma_cols=None,
    past_window_sizes=(7, 28),
    future_window_sizes=(7, 28),
):
    past_dict = {}
    future_dict = {}
    covs_pipe = get_pipeline()

    for fam in tqdm(data.family.unique(), desc="Extracting covariates"):
        df = data[data.family.eq(fam)]

        past_covs = TimeSeries.from_group_dataframe(
            df=df[df.date.le(train_end)],
            time_col="date",
            value_cols=past_cols,
            group_cols="store_nbr",
        )
        past_covs = [p.with_static_covariates(None) for p in past_covs]
        past_covs = covs_pipe.fit_transform(past_covs)

        if past_ma_cols is not None:
            for size in past_window_sizes:
                ma_filter = MovingAverageFilter(window=size)
                old_names = [f"rolling_mean_{size}_{col}" for col in past_ma_cols]
                new_names = [f"{col}_ma{size}" for col in past_ma_cols]
                past_ma_covs = [
                    ma_filter.filter(p[past_ma_cols]).with_columns_renamed(old_names, new_names)
                    for p in past_covs
                ]
                past_covs = [p.stack(p_ma) for p, p_ma in zip(past_covs, past_ma_covs)]

        future_covs = TimeSeries.from_group_dataframe(
            df=df,
            time_col="date",
            value_cols=future_cols,
            group_cols="store_nbr",
        )
        future_covs = [f.with_static_covariates(None) for f in future_covs]
        future_covs = covs_pipe.fit_transform(future_covs)

        if future_ma_cols is not None:
            for size in future_window_sizes:
                ma_filter = MovingAverageFilter(window=size)
                old_names = [f"rolling_mean_{size}_{col}" for col in future_ma_cols]
                new_names = [f"{col}_ma{size}" for col in future_ma_cols]
                future_ma_covs = [
                    ma_filter.filter(f[future_ma_cols]).with_columns_renamed(old_names, new_names)
                    for f in future_covs
                ]
                future_covs = [
                    f.stack(f_ma) for f, f_ma in zip(future_covs, future_ma_covs)
                ]

        past_dict[fam] = [p.astype(np.float32) for p in past_covs]
        future_dict[fam] = [f.astype(np.float32) for f in future_covs]

    return past_dict, future_dict


class Trainer:
    def __init__(
        self,
        target_dict,
        pipe_dict,
        id_dict,
        past_dict,
        future_dict,
        forecast_horizon,
        folds,
        zero_fc_window,
        static_covs=None,
        past_covs=None,
        future_covs=None,
    ):
        self.target_dict = target_dict.copy()
        self.pipe_dict = pipe_dict.copy()
        self.id_dict = id_dict.copy()
        self.past_dict = past_dict.copy()
        self.future_dict = future_dict.copy()
        self.forecast_horizon = forecast_horizon
        self.folds = folds
        self.zero_fc_window = zero_fc_window
        self.static_covs = static_covs
        self.past_covs = past_covs
        self.future_covs = future_covs
        self.setup()

    def setup(self):
        for fam in tqdm(self.target_dict.keys(), desc="Setting up"):
            if self.static_covs != "keep_all":
                if self.static_covs is not None:
                    target = self.target_dict[fam]
                    keep_static = [
                        col
                        for col in target[0].static_covariates.columns
                        if col.startswith(tuple(self.static_covs))
                    ]
                    static_covs_df = [t.static_covariates[keep_static] for t in target]
                    self.target_dict[fam] = [
                        t.with_static_covariates(d) for t, d in zip(target, static_covs_df)
                    ]
                else:
                    self.target_dict[fam] = [
                        t.with_static_covariates(None) for t in self.target_dict[fam]
                    ]

            if self.past_covs != "keep_all":
                self.past_dict[fam] = (
                    [p[self.past_covs] for p in self.past_dict[fam]]
                    if self.past_covs is not None
                    else None
                )

            if self.future_covs != "keep_all":
                self.future_dict[fam] = (
                    [p[self.future_covs] for p in self.future_dict[fam]]
                    if self.future_covs is not None
                    else None
                )

    @staticmethod
    def clip(array):
        return np.clip(array, a_min=0.0, a_max=None)

    def train_valid_split(self, target, length):
        train = [t[:-length] for t in target]
        valid_end_idx = -length + self.forecast_horizon
        if valid_end_idx >= 0:
            valid_end_idx = None
        valid = [t[-length:valid_end_idx] for t in target]
        return train, valid

    def get_models(self, model_names, model_configs):
        models = {
            "lgbm": LightGBMModel,
        }
        if len(model_names) != len(model_configs):
            raise ValueError("model_names and model_configs length mismatch")

        configs = []
        for name, config in zip(model_names, model_configs):
            cfg = dict(config)
            if name == "lgbm":
                cfg.setdefault("n_jobs", -1)
                if USE_LGB_GPU:
                    cfg.setdefault("device", "gpu")
            if name == "xgb":
                cfg = {"tree_method": "hist", **cfg}
            configs.append(cfg)

        return [models[name](**configs[j]) for j, name in enumerate(model_names)]

    def generate_forecasts(self, models, train, pipe, past_covs, future_covs, drop_before):
        if drop_before is not None:
            date = pd.Timestamp(drop_before) - pd.Timedelta(days=1)
            train = [t.drop_before(date) for t in train]

        inputs = {
            "series": train,
            "past_covariates": past_covs,
            "future_covariates": future_covs,
        }
        zero_pred = TimeSeries.from_dataframe(
            df=pd.DataFrame(
                {
                    "date": pd.date_range(train[0].end_time(), periods=self.forecast_horizon + 1)[1:],
                    "sales": np.zeros(self.forecast_horizon),
                }
            ),
            time_col="date",
            value_cols="sales",
        )

        pred_list = []
        ens_pred = [0 for _ in range(len(train))]

        for idx, model in enumerate(models, start=1):
            log(f"    fitting member {idx}/{len(models)}: {model.__class__.__name__}")
            model.fit(**inputs)
            pred = model.predict(n=self.forecast_horizon, **inputs)
            pred = pipe.inverse_transform(pred)

            for j in range(len(train)):
                if train[j][-self.zero_fc_window :].values().sum() == 0:
                    pred[j] = zero_pred

            pred = [p.map(self.clip) for p in pred]
            pred_list.append(pred)
            for j in range(len(ens_pred)):
                ens_pred[j] += pred[j] / len(models)

            gc.collect()

        return pred_list, ens_pred

    @staticmethod
    def metric(valid, pred):
        return rmsle(valid, pred, inter_reduction=np.mean)

    @staticmethod
    def series_to_dataframe(series):
        if hasattr(series, "pd_dataframe"):
            return series.pd_dataframe()
        return series.to_dataframe()

    def validate(self, model_names, model_configs, drop_before=None):
        model_metrics_history = []
        ens_metric_history = []

        for fam in tqdm(self.target_dict, desc="Validation"):
            target = self.target_dict[fam]
            pipe = self.pipe_dict[fam]
            past_covs = self.past_dict[fam]
            future_covs = self.future_dict[fam]

            model_metrics = []
            ens_metric = 0.0
            for fold in range(self.folds):
                length = (self.folds - fold) * self.forecast_horizon
                train, valid = self.train_valid_split(target, length)
                valid = pipe.inverse_transform(valid)

                models = self.get_models(model_names, model_configs)
                pred_list, ens_pred = self.generate_forecasts(
                    models, train, pipe, past_covs, future_covs, drop_before
                )
                metric_list = [self.metric(valid, pred) / self.folds for pred in pred_list]
                model_metrics.append(metric_list)
                if len(models) > 1:
                    ens_metric += self.metric(valid, ens_pred) / self.folds

            model_metrics = np.sum(model_metrics, axis=0)
            model_metrics_history.append(model_metrics)
            ens_metric_history.append(ens_metric)
            log(
                f"{fam}: "
                + " - ".join(
                    f"{model}: {metric:.5f}"
                    for model, metric in zip(model_names, model_metrics)
                )
                + (f" - ens: {ens_metric:.5f}" if len(model_names) > 1 else "")
            )

        log(
            "Average RMSLE | "
            + " - ".join(
                f"{model}: {metric:.5f}"
                for model, metric in zip(model_names, np.mean(model_metrics_history, axis=0))
            )
            + (
                f" - ens: {np.mean(ens_metric_history):.5f}"
                if len(model_names) > 1
                else ""
            )
        )

    def ensemble_predict(self, model_names, model_configs, drop_before=None):
        forecasts = []

        for fam in tqdm(self.target_dict.keys(), desc="Generating forecasts"):
            log(f"  family={fam}, drop_before={drop_before}")
            target = self.target_dict[fam]
            pipe = self.pipe_dict[fam]
            target_id = self.id_dict[fam]
            past_covs = self.past_dict[fam]
            future_covs = self.future_dict[fam]

            models = self.get_models(model_names, model_configs)
            _, ens_pred = self.generate_forecasts(
                models, target, pipe, past_covs, future_covs, drop_before
            )
            ens_pred = [
                self.series_to_dataframe(p).assign(**i)
                for p, i in zip(ens_pred, target_id)
            ]
            forecasts.append(pd.concat(ens_pred, axis=0))
            gc.collect()

        forecasts = pd.concat(forecasts, axis=0)
        return forecasts.rename_axis(None, axis=1).reset_index(names="date")


def build_configs(recipe, past_covs_enabled=True, future_covs_enabled=True):
    if recipe == "chong":
        base_lags = 63
        past_lags = list(range(-16, -23, -1))
        extra_params = {}
    elif recipe == "dang":
        base_lags = 56
        past_lags = list(range(-17, -24, -1))
        extra_params = {
            "n_estimators": 100,
            "learning_rate": 0.065,
            "max_depth": 6,
        }
    else:
        raise ValueError("DARTS_RECIPE must be one of: chong, dang")

    base_config = {
        "random_state": RANDOM_STATE,
        "lags": base_lags,
        "lags_past_covariates": past_lags if past_covs_enabled else None,
        "lags_future_covariates": (14, 1) if future_covs_enabled else None,
        "output_chunk_length": 1,
    }

    gbdt_config1 = {**base_config, **extra_params}
    gbdt_config2 = dict(gbdt_config1, lags=7)
    gbdt_config3 = dict(gbdt_config1, lags=365)
    gbdt_config4 = dict(gbdt_config1, lags=730)

    model_names = ["lgbm", "lgbm", "lgbm", "lgbm"]
    model_configs = [gbdt_config1, gbdt_config2, gbdt_config3, gbdt_config4]
    return model_names, model_configs


def prepare_submission(test: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["store_nbr"] = predictions["store_nbr"].replace(
        "store_nbr_", "", regex=True
    ).astype(int)
    submission = test.merge(
        predictions,
        on=["date", "store_nbr", "family"],
        how="left",
    )[["id", "sales"]]
    submission["sales"] = submission["sales"].fillna(0).clip(lower=0)
    submission["id"] = submission["id"].astype(int)
    assert_unique_submission(submission, "final submission")
    return submission


def main() -> None:
    start = time.time()
    out_dir = find_output_dir()
    data_dir = find_data_dir()

    log("=" * 72)
    log("Store Sales Darts + LightGBM 0.38 repro")
    log(f"Output path: {out_dir}")
    log(f"DARTS_RECIPE={DARTS_RECIPE}, OUTPUT_NAME={OUTPUT_NAME}")
    log(f"USE_LGB_GPU={USE_LGB_GPU}, RUN_VALIDATION={RUN_VALIDATION}")
    log("=" * 72)

    train, test, oil, store, transaction, holiday = load_raw_data(data_dir)
    data, selected_holidays, train_end, raw_test = prepare_data(
        train, test, oil, store, transaction, holiday
    )

    log("\n[2/6] Building target series...")
    static_cols = ["city", "state", "type", "cluster"]
    target_dict, pipe_dict, id_dict = get_target_series(data, train_end, static_cols)

    log("\n[3/6] Building covariates...")
    past_cols = ["transactions"]
    future_cols = [
        "oil",
        "onpromotion",
        "day",
        "month",
        "year",
        "day_of_week",
        "day_of_year",
        "week_of_year",
        "date_index",
        "work_day",
        *selected_holidays,
    ]
    past_dict, future_dict = get_covariates(
        data,
        train_end,
        past_cols=past_cols,
        future_cols=future_cols,
        past_ma_cols=None,
        future_ma_cols=["oil", "onpromotion"],
    )

    log("\n[4/6] Initializing trainer...")
    trainer = Trainer(
        target_dict=target_dict,
        pipe_dict=pipe_dict,
        id_dict=id_dict,
        past_dict=past_dict,
        future_dict=future_dict,
        forecast_horizon=FORECAST_HORIZON,
        folds=1,
        zero_fc_window=ZERO_FC_WINDOW,
        static_covs="keep_all",
        past_covs="keep_all",
        future_covs="keep_all",
    )

    model_names, model_configs = build_configs(
        recipe=DARTS_RECIPE,
        past_covs_enabled=True,
        future_covs_enabled=True,
    )

    if RUN_VALIDATION:
        log("\n[optional] Validation on last 16 days, drop_before=2015-01-01...")
        trainer.validate(model_names=model_names, model_configs=model_configs, drop_before=DROP_BEFORE)

    log("\n[5/6] Generating full-history prediction...")
    predictions_full = trainer.ensemble_predict(
        model_names=model_names,
        model_configs=model_configs,
        drop_before=None,
    )

    log("\n[5/6] Generating 2015+ prediction...")
    predictions_2015 = trainer.ensemble_predict(
        model_names=model_names,
        model_configs=model_configs,
        drop_before=DROP_BEFORE,
    )

    log("\n[6/6] Averaging predictions and saving submission...")
    final_predictions = predictions_full.merge(
        predictions_2015,
        on=["date", "store_nbr", "family"],
        how="left",
    )
    final_predictions["sales"] = final_predictions[["sales_x", "sales_y"]].mean(axis=1)
    final_predictions = final_predictions.drop(columns=["sales_x", "sales_y"])

    submission = prepare_submission(raw_test, final_predictions)

    out_main = out_dir / OUTPUT_NAME
    out_default = out_dir / "submission.csv"
    submission.to_csv(out_main, index=False)
    submission.to_csv(out_default, index=False)

    stats = pd.DataFrame(
        [
            {
                "rows": len(submission),
                "zero_rows": int((submission["sales"] == 0).sum()),
                "sales_min": float(submission["sales"].min()),
                "sales_mean": float(submission["sales"].mean()),
                "sales_max": float(submission["sales"].max()),
                "elapsed_minutes": round((time.time() - start) / 60, 2),
            }
        ]
    )
    stats_path = out_dir / f"{Path(OUTPUT_NAME).stem}_stats.csv"
    stats.to_csv(stats_path, index=False)

    log(f"Saved: {out_main}")
    log(f"Saved: {out_default}")
    log(f"Saved: {stats_path}")
    log(stats.to_string(index=False))
    log("Done.")


if __name__ == "__main__":
    main()
