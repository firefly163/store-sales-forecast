"""
Reproduce Nina's public h-blend for Store Sales.

Source notebook:
https://www.kaggle.com/code/nina2025/store-sales-time-series-forecasting-h-blend

Expected input files:
- 0.37982.csv
- 0.37984.csv
- 0.38006.csv
- 0.38040.csv

The notebook's final submission uses Version 2/3/5:
- files: 0.37982, 0.38006, 0.38040
- base weights: 0.37, 0.33, 0.30
- rank correction weights: +0.10, -0.03, -0.07
- asc/desc mix: 0.30 / 0.70

Outputs:
- submission_hblend_037946.csv, recommended, public table score 0.37946
- submission_hblend_v1_037950.csv, public table score 0.37950
- submission_hblend_v2_037946.csv, public table score 0.37946
- submission_hblend_v3_037967.csv, public table score 0.37967
- submission_hblend_v4_037948.csv, public table score 0.37948
- submission_hblend_v5_experimental.csv, notebook final cell, no known public score
- submission.csv, same as recommended v2
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_SLUG = "nina2025/2025-12-22-store-sales-time-series-forecasting"
INPUT_DIR_NAME = "2025-12-22-store-sales-time-series-forecasting"


def log(msg: str) -> None:
    print(msg, flush=True)


def find_input_dir() -> Path:
    env_path = os.environ.get("HBLEND_DATA") or os.environ.get("STORE_SALES_HBLEND_DATA")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path("/kaggle/input") / INPUT_DIR_NAME,
            Path("/root/autodl-tmp") / INPUT_DIR_NAME,
            Path("/root/datasets") / INPUT_DIR_NAME,
            Path.cwd() / INPUT_DIR_NAME,
            Path.cwd() / "_kaggle_research" / "hblend_inputs",
            Path.cwd(),
        ]
    )
    required = {"0.37982.csv", "0.37984.csv", "0.38006.csv", "0.38040.csv"}
    for path in candidates:
        if path.exists() and required.issubset({p.name for p in path.iterdir() if p.is_file()}):
            return path
    raise FileNotFoundError(
        "Could not find h-blend input CSVs. Download them with:\n"
        f"  kaggle datasets download -d {DATASET_SLUG} -p /root/autodl-tmp/hblend_inputs --unzip\n"
        "Then run with:\n"
        "  HBLEND_DATA=/root/autodl-tmp/hblend_inputs python -u store_sales_h_blend_03795.py"
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


def read_submission(input_dir: Path, name: str) -> pd.DataFrame:
    path = input_dir / f"{name}.csv"
    df = pd.read_csv(path)
    if "id" not in df.columns or "sales" not in df.columns:
        raise ValueError(f"{path} must contain id,sales columns")
    return df[["id", "sales"]].rename(columns={"sales": name})


def merge_submissions(input_dir: Path, names: list[str]) -> pd.DataFrame:
    merged = read_submission(input_dir, names[0])
    for name in names[1:]:
        merged = merged.merge(read_submission(input_dir, name), on="id", how="inner")
    if len(merged) != 28512:
        log(f"[WARN] merged rows={len(merged)}, expected 28512")
    if merged["id"].duplicated().any():
        raise AssertionError("duplicated ids in merged submissions")
    return merged


def directional_blend(
    df: pd.DataFrame,
    names: list[str],
    base_weights: list[float],
    correction_weights: list[float],
    direction: str,
) -> np.ndarray:
    values = df[names].to_numpy(dtype=np.float64)
    base = np.asarray(base_weights, dtype=np.float64)
    corr = np.asarray(correction_weights, dtype=np.float64)

    if len(base) != len(names) or len(corr) != len(names):
        raise ValueError("weights length must match names length")
    if abs(base.sum() - 1.0) > 1e-9:
        raise ValueError(f"base weights should sum to 1, got {base.sum()}")
    if abs(corr.sum()) > 1e-9:
        raise ValueError(f"correction weights should sum to 0, got {corr.sum()}")

    order = np.argsort(values, axis=1)
    if direction == "desc":
        order = order[:, ::-1]
    elif direction != "asc":
        raise ValueError("direction must be asc or desc")

    rank_pos = np.empty_like(order)
    rows = np.arange(values.shape[0])[:, None]
    rank_pos[rows, order] = np.arange(values.shape[1])

    weights = base[None, :] + corr[rank_pos]
    return np.sum(values * weights, axis=1)


def h_blend(
    input_dir: Path,
    names: list[str],
    base_weights: list[float],
    correction_weights: list[float],
    asc_weight: float,
    desc_weight: float,
) -> pd.DataFrame:
    df = merge_submissions(input_dir, names)
    asc = directional_blend(df, names, base_weights, correction_weights, "asc")
    desc = directional_blend(df, names, base_weights, correction_weights, "desc")
    sales = asc_weight * asc + desc_weight * desc
    sales = np.clip(sales, 0, None)
    return pd.DataFrame({"id": df["id"].astype(int), "sales": sales})


def save_submission(df: pd.DataFrame, out_dir: Path, filename: str) -> Path:
    path = out_dir / filename
    if df["id"].duplicated().any():
        raise AssertionError(f"duplicated ids before saving {filename}")
    df.to_csv(path, index=False)
    log(
        f"Saved {path} | rows={len(df):,}, mean={df.sales.mean():.4f}, "
        f"zero={(df.sales == 0).sum():,}"
    )
    return path


def main() -> None:
    input_dir = find_input_dir()
    out_dir = find_output_dir()
    log(f"h-blend input: {input_dir}")
    log(f"output path: {out_dir}")

    # Table v1: public score 0.37950.
    blend_v1 = h_blend(
        input_dir=input_dir,
        names=["0.37982", "0.37984", "0.38006", "0.38040"],
        base_weights=[0.25, 0.25, 0.25, 0.25],
        correction_weights=[0.11, -0.01, -0.03, -0.07],
        asc_weight=0.30,
        desc_weight=0.70,
    )

    # Table v2: public score 0.37946. This is the recommended first submit.
    blend_v2 = h_blend(
        input_dir=input_dir,
        names=["0.37982", "0.38006", "0.38040"],
        base_weights=[0.334, 0.333, 0.333],
        correction_weights=[0.10, -0.03, -0.07],
        asc_weight=0.30,
        desc_weight=0.70,
    )

    # Table v3: public score 0.37967.
    blend_v3 = h_blend(
        input_dir=input_dir,
        names=["0.37982", "0.38006", "0.38040"],
        base_weights=[0.70, 0.20, 0.10],
        correction_weights=[0.10, -0.03, -0.07],
        asc_weight=0.30,
        desc_weight=0.70,
    )

    # Table v4: public score 0.37948.
    blend_v4 = h_blend(
        input_dir=input_dir,
        names=["0.37982", "0.37984", "0.38006", "0.38040"],
        base_weights=[0.30, 0.10, 0.30, 0.30],
        correction_weights=[0.14, -0.01, -0.05, -0.08],
        asc_weight=0.30,
        desc_weight=0.70,
    )

    # Notebook final cell v5: no public score shown in the table.
    blend_v5 = h_blend(
        input_dir=input_dir,
        names=["0.37982", "0.38006", "0.38040"],
        base_weights=[0.37, 0.33, 0.30],
        correction_weights=[0.10, -0.03, -0.07],
        asc_weight=0.30,
        desc_weight=0.70,
    )

    save_submission(blend_v1, out_dir, "submission_hblend_v1_037950.csv")
    save_submission(blend_v2, out_dir, "submission_hblend_v2_037946.csv")
    save_submission(blend_v3, out_dir, "submission_hblend_v3_037967.csv")
    save_submission(blend_v4, out_dir, "submission_hblend_v4_037948.csv")
    save_submission(blend_v5, out_dir, "submission_hblend_v5_experimental.csv")
    save_submission(blend_v2, out_dir, "submission_hblend_037946.csv")
    save_submission(blend_v2, out_dir, "submission.csv")
    log("Final recommended file: submission_hblend_037946.csv")


if __name__ == "__main__":
    main()
