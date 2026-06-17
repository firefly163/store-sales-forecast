from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def first_existing(candidates: list[str]) -> Path | None:
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return path
    return None


def read_submission(name: str, candidates: list[str]) -> pd.DataFrame:
    path = first_existing(candidates)
    if path is None:
        raise FileNotFoundError(
            f"Missing {name}. Checked:\n" + "\n".join(f"  - {c}" for c in candidates)
        )
    df = pd.read_csv(path)
    if list(df.columns) != ["id", "sales"]:
        raise ValueError(f"{name} at {path} must have columns: id,sales")
    if df["id"].duplicated().any():
        raise ValueError(f"{name} at {path} has duplicated id values")
    print(f"[load] {name:<10} {path} rows={len(df)} mean={df.sales.mean():.6f}")
    return df


def assert_aligned(anchor: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    if len(anchor) != len(other) or not anchor["id"].equals(other["id"]):
        raise ValueError(f"{name} id order does not match xiewenwei anchor")


def linear_blend(anchor: pd.DataFrame, parts: dict[str, pd.Series], weights: dict[str, float]) -> pd.DataFrame:
    total = sum(weights.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"weights must sum to 1.0, got {total}")
    out = anchor[["id"]].copy()
    sales = np.zeros(len(anchor), dtype=np.float64)
    for name, weight in weights.items():
        sales += weight * parts[name].to_numpy(dtype=np.float64)
    out["sales"] = np.maximum(sales, 0.0)
    return out


def log_blend(anchor: pd.DataFrame, parts: dict[str, pd.Series], weights: dict[str, float]) -> pd.DataFrame:
    total = sum(weights.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"weights must sum to 1.0, got {total}")
    out = anchor[["id"]].copy()
    log_sales = np.zeros(len(anchor), dtype=np.float64)
    for name, weight in weights.items():
        values = np.maximum(parts[name].to_numpy(dtype=np.float64), 0.0)
        log_sales += weight * np.log1p(values)
    out["sales"] = np.maximum(np.expm1(log_sales), 0.0)
    return out


def save(out_dir: Path, name: str, df: pd.DataFrame, rows: list[dict[str, object]]) -> None:
    path = out_dir / name
    df.to_csv(path, index=False)
    rows.append(
        {
            "file": name,
            "mean": df["sales"].mean(),
            "median": df["sales"].median(),
            "zeros": int((df["sales"] == 0).sum()),
            "min": df["sales"].min(),
            "max": df["sales"].max(),
        }
    )
    print(f"[save] {path} mean={df.sales.mean():.6f} zeros={(df.sales == 0).sum()}")


def main() -> None:
    out_dir = Path(os.environ.get("STORE_SALES_OUT", "/root/output"))
    if os.name == "nt" and not out_dir.exists():
        out_dir = Path("score_push_blends")
    out_dir.mkdir(parents=True, exist_ok=True)

    xw = read_submission(
        "xiewenwei",
        [
            "/root/output/BEST_037936_xiewenwei.csv",
            "/root/output/submission_xiewenwei_scoretop.csv",
            "/root/autodl-tmp/xiewenwei_output/submission",
            "_kaggle_research/score_top_xiewenwei_output/submission",
        ],
    )
    hb = read_submission(
        "hblend",
        [
            "/root/output/submission_hblend_037946.csv",
            "/root/output/submission.csv",
            "_kaggle_research/hblend_outputs_v2/submission_hblend_037946.csv",
        ],
    )
    assert_aligned(xw, hb, "hblend")

    parts = {
        "xw": xw["sales"],
        "hb": hb["sales"],
    }

    t127_path = first_existing(
        [
            "/root/output/submission_t127.csv",
            "/root/autodl-tmp/t127_output/submission_tier127.csv",
            "_kaggle_research/score_top_t127_output/submission_tier127.csv",
        ]
    )
    if t127_path is not None:
        t127 = pd.read_csv(t127_path)
        assert_aligned(xw, t127, "t127")
        parts["t127"] = t127["sales"]
        print(f"[load] {'t127':<10} {t127_path} rows={len(t127)} mean={t127.sales.mean():.6f}")
    else:
        print("[skip] t127 not found; generating xiewenwei+hblend blends only")

    lgbm_path = first_existing(
        [
            "/root/output/submission_lgbm_final.csv",
            "_kaggle_research/xiewenwei_clean_outputs/submission_lgbm_final.csv",
        ]
    )
    if lgbm_path is not None:
        lgbm = pd.read_csv(lgbm_path)
        assert_aligned(xw, lgbm, "lgbm_final")
        parts["lgbm"] = lgbm["sales"]
        print(f"[load] {'lgbm':<10} {lgbm_path} rows={len(lgbm)} mean={lgbm.sales.mean():.6f}")
    else:
        print("[skip] lgbm_final not found; skipping xiewenwei+hblend+lgbm blends")

    summary: list[dict[str, object]] = []

    # Low-risk blends around the new 0.37936 anchor.
    for hb_w in [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]:
        weights = {"xw": 1.0 - hb_w, "hb": hb_w}
        tag = f"{int(round(hb_w * 100)):02d}"
        save(out_dir, f"submission_xw_hb_linear_w{tag}.csv", linear_blend(xw, parts, weights), summary)
        save(out_dir, f"submission_xw_hb_log_w{tag}.csv", log_blend(xw, parts, weights), summary)

    if "t127" in parts:
        for xw_w, hb_w, t127_w in [
            (0.92, 0.06, 0.02),
            (0.90, 0.08, 0.02),
            (0.90, 0.07, 0.03),
            (0.88, 0.10, 0.02),
        ]:
            weights = {"xw": xw_w, "hb": hb_w, "t127": t127_w}
            tag = f"{int(xw_w * 100):02d}{int(hb_w * 100):02d}{int(t127_w * 100):02d}"
            save(out_dir, f"submission_xw_hb_t127_linear_{tag}.csv", linear_blend(xw, parts, weights), summary)
            save(out_dir, f"submission_xw_hb_t127_log_{tag}.csv", log_blend(xw, parts, weights), summary)

    if "lgbm" in parts:
        # These are centered around the current public best:
        # 95% xiewenwei + 5% hblend in log space scored 0.37932.
        # Keep the added LGBM component tiny; it is useful as a row-level nudge,
        # not as a replacement for the public xiewenwei anchor.
        for xw_w, hb_w, lgbm_w in [
            (0.93, 0.05, 0.02),
            (0.92, 0.05, 0.03),
            (0.90, 0.05, 0.05),
            (0.92, 0.03, 0.05),
            (0.95, 0.00, 0.05),
            (0.97, 0.00, 0.03),
        ]:
            weights = {"xw": xw_w, "hb": hb_w, "lgbm": lgbm_w}
            tag = f"{int(xw_w * 100):02d}{int(hb_w * 100):02d}{int(lgbm_w * 100):02d}"
            save(out_dir, f"submission_xw_hb_lgbm_linear_{tag}.csv", linear_blend(xw, parts, weights), summary)
            save(out_dir, f"submission_xw_hb_lgbm_log_{tag}.csv", log_blend(xw, parts, weights), summary)

    summary_df = pd.DataFrame(summary)
    summary_path = out_dir / "score_push_blend_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[done] wrote {len(summary_df)} candidates")
    print(summary_df[["file", "mean", "median", "zeros"]].to_string(index=False))
    print(f"\n[summary] {summary_path}")


if __name__ == "__main__":
    main()
