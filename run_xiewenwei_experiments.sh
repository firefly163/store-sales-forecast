#!/usr/bin/env bash
set -euo pipefail

# Store Sales xiewenwei experiment runner for AutoDL.
#
# Usage examples:
#   bash run_xiewenwei_experiments.sh weights
#   bash run_xiewenwei_experiments.sh zero
#   bash run_xiewenwei_experiments.sh xgb
#   bash run_xiewenwei_experiments.sh lag
#
# The script writes each experiment into a separate folder under $STORE_SALES_OUT.

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/store_sales}"
BASE_OUT="${STORE_SALES_OUT:-/root/output}"
DATA_DIR="${STORE_SALES_DATA:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$ROOT_DIR"
mkdir -p "$BASE_OUT"

run_one() {
  local name="$1"
  shift
  local out="$BASE_OUT/$name"
  mkdir -p "$out"
  echo "================================================================"
  echo "Experiment: $name"
  echo "Output: $out"
  echo "Extra env: $*"
  echo "================================================================"
  env STORE_SALES_OUT="$out" STORE_SALES_DATA="$DATA_DIR" "$@" \
    "$PYTHON_BIN" -u store_sales_xiewenwei_clean.py 2>&1 | tee "$out/run.log"
}

case "${1:-weights}" in
  weights)
    # Step 1 + 2: tune the four model components and prefer log blend.
    # Component order is:
    #   lgb_full,lgb_2015,xgb_full,xgb_2015
    run_one weights_log_30251530 \
      COMPONENT_BLEND_SPACE=log \
      COMPONENT_WEIGHTS=0.30,0.25,0.15,0.30
    run_one weights_log_35251030 \
      COMPONENT_BLEND_SPACE=log \
      COMPONENT_WEIGHTS=0.35,0.25,0.10,0.30
    run_one weights_log_30301030 \
      COMPONENT_BLEND_SPACE=log \
      COMPONENT_WEIGHTS=0.30,0.30,0.10,0.30
    ;;

  zero)
    # Step 3: zero-sales rule window. Use the current best local-opt component
    # weights found by grid search:
    #   lgb_full=0.325, lgb_2015=0.325, xgb_full=0.350, xgb_2015=0.000
    for zw in 14 21 28; do
      run_one zero_${zw}_log_325325350000 \
        ZERO_FC_WINDOW="$zw" \
        COMPONENT_BLEND_SPACE=log \
        COMPONENT_WEIGHTS=0.325,0.325,0.350,0.000
    done
    ;;

  xgb)
    # Step 4: conservative XGB micro-tuning.
    run_one xgb_lr008_depth6_log_30301030 \
      XGB_LEARNING_RATE=0.08 \
      XGB_MAX_DEPTH=6 \
      XGB_N_ESTIMATORS=140 \
      COMPONENT_BLEND_SPACE=log \
      COMPONENT_WEIGHTS=0.30,0.30,0.10,0.30
    run_one xgb_lr006_depth5_log_30301030 \
      XGB_LEARNING_RATE=0.06 \
      XGB_MAX_DEPTH=5 \
      XGB_N_ESTIMATORS=180 \
      COMPONENT_BLEND_SPACE=log \
      COMPONENT_WEIGHTS=0.30,0.30,0.10,0.30
    run_one xgb_reg_log_30301030 \
      XGB_LEARNING_RATE=0.08 \
      XGB_MAX_DEPTH=6 \
      XGB_N_ESTIMATORS=150 \
      XGB_MIN_CHILD_WEIGHT=2 \
      XGB_REG_LAMBDA=2 \
      COMPONENT_BLEND_SPACE=log \
      COMPONENT_WEIGHTS=0.30,0.30,0.10,0.30
    ;;

  lag)
    # Step 5: first lag variants. Keep 7/365/730 anchors.
    for base_lag in 56 63 70; do
      run_one lag_${base_lag}_log_30301030 \
        DARTS_LAGS="${base_lag},7,365,730" \
        COMPONENT_BLEND_SPACE=log \
        COMPONENT_WEIGHTS=0.30,0.30,0.10,0.30
    done
    ;;

  *)
    echo "Unknown mode: ${1:-}" >&2
    echo "Use one of: weights, zero, xgb, lag" >&2
    exit 2
    ;;
esac
