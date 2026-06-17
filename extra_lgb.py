# ============================================================
# 额外 LGB (seed=123, 更低 lr) — 在 V4 Cat 跑完后运行
# 和原来的 LGB (seed=42) 平均, 增加模型多样性
# ============================================================
print("=== Extra LGB (seed=123) ===")

# 使用 Optuna 调出的参数结构, 但用更保守的 learning_rate
lgb_params_v2 = {
    **{k: v for k, v in lgb_params.items() if k not in ["random_state", "learning_rate"]},
    "learning_rate": 0.015,  # 更低lr → 更多树 → 不同bias
    "random_state": 123,
    "num_leaves": min(lgb_params.get("num_leaves", 256), 400),
}
dtrain2 = lgb.Dataset(X_tr, label=y_tr)
dval2 = lgb.Dataset(X_val, label=y_val_log, reference=dtrain2)

lgb_model2 = lgb.train(
    params=lgb_params_v2, train_set=dtrain2, num_boost_round=5000,
    valid_sets=[dtrain2, dval2], valid_names=["train", "valid"],
    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)]
)

lgb2_v = np.expm1(np.clip(lgb_model2.predict(X_val, num_iteration=lgb_model2.best_iteration), 0, None))
print(f"LGB2 (seed=123) RMSLE: {rmsle(y_val_real, lgb2_v):.6f}  iter={lgb_model2.best_iteration}")

# 两个LGB平均
lgb1_raw = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
lgb2_raw = lgb_model2.predict(X_val, num_iteration=lgb_model2.best_iteration)
lgb_avg = np.expm1(np.clip((lgb1_raw + lgb2_raw) / 2, 0, None))
print(f"LGB Avg (2 seeds) RMSLE: {rmsle(y_val_real, lgb_avg):.6f}")

# 新3-tree ensemble (LGB_avg + XGB + CatBoost)
xgb_raw = xgb_model.predict(dv_xgb)
cat_raw = cat_model.predict(X_val.values)
lgb_avg_raw = (lgb1_raw + lgb2_raw) / 2

best_new, bw_new = float("inf"), (0.4, 0.3, 0.3)
for w1 in np.arange(0.0, 1.01, 0.05):
    for w2 in np.arange(0.0, 1.01 - w1, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0: continue
        ep = np.expm1(np.clip(w1*lgb_avg_raw + w2*xgb_raw + w3*cat_raw, 0, None))
        s = rmsle(y_val_real, ep)
        if s < best_new: best_new, bw_new = s, (w1, w2, w3)

print(f"New Ensemble (LGB×2+XGB+CAT) RMSLE: {best_new:.6f}")
print(f"Weights: LGB_avg={bw_new[0]:.2f} XGB={bw_new[1]:.2f} CAT={bw_new[2]:.2f}")
