import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import os

from path import *

def tuning():
    # ─── Load Data ────────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH, index_col=False)
    X  = df.drop(columns=['price'])
    y  = df['price']

    # ─── Recover original price scale for MAE/RMSE readability ───────────────────
    scaler  = joblib.load(PRICE_SCALER_PATH)
    p_min   = scaler.inverse_transform([[0]])[0][0]
    p_max   = scaler.inverse_transform([[1]])[0][0]
    price_range = p_max - p_min

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # ─── Utility: evaluate and print cross-val metrics ───────────────────────────
    def evaluate(model, name, stage=""):
        cv = cross_validate(
            model, X, y, cv=kf,
            scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']
        )
        mae  = (-cv['test_neg_mean_absolute_error'].mean())  * price_range
        rmse = (-cv['test_neg_root_mean_squared_error'].mean()) * price_range
        r2   =   cv['test_r2'].mean()
        label = f"{name} {stage}".strip()
        print(f"\n--- {label} ---")
        print(f"  MAE : {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R2  : {r2:.4f}")
        return mae, rmse, r2

    # ─── Utility: run GridSearchCV and return best estimator ─────────────────────
    def grid_tune(model, params, name):
        gs = GridSearchCV(
            estimator=model,
            param_grid=params,
            cv=kf,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        gs.fit(X, y)
        print(f"\n[{name}] Best params: {gs.best_params_}")
        return gs.best_estimator_, gs.best_params_

    # ═══════════════════════════════════════════════════════════════════════════════
    # STAGE 1 — COARSE TUNING
    # Models: ExtraTrees, RandomForest, GradientBoosting, XGBoost
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("STAGE 1 — COARSE TUNING")
    print("="*60)

    # ── ExtraTrees ────────────────────────────────────────────────────────────────
    et_params_coarse = {
        'n_estimators':  [100, 200, 300],
        'max_depth':     [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'max_features':  ['sqrt', 'log2', 0.8],
    }
    print("\nTuning ExtraTrees (coarse)...")
    et_best, et_best_params = grid_tune(ExtraTreesRegressor(random_state=42), et_params_coarse, "ExtraTrees")
    evaluate(et_best, "ExtraTrees", "Coarse")

    # ── RandomForest ──────────────────────────────────────────────────────────────
    rf_params_coarse = {
        'n_estimators':  [100, 200, 300],
        'max_depth':     [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'max_features':  ['sqrt', 'log2', 0.8],
    }
    print("\nTuning RandomForest (coarse)...")
    rf_best, rf_best_params = grid_tune(RandomForestRegressor(random_state=42), rf_params_coarse, "RandomForest")
    evaluate(rf_best, "RandomForest", "Coarse")

    # ── GradientBoosting ──────────────────────────────────────────────────────────
    gb_params_coarse = {
        'n_estimators':      [100, 200, 300],
        'learning_rate':     [0.05, 0.1, 0.2],
        'max_depth':         [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'subsample':         [0.8, 1.0],
    }
    print("\nTuning GradientBoosting (coarse)...")
    gb_best, gb_best_params = grid_tune(GradientBoostingRegressor(random_state=42), gb_params_coarse, "GradientBoosting")
    evaluate(gb_best, "GradientBoosting", "Coarse")

    # ── XGBoost coarse ────────────────────────────────────────────────────────────
    xgb_params_coarse = {
        'n_estimators':     [100, 200, 300],
        'learning_rate':    [0.05, 0.1, 0.2],
        'max_depth':        [3, 4, 5],
        'subsample':        [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }
    print("\nTuning XGBoost (coarse)...")
    xgb_best, xgb_best_params = grid_tune(XGBRegressor(), xgb_params_coarse, "XGBoost")
    evaluate(xgb_best, "XGBoost", "Coarse")


    # ─── Save coarse-tuned models ─────────────────────────────────────────────────
    print("\nSaving coarse-tuned models...")
    coarse_models = {
        'ExtraTrees_coarse':       et_best,
        'RandomForest_coarse':     rf_best,
        'GradientBoosting_coarse': gb_best,
        'XGBoost_coarse':          xgb_best,
    }
    for name, model in coarse_models.items():
        out_path = os.path.join(MODELS_DIR, f'{name}.pkl')
        joblib.dump(model, out_path)
        print(f"  Saved: {out_path}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # STAGE 2 — FINE TUNING (narrow search around best params from stage 1)
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("STAGE 2 — FINE TUNING")
    print("="*60)

    # ── ExtraTrees fine ───────────────────────────────────────────────────────────
    # Narrow around best n_estimators / min_samples_split
    et_n   = et_best_params['n_estimators']
    et_mss = et_best_params['min_samples_split']
    et_params_fine = {
        'n_estimators':      sorted(set([max(50, et_n - 50), et_n, et_n + 50])),
        'max_depth':         [et_best_params['max_depth']],   # keep winner
        'min_samples_split': sorted(set([max(2, et_mss - 1), et_mss, et_mss + 1])),
        'max_features':      [et_best_params['max_features']],
        'min_samples_leaf':  [1, 2, 4],      # new dimension to explore
    }
    print("\nFine-tuning ExtraTrees...")
    et_fine, et_fine_params = grid_tune(ExtraTreesRegressor(random_state=42), et_params_fine, "ExtraTrees")
    evaluate(et_fine, "ExtraTrees", "Fine")

    # ── RandomForest fine ─────────────────────────────────────────────────────────
    rf_n   = rf_best_params['n_estimators']
    rf_mss = rf_best_params['min_samples_split']
    rf_params_fine = {
        'n_estimators':      sorted(set([max(50, rf_n - 50), rf_n, rf_n + 50])),
        'max_depth':         [rf_best_params['max_depth']],
        'min_samples_split': sorted(set([max(2, rf_mss - 1), rf_mss, rf_mss + 1])),
        'max_features':      [rf_best_params['max_features']],
        'min_samples_leaf':  [1, 2, 4],
        'max_samples':       [0.8, 0.9, None],  # bagging fraction
    }
    print("\nFine-tuning RandomForest...")
    rf_fine, rf_fine_params = grid_tune(RandomForestRegressor(random_state=42), rf_params_fine, "RandomForest")
    evaluate(rf_fine, "RandomForest", "Fine")

    # ── GradientBoosting fine ─────────────────────────────────────────────────────
    gb_lr  = gb_best_params['learning_rate']
    gb_n   = gb_best_params['n_estimators']
    gb_params_fine = {
        'n_estimators':      sorted(set([max(50, gb_n - 50), gb_n, gb_n + 50])),
        'learning_rate':     sorted(set([round(gb_lr - 0.02, 3), gb_lr, round(gb_lr + 0.02, 3)])),
        'max_depth':         [gb_best_params['max_depth']],
        'min_samples_split': [gb_best_params['min_samples_split']],
        'subsample':         [gb_best_params['subsample']],
        'min_samples_leaf':  [1, 3, 5],
    }
    print("\nFine-tuning GradientBoosting...")
    gb_fine, gb_fine_params = grid_tune(GradientBoostingRegressor(random_state=42), gb_params_fine, "GradientBoosting")
    evaluate(gb_fine, "GradientBoosting", "Fine")

    # ── XGBoost fine ──────────────────────────────────────────────────────────────
    # Extract current best params from loaded model
    xp = xgb_best.get_params()
    xgb_params_fine = {
        'n_estimators':      sorted(set([max(50, xp['n_estimators'] - 50), xp['n_estimators'], xp['n_estimators'] + 50])),
        'learning_rate':     sorted(set([round(xp['learning_rate'] - 0.02, 3), xp['learning_rate'], round(xp['learning_rate'] + 0.02, 3)])),
        'max_depth':         [xp['max_depth']],
        'subsample':         sorted(set([round(min(1.0, xp['subsample'] - 0.05), 2), xp['subsample']])),
        'colsample_bytree':  sorted(set([round(min(1.0, xp['colsample_bytree'] - 0.05), 2), xp['colsample_bytree']])),
        'min_child_weight':  [xp.get('min_child_weight', 1)],
        'gamma':             [xp.get('gamma', 0)],
        'reg_alpha':         [0, 0.01, 0.1],   # L1 regularization — new dimension
        'reg_lambda':        [1, 1.5, 2],       # L2 regularization — new dimension
    }
    print("\nFine-tuning XGBoost...")
    xgb_fine, xgb_fine_params = grid_tune(XGBRegressor(), xgb_params_fine, "XGBoost")
    evaluate(xgb_fine, "XGBoost", "Fine")

    # ═══════════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R2':>8}")
    print("-" * 55)

    final_models = {
        'ExtraTrees_fine':       et_fine,
        'RandomForest_fine':     rf_fine,
        'GradientBoosting_fine': gb_fine,
        'XGBoost_fine':          xgb_fine,
    }

    for name, model in final_models.items():
        cv = cross_validate(model, X, y, cv=kf,
                            scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'])
        mae  = (-cv['test_neg_mean_absolute_error'].mean())     * price_range
        rmse = (-cv['test_neg_root_mean_squared_error'].mean()) * price_range
        r2   =   cv['test_r2'].mean()
        print(f"{name:<25} {mae:>10.2f} {rmse:>10.2f} {r2:>8.4f}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # SAVE FINAL MODELS
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\nSaving final models...")
    for name, model in final_models.items():
        out_path = os.path.join(MODELS_DIR, f'{name}.pkl')
        joblib.dump(model, out_path)
        print(f"  Saved: {out_path}")

    print("\nDone!")

if __name__ == '__main__':
    tuning()