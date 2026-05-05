import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import joblib

df = pd.read_csv(r'c:\Users\USER\Documents\Auto-Price/data/preprocessing.csv')

X = df.iloc[:, 1:]
X = X.drop(columns=['price'])
y = df['price']

scaler_loaded = joblib.load(r'c:\Users\USER\Documents\Auto-Price/data/minmax_scaler.pkl')

dummy_min = scaler_loaded.inverse_transform([[0]])
dummy_max = scaler_loaded.inverse_transform([[1]])
price_range = dummy_max[0][0] - dummy_min[0][0]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- GradientBoosting parameter grid ---
gb_params = {
    'n_estimators':      [100, 200, 300],
    'learning_rate':     [0.05, 0.1, 0.2],
    'max_depth':         [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'subsample':         [0.8, 1.0],
}

# --- XGBoost parameter grid ---
xgb_params = {
    'n_estimators':  [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth':     [3, 4, 5],
    'subsample':     [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

models = {
    'GradientBoosting': (GradientBoostingRegressor(), gb_params),
    'XGBoost':          (XGBRegressor(), xgb_params),
}

for name, (model, params) in models.items():
    print(f"Tuning {name}...")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=kf,
        scoring='r2',
        n_jobs=-1,      # Use all CPU cores
        verbose=1
    )

    grid_search.fit(X, y)

    best = grid_search.best_estimator_
    best_r2 = grid_search.best_score_

    # --- Get MAE and RMSE for the best model ---
    from sklearn.model_selection import cross_validate
    cv_results = cross_validate(
        best, X, y, cv=kf,
        scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']
    )

    mae  = (-cv_results['test_neg_mean_absolute_error'].mean()) * price_range
    rmse = (-cv_results['test_neg_root_mean_squared_error'].mean()) * price_range
    r2   = cv_results['test_r2'].mean()

    print(f"\n--- {name} Best Results ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")
    print()


    model.fit(X, y)
    joblib.dump(model, rf'c:\Users\USER\Documents\Auto-Price\Models\{name}_tuned.pkl')

