import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from xgboost import XGBRegressor
import joblib

df = pd.read_csv(r'C:\Users\USER\Documents\Projects\AutoPrice\Auto-Price\data/preprocessing.csv', index_col=False)

X = df.drop(columns=['price'])
y = df['price']

scaler_loaded = joblib.load(r'C:\Users\USER\Documents\Projects\AutoPrice\Auto-Price\data/minmax_scaler.pkl')

dummy_min = scaler_loaded.inverse_transform([[0]])
dummy_max = scaler_loaded.inverse_transform([[1]])
price_range = dummy_max[0][0] - dummy_min[0][0]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- Fine tuning around the best params found earlier ---
xgb_params = {
    'n_estimators':     [250, 300, 350, 400],
    'learning_rate':    [0.18, 0.19, 0.2, 0.21, 0.22],
    'max_depth':        [5],
    'subsample':        [0.9, 1.0, 1.1],
    'colsample_bytree': [0.7,0.8, 0.9],
    'min_child_weight': [1, 3, 5],    # New parameter - controls overfitting
    'gamma':            [0, 0.1, 0.2] # New parameter - minimum split gain
}

print("Fine tuning XGBoost...")

grid_search = GridSearchCV(
    estimator=XGBRegressor(),
    param_grid=xgb_params,
    cv=kf,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)

best = grid_search.best_estimator_

cv_results = cross_validate(
    best, X, y, cv=kf,
    scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']
)

mae  = (-cv_results['test_neg_mean_absolute_error'].mean()) * price_range
rmse = (-cv_results['test_neg_root_mean_squared_error'].mean()) * price_range
r2   = cv_results['test_r2'].mean()

print(f"\n--- XGBoost Fine Tuned Results ---")
print(f"Best Params: {grid_search.best_params_}")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.4f}")

