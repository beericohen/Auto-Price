import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
import joblib
import os

df = pd.read_csv(r'c:\Users\USER\Documents\Auto-Price/data/preprocessing.csv')

X = df.iloc[:, 1:]
X = X.drop(columns=['price'])
y = df['price']

scaler_loaded = joblib.load(r'c:\Users\USER\Documents\Auto-Price/data/minmax_scaler.pkl')

dummy_min = scaler_loaded.inverse_transform([[0]])
dummy_max = scaler_loaded.inverse_transform([[1]])
price_range = dummy_max[0][0] - dummy_min[0][0]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'LinearRegression':  LinearRegression(),
    'Ridge':             Ridge(),
    'Lasso':             Lasso(),
    'ElasticNet':        ElasticNet(),
    'BayesianRidge':     BayesianRidge(),
    'KNN':               KNeighborsRegressor(),
    'ExtraTrees':        ExtraTreesRegressor(),
    'RandomForest':      RandomForestRegressor(),
    'GradientBoosting':  GradientBoostingRegressor(),
    'XGBoost':           XGBRegressor(),
}

os.makedirs(r'c:\Users\USER\Documents\Auto-Price\Models', exist_ok=True)

print(f"{'Model':<20} {'MAE':>12} {'RMSE':>12} {'R2':>8}")
print("-" * 56)

for name, model in models.items():
    cv_results = cross_validate(
        model, X, y, cv=kf,
        scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']
    )

    mae  = (-cv_results['test_neg_mean_absolute_error'].mean()) * price_range
    rmse = (-cv_results['test_neg_root_mean_squared_error'].mean()) * price_range
    r2   = cv_results['test_r2'].mean()

    print(f"{name:<20} {mae:>12.2f} {rmse:>12.2f} {r2:>8.4f}")

    # --- Train on ALL data and save ---
    model.fit(X, y)
    joblib.dump(model, rf'c:\Users\USER\Documents\Auto-Price\Models\{name}.pkl')

print("\nAll models saved to Auto-Price/Models/")