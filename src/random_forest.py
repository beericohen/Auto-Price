import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv(r'c:\Users\USER\Documents\Auto-Price/data/preprocessing.csv')

X = df.iloc[:, 1:]
X = X.drop(columns=['price'])
y = df['price']

scaler_loaded = joblib.load(r'c:\Users\USER\Documents\Auto-Price/data/minmax_scaler.pkl')

# --- Cross-Validation ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestRegressor()

cv_results = cross_validate(
    rf, X, y,
    cv=kf,
    scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'],
    return_train_score=False
)

# --- Convert scaled metrics back to original scale ---
# The scaler was fit on price column, so we use it to inverse the error margins
dummy_min = scaler_loaded.inverse_transform([[0]])
dummy_max = scaler_loaded.inverse_transform([[1]])
price_range = dummy_max[0][0] - dummy_min[0][0]

mae_scaled = -cv_results['test_neg_mean_absolute_error']
rmse_scaled = -cv_results['test_neg_root_mean_squared_error']

mae_original = mae_scaled * price_range
rmse_original = rmse_scaled * price_range

print("--- Cross-Validation Metrics (Original Scale) ---")
print(f"MAE:  {mae_original.mean():.2f} (+/- {mae_original.std():.2f})")
print(f"RMSE: {rmse_original.mean():.2f} (+/- {rmse_original.std():.2f})")
print(f"R2:   {cv_results['test_r2'].mean():.4f} (+/- {cv_results['test_r2'].std():.4f})")

# --- Train final model on ALL data ---
rf.fit(X, y)