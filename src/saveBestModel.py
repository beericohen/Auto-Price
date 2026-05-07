import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from xgboost import XGBRegressor
import joblib
import os


df = pd.read_csv(r'C:\Users\USER\Documents\Projects\AutoPrice\Auto-Price\data/preprocessing.csv', index_col=False)

X = df.drop(columns=['price'])
y = df['price']

scaler_loaded = joblib.load(r'C:\Users\USER\Documents\Projects\AutoPrice\Auto-Price\data/minmax_scaler.pkl')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- Best params from fine tuning ---
xgb_fine_tuned = XGBRegressor(
    colsample_bytree=0.8,
    gamma=0,
    learning_rate=0.19,
    max_depth=5,
    min_child_weight=1,
    n_estimators=350,
    subsample=0.9
)



# --- Train on ALL data ---
xgb_fine_tuned.fit(X, y)


# --- Save model ---
joblib.dump(xgb_fine_tuned, r'C:\Users\USER\Documents\Projects\AutoPrice\Auto-Price\Models\xgb_fine_tuned.pkl')


