import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from xgboost import XGBRegressor
import joblib
import os


df = pd.read_csv(r'c:\Users\USER\Documents\Auto-Price/data/preprocessing.csv')

X = df.iloc[:, 1:]
X = X.drop(columns=['price'])
y = df['price']

scaler_loaded = joblib.load(r'c:\Users\USER\Documents\Auto-Price/data/minmax_scaler.pkl')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- Best params from fine tuning ---
xgb_fine_tuned = XGBRegressor(
    colsample_bytree=1.0,
    gamma=0,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=3,
    n_estimators=400,
    subsample=1.0
)



# --- Train on ALL data ---
xgb_fine_tuned.fit(X, y)


# --- Save model ---
joblib.dump(xgb_fine_tuned, r'c:\Users\USER\Documents\Auto-Price\Models\xgb_fine_tuned.pkl')


