import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import uniform
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from sklearn.neural_network import MLPRegressor

# Assuming path.py contains DATA_PATH, PRICE_SCALER_PATH, and MODELS_DIR
from path import * 

def train_and_tune_neural_network():
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

    # ─── MLPRegressor Hyperparameter Space ───────────────────────────────────────
    param_dist = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (64, 32), (128, 64)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': uniform(0.0001, 0.01),          # L2 regularization strength
        'learning_rate_init': uniform(0.001, 0.01),
        'batch_size': [32, 64, 128, 'auto']
    }

    print("Starting Neural Network Hyperparameter Tuning...")
    
    mlp = MLPRegressor(max_iter=500, random_state=42, early_stopping=True)
    
    rs = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=param_dist,
        n_iter=30, 
        cv=kf,
        scoring='r2',
        n_jobs=1,
        verbose=1,
        random_state=42
    )

    rs.fit(X, y)
    best_model = rs.best_estimator_
    
    print(f"\nBest Parameters Found:\n{rs.best_params_}")

    # ─── Evaluate Final Model ────────────────────────────────────────────────────
    print("\nEvaluating Best Model via Cross-Validation...")
    cv_results = cross_validate(
        best_model, X, y, cv=kf,
        scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']
    )

    mae  = (-cv_results['test_neg_mean_absolute_error'].mean()) * price_range
    rmse = (-cv_results['test_neg_root_mean_squared_error'].mean()) * price_range
    r2   = cv_results['test_r2'].mean()

    print("\n" + "="*40)
    print("FINAL NEURAL NETWORK RESULTS")
    print("="*40)
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    # ─── Train on ALL data and save ──────────────────────────────────────────────
    print("\nRetraining on full dataset and saving...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    best_model.fit(X, y)
    out_path = os.path.join(MODELS_DIR, 'MLPRegressor_tuned.pkl')
    joblib.dump(best_model, out_path)
    
    print(f"Model successfully saved to: {out_path}")

if __name__ == '__main__':
    train_and_tune_neural_network()