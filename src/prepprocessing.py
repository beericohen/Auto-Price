import pandas as pd
import numpy as np
import joblib
from path import *
import os
from sklearn.preprocessing import MinMaxScaler


def prepprocessing():
    clean_path = os.path.join(DATA_DIR, 'autoboom_clean.csv')
    df = pd.read_csv(clean_path, index_col=False)

    # Fill NaN (engine_liters / horsepower for electrics)
    df = df.fillna(0)

    # One-hot encode categorical columns
    dummies = pd.get_dummies(df, dtype=int)

    # Scale numerical features and price
    scaler   = MinMaxScaler()
    Yscaler  = MinMaxScaler()

    cols_to_scale = ['year', 'hand', 'engine_liters', 'horsepower', 'mileage']
    dummies[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    dummies['price']       = Yscaler.fit_transform(df[['price']])

    # Save scalers
    joblib.dump(Yscaler, PRICE_SCALER_PATH)
    joblib.dump(scaler,  FEATURES_SCALER_PATH)
    print("Scalers saved.")

    # Save preprocessed data
    dummies.to_csv(DATA_PATH, index=False)
    print(f"Saved: preprocessing.csv  ({len(dummies)} rows, {len(dummies.columns)} columns)")


prepprocessing()