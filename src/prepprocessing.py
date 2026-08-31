import pandas as pd
import numpy as np
import joblib
from path import *
import os
from sklearn.preprocessing import MinMaxScaler


def prepprocessing():
    clean_path = os.path.join(DATA_DIR, 'autoboom_clean.csv')
    df = pd.read_csv(clean_path, index_col=False)

    # 1. Fill NaN values
    df = df.fillna(0)

    # 2. Drop high-cardinality non-predictive metadata if present
    cols_to_drop = ['id', 'title', 'url', 'description']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # # 3. Collapse rare categories (< 1% frequency) into 'Other' to limit column explosion
    # categorical_cols = df.select_dtypes(include=['object']).columns
    # for col in categorical_cols:
    #     freq = df[col].value_counts(normalize=True)
    #     frequent_cats = freq[freq >= 0.01].index
    #     df[col] = df[col].where(df[col].isin(frequent_cats), other='Other')

    # 4. One-hot encode using uint8 (8-bit) instead of default int64 (64-bit)
    dummies = pd.get_dummies(df, dtype=np.uint8)

    # 5. Scale numerical features and target price
    scaler = MinMaxScaler()
    Yscaler = MinMaxScaler()

    cols_to_scale = ['year', 'hand', 'engine_liters', 'horsepower', 'mileage']
    existing_scales = [c for c in cols_to_scale if c in dummies.columns]

    dummies[existing_scales] = scaler.fit_transform(dummies[existing_scales])
    dummies['price'] = Yscaler.fit_transform(df[['price']])

    # 6. Save scalers and processed data
    os.makedirs(os.path.dirname(PRICE_SCALER_PATH), exist_ok=True)
    joblib.dump(Yscaler, PRICE_SCALER_PATH)
    joblib.dump(scaler, FEATURES_SCALER_PATH)
    print("Scalers saved.")

    dummies.to_csv(DATA_PATH, index=False)
    print(f"Saved: data.csv ({len(dummies)} rows, {len(dummies.columns)} columns)")


if __name__ == '__main__':
    prepprocessing()