import json
import numpy as np
import joblib

# פונקציית עזר להמרת מערכי NumPy לרשימות פייתון
def convert_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_list(i) for i in obj]
    return obj

# פונקציה גנרית לחילוץ נתונים מ-MinMaxScaler
def extract_minmax_data(scaler):
    return {
        "min_": convert_to_list(scaler.min_),
        "scale_": convert_to_list(scaler.scale_),
        "data_min_": convert_to_list(scaler.data_min_),
        "data_max_": convert_to_list(scaler.data_max_),
        "data_range_": convert_to_list(scaler.data_range_)
    }

# 1. המרת minmax_scaler.pkl
try:
    scaler = joblib.load('minmax_scaler.pkl')
    scaler_data = extract_minmax_data(scaler)
    with open('minmax_scaler.json', 'w', encoding='utf-8') as f:
        json.dump(scaler_data, f, indent=4)
    print("Successfully converted minmax_scaler.pkl to JSON.")
except Exception as e:
    print(f"Error converting minmax_scaler.pkl: {e}")

# 2. המרת scaler.pkl (תוקן ל-MinMaxScaler בעקבות השגיאה)
try:
    scaler = joblib.load('scaler.pkl')
    scaler_data = extract_minmax_data(scaler)
    with open('scaler.json', 'w', encoding='utf-8') as f:
        json.dump(scaler_data, f, indent=4)
    print("Successfully converted scaler.pkl to JSON.")
except Exception as e:
    print(f"Error converting scaler.pkl: {e}")

# 3. המרת MLPRegressor_tuned.pkl
try:
    model = joblib.load('MLPRegressor_tuned.pkl')
    model_data = {
        "weights": convert_to_list(model.coefs_),
        "biases": convert_to_list(model.intercepts_),
        "n_layers_": model.n_layers_,
        "n_outputs_": model.n_outputs_,
        "out_activation_": model.out_activation_
    }
    with open('MLPRegressor_tuned.json', 'w', encoding='utf-8') as f:
        json.dump(model_data, f, indent=4)
    print("Successfully converted MLPRegressor_tuned.pkl to JSON.")
except Exception as e:
    print(f"Error converting MLPRegressor_tuned.pkl: {e}")