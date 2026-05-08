import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Goes one level back from src to main directory
MODEL_PATH = os.path.join(BASE_DIR, "Models", "XGBoost_fine.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessing.csv")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
DATA_DIR = os.path.join(BASE_DIR, "data")
PRICE_SCALER_PATH = os.path.join(BASE_DIR, "data", "minmax_scaler.pkl")
FEATURES_SCALER_PATH = os.path.join(BASE_DIR, "data", "scaler.pkl")
GRAPH_PATH = os.path.join(BASE_DIR, "graphs")