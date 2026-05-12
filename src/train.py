"""
retrain.py — Full retraining pipeline.
Steps:
  1. Merge user feedback into autoboom_raw.csv
  2. Clean data  (DataCleaner logic)
  3. Preprocess  (Preprocessing notebook logic)
  4. Baseline model evaluation  (models logic)
  5. Coarse + fine tuning  (tuning logic)
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from path import *

from DataCleaner import cleanData
from EDA import eda
from prepprocessing import prepprocessing
from models import models
from Tuning import tuning

FEEDBACK_PATH = os.path.join(DATA_DIR, 'autoboom_raw.csv')

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Clean data
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 2 — Cleaning data")
print("="*60)

cleanData()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 2 — EDA")
print("="*60)

eda()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Preprocessing  (Preprocessing.ipynb logic)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 3 — Preprocessing")
print("="*60)

prepprocessing()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Baseline model evaluation  (models.py logic)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 4 — Baseline model evaluation")
print("="*60)

models()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Coarse + Fine tuning  (tuning.py logic)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 5 — Tuning")
print("="*60)

tuning()

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)