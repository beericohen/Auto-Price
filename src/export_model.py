"""
export_model.py
----------------
Converts the trained scikit-learn MLPRegressor + MinMaxScalers into plain
JSON files that a static website (no Python backend) can load and run
entirely in the browser with JavaScript.

Usage:
    python export_model.py

Expects, in the same folder (or edit the paths below):
    MLPRegressor_tuned.pkl   - trained sklearn MLPRegressor
    scaler.pkl               - MinMaxScaler fit on [year, hand, engine_liters, horsepower, mileage]
    minmax_scaler.pkl        - MinMaxScaler fit on [price]
    preprocessing.csv        - the one-hot encoded training dataframe (used to
                                recover category lists + build the dataset used
                                for the "where does your car stand" scatter plot)

Produces:
    site/assets/data/model.json   - network weights/biases + scaler params
    site/assets/data/dataset.json - decoded dataset + category options for the UI
"""
import json
import os
import numpy as np
import pandas as pd
import joblib

from path import *

# ---- EDIT THESE IF YOUR FILES LIVE ELSEWHERE -----------------------------
OUT_DIR = os.path.join("site", "assets", "data")
# ---------------------------------------------------------------------------

NUMERIC_COLS = ["year", "hand", "engine_liters", "horsepower", "mileage"]
GROUP_PREFIXES = [
    "manufacturer_",
    "submodel_",   # must come before "model_" since both start differently but keep order safe
    "model_",
    "fuel_",
    "transmission_",
    "drive_type_",
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(FEATURES_SCALER_PATH)
    yscaler = joblib.load(PRICE_SCALER_PATH)
    df = pd.read_csv(DATA_PATH)

    feature_cols = [c for c in df.columns if c != "price"]
    assert len(feature_cols) == model.n_features_in_, "Column count mismatch vs model input size"

    # ---- group one-hot columns by category ----
    groups = {p: [] for p in GROUP_PREFIXES}
    for c in feature_cols:
        if c in NUMERIC_COLS:
            continue
        for p in GROUP_PREFIXES:
            if c.startswith(p):
                groups[p].append(c)
                break

    # ================= 1. MODEL WEIGHTS (model.json) =================
    model_json = {
        "feature_order": feature_cols,
        "numeric_cols": NUMERIC_COLS,
        "numeric_scaler": {
            "cols": list(scaler.feature_names_in_),
            "data_min": scaler.data_min_.tolist(),
            "data_max": scaler.data_max_.tolist(),
        },
        "price_scaler": {
            "data_min": yscaler.data_min_.tolist(),
            "data_max": yscaler.data_max_.tolist(),
        },
        "architecture": {
            "hidden_layer_sizes": list(model.hidden_layer_sizes),
            "activation": model.activation,
            "out_activation": model.out_activation_,
            "n_layers": model.n_layers_,
        },
        "weights": [w.tolist() for w in model.coefs_],
        "biases": [b.tolist() for b in model.intercepts_],
        "groups": {p.rstrip("_"): [c[len(p):] for c in cols] for p, cols in groups.items()},
    }

    with open(os.path.join(OUT_DIR, "model.json"), "w", encoding="utf-8") as f:
        json.dump(model_json, f)
    print(f"Wrote {OUT_DIR}/model.json  ({os.path.getsize(os.path.join(OUT_DIR,'model.json'))/1024:.1f} KB)")

    # ================= 2. DECODED DATASET (dataset.json) =================
    def decode_group(row, prefix, cols):
        for c in cols:
            if row[c] == 1:
                return c[len(prefix):]
        return "Other"

    inv_num = scaler.inverse_transform(df[NUMERIC_COLS])
    inv_price = yscaler.inverse_transform(df[["price"]])

    records = []
    model_to_manufacturer = {}
    model_to_submodels = {}

    for i, row in df.iterrows():
        manufacturer = decode_group(row, "manufacturer_", groups["manufacturer_"])
        model_name = decode_group(row, "model_", groups["model_"])
        submodel = decode_group(row, "submodel_", groups["submodel_"])
        fuel = decode_group(row, "fuel_", groups["fuel_"])
        transmission = decode_group(row, "transmission_", groups["transmission_"])
        drive = decode_group(row, "drive_type_", groups["drive_type_"])

        rec = {
            "manufacturer": manufacturer,
            "model": model_name,
            "submodel": submodel,
            "fuel": fuel,
            "transmission": transmission,
            "drive_type": drive,
            "year": round(float(inv_num[i, 0])),
            "hand": round(float(inv_num[i, 1])),
            "engine_liters": round(float(inv_num[i, 2]), 1),
            "horsepower": round(float(inv_num[i, 3])),
            "mileage": round(float(inv_num[i, 4])),
            "price": round(float(inv_price[i, 0])),
        }
        records.append(rec)

        model_to_manufacturer.setdefault(model_name, {})
        model_to_manufacturer[model_name][manufacturer] = model_to_manufacturer[model_name].get(manufacturer, 0) + 1
        model_to_submodels.setdefault(model_name, set()).add(submodel)

    # collapse manufacturer counts -> most likely manufacturer per model
    model_manufacturer_map = {
        m: max(counts.items(), key=lambda kv: kv[1])[0] for m, counts in model_to_manufacturer.items()
    }
    model_submodel_map = {m: sorted(list(s)) for m, s in model_to_submodels.items()}

    # reverse map: manufacturer -> list of real models (the catch-all "Other" model
    # is available under every manufacturer since it pools rare models from all brands)
    all_manufacturers = sorted([c[len("manufacturer_"):] for c in groups["manufacturer_"]])
    manufacturer_to_models = {manu: [] for manu in all_manufacturers}
    for m, manu in model_manufacturer_map.items():
        if m == "Other":
            continue
        manufacturer_to_models.setdefault(manu, []).append(m)
    for manu in manufacturer_to_models:
        manufacturer_to_models[manu] = sorted(manufacturer_to_models[manu]) + ["Other"]

    dataset_json = {
        "records": records,
        "options": {
            "manufacturer": sorted(groups["manufacturer_"] and [c[len("manufacturer_"):] for c in groups["manufacturer_"]]),
            "model": sorted([c[len("model_"):] for c in groups["model_"]]),
            "submodel": sorted([c[len("submodel_"):] for c in groups["submodel_"]]),
            "fuel": sorted([c[len("fuel_"):] for c in groups["fuel_"]]),
            "transmission": sorted([c[len("transmission_"):] for c in groups["transmission_"]]),
            "drive_type": sorted([c[len("drive_type_"):] for c in groups["drive_type_"]]),
        },
        "model_to_manufacturer": model_manufacturer_map,
        "model_to_submodels": model_submodel_map,
        "manufacturer_to_models": manufacturer_to_models,
        "stats": {
            "n_rows": len(records),
            "year_min": int(min(r["year"] for r in records)),
            "year_max": int(max(r["year"] for r in records)),
            "price_min": int(min(r["price"] for r in records)),
            "price_max": int(max(r["price"] for r in records)),
            "mileage_max": int(max(r["mileage"] for r in records)),
        },
    }

    with open(os.path.join(OUT_DIR, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_json, f)
    print(f"Wrote {OUT_DIR}/dataset.json  ({os.path.getsize(os.path.join(OUT_DIR,'dataset.json'))/1024:.1f} KB)")


if __name__ == "__main__":
    main()
