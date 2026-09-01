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
    site/assets/data/model.json   - network weights/biases + scaler params + validation metadata
    site/assets/data/dataset.json - decoded dataset + category options + market statistics
"""
import json
import os
import numpy as np
import pandas as pd
import joblib

from path import *

# Always write to the website's actual data directory, regardless of the
# directory from which this script is executed.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "site", "assets", "data")

# These are the project's documented held-out validation metrics.
# They are metadata only; this exporter does not recompute cross-validation
# from the final MLPRegressor pickle.
VALIDATION_METRICS = {
    "mae": 8480,
    "rmse": 12023,
    "r2": 0.8967,
    "folds": 5,
}

NUMERIC_COLS = ["year", "hand", "engine_liters", "horsepower", "mileage"]
GROUP_PREFIXES = [
    "manufacturer_",
    "submodel_",
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
        "validation": VALIDATION_METRICS,
        "weights": [w.tolist() for w in model.coefs_],
        "biases": [b.tolist() for b in model.intercepts_],
        "groups": {p.rstrip("_"): [c[len(p):] for c in cols] for p, cols in groups.items()},
        "export_version": 2,
    }

    model_path = os.path.join(OUT_DIR, "model.json")
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model_json, f)
    print(f"Wrote {model_path}  ({os.path.getsize(model_path)/1024:.1f} KB)")

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

    model_manufacturer_map = {
        m: max(counts.items(), key=lambda kv: kv[1])[0]
        for m, counts in model_to_manufacturer.items()
    }
    model_submodel_map = {m: sorted(list(s)) for m, s in model_to_submodels.items()}

    all_manufacturers = sorted([c[len("manufacturer_"):] for c in groups["manufacturer_"]])
    manufacturer_to_models = {manu: [] for manu in all_manufacturers}
    for m, manu in model_manufacturer_map.items():
        if m == "Other":
            continue
        manufacturer_to_models.setdefault(manu, []).append(m)
    for manu in manufacturer_to_models:
        manufacturer_to_models[manu] = sorted(manufacturer_to_models[manu]) + ["Other"]

    prices = [r["price"] for r in records]
    mileages = [r["mileage"] for r in records]
    manufacturers = sorted({r["manufacturer"] for r in records})
    models = sorted({r["model"] for r in records})

    def mean(values):
        return float(sum(values) / len(values)) if values else 0.0

    def median(values):
        if not values:
            return 0.0
        ordered = sorted(values)
        n = len(ordered)
        mid = n // 2
        return float(ordered[mid]) if n % 2 else float((ordered[mid - 1] + ordered[mid]) / 2)

    def top_counts(key, limit=6):
        counts = {}
        for record in records:
            value = record[key]
            counts[value] = counts.get(value, 0) + 1
        return sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[:limit]

    dataset_json = {
        "records": records,
        "options": {
            "manufacturer": sorted([c[len("manufacturer_"):] for c in groups["manufacturer_"]]),
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
            "price_min": int(min(prices)),
            "price_max": int(max(prices)),
            "mileage_max": int(max(mileages)),
            "average_price": mean(prices),
            "median_price": median(prices),
            "average_mileage": mean(mileages),
            "manufacturer_count": len(manufacturers),
            "model_count": len(models),
            "top_manufacturers": top_counts("manufacturer"),
            "top_years": top_counts("year"),
        },
        "validation": VALIDATION_METRICS,
        "export_version": 2,
    }

    dataset_path = os.path.join(OUT_DIR, "dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f)
    print(f"Wrote {dataset_path}  ({os.path.getsize(dataset_path)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
