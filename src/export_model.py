"""
export_model.py
----------------
Exports the trained scikit-learn MLPRegressor and its preprocessing assets
into JSON files consumed by the static GitHub Pages website.

IMPORTANT: output paths are anchored to the repository root, so running
`python src/export_model.py` from either the repository root or src/ produces
the same website assets.
"""
import json
import os
import numpy as np
import pandas as pd
import joblib

from path import *

REPO_DIR = BASE_DIR
SITE_DIR = os.path.join(REPO_DIR, "site") if os.path.isdir(os.path.join(REPO_DIR, "site")) else REPO_DIR
OUT_DIR = os.path.join(SITE_DIR, "assets", "data")

NUMERIC_COLS = ["year", "hand", "engine_liters", "horsepower", "mileage"]
GROUP_PREFIXES = [
    "manufacturer_", "submodel_", "model_", "fuel_", "transmission_", "drive_type_"
]

# These are the project's documented 5-fold cross-validation results.
# They describe validation performance, not a confidence interval.
VALIDATION_METRICS = {
    "mae": 8480,
    "rmse": 12023,
    "r2": 0.8967,
    "method": "5-fold cross-validation",
}


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
        "validation": VALIDATION_METRICS,
        "export_version": 2,
    }

    model_path = os.path.join(OUT_DIR, "model.json")
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model_json, f, separators=(",", ":"))
    print(f"Wrote {model_path} ({os.path.getsize(model_path) / 1024:.1f} KB)")

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

        records.append({
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
        })

        model_to_manufacturer.setdefault(model_name, {})
        model_to_manufacturer[model_name][manufacturer] = (
            model_to_manufacturer[model_name].get(manufacturer, 0) + 1
        )
        model_to_submodels.setdefault(model_name, set()).add(submodel)

    model_manufacturer_map = {
        m: max(counts.items(), key=lambda kv: kv[1])[0]
        for m, counts in model_to_manufacturer.items()
    }
    model_submodel_map = {m: sorted(s) for m, s in model_to_submodels.items()}

    all_manufacturers = sorted(c[len("manufacturer_"):] for c in groups["manufacturer_"])
    manufacturer_to_models = {manu: [] for manu in all_manufacturers}
    for m, manu in model_manufacturer_map.items():
        if m != "Other":
            manufacturer_to_models.setdefault(manu, []).append(m)
    for manu in manufacturer_to_models:
        manufacturer_to_models[manu] = sorted(set(manufacturer_to_models[manu])) + ["Other"]

    prices = [r["price"] for r in records]
    sorted_prices = sorted(prices)
    median = (
        sorted_prices[len(sorted_prices) // 2]
        if len(sorted_prices) % 2
        else (sorted_prices[len(sorted_prices) // 2 - 1] + sorted_prices[len(sorted_prices) // 2]) / 2
    )

    def top_counts(key, limit=6):
        counts = {}
        for r in records:
            counts[r[key]] = counts.get(r[key], 0) + 1
        return sorted(counts.items(), key=lambda x: (-x[1], str(x[0])))[:limit]

    dataset_json = {
        "records": records,
        "options": {
            "manufacturer": sorted(c[len("manufacturer_"):] for c in groups["manufacturer_"]),
            "model": sorted(c[len("model_"):] for c in groups["model_"]),
            "submodel": sorted(c[len("submodel_"):] for c in groups["submodel_"]),
            "fuel": sorted(c[len("fuel_"):] for c in groups["fuel_"]),
            "transmission": sorted(c[len("transmission_"):] for c in groups["transmission_"]),
            "drive_type": sorted(c[len("drive_type_"):] for c in groups["drive_type_"]),
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
            "mileage_max": int(max(r["mileage"] for r in records)),
            "average_price": float(np.mean(prices)),
            "median_price": float(median),
            "average_mileage": float(np.mean([r["mileage"] for r in records])),
            "manufacturer_count": len(set(r["manufacturer"] for r in records)),
            "model_count": len(set(r["model"] for r in records)),
            "top_manufacturers": top_counts("manufacturer"),
            "top_years": top_counts("year"),
        },
        "validation": VALIDATION_METRICS,
    }

    dataset_path = os.path.join(OUT_DIR, "dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, separators=(",", ":"))
    print(f"Wrote {dataset_path} ({os.path.getsize(dataset_path) / 1024:.1f} KB)")
    print(f"Website data directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
