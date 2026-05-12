import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import subprocess
import sys
import threading

from path import *

MAE_THRESHOLD = 8928  # Update this after retraining — current model MAE in ILS



st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

@st.cache_resource
def load_assets():
    model         = joblib.load(MODEL_PATH)
    price_scaler  = joblib.load(PRICE_SCALER_PATH)
    feat_scaler   = joblib.load(FEATURES_SCALER_PATH)
    return model, price_scaler, feat_scaler

@st.cache_data
def load_manufacturer_model_map():
    df   = pd.read_csv(DATA_PATH)
    cols = df.columns.tolist()

    manufacturers  = sorted([c.replace('manufacturer_', '') for c in cols if c.startswith('manufacturer_')])
    models_all     = [c.replace('model_', '') for c in cols if c.startswith('model_')]
    fuels          = sorted([c.replace('fuel_', '') for c in cols if c.startswith('fuel_')])
    transmissions  = sorted([c.replace('transmission_', '') for c in cols if c.startswith('transmission_')])
    drive_types  = sorted([c.replace('drive_type_', '') for c in cols if c.startswith('drive_type_')])


    mfg_to_models = {}
    for mfg in manufacturers:
        mfg_col = f'manufacturer_{mfg}'
        rows    = df[df[mfg_col] == 1]
        mfg_to_models[mfg] = sorted([
            m for m in models_all
            if f'model_{m}' in df.columns and rows[f'model_{m}'].sum() > 0
        ])

    return mfg_to_models, fuels, transmissions, drive_types


def load_submodels():
    df   = pd.read_csv(DATA_PATH)
    cols = df.columns.tolist()


    models  = sorted([c.replace('model_', '') for c in cols if c.startswith('model_')])
    submodels_all     = [c.replace('submodel_', '') for c in cols if c.startswith('submodel_')]
    model_to_submodels = {}

    for model in models:
        model_col = f'model_{model}'
        rows    = df[df[model_col] == 1]
        model_to_submodels[model] = sorted([
            m for m in submodels_all
            if f'submodel_{m}' in df.columns and rows[f'submodel_{m}'].sum() > 0
        ])

    return model_to_submodels


@st.cache_data
def get_feature_columns(_model):
    return list(_model.get_booster().feature_names)



# ─── Main UI ──────────────────────────────────────────────────────────────────
try:
    model, price_scaler, features_scaler = load_assets()
    mfg_to_models, fuels, transmissions, drive_types  = load_manufacturer_model_map()
    model_to_submodels = load_submodels()
    feature_columns = get_feature_columns(model)
    manufacturers   = sorted(mfg_to_models.keys())

    st.title("Car Price Predictor 🚗")

    # ── Manufacturer + model  + submodel selectors ────────────────────────────────────────
    col_top1, col_top2, col_top3 = st.columns(3)
    with col_top1:
        selected_mfg = st.selectbox("Manufacturer", manufacturers)
    with col_top2:
        available_models = mfg_to_models.get(selected_mfg, [])
        selected_model   = st.selectbox("Model", available_models)
    with col_top3:
        available_submodels = model_to_submodels.get(selected_model, [])
        selected_submodel   = st.selectbox("Submodel", available_submodels)

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            year     = st.number_input("Year", 2000, 2025, 2020)
            hand     = st.number_input("Hand", 1, 10, 1)
            engine   = st.number_input("Engine Size (Liters)", 0.1, 6.0, 1.6, step=0.1)
            drive = st.selectbox("Drive Type", drive_types)
        with col2:
            hp       = st.number_input("Horsepower", 50, 600, 110)
            mileage  = st.number_input("Mileage", 0, 500000, 50000, step=1000)
            fuel         = st.selectbox("Fuel Type", fuels)
            transmission = st.selectbox("Transmission", transmissions)

        submit = st.form_submit_button("💰 Calculate Price")

    # ── On submit: run prediction and store everything in session_state ────────
    if submit:
        input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

        num_cols = ['year', 'hand', 'engine_liters', 'horsepower', 'mileage']
        input_df['year']          = year
        input_df['hand']          = hand
        input_df['engine_liters'] = engine
        input_df['horsepower']    = hp
        input_df['mileage']       = mileage

        for prefix, val in [
            ('manufacturer', selected_mfg),
            ('model',        selected_model),
            ('submodel',        selected_submodel),
            ('fuel',         fuel),
            ('transmission', transmission),
            ('drive_type', drive)
        ]:
            col_name = f"{prefix}_{val}"
            if col_name in input_df.columns:
                input_df[col_name] = 1

        source_col = f"source_{selected_mfg.lower()}"
        if source_col in input_df.columns:
            input_df[source_col] = 1

        input_df[num_cols] = features_scaler.transform(input_df[num_cols])

        scaled_pred = model.predict(input_df)
        final_price = price_scaler.inverse_transform(
            np.array(scaled_pred).reshape(-1, 1)
        )[0][0]

        # Store result in session_state so it survives button clicks
        st.session_state['prediction'] = round(final_price)
        st.session_state['car_row'] = dict(
            manufacturer=selected_mfg, model=selected_model, submodel = selected_submodel,
            year=year, hand=hand, fuel=fuel,
            engine_liters=engine, horsepower=hp,
            transmission=transmission, mileage=mileage, drive_type = drive
        )

        # ── Show prediction ────
    if 'prediction' in st.session_state:
        final_price = st.session_state['prediction']
        car_row     = st.session_state['car_row']

        st.success(f"### Estimated Price: ₪{final_price:,}")


except Exception as e:
    st.error(f"Error: {e}")