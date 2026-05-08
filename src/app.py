import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import subprocess
import sys
import threading

from path import *

FEEDBACK_PATH = os.path.join(DATA_DIR, 'feedback.csv')

# Feedback CSV columns — must match autoboom_raw.csv structure
FEEDBACK_COLS = ['manufacturer', 'model', 'year', 'price', 'hand',
                 'fuel', 'engine_liters', 'horsepower', 'transmission', 'mileage', 'source']

st.set_page_config(page_title="חיזוי מחיר רכב מדויק", page_icon="🚗", layout="centered")

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

    mfg_to_models = {}
    for mfg in manufacturers:
        mfg_col = f'manufacturer_{mfg}'
        rows    = df[df[mfg_col] == 1]
        mfg_to_models[mfg] = sorted([
            m for m in models_all
            if f'model_{m}' in df.columns and rows[f'model_{m}'].sum() > 0
        ])

    return mfg_to_models, fuels, transmissions

@st.cache_data
def get_feature_columns(_model):
    return list(_model.get_booster().feature_names)

def save_feedback(row: dict, actual_price: float):
    """Append one corrected row to feedback.csv."""
    row['price'] = actual_price
    row['source'] = 'feedback'
    df_new = pd.DataFrame([row], columns=FEEDBACK_COLS)

    if os.path.exists(FEEDBACK_PATH):
        df_existing = pd.read_csv(FEEDBACK_PATH)
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(FEEDBACK_PATH, index=False)

def count_feedback():
    if not os.path.exists(FEEDBACK_PATH):
        return 0
    return len(pd.read_csv(FEEDBACK_PATH))

def run_retrain():
    """Run retrain.py in a subprocess and stream output to a file."""
    log_path = os.path.join(DATA_DIR, 'retrain_log.txt')
    script   = os.path.join(os.path.dirname(__file__), 'retrain.py')
    with open(log_path, 'w', encoding='utf-8') as log:
        subprocess.run([sys.executable, script], stdout=log, stderr=log)

# ─── Main UI ──────────────────────────────────────────────────────────────────
try:
    model, price_scaler, features_scaler = load_assets()
    mfg_to_models, fuels, transmissions  = load_manufacturer_model_map()
    feature_columns = get_feature_columns(model)
    manufacturers   = sorted(mfg_to_models.keys())

    st.title("🚗 מחשבון חיזוי מחיר רכב")

    # ── Manufacturer + model selectors ────────────────────────────────────────
    col_top1, col_top2 = st.columns(2)
    with col_top1:
        selected_mfg = st.selectbox("יצרן", manufacturers)
    with col_top2:
        available_models = mfg_to_models.get(selected_mfg, [])
        selected_model   = st.selectbox("דגם", available_models)

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            year     = st.number_input("שנת ייצור", 2000, 2025, 2020)
            hand     = st.number_input("יד", 1, 10, 1)
            engine   = st.number_input("נפח מנוע (ליטר)", 0.1, 6.0, 1.6, step=0.1)
        with col2:
            hp       = st.number_input("כוחות סוס", 50, 600, 110)
            mileage  = st.number_input("קילומטראז'", 0, 500000, 50000, step=1000)
            fuel         = st.selectbox("סוג דלק", fuels)
            transmission = st.selectbox("תיבת הילוכים", transmissions)

        submit = st.form_submit_button("💰 חשב מחיר")

    # ── Prediction ────────────────────────────────────────────────────────────
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
            ('fuel',         fuel),
            ('transmission', transmission),
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

        st.success(f"### המחיר המוערך: ₪{round(final_price):,}")
        st.balloons()

        # ── Feedback section ──────────────────────────────────────────────────
        st.divider()
        st.subheader("📊 האם המחיר מדויק?")

        feedback_col1, feedback_col2 = st.columns(2)

        with feedback_col1:
            if st.button("✅ כן, המחיר נכון"):
                # Save the prediction itself as confirmed feedback
                save_feedback(
                    row=dict(manufacturer=selected_mfg, model=selected_model,
                             year=year, hand=hand, fuel=fuel,
                             engine_liters=engine, horsepower=hp,
                             transmission=transmission, mileage=mileage),
                    actual_price=round(final_price)
                )
                st.toast("תודה! המשוב נשמר ✓", icon="✅")

        with feedback_col2:
            with st.expander("❌ לא, המחיר שגוי — הכנס מחיר אמיתי"):
                actual = st.number_input("המחיר האמיתי (₪)", min_value=5000,
                                         max_value=2000000, step=1000, key="actual_price")
                if st.button("שמור משוב"):
                    save_feedback(
                        row=dict(manufacturer=selected_mfg, model=selected_model,
                                 year=year, hand=hand, fuel=fuel,
                                 engine_liters=engine, horsepower=hp,
                                 transmission=transmission, mileage=mileage),
                        actual_price=actual
                    )
                    st.toast(f"תודה! שמרנו את המחיר האמיתי: ₪{actual:,} ✓", icon="💾")

    # ── Admin: Retrain section ─────────────────────────────────────────────────
    st.divider()
    with st.expander("⚙️ ניהול מודל"):
        n_feedback = count_feedback()
        st.write(f"**משובים שנאספו עד כה:** {n_feedback}")

        if n_feedback == 0:
            st.info("אין עדיין משובים — אסוף לפחות כמה לפני אימון מחדש.")

        col_r1, col_r2 = st.columns([2, 1])
        with col_r1:
            retrain_clicked = st.button(
                "🔄 אמן מחדש את המודל",
                disabled=(n_feedback == 0),
                help="ימזג את המשובים עם הנתונים המקוריים ויריץ את כל הפייפליין מחדש"
            )
        with col_r2:
            log_path = os.path.join(DATA_DIR, 'retrain_log.txt')
            if os.path.exists(log_path):
                with open(log_path, encoding='utf-8') as f:
                    log_text = f.read()
                st.download_button("📄 הורד לוג", data=log_text,
                                   file_name="retrain_log.txt", mime="text/plain")

        if retrain_clicked:
            st.warning(
                "⏳ האימון מחדש החל ברקע. התהליך לוקח **30-90 דקות**.\n\n"
                "תוכל לסגור את הדף — המודל יישמר אוטומטית כשהאימון יסתיים.\n"
                "לאחר הסיום, רענן את הדף כדי להשתמש במודל החדש."
            )
            # Run in background thread so Streamlit doesn't freeze
            t = threading.Thread(target=run_retrain, daemon=True)
            t.start()
            st.session_state['retraining'] = True

        if st.session_state.get('retraining'):
            st.info("🔄 האימון רץ ברקע...")

except Exception as e:
    st.error(f"Error: {e}")