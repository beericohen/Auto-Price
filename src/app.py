import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# הגדרות עמוד
st.set_page_config(page_title="חיזוי מחיר רכב מדויק", page_icon="🚗", layout="centered")

# נתיבים - וודא שהם תואמים למבנה שלך
# הגדרות נתיבים מתוקנות
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) # הולך רמה אחת אחורה מ-src לתיקייה הראשית

MODEL_PATH = os.path.join(BASE_DIR, "Models", "xgb_fine_tuned.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessing.csv")
PRICE_SCALER_PATH = os.path.join(BASE_DIR, "data", "minmax_scaler.pkl")
FEATURES_SCALER_PATH = os.path.join(BASE_DIR, "data", "scaler.pkl")

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    price_scaler = joblib.load(PRICE_SCALER_PATH)
    features_scaler = joblib.load(FEATURES_SCALER_PATH)
    return model, price_scaler, features_scaler

@st.cache_data
def get_options():
    df = pd.read_csv(DATA_PATH, nrows=0, index_col=False)
    cols = [c for c in df.columns if c not in ['Unnamed: 0', 'price']]
    
    manufacturers = sorted([c.replace('manufacturer_', '') for c in cols if 'manufacturer_' in c])
    models_list = sorted([c.replace('model_', '') for c in cols if 'model_' in c])
    fuels = sorted([c.replace('fuel_', '') for c in cols if 'fuel_' in c])
    transmissions = sorted([c.replace('transmission_', '') for c in cols if 'transmission_' in c])
    
    return manufacturers, models_list, fuels, transmissions, cols

try:
    model, price_scaler, features_scaler = load_assets()
    manufacturers, models_list, fuels, transmissions, feature_columns = get_options()

    st.title("🚗 מחשבון חיזוי מחיר רכב")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            selected_mfg = st.selectbox("יצרן", manufacturers)
            selected_model = st.selectbox("דגם", models_list)
            year = st.number_input("שנת ייצור", 2000, 2025, 2020)
            hand = st.number_input("יד", 1, 10, 1)
        with col2:
            engine = st.number_input("נפח מנוע", 0.1, 6.0, 1.6)
            hp = st.number_input("כוחות סוס", 50, 600, 110)
            mileage = st.number_input("קילומטראז'", 0, 500000, 50000)
            fuel = st.selectbox("סוג דלק", fuels)
            transmission = st.selectbox("תיבת הילוכים", transmissions)

        submit = st.form_submit_button("חשב מחיר")

    if submit:
        # 1. בניית הנתונים הגולמיים
        input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        # הגדרת העמודות המספריות שעוברות Scaling
        num_cols = ['year', 'hand', 'engine_liters', 'horsepower', 'mileage']
        
        input_df['year'] = year
        input_df['hand'] = hand
        input_df['engine_liters'] = engine
        input_df['horsepower'] = hp
        input_df['mileage'] = mileage
        
        # עדכון One-Hot (נשארים 0 או 1, לא עוברים Scaling)
        for prefix, val in [('manufacturer', selected_mfg), ('model', selected_model), 
                            ('fuel', fuel), ('transmission', transmission)]:
            col_name = f"{prefix}_{val}"
            if col_name in input_df.columns:
                input_df[col_name] = 1

        # 2. התיקון: נרמול רק של העמודות המספריות!
        # אנחנו מעדכנים רק את 5 העמודות הראשונות בתוך ה-DataFrame
        input_df[num_cols] = features_scaler.transform(input_df[num_cols])

        # 3. חיזוי (המודל מקבל את כל 75 העמודות כשהמספריות מנורמלות)
        scaled_prediction = model.predict(input_df)
        
        # 4. המרה חזרה למחיר שקלי
        prediction_reshaped = np.array(scaled_prediction).reshape(-1, 1)
        final_price = price_scaler.inverse_transform(prediction_reshaped)[0][0]

        st.success(f"### המחיר המוערך: ₪{round(final_price):,}")
        st.balloons()

except Exception as e:
    st.error(f"שגיאה: {e}")