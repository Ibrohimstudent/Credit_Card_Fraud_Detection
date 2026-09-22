import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ------------------------------------------------------------------
# SOZLAMALAR — kerak bo'lsa o'zgartiring
# ------------------------------------------------------------------
DATA_PATH = "creditcard.csv"        # creditcard.csv shu fayl bilan bir papkada bo'lishi kerak
MODEL_PATH = "xgb_fraud_model.joblib"

st.set_page_config(page_title="Kredit karta firibgarligini aniqlash", page_icon="💳", layout="centered")


@st.cache_resource
def load_or_train_model():
    """Modelni diskdan yuklaydi, topilmasa datasetdan o'qitib saqlab qo'yadi."""
    if not os.path.exists(DATA_PATH):
        st.error(
            f"'{DATA_PATH}' topilmadi. Kaggle'dan creditcard.csv faylini yuklab, "
            "shu skript bilan bir papkaga joylashtiring."
        )
        st.stop()

    df = pd.read_csv(DATA_PATH)

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        X = df.drop(columns=["Class"])
        y = df["Class"]
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)

    return model, df


model, df = load_or_train_model()
feature_cols = [c for c in df.columns if c != "Class"]
v_cols = [c for c in feature_cols if c.startswith("V")]

st.title("💳 Kredit karta firibgarligini aniqlash")
st.write(
    "Tranzaksiya ma'lumotlarini kiriting — model uni **oddiy** yoki "
    "**firibgarlik (fraud)** sifatida bashorat qiladi (XGBoost, test aniqligi ≈ 99.9%)."
)

st.subheader("Asosiy qiymatlar")
col1, col2 = st.columns(2)
with col1:
    time_val = st.number_input("Time (sekundlarda)", value=float(df["Time"].mean()))
with col2:
    amount_val = st.number_input("Amount ($)", value=float(df["Amount"].mean()), min_value=0.0)

with st.expander("V1–V28 (PCA komponentlari) — ixtiyoriy, kerak bo'lsa o'zgartiring"):
    st.caption("Bu ustunlar dataset yaratuvchilari tomonidan maxfiylik uchun PCA bilan o'zgartirilgan.")
    v_values = {}
    cols = st.columns(4)
    for i, col_name in enumerate(v_cols):
        with cols[i % 4]:
            v_values[col_name] = st.number_input(
                col_name, value=float(df[col_name].mean()), format="%.4f", key=col_name
            )

if st.button("Bashorat qilish", type="primary"):
    input_data = {"Time": time_val, **v_values, "Amount": amount_val}
    input_df = pd.DataFrame([input_data])[feature_cols]

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.divider()
    if pred == 1:
        st.error(f"⚠️ Firibgarlik (Fraud) ehtimoli yuqori! Ehtimollik: **{proba:.4%}**")
    else:
        st.success(f"✅ Oddiy tranzaksiya. Firibgarlik ehtimoli: **{proba:.4%}**")

st.divider()
st.caption(
    "Ishga tushirish: `pip install streamlit xgboost pandas scikit-learn joblib` "
    "va `streamlit run fraud_app.py`"
)
