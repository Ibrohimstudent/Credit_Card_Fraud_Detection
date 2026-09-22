import pandas as pd
import joblib
import streamlit as st

# ------------------------------------------------------------------
# Bu versiya creditcard.csv'ga muhtoj emas — model oldindan o'qitilgan
# holda "xgb_fraud_model.joblib" fayli sifatida repo'ga qo'shilishi kerak.
# ------------------------------------------------------------------
MODEL_PATH = "xgb_fraud_model.joblib"

st.set_page_config(page_title="Kredit karta firibgarligini aniqlash", page_icon="💳", layout="centered")

# Dataset ustunlari tartibi (creditcardfraud datasetidagi asl tartib)
FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Har bir ustun uchun taxminiy o'rtacha qiymatlar (default sifatida ko'rsatiladi).
# V1-V28 PCA komponentlari bo'lgani uchun o'rtachasi ~0 ga yaqin.
DEFAULTS = {"Time": 94813.0, "Amount": 88.35}
DEFAULTS.update({f"V{i}": 0.0 for i in range(1, 29)})


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(
            f"'{MODEL_PATH}' topilmadi. Modelni oldindan o'qitib, shu fayl bilan "
            "bir papkaga (repo'ga) qo'shishingiz kerak."
        )
        st.stop()


model = load_model()
v_cols = [c for c in FEATURE_COLS if c.startswith("V")]

st.title("💳 Kredit karta firibgarligini aniqlash")
st.write(
    "Tranzaksiya ma'lumotlarini kiriting — model uni **oddiy** yoki "
    "**firibgarlik (fraud)** sifatida bashorat qiladi (XGBoost)."
)

st.subheader("Asosiy qiymatlar")
col1, col2 = st.columns(2)
with col1:
    time_val = st.number_input("Time (sekundlarda)", value=DEFAULTS["Time"])
with col2:
    amount_val = st.number_input("Amount ($)", value=DEFAULTS["Amount"], min_value=0.0)

with st.expander("V1–V28 (PCA komponentlari) — ixtiyoriy, kerak bo'lsa o'zgartiring"):
    st.caption("Bu ustunlar dataset yaratuvchilari tomonidan maxfiylik uchun PCA bilan o'zgartirilgan.")
    v_values = {}
    cols = st.columns(4)
    for i, col_name in enumerate(v_cols):
        with cols[i % 4]:
            v_values[col_name] = st.number_input(
                col_name, value=DEFAULTS[col_name], format="%.4f", key=col_name
            )

if st.button("Bashorat qilish", type="primary"):
    input_data = {"Time": time_val, **v_values, "Amount": amount_val}
    input_df = pd.DataFrame([input_data])[FEATURE_COLS]

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.divider()
    if pred == 1:
        st.error(f"⚠️ Firibgarlik (Fraud) ehtimoli yuqori! Ehtimollik: **{proba:.4%}**")
    else:
        st.success(f"✅ Oddiy tranzaksiya. Firibgarlik ehtimoli: **{proba:.4%}**")
