import streamlit as st
import numpy as np
import joblib
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="Smart Fertilizer Recommendation",
    page_icon="🌱",
    layout="centered"
)

st.markdown("""
<style>
body {
    background: linear-gradient(120deg,#e0f7fa,#e8f5e9);
}
.main {
    background-color: #ffffff;
    padding: 30px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("🌾 Smart Fertilizer Recommendation System")
st.caption("AI-powered recommendation using Deep Learning")

# ---------------- LOAD MODELS ----------------
tabnet = TabNetClassifier()
tabnet.load_model("tabnet_model.zip")

scaler = joblib.load("scaler.pkl")
le_target = joblib.load("target_encoder.pkl")
cat_cols = joblib.load("categorical_cols.pkl")
num_cols = joblib.load("numerical_cols.pkl")

# ---------------- USER INPUT ----------------
st.subheader("🧪 Enter Soil & Crop Details")

user_input = {}

for col in cat_cols:
    user_input[col] = st.text_input(f"{col}")

for col in num_cols:
    user_input[col] = st.number_input(f"{col}", value=0.0)

# ---------------- PREDICTION ----------------
if st.button("🌱 Recommend Fertilizer"):
    df = pd.DataFrame([user_input])

    # Encode categorical
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes

    df[num_cols] = scaler.transform(df[num_cols])

    prediction = tabnet.predict(df.values)
    fertilizer = le_target.inverse_transform(prediction)[0]

    st.success(f"✅ **Recommended Fertilizer:** {fertilizer}")