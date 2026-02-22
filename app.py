import streamlit as st
import joblib
import numpy as np
import os
from googletrans import Translator

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crop Recommendation System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- MOBILE CSS ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
.stButton>button {
    width: 100%;
    height: 3.2em;
    font-size: 18px;
    border-radius: 10px;
}
.stNumberInput input {
    font-size: 16px !important;
}
h1, h2, h3 {
    text-align: center;
}
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------- LANGUAGE TOGGLE ----------------
language = st.radio(
    "Language / اللغة",
    ["English", "عربية"],
    horizontal=True
)

# RTL for Arabic
if language == "عربية":
    st.markdown(
        """
        <style>
        body { direction: RTL; text-align: right; }
        </style>
        """,
        unsafe_allow_html=True
    )

translator = Translator()

def auto_translate(text):
    if language == "عربية":
        return translator.translate(text, dest='ar').text
    return text

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("crop_model.pkl")

model = load_model()

# ---------------- CROP LABELS ----------------
crop_dict = {
    0: "Rice", 1: "Maize", 2: "Chickpea", 3: "Kidneybeans", 4: "Pigeonpeas",
    5: "Mothbeans", 6: "Mungbean", 7: "Blackgram", 8: "Lentil",
    9: "Pomegranate", 10: "Banana", 11: "Mango", 12: "Grapes",
    13: "Watermelon", 14: "Muskmelon", 15: "Apple", 16: "Orange",
    17: "Papaya", 18: "Coconut", 19: "Cotton", 20: "Jute", 21: "Coffee"
}

# ---------------- CROP INFO ----------------
# (Keep your full crop_info dictionary here exactly as provided earlier)
crop_info = { ... }

# ---------------- HEADER ----------------
st.title(auto_translate("🌱 AI Crop Recommendation System"))
st.write(auto_translate("Enter soil nutrients and environmental conditions:"))

st.markdown("---")

# ---------------- INPUTS WITH UNITS ----------------
N = st.number_input(
    auto_translate("Nitrogen (N) [kg/ha]"),
    min_value=0,
    step=1
)

P = st.number_input(
    auto_translate("Phosphorus (P) [kg/ha]"),
    min_value=0,
    step=1
)

K = st.number_input(
    auto_translate("Potassium (K) [kg/ha]"),
    min_value=0,
    step=1
)

temperature = st.number_input(
    auto_translate("Temperature [°C]"),
    step=0.1
)

humidity = st.number_input(
    auto_translate("Humidity [%]"),
    min_value=0.0,
    max_value=100.0,
    step=0.1
)

ph = st.number_input(
    auto_translate("Soil pH [0–14]"),
    min_value=0.0,
    max_value=14.0,
    step=0.1
)

rainfall = st.number_input(
    auto_translate("Rainfall [mm]"),
    min_value=0.0,
    step=1.0
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if st.button(auto_translate("🔍 Recommend Crop")):

    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    crop_name = crop_dict[int(prediction[0])]
    info = crop_info[crop_name]

    st.markdown("---")
    st.subheader(auto_translate("✅ Recommended Crop"))

    # Mobile friendly result layout
    if os.path.exists(info["image"]):
        st.image(info["image"], use_container_width=True)

    st.markdown(f"### {auto_translate(crop_name)}")
    st.write(auto_translate(info["desc"]))

    st.success(
        auto_translate(f"{crop_name} is suitable based on the given soil and climate conditions.")
    )
