import streamlit as st
import joblib
import numpy as np
crop_dict = {
    0: "Rice",
    1: "Maize",
    2: "Chickpea",
    3: "Kidneybeans",
    4: "Pigeonpeas",
    5: "Mothbeans",
    6: "Mungbean",
    7: "Blackgram",
    8: "Lentil",
    9: "Pomegranate",
    10: "Banana",
    11: "Mango",
    12: "Grapes",
    13: "Watermelon",
    14: "Muskmelon",
    15: "Apple",
    16: "Orange",
    17: "Papaya",
    18: "Coconut",
    19: "Cotton",
    20: "Jute",
    21: "Coffee"
}

# Load trained model
model = joblib.load("crop_model.pkl")

st.set_page_config(page_title="Crop Recommendation System")

st.title("🌱 Crop Recommendation System")

st.write("Enter soil and climate parameters:")

N = st.number_input("Nitrogen (N)", min_value=0)
P = st.number_input("Phosphorus (P)", min_value=0)
K = st.number_input("Potassium (K)", min_value=0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("Soil pH")
rainfall = st.number_input("Rainfall (mm)")

if st.button("Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    crop_name = crop_dict[int(prediction[0])]
    st.success(f"✅ Recommended Crop: **{crop_name}**")
    st.success(f"✅ Recommended Crop: **{prediction[0]}**")