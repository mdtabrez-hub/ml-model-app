import os
import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Crop Recommendation System", layout="centered")

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
crop_info = {
    "Rice": {"image": "images/rice.jpeg", "desc": "Needs high rainfall and warm climate. Suitable for clayey soil."},
    "Maize": {"image": "images/maize.jpeg", "desc": "Grows best in warm weather with moderate rainfall."},
    "Chickpea": {"image": "images/chickpea.jpeg", "desc": "Rabi crop requiring cool climate and low humidity."},
    "Kidneybeans": {"image": "images/kidneybeans.jpeg", "desc": "Needs moderate rainfall and well-drained soil."},
    "Pigeonpeas": {"image": "images/pigeaonpeas.jpeg", "desc": "Drought-resistant crop grown in semi-arid regions."},
    "Mothbeans": {"image": "images/mothbean.jpeg", "desc": "Suitable for dry areas with sandy soil."},
    "Mungbean": {"image": "images/mungbean.jpeg", "desc": "Short duration pulse crop for warm climate."},
    "Blackgram": {"image": "images/blackgram.jpeg", "desc": "Needs warm conditions and fertile loamy soil."},
    "Lentil": {"image": "images/lentil.jpeg", "desc": "Cool-season crop requiring less water."},
    "Pomegranate": {"image": "images/pomogranate.jpeg", "desc": "Fruit crop suited for arid and semi-arid climate."},
    "Banana": {"image": "images/banana.jpeg", "desc": "Needs high humidity, temperature, and rainfall."},
    "Mango": {"image": "images/mango.jpeg", "desc": "Tropical fruit requiring warm climate and moderate rainfall."},
    "Grapes": {"image": "images/grapes.jpeg", "desc": "Needs dry climate during fruit development."},
    "Watermelon": {"image": "images/watermelon.jpeg", "desc": "Grows best in warm temperature and sandy soil."},
    "Muskmelon": {"image": "images/muskmelon.jpeg", "desc": "Requires warm dry climate and good sunlight."},
    "Apple": {"image": "images/apple.jpeg", "desc": "Needs cold climate and well-drained soil."},
    "Orange": {"image": "images/orange.jpeg", "desc": "Subtropical crop needing moderate rainfall."},
    "Papaya": {"image": "images/papaya.jpeg", "desc": "Grows in tropical climate with good drainage."},
    "Coconut": {"image": "images/coconut.jpeg", "desc": "Needs coastal humid climate and sandy soil."},
    "Cotton": {"image": "images/cotton.jpeg", "desc": "Requires warm climate and black soil."},
    "Jute": {"image": "images/jute.jpeg", "desc": "Needs hot humid climate and heavy rainfall."},
    "Coffee": {"image": "images/coffee.jpeg", "desc": "Grows in cool tropical hills with shade."}
}

# ---------------- UI HEADER ----------------
st.title("🌱 AI Crop Recommendation System")
st.write("Enter soil nutrients and environmental conditions:")

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0)
    P = st.number_input("Phosphorus (P)", min_value=0)
    K = st.number_input("Potassium (K)", min_value=0)
    temperature = st.number_input("Temperature (°C)")

with col2:
    humidity = st.number_input("Humidity (%)")
    ph = st.number_input("Soil pH")
    rainfall = st.number_input("Rainfall (mm)")

# ---------------- PREDICT BUTTON ----------------
if st.button("🔍 Recommend Crop"):

    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    crop_name = crop_dict[int(prediction[0])]

    # ---------- RESULT CARD ----------
    st.markdown("---")
    st.subheader("✅ Recommended Crop")

    info = crop_info.get(crop_name, None)

    col_img, col_text = st.columns([1, 2])

    with col_img:
        if info:
            st.image(info["image"], use_container_width=True)

    with col_text:
        st.markdown(f"### 🌾 {crop_name}")
        if info:
            st.write(info["desc"])
        else:
            st.write("No description available.")

    # ---------- OPTIONAL NICE SUCCESS BOX ----------
    st.success(f"{crop_name} is suitable based on given soil and climate conditions.")