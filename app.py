import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Crop Recommendation System", layout="centered")

# ---------------- LANGUAGE TOGGLE ----------------
if "language" not in st.session_state:
    st.session_state.language = "English"

if st.button("🌐 Translate"):
    st.session_state.language = "Arabic" if st.session_state.language == "English" else "English"

language = st.session_state.language

# RTL support for Arabic
if language == "Arabic":
    st.markdown(
        """
        <style>
        body { direction: RTL; text-align: right; }
        </style>
        """,
        unsafe_allow_html=True
    )

def tr(en, ar):
    return ar if language == "Arabic" else en

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

# ---------------- CROP INFORMATION ----------------
crop_info = {

    "Rice": {
        "image": "images/rice.jpeg",
        "desc": "Rice is a staple cereal crop cultivated mainly in tropical and subtropical regions. It requires high rainfall, warm temperatures (20–35°C), and clayey or loamy soils capable of retaining water under flooded conditions. Proper nitrogen management and irrigation control significantly improve yield and grain quality."
    },

    "Maize": {
        "image": "images/maize.jpeg",
        "desc": "Maize is a versatile cereal crop grown across diverse agro-climatic zones. It thrives in well-drained loamy soils with moderate rainfall and temperatures between 18–27°C. Balanced fertilization, especially nitrogen and phosphorus, plays a crucial role in maximizing productivity."
    },

    "Chickpea": {
        "image": "images/chickpea.jpeg",
        "desc": "Chickpea is a cool-season pulse crop typically grown during the rabi season. It prefers well-drained sandy-loam soils and moderate temperatures (20–25°C) with low humidity. As a legume, it enhances soil fertility through biological nitrogen fixation."
    },

    "Kidneybeans": {
        "image": "images/kidneybeans.jpeg",
        "desc": "Kidney beans are warm-season legumes requiring fertile, well-drained soils and moderate rainfall. Optimal growth occurs between 18–30°C. They are valued for their high protein content and contribution to soil nitrogen enrichment."
    },

    "Pigeonpeas": {
        "image": "images/pigeaonpeas.jpeg",
        "desc": "Pigeonpea is a drought-tolerant pulse crop suited for semi-arid and tropical climates. It grows well in loamy soils under moderate rainfall and temperatures of 20–30°C. The crop supports sustainable agriculture through nitrogen fixation."
    },

    "Mothbeans": {
        "image": "images/mothbean.jpeg",
        "desc": "Mothbean is a hardy, drought-resistant crop adapted to arid and semi-arid regions. It performs well in sandy soils with minimal rainfall and high temperatures. Its resilience makes it ideal for dryland farming systems."
    },

    "Mungbean": {
        "image": "images/mungbean.jpeg",
        "desc": "Mungbean is a short-duration pulse crop suited for warm climates and moderate rainfall. It grows best in well-drained loamy soils with temperatures between 25–35°C. It improves soil fertility by fixing atmospheric nitrogen."
    },

    "Blackgram": {
        "image": "images/blackgram.jpeg",
        "desc": "Blackgram is a warm-season legume that thrives in fertile loamy soils with moderate moisture. Ideal temperatures range between 25–35°C. It plays an important role in crop rotation and sustainable soil management."
    },

    "Lentil": {
        "image": "images/lentil.jpeg",
        "desc": "Lentil is a cool-season pulse crop cultivated in regions with low to moderate rainfall. It prefers well-drained soils and temperatures of 15–25°C. Lentils enhance soil nitrogen levels and are nutritionally rich in protein."
    },

    "Pomegranate": {
        "image": "images/pomogranate.jpeg",
        "desc": "Pomegranate is a drought-tolerant fruit crop suited for arid and semi-arid climates. It grows well in well-drained loamy soils and moderate temperatures. Proper irrigation management enhances fruit size and sweetness."
    },

    "Banana": {
        "image": "images/banana.jpeg",
        "desc": "Banana is a tropical crop requiring high humidity, consistent irrigation, and temperatures between 25–35°C. It thrives in fertile, well-drained soils rich in organic matter. Adequate nutrient supply is essential for high yield."
    },

    "Mango": {
        "image": "images/mango.jpeg",
        "desc": "Mango is a tropical fruit tree suited to warm climates with moderate rainfall. It grows best in well-drained loamy soils and temperatures between 24–30°C. Proper nutrient and irrigation management improve fruit quality."
    },

    "Grapes": {
        "image": "images/grapes.jpeg",
        "desc": "Grapes are cultivated in warm climates with dry conditions during fruit maturation. They prefer well-drained sandy-loam soils and moderate irrigation. Climate conditions strongly influence sugar accumulation and flavor."
    },

    "Watermelon": {
        "image": "images/watermelon.jpeg",
        "desc": "Watermelon is a warm-season crop requiring sandy soils and high sunlight exposure. It grows best at temperatures between 22–30°C with moderate irrigation. Proper drainage enhances fruit development and sweetness."
    },

    "Muskmelon": {
        "image": "images/muskmelon.jpeg",
        "desc": "Muskmelon thrives in warm, dry climates with sandy-loam soils and good sunlight. It requires moderate irrigation and temperatures between 20–30°C. Balanced nutrition improves fruit aroma and overall quality."
    },

    "Apple": {
        "image": "images/apple.jpeg",
        "desc": "Apple is a temperate fruit crop requiring sufficient chilling hours and cool climate conditions. It grows best in well-drained loamy soils with moderate rainfall. Proper orchard management enhances yield and fruit quality."
    },

    "Orange": {
        "image": "images/orange.jpeg",
        "desc": "Orange is a subtropical citrus crop requiring moderate rainfall and well-drained soils. Ideal temperatures range between 15–30°C. Balanced fertilization and proper drainage are essential for healthy fruit production."
    },

    "Papaya": {
        "image": "images/papaya.jpeg",
        "desc": "Papaya is a tropical fruit crop grown in warm climates with good drainage. It prefers fertile soils and moderate irrigation. Continuous nutrient supply supports steady fruit production."
    },

    "Coconut": {
        "image": "images/coconut.jpeg",
        "desc": "Coconut thrives in coastal humid climates with sandy soils and high rainfall. It requires consistent moisture and temperatures between 25–32°C. Proper nutrient management ensures sustained nut production."
    },

    "Cotton": {
        "image": "images/cotton.jpeg",
        "desc": "Cotton is a fiber crop grown in warm climates with moderate rainfall and high sunlight. It performs well in black or well-drained loamy soils. Effective nutrient and pest management improve fiber yield and quality."
    },

    "Jute": {
        "image": "images/jute.jpeg",
        "desc": "Jute is cultivated in hot, humid climates with heavy rainfall. It grows best in fertile alluvial soils with good drainage. The crop is economically significant for producing biodegradable fiber."
    },

    "Coffee": {
        "image": "images/coffee.jpeg",
        "desc": "Coffee is grown in cool tropical highlands under partial shade. It requires well-drained soils, moderate rainfall, and temperatures between 18–24°C. Altitude and climate strongly influence bean quality and flavor."
    }
}

# ---------------- UI HEADER ----------------
st.title(tr("🌱 AI Crop Recommendation System",
            "🌱 نظام التوصية بالمحاصيل بالذكاء الاصطناعي"))

st.write(tr("Enter soil nutrients and environmental conditions:",
            "أدخل عناصر التربة والظروف البيئية:"))

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    N = st.number_input(tr("Nitrogen (N)", "النيتروجين (N)"), min_value=0)
    P = st.number_input(tr("Phosphorus (P)", "الفوسفور (P)"), min_value=0)
    K = st.number_input(tr("Potassium (K)", "البوتاسيوم (K)"), min_value=0)
    temperature = st.number_input(tr("Temperature (°C)", "درجة الحرارة (°م)"))

with col2:
    humidity = st.number_input(tr("Humidity (%)", "الرطوبة (%)"))
    ph = st.number_input(tr("Soil pH", "درجة حموضة التربة"))
    rainfall = st.number_input(tr("Rainfall (mm)", "هطول الأمطار (ملم)"))

# ---------------- PREDICTION ----------------
if st.button(tr("🔍 Recommend Crop", "🔍 توصية بالمحصول")):

    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    crop_name = crop_dict[int(prediction[0])]
    info = crop_info[crop_name]

    st.markdown("---")
    st.subheader(tr("✅ Recommended Crop", "✅ المحصول الموصى به"))

    col_img, col_text = st.columns([1, 2])

    with col_img:
        if os.path.exists(info["image"]):
            st.image(info["image"], use_container_width=True)

    with col_text:
        st.markdown(f"### 🌾 {crop_name}")
        st.write(info["desc"])

    st.success(tr(
        f"{crop_name} is suitable based on given soil and climate conditions.",
        f"{crop_name} مناسب بناءً على ظروف التربة والمناخ المدخلة."
    ))

    # -------- Feature Table --------
    st.markdown("---")
    st.subheader(tr("📊 Input Feature Explanation",
                    "📊 شرح خصائص الإدخال"))

    feature_df = pd.DataFrame({
        tr("Feature", "الميزة"): [
            tr("Nitrogen", "النيتروجين"),
            tr("Phosphorus", "الفوسفور"),
            tr("Potassium", "البوتاسيوم"),
            tr("Temperature", "درجة الحرارة"),
            tr("Humidity", "الرطوبة"),
            tr("Soil pH", "درجة الحموضة"),
            tr("Rainfall", "هطول الأمطار")
        ],
        tr("Entered Value", "القيمة المدخلة"): [
            N, P, K, temperature, humidity, ph, rainfall
        ]
    })

    st.dataframe(feature_df, use_container_width=True)