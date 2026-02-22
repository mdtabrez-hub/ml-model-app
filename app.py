import streamlit as st
import joblib
import numpy as np
import os

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
    max-width: 700px;
}
.stButton>button {
    width: 100%;
    height: 3.2em;
    font-size: 18px;
    border-radius: 12px;
}
.stNumberInput input {
    font-size: 16px !important;
}
h1, h2, h3 {
    text-align: center;
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
    st.markdown("""
        <style>
        body { direction: RTL; text-align: right; }
        </style>
    """, unsafe_allow_html=True)

def tr(en, ar):
    return ar if language == "عربية" else en

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

# ---------------- CROP INFO (4–5 LINE DESCRIPTIONS) ----------------
crop_info = {
    "Rice": {
        "image": "images/rice.jpeg",
        "en": """Rice is a staple cereal cultivated in tropical and subtropical climates.
It requires high rainfall (100–200 cm annually) and warm temperatures between 20–35°C.
Clayey or loamy soils capable of retaining water are ideal for flooded cultivation.
Nitrogen-rich fertilization significantly enhances vegetative growth and grain yield.
Proper irrigation and drainage management directly affect productivity.""",
        "ar": """الأرز محصول حبوب أساسي يُزرع في المناخات الاستوائية وشبه الاستوائية.
يحتاج إلى أمطار غزيرة ودرجات حرارة بين 20–35°م.
تعد التربة الطينية أو الطميية القادرة على الاحتفاظ بالمياه مثالية لزراعته.
يساعد التسميد الغني بالنيتروجين في تحسين النمو والإنتاجية.
تؤثر إدارة الري والصرف بشكل مباشر على جودة المحصول."""
    },

    "Maize": {
        "image": "images/maize.jpeg",
        "en": """Maize is a versatile cereal grown across diverse agro-climatic zones.
It thrives in well-drained loamy soils with moderate rainfall.
Optimal growth temperature ranges from 18–27°C.
Balanced nitrogen and phosphorus supply improves kernel formation.
It is widely used for food, feed, and industrial applications.""",
        "ar": """الذرة محصول حبوب متعدد الاستخدامات يُزرع في مناطق مناخية مختلفة.
تنمو جيدًا في التربة الطميية جيدة التصريف مع أمطار معتدلة.
تتراوح درجة الحرارة المثلى للنمو بين 18–27°م.
يساعد التوازن في النيتروجين والفوسفور على تحسين تكوين الحبوب.
تُستخدم في الغذاء والأعلاف والصناعات المختلفة."""
    },

    "Chickpea": {
        "image": "images/chickpea.jpeg",
        "en": """Chickpea is a cool-season pulse crop commonly grown in the rabi season.
It prefers sandy-loam soils with moderate temperatures (20–25°C).
Low humidity conditions reduce disease incidence.
As a legume, it fixes atmospheric nitrogen and improves soil fertility.
It is an important source of plant protein.""",
        "ar": """الحمص محصول بقولي شتوي يُزرع غالبًا في موسم الربيع.
يفضل التربة الرملية الطميية ودرجات حرارة بين 20–25°م.
تقلل الرطوبة المنخفضة من انتشار الأمراض.
يثبت النيتروجين الجوي ويحسن خصوبة التربة.
يُعد مصدرًا مهمًا للبروتين النباتي."""
    },

    "Banana": {
        "image": "images/banana.jpeg",
        "en": """Banana is a tropical fruit crop requiring high humidity and consistent irrigation.
It grows best at temperatures between 25–35°C.
Fertile, well-drained soils rich in organic matter enhance productivity.
It is a heavy nutrient feeder, especially nitrogen and potassium.
Regular water supply ensures uniform fruit development.""",
        "ar": """الموز محصول فاكهة استوائي يحتاج إلى رطوبة عالية وري منتظم.
ينمو أفضل بين 25–35°م.
تزيد التربة الخصبة الغنية بالمواد العضوية من الإنتاجية.
يحتاج إلى كميات عالية من النيتروجين والبوتاسيوم.
يساعد الري المنتظم على نمو الثمار بشكل متوازن."""
    },

    "Coffee": {
        "image": "images/coffee.jpeg",
        "en": """Coffee is cultivated in cool tropical highlands under partial shade.
It requires moderate rainfall and temperatures between 18–24°C.
Well-drained acidic soils support healthy root development.
Altitude and climate significantly influence bean flavor and quality.
Proper shade management improves yield stability.""",
        "ar": """تُزرع القهوة في المرتفعات الاستوائية الباردة تحت الظل الجزئي.
تحتاج إلى أمطار معتدلة ودرجات حرارة بين 18–24°م.
تدعم التربة الحمضية جيدة التصريف نمو الجذور.
يؤثر الارتفاع والمناخ على جودة ونكهة الحبوب.
تحسن إدارة الظل استقرار الإنتاج."""
    }
}

# ---------------- HEADER ----------------
st.title(tr("🌱 AI Crop Recommendation System",
            "🌱 نظام التوصية بالمحاصيل بالذكاء الاصطناعي"))

st.write(tr("Enter soil nutrients and environmental conditions:",
            "أدخل عناصر التربة والظروف البيئية:"))

st.markdown("---")

# ---------------- INPUTS WITH UNITS ----------------
N = st.number_input(tr("Nitrogen (N) [kg/ha]", "النيتروجين (كجم/هكتار]"), min_value=0)
P = st.number_input(tr("Phosphorus (P) [kg/ha]", "الفوسفور (كجم/هكتار]"), min_value=0)
K = st.number_input(tr("Potassium (K) [kg/ha]", "البوتاسيوم (كجم/هكتار]"), min_value=0)
temperature = st.number_input(tr("Temperature [°C]", "درجة الحرارة [°م]"))
humidity = st.number_input(tr("Humidity [%]", "الرطوبة [%]"), min_value=0.0, max_value=100.0)
ph = st.number_input(tr("Soil pH [0–14]", "درجة حموضة التربة [0–14]"),
                     min_value=0.0, max_value=14.0)
rainfall = st.number_input(tr("Rainfall [mm]", "هطول الأمطار [ملم]"), min_value=0.0)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if st.button(tr("🔍 Recommend Crop", "🔍 توصية بالمحصول")):

    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    crop_name = crop_dict[int(prediction[0])]
    info = crop_info.get(crop_name)

    st.markdown("---")
    st.subheader(tr("✅ Recommended Crop", "✅ المحصول الموصى به"))

    if info:
        if os.path.exists(info["image"]):
            st.image(info["image"], use_container_width=True)

        st.markdown(f"### {crop_name}")
        st.write(info["ar"] if language == "عربية" else info["en"])

        st.success(tr(
            f"{crop_name} is suitable based on the given soil and climate conditions.",
            f"{crop_name} مناسب بناءً على ظروف التربة والمناخ المدخلة."
        ))