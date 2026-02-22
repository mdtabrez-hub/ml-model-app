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
        "en": """Rice is a staple cereal crop cultivated mainly in tropical and subtropical regions.
It requires high rainfall (100–200 cm) and warm temperatures between 20–35°C.
Clayey or loamy soils capable of retaining water are ideal for flooded cultivation.
Nitrogen-rich fertilization significantly enhances vegetative growth and grain yield.
Proper irrigation and drainage management directly affect productivity.""",
        "ar": """الأرز محصول حبوب أساسي يُزرع في المناطق الاستوائية وشبه الاستوائية.
يحتاج إلى أمطار غزيرة ودرجات حرارة بين 20–35°م.
تعد التربة الطينية أو الطميية القادرة على الاحتفاظ بالمياه مثالية.
يساعد التسميد الغني بالنيتروجين على زيادة الإنتاجية.
تؤثر إدارة الري والصرف على جودة المحصول."""
    },

    "Maize": {
        "image": "images/maize.jpeg",
        "en": """Maize is a versatile cereal grown across diverse agro-climatic zones.
It thrives in well-drained loamy soils with moderate rainfall.
Optimal temperature ranges from 18–27°C for healthy growth.
Balanced nitrogen and phosphorus improve kernel formation.
It is widely used for food, feed, and industrial products.""",
        "ar": """الذرة محصول حبوب متعدد الاستخدامات يُزرع في مناطق مناخية مختلفة.
تنمو جيدًا في التربة الطميية جيدة التصريف مع أمطار معتدلة.
تتراوح درجة الحرارة المثلى بين 18–27°م.
يحسن التوازن بين النيتروجين والفوسفور تكوين الحبوب.
تستخدم في الغذاء والأعلاف والصناعة."""
    },

    "Chickpea": {
        "image": "images/chickpea.jpeg",
        "en": """Chickpea is a cool-season pulse crop grown mainly in the rabi season.
It prefers sandy-loam soils and moderate temperatures (20–25°C).
Low humidity reduces disease incidence.
Being a legume, it fixes atmospheric nitrogen.
It improves soil fertility and provides high protein yield.""",
        "ar": """الحمص محصول بقولي شتوي يُزرع غالبًا في موسم الربيع.
يفضل التربة الرملية الطميية ودرجات حرارة معتدلة.
تقل الرطوبة المنخفضة من انتشار الأمراض.
يثبت النيتروجين الجوي في التربة.
يحسن خصوبة التربة ويوفر بروتينًا عالي القيمة."""
    },

    "Kidneybeans": {
        "image": "images/kidneybeans.jpeg",
        "en": """Kidney beans are warm-season legumes requiring fertile soils.
They grow well under temperatures between 18–30°C.
Moderate rainfall supports healthy plant development.
They contribute to nitrogen enrichment in soil.
High protein content makes them nutritionally valuable.""",
        "ar": """الفاصولياء الحمراء محصول بقولي يحتاج إلى تربة خصبة.
تنمو بين 18–30°م بشكل مثالي.
تساعد الأمطار المعتدلة على النمو الجيد.
تسهم في زيادة النيتروجين في التربة.
تتميز بقيمة غذائية عالية لاحتوائها على البروتين."""
    },

    "Pigeonpeas": {
        "image": "images/pigeaonpeas.jpeg",
        "en": """Pigeonpea is drought-tolerant and suited for semi-arid climates.
It grows well in loamy soils with moderate rainfall.
Temperature range of 20–30°C is ideal.
It enhances soil fertility through nitrogen fixation.
Often cultivated in intercropping systems.""",
        "ar": """اللوبيا محصول يتحمل الجفاف ومناسب للمناخات شبه القاحلة.
ينمو في التربة الطميية مع أمطار معتدلة.
تعد درجات الحرارة بين 20–30°م مثالية.
يثبت النيتروجين ويحسن خصوبة التربة.
يُزرع غالبًا مع محاصيل أخرى."""
    },

    "Mothbeans": {
        "image": "images/mothbean.jpeg",
        "en": """Mothbean is highly drought-resistant and adapted to arid regions.
It thrives in sandy soils with minimal rainfall.
High temperatures do not affect its productivity significantly.
It supports dryland farming systems.
It is used both as pulse and fodder crop.""",
        "ar": """الموثبين محصول مقاوم للجفاف ومناسب للمناطق القاحلة.
ينمو في التربة الرملية مع أمطار قليلة.
يتحمل درجات الحرارة المرتفعة.
يدعم أنظمة الزراعة الجافة.
يُستخدم كمحصول غذائي وعلفي."""
    },

    "Mungbean": {
        "image": "images/mungbean.jpeg",
        "en": """Mungbean is a short-duration pulse crop suited for warm climates.
It prefers well-drained soils and moderate rainfall.
Ideal temperature ranges from 25–35°C.
It improves soil nitrogen content.
Commonly included in crop rotation systems.""",
        "ar": """الفاصولياء الخضراء محصول بقولي قصير المدة.
يفضل التربة جيدة التصريف وأمطار معتدلة.
تتراوح الحرارة المثلى بين 25–35°م.
يحسن محتوى النيتروجين في التربة.
يستخدم في تناوب المحاصيل."""
    },

    "Blackgram": {
        "image": "images/blackgram.jpeg",
        "en": """Blackgram grows well in warm climates with fertile loamy soils.
Optimal temperatures range between 25–35°C.
Moderate rainfall enhances yield.
It fixes nitrogen and improves soil structure.
It is widely consumed as protein-rich food.""",
        "ar": """العدس الأسود ينمو في المناخات الدافئة والتربة الخصبة.
تتراوح الحرارة المثلى بين 25–35°م.
تزيد الأمطار المعتدلة الإنتاج.
يثبت النيتروجين ويحسن بنية التربة.
يعد مصدرًا غنيًا بالبروتين."""
    },

    "Lentil": {
        "image": "images/lentil.jpeg",
        "en": """Lentil is a cool-season crop grown in moderate climates.
It prefers well-drained soils and temperatures of 15–25°C.
Low rainfall conditions are suitable.
It enhances soil fertility naturally.
It is nutritionally rich in protein and fiber.""",
        "ar": """العدس محصول شتوي يُزرع في مناخات معتدلة.
يفضل التربة جيدة التصريف وحرارة 15–25°م.
يناسب المناطق ذات الأمطار القليلة.
يحسن خصوبة التربة طبيعيًا.
غني بالبروتين والألياف."""
    },

    "Pomegranate": {
        "image": "images/pomogranate.jpeg",
        "en": """Pomegranate is suited for arid and semi-arid climates.
It grows well in well-drained loamy soils.
Temperature range between 25–35°C is favorable.
It is drought tolerant once established.
Fruits are valued for nutritional and medicinal properties.""",
        "ar": """الرمان مناسب للمناخات القاحلة وشبه القاحلة.
ينمو في التربة الطميية جيدة التصريف.
تعد الحرارة بين 25–35°م مناسبة.
يتحمل الجفاف بعد تثبيت الجذور.
ثماره ذات قيمة غذائية وطبية عالية."""
    },

    "Banana": {
        "image": "images/banana.jpeg",
        "en": """Banana requires tropical climate with high humidity.
Optimal temperature ranges between 25–35°C.
Fertile soils rich in organic matter improve yield.
It requires continuous irrigation.
It is a heavy feeder of nitrogen and potassium.""",
        "ar": """الموز يحتاج إلى مناخ استوائي ورطوبة عالية.
ينمو بين 25–35°م بشكل مثالي.
تحسن التربة الخصبة الغنية بالمواد العضوية الإنتاج.
يتطلب ريًا مستمرًا.
يحتاج إلى كميات عالية من النيتروجين والبوتاسيوم."""
    },

    "Mango": {
        "image": "images/mango.jpeg",
        "en": """Mango thrives in warm tropical climates.
It grows best in well-drained loamy soils.
Moderate rainfall with dry period improves flowering.
Optimal temperature is 24–30°C.
It is a major commercial fruit crop.""",
        "ar": """المانجو تنمو في المناخات الاستوائية الدافئة.
تفضل التربة الطميية جيدة التصريف.
تساعد الأمطار المعتدلة مع فترة جفاف على الإزهار.
الحرارة المثلى بين 24–30°م.
تعد من أهم محاصيل الفاكهة التجارية."""
    },

    "Grapes": {
        "image": "images/grapes.jpeg",
        "en": """Grapes grow best in warm climates with dry maturity period.
Well-drained sandy-loam soils are preferred.
Moderate irrigation is required.
Temperature between 20–30°C is suitable.
Used for fresh fruit, raisins, and wine production.""",
        "ar": """العنب ينمو في المناخات الدافئة مع فترة نضج جافة.
يفضل التربة الرملية الطميية جيدة التصريف.
يحتاج إلى ري معتدل.
تتراوح الحرارة المناسبة بين 20–30°م.
يستخدم طازجًا ولصناعة الزبيب والنبيذ."""
    },

    "Watermelon": {
        "image": "images/watermelon.jpeg",
        "en": """Watermelon requires warm temperatures and sandy soils.
It thrives between 22–30°C.
Adequate sunlight is essential.
Moderate irrigation enhances fruit sweetness.
Proper drainage prevents root diseases.""",
        "ar": """البطيخ يحتاج إلى حرارة دافئة وتربة رملية.
ينمو بين 22–30°م.
يتطلب تعرضًا جيدًا لأشعة الشمس.
يساعد الري المعتدل على زيادة الحلاوة.
يمنع التصريف الجيد أمراض الجذور."""
    },

    "Muskmelon": {
        "image": "images/muskmelon.jpeg",
        "en": """Muskmelon grows in warm dry climates.
It prefers sandy-loam soils.
Temperature of 20–30°C is optimal.
Moderate irrigation improves fruit quality.
Balanced nutrition enhances aroma and sweetness.""",
        "ar": """الشمام ينمو في المناخات الدافئة الجافة.
يفضل التربة الرملية الطميية.
الحرارة المثلى بين 20–30°م.
يحسن الري المعتدل جودة الثمار.
يعزز التسميد المتوازن الطعم والرائحة."""
    },

    "Apple": {
        "image": "images/apple.jpeg",
        "en": """Apple requires cool climate and chilling hours.
It grows in well-drained loamy soils.
Temperature between 10–24°C is ideal.
Proper pruning enhances yield.
Suitable for temperate regions.""",
        "ar": """التفاح يحتاج إلى مناخ بارد وساعات برودة.
ينمو في التربة الطميية جيدة التصريف.
الحرارة المثلى بين 10–24°م.
يزيد التقليم الصحيح الإنتاج.
مناسب للمناطق المعتدلة."""
    },

    "Orange": {
        "image": "images/orange.jpeg",
        "en": """Orange grows in subtropical climates.
Moderate rainfall is required.
Temperature between 15–30°C is ideal.
Well-drained soils improve fruit quality.
Balanced fertilization is essential.""",
        "ar": """البرتقال ينمو في المناخات شبه الاستوائية.
يحتاج إلى أمطار معتدلة.
الحرارة المثلى بين 15–30°م.
تحسن التربة جيدة التصريف جودة الثمار.
التسميد المتوازن ضروري."""
    },

    "Papaya": {
        "image": "images/papaya.jpeg",
        "en": """Papaya thrives in tropical climates.
It requires well-drained fertile soils.
Temperature range of 22–35°C is suitable.
Continuous nutrient supply improves yield.
It produces fruits throughout the year.""",
        "ar": """البابايا تزدهر في المناخات الاستوائية.
تحتاج إلى تربة خصبة جيدة التصريف.
الحرارة المناسبة بين 22–35°م.
يزيد التسميد المنتظم الإنتاج.
تعطي ثمارًا على مدار العام."""
    },

    "Coconut": {
        "image": "images/coconut.jpeg",
        "en": """Coconut grows in humid coastal regions.
It prefers sandy soils and high rainfall.
Temperature between 25–32°C is ideal.
Requires continuous moisture supply.
Used for food, oil, and industrial purposes.""",
        "ar": """جوز الهند ينمو في المناطق الساحلية الرطبة.
يفضل التربة الرملية وأمطار غزيرة.
الحرارة المثلى بين 25–32°م.
يحتاج إلى رطوبة مستمرة.
يستخدم في الغذاء والزيوت والصناعة."""
    },

    "Cotton": {
        "image": "images/cotton.jpeg",
        "en": """Cotton is a fiber crop grown in warm climates.
It requires moderate rainfall and high sunlight.
Black soils are ideal for cultivation.
Temperature between 21–30°C is suitable.
Important for textile industries.""",
        "ar": """القطن محصول ألياف يُزرع في المناخات الدافئة.
يحتاج إلى أمطار معتدلة وضوء شمس وفير.
تعد التربة السوداء مناسبة لزراعته.
الحرارة بين 21–30°م مثالية.
مهم لصناعة المنسوجات."""
    },

    "Jute": {
        "image": "images/jute.jpeg",
        "en": """Jute grows in hot and humid climates.
It requires heavy rainfall and fertile soils.
Temperature between 24–35°C is ideal.
Used for biodegradable fiber production.
Common in river basin regions.""",
        "ar": """الجوت ينمو في المناخات الحارة والرطبة.
يحتاج إلى أمطار غزيرة وتربة خصبة.
الحرارة بين 24–35°م مناسبة.
يستخدم لإنتاج الألياف القابلة للتحلل.
ينتشر في مناطق الأنهار."""
    },

    "Coffee": {
        "image": "images/coffee.jpeg",
        "en": """Coffee is cultivated in cool tropical highlands.
It requires moderate rainfall and partial shade.
Temperature between 18–24°C is optimal.
Well-drained acidic soils are preferred.
Altitude influences flavor and quality.""",
        "ar": """تزرع القهوة في المرتفعات الاستوائية الباردة.
تحتاج إلى أمطار معتدلة وظل جزئي.
الحرارة المثلى بين 18–24°م.
تفضل التربة الحمضية جيدة التصريف.
يؤثر الارتفاع على النكهة والجودة."""
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