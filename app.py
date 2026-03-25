import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
import os

st.set_page_config(page_title="IS Project", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #f5f4f0; }
section[data-testid="stSidebar"] { background-color: #1a1a2e; }
section[data-testid="stSidebar"] * { color: #f0f0f0 !important; }
section[data-testid="stSidebar"] hr { border-color: #333 !important; }

/* Shrink metric values */
[data-testid="stMetricValue"] {
    font-size: 1rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
}
/* Main content buttons */
.stButton > button {
    background-color: #1a1a2e !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100% !important;
}
.stButton > button:hover {
    background-color: #e8a838 !important;
    color: #1a1a2e !important;
}

/* Sidebar nav: inactive */
section[data-testid="stSidebar"] .stButton > button {
    background-color: transparent !important;
    color: #aaaaaa !important;
    border: 1px solid #2e2e4e !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 8px !important;
    margin-bottom: 4px !important;
    text-align: center !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #2a2a40 !important;
    color: #e8a838 !important;
    border-color: #e8a838 !important;
}

/* Sidebar nav: active */
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background-color: #e8a838 !important;
    color: #1a1a2e !important;
    border-color: #e8a838 !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Models
# --------------------------------------------------
@st.cache_resource
def load_models():
    rf, nn, ai_text_nn, tfidf = None, None, None, None
    try:
        rf = joblib.load("rf_model.pkl")
    except: pass
    try:
        model_path = "nn_model.keras" if os.path.exists("nn_model.keras") else "nn_model.h5"
        nn = tf.keras.models.load_model(model_path)
    except: pass
    try:
        ai_text_nn = tf.keras.models.load_model("ai_text_nn.keras")
    except: pass
    try:
        tfidf = joblib.load("tfidf.pkl")
    except: pass
    return rf, nn, ai_text_nn, tfidf

rf, nn, ai_text_nn, tfidf = load_models()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "SuperStore Info"

menu_items = [
    "SuperStore Info",
    "AI Text Info",
    "Profit Prediction",
    "Text Prediction",
]

with st.sidebar:
    st.markdown("")
    st.caption("")
    for item in menu_items:
        is_active = st.session_state.page == item
        if st.button(item, key=f"nav_{item}",
                     type="primary" if is_active else "secondary",
                     use_container_width=True):
            st.session_state.page = item
            st.rerun()

page = st.session_state.page

# ==================================================
# PAGE 1 — SuperStore Info
# ==================================================
if page == "SuperStore Info":

    st.title("SuperStore Dataset")
    st.markdown("ข้อมูลชุดนี้ใช้สำหรับวิเคราะห์และทำนาย **กำไร/ขาดทุน** ของธุรกิจค้าปลีก")
    st.divider()

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("รายละเอียด Dataset")
            st.markdown("""
ชุดข้อมูล **SuperStore Sales Analytics** เป็นข้อมูลการขายสินค้าของร้านค้าปลีก
ประกอบด้วยข้อมูลการสั่งซื้อ ลูกค้า สินค้า และผลกำไร
เหมาะสำหรับฝึกฝนโมเดล Machine Learning ประเภท Classification
            """)
            st.markdown("**Features ที่ใช้**")
            features = {
                "Sales": "ยอดขายรวมต่อรายการ",
                "Quantity": "จำนวนสินค้าที่สั่ง",
                "Discount": "ส่วนลดที่ให้ (0.0–1.0)",
                "Shipping Cost": "ค่าจัดส่ง",
                "Category": "หมวดหมู่สินค้าหลัก (0, 1, 2)",
                "Sub-Category": "หมวดหมู่ย่อยของสินค้า",
                "Region": "ภูมิภาค (0–3)",
                "Segment": "กลุ่มลูกค้า (0, 1, 2)",
            }
            for k, v in features.items():
                st.markdown(f"- **{k}** — {v}")
            st.markdown("**Target Variable**")
            c1, c2 = st.columns(2)
            c1.success("1 = Profit (กำไร)")
            c2.error("0 = Loss (ขาดทุน)")
            st.markdown("**โมเดลที่ใช้**")
            st.info("Random Forest  |  Neural Network")

    with col2:
        with st.container(border=True):
            st.subheader("สถิติเบื้องต้น")
            st.metric("จำนวนแถว", "~9,994")
            st.metric("จำนวน Features", "8")
            st.metric("ประเภทปัญหา", "Binary Classification")
            st.metric("Profit : Loss", "≈ 70 : 30")
            st.markdown("**การเตรียมข้อมูล**")
            st.markdown("""
- Label Encoding สำหรับ Categorical
- MinMax Scaling สำหรับตัวเลข
- ตัดข้อมูลที่หายออก
            """)

        with st.container(border=True):
            st.markdown("**แหล่งข้อมูล (Dataset Credit)**")
            st.markdown("[ thuandao/superstore-sales-analytics](https://www.kaggle.com/datasets/thuandao/superstore-sales-analytics)")

# ==================================================
# PAGE 2 — AI Text Info
# ==================================================
elif page == "AI Text Info":

    st.title("AI Text Dataset")
    st.markdown("ชุดข้อมูลสำหรับตรวจสอบว่าข้อความถูกเขียนโดย **AI** หรือ **มนุษย์**")
    st.divider()

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("รายละเอียด Dataset")
            st.markdown("""
ชุดข้อมูล **AI and Human Text Dataset** รวบรวมข้อความจากทั้ง AI (เช่น ChatGPT, GPT-4)
และมนุษย์จริงๆ เพื่อนำมาฝึกโมเดลให้แยกแยะรูปแบบการเขียน
ใช้ประโยชน์ในงานตรวจสอบความถูกต้องของเนื้อหา
            """)
            st.markdown("**Feature ที่ใช้**")
            st.markdown("- **Text** — ข้อความดิบ (raw text) ที่ต้องการตรวจสอบ")
            st.markdown("**Target Variable**")
            c1, c2 = st.columns(2)
            c1.warning("AI")
            c2.success("Human")
            st.markdown("**วิธีการประมวลผล**")
            st.markdown("""
- **TF-IDF Vectorization** — แปลงข้อความเป็น Vector ตัวเลข
- **Neural Network** — โมเดลจำแนกประเภท
- Threshold: 0.5 (> 0.5 = AI, ≤ 0.5 = Human)
            """)
            st.markdown("**ข้อจำกัด**")
            st.markdown("""
- ข้อความสั้นมากอาจให้ผลไม่แม่นยำ
- รองรับเฉพาะภาษาอังกฤษ
- AI รุ่นใหม่ๆ อาจทำให้โมเดลสับสนได้
            """)

    with col2:
        with st.container(border=True):
            st.subheader("รายละเอียดโมเดล")
            st.metric("วิธีการ", "TF-IDF + Neural Network")
            st.metric("Input", "Raw text ความยาวอิสระ")
            st.metric("Output", "ความน่าจะเป็น 0–1")
            st.metric("ประเภทปัญหา", "Binary Classification")

        with st.container(border=True):
            st.markdown("**แหล่งข้อมูล (Dataset Credit)**")
            st.markdown("[ hasanyiitakbulut/ai-and-human-text-dataset](https://www.kaggle.com/datasets/hasanyiitakbulut/ai-and-human-text-dataset)")

# ==================================================
# PAGE 3 — Profit Prediction
# ==================================================
elif page == "Profit Prediction":

    st.title("Profit Prediction")
    st.markdown("ทำนาย **กำไร/ขาดทุน** จากข้อมูลการขาย SuperStore")
    st.divider()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("ข้อมูลการขาย")
            sales    = st.number_input("Sales — ยอดขาย", min_value=0.0, step=10.0)
            quantity = st.number_input("Quantity — จำนวน", min_value=1, step=1)
            discount = st.slider("Discount — ส่วนลด", 0.0, 1.0, 0.1, step=0.05)
            shipping = st.number_input("Shipping Cost — ค่าจัดส่ง", min_value=0.0, step=1.0)

    with col2:
        with st.container(border=True):
            st.subheader("ข้อมูลหมวดหมู่")
            category     = st.selectbox("Category — หมวดหมู่", [0, 1, 2],
                               format_func=lambda x: ["Furniture", "Office Supplies", "Technology"][x])
            sub_category = st.number_input("Sub Category — หมวดย่อย", min_value=0, step=1)
            region       = st.selectbox("Region — ภูมิภาค", [0, 1, 2, 3],
                               format_func=lambda x: ["Central", "East", "South", "West"][x])
            segment      = st.selectbox("Segment — กลุ่มลูกค้า", [0, 1, 2],
                               format_func=lambda x: ["Consumer", "Corporate", "Home Office"][x])

    st.divider()
    data = np.array([[sales, quantity, discount, shipping, category, sub_category, region, segment]])

    col_btn1, col_btn2 = st.columns(2, gap="large")

    with col_btn1:
        if st.button("ทำนายด้วย Random Forest", use_container_width=True):
            if rf:
                pred = rf.predict(data)[0]
                if pred == 1:
                    st.success("## PROFIT — มีกำไร")
                    st.markdown("**Random Forest** คาดว่ารายการนี้จะ **มีกำไร**")
                else:
                    st.error("## LOSS — ขาดทุน")
                    st.markdown("**Random Forest** คาดว่ารายการนี้จะ **ขาดทุน**")
            else:
                st.warning("ไม่พบโมเดล rf_model.pkl")

    with col_btn2:
        if st.button("ทำนายด้วย Neural Network", use_container_width=True):
            if nn:
                pred = nn.predict(data)[0][0]
                confidence = pred if pred > 0.5 else (1 - pred)
                if pred > 0.5:
                    st.success(f"## PROFIT — มีกำไร")
                    st.markdown(f"**Neural Network** · ความมั่นใจ **{confidence:.1%}**")
                    st.progress(float(confidence))
                else:
                    st.error(f"## LOSS — ขาดทุน")
                    st.markdown(f"**Neural Network** · ความมั่นใจ **{confidence:.1%}**")
                    st.progress(float(confidence))
            else:
                st.warning("ไม่พบโมเดล Neural Network")

# ==================================================
# PAGE 4 — Text Prediction
# ==================================================
elif page == "Text Prediction":

    st.title("AI Text Detector")
    st.markdown("ตรวจสอบว่าข้อความถูกเขียนโดย **AI** หรือ **มนุษย์**")
    st.divider()

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("ใส่ข้อความที่ต้องการตรวจสอบ")
            text = st.text_area(
                "วางข้อความภาษาอังกฤษที่นี่",
                height=220,
                placeholder="Paste your English text here to detect if it was written by AI or a human..."
            )
            if st.button("วิเคราะห์ข้อความ", use_container_width=True):
                if not (ai_text_nn and tfidf):
                    st.error("ไม่พบโมเดลหรือ TF-IDF vectorizer")
                elif text.strip() == "":
                    st.warning("กรุณากรอกข้อความก่อนทำการวิเคราะห์")
                else:
                    with st.spinner("กำลังวิเคราะห์..."):
                        vec  = tfidf.transform([text]).toarray()
                        pred = ai_text_nn.predict(vec)[0][0]
                    st.divider()
                    if pred > 0.5:
                        st.warning("## AI Generated")
                        st.markdown(f"ข้อความนี้น่าจะเขียนโดย **AI** · ความมั่นใจ **{pred:.1%}**")
                        st.progress(float(pred))
                    else:
                        confidence = 1 - pred
                        st.success("## Human Written")
                        st.markdown(f"ข้อความนี้น่าจะเขียนโดย **มนุษย์** · ความมั่นใจ **{confidence:.1%}**")
                        st.progress(float(confidence))

    with col2:
        with st.container(border=True):
            st.subheader("คำแนะนำ")
            st.markdown("""
**ข้อความที่ดีควร:**
- ความยาวอย่างน้อย 50 คำ
- เป็นภาษาอังกฤษ
- เป็นย่อหน้าต่อเนื่อง

**ผลลัพธ์:**
- AI — เขียนโดย AI
- Human — เขียนโดยมนุษย์

**หมายเหตุ:**
ความแม่นยำโดยประมาณ 85–90%
ขึ้นอยู่กับลักษณะข้อความ
            """)
