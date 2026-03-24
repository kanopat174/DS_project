import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="California Housing Price Prediction", page_icon="🏠", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('model.joblib')

@st.cache_data
def load_data():
    df = pd.read_csv('housing.csv')
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    return df

model = load_model()
df = load_data()

st.sidebar.title("🏠 Housing Price Prediction")
st.sidebar.markdown("ทำนายราคาบ้านใน California")
st.sidebar.markdown("---")
page = st.sidebar.radio("เลือกหน้า", ["🔮 ทำนายราคาบ้าน", "📊 สำรวจข้อมูล", "ℹ️ เกี่ยวกับโปรเจค"])

if page == "🔮 ทำนายราคาบ้าน":
    st.title("🔮 ทำนายราคาบ้านใน California")
    st.markdown("กรอกข้อมูลพื้นที่ แล้วกดทำนายราคาบ้าน")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 ข้อมูลทำเล")
        longitude = st.slider("Longitude (พิกัดตะวันออก-ตก)", -124.3, -114.3, -119.6,
                              help="California อยู่ระหว่าง -124 ถึง -114")
        latitude = st.slider("Latitude (พิกัดเหนือ-ใต้)", 32.5, 42.0, 35.6,
                             help="California อยู่ระหว่าง 32 ถึง 42")
        ocean_proximity = st.selectbox("Ocean Proximity (ความใกล้ทะเล)",
                                       ['<1H OCEAN', 'INLAND', 'NEAR BAY', 'NEAR OCEAN', 'ISLAND'],
                                       help="INLAND = ในแผ่นดิน, <1H OCEAN = ห่างทะเลไม่เกิน 1 ชม.")
        housing_median_age = st.slider("Housing Median Age (อายุบ้านเฉลี่ย ปี)", 1, 52, 28)

    with col2:
        st.subheader("👥 ข้อมูลพื้นที่")
        median_income = st.slider("Median Income (รายได้เฉลี่ย x$10,000)", 0.5, 15.0, 3.9, step=0.1,
                                  help="เช่น 3.0 = $30,000/ปี, 8.0 = $80,000/ปี")
        total_rooms = st.number_input("Total Rooms (ห้องทั้งหมดในพื้นที่)", 2, 40000, 2600)
        total_bedrooms = st.number_input("Total Bedrooms (ห้องนอนทั้งหมด)", 1, 6500, 540)
        population = st.number_input("Population (ประชากร)", 3, 36000, 1400)
        households = st.number_input("Households (ครัวเรือน)", 1, 6100, 500)

    if median_income < 2.0:
        st.warning("⚠️ รายได้เฉลี่ยต่ำกว่า $20,000 — พื้นที่นี้อาจมีราคาบ้านต่ำ")

    st.markdown("---")

    if st.button("🔮 ทำนายราคาบ้าน", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            'longitude': longitude, 'latitude': latitude,
            'housing_median_age': housing_median_age,
            'total_rooms': total_rooms, 'total_bedrooms': total_bedrooms,
            'population': population, 'households': households,
            'median_income': median_income, 'ocean_proximity': ocean_proximity
        }])

        prediction = model.predict(input_data)[0]
        prediction = max(prediction, 0)

        st.markdown("---")
        st.subheader("📋 ผลการทำนาย")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("ราคาบ้านที่ทำนาย", f"${prediction:,.0f}")
        col_b.metric("รายได้เฉลี่ย", f"${median_income * 10000:,.0f}/ปี")
        col_c.metric("ทำเล", ocean_proximity)

        if prediction > 300000:
            st.success(f"พื้นที่นี้ราคาบ้านค่อนข้างสูง (${prediction:,.0f}) — น่าจะเป็นทำเลดีใกล้ทะเลหรือรายได้สูง")
        elif prediction > 150000:
            st.info(f"พื้นที่นี้ราคาบ้านปานกลาง (${prediction:,.0f})")
        else:
            st.warning(f"พื้นที่นี้ราคาบ้านค่อนข้างต่ำ (${prediction:,.0f}) — อาจเป็นพื้นที่ INLAND หรือรายได้เฉลี่ยต่ำ")

        st.caption("⚠️ ผลทำนายเป็นการประมาณจาก ML model (ข้อมูลปี 1990) ไม่ใช่ราคาจริงปัจจุบัน")


elif page == "📊 สำรวจข้อมูล":
    st.title("📊 สำรวจข้อมูล California Housing")
    st.markdown(f"Dataset มี **{len(df):,} พื้นที่** ใน California")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("จำนวนพื้นที่", f"{len(df):,}")
    col2.metric("ราคาเฉลี่ย", f"${df['median_house_value'].mean():,.0f}")
    col3.metric("ราคาต่ำสุด", f"${df['median_house_value'].min():,.0f}")
    col4.metric("ราคาสูงสุด", f"${df['median_house_value'].max():,.0f}")

    st.markdown("---")
    chart = st.selectbox("เลือกกราฟ", [
        "ราคาบ้าน ตาม Ocean Proximity",
        "รายได้ vs ราคาบ้าน",
        "Correlation กับราคาบ้าน",
        "Feature Importance"
    ])

    fig, ax = plt.subplots(figsize=(10, 6))

    if chart == "ราคาบ้าน ตาม Ocean Proximity":
        ocean_price = df.groupby('ocean_proximity')['median_house_value'].median().sort_values()
        ax.barh(ocean_price.index, ocean_price.values, color='steelblue', edgecolor='white')
        ax.set_title('Median House Value by Ocean Proximity')
        ax.set_xlabel('Price ($)')

    elif chart == "รายได้ vs ราคาบ้าน":
        ax.scatter(df['median_income'], df['median_house_value'], alpha=0.1, s=5, color='steelblue')
        ax.set_title('Median Income vs House Value')
        ax.set_xlabel('Median Income (x$10,000)')
        ax.set_ylabel('House Value ($)')

    elif chart == "Correlation กับราคาบ้าน":
        corr = df.select_dtypes(include=[np.number]).corr()['median_house_value'].drop('median_house_value').sort_values()
        colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in corr.values]
        ax.barh(corr.index, corr.values, color=colors, edgecolor='white')
        ax.set_title('Correlation with House Value')
        ax.axvline(0, color='black', linewidth=0.8)

    elif chart == "Feature Importance":
        model_step = model.named_steps['model']
        preprocessor_step = model.named_steps['preprocessor']
        cat_encoder = preprocessor_step.named_transformers_['cat']
        cat_names = cat_encoder.get_feature_names_out(['ocean_proximity']).tolist()
        num_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                        'total_bedrooms', 'population', 'households', 'median_income']
        all_names = num_features + cat_names
        feat_imp = pd.Series(model_step.feature_importances_, index=all_names).sort_values(ascending=False)
        top = feat_imp.head(10).sort_values()
        ax.barh(top.index, top.values, color='steelblue', edgecolor='white')
        ax.set_title('Top 10 Feature Importance')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


elif page == "ℹ️ เกี่ยวกับโปรเจค":
    st.title("ℹ️ เกี่ยวกับโปรเจค")
    st.markdown("""
    ### 🎯 เป้าหมาย
    ทำนายราคาบ้าน (median house value) ในแต่ละพื้นที่ของ California

    ### 📊 Dataset
    - **ชื่อ:** California Housing Dataset
    - **จำนวน:** 20,640 พื้นที่
    - **ช่วงเวลา:** สำมะโนประชากร 1990

    ### 🤖 Model
    - **Algorithm:** Gradient Boosting Regressor
    - **Features:** 9 ตัว (8 numerical + 1 categorical)
    - **Missing Values:** total_bedrooms 207 แถว → เติมด้วย median
    - **Pipeline:** SimpleImputer + StandardScaler + OneHotEncoder

    ### 💡 Key Insights
    1. **median_income** คือปัจจัยสำคัญที่สุดต่อราคาบ้าน
    2. **INLAND** ราคาถูกที่สุด, **ISLAND** แพงที่สุด
    3. ยิ่งใกล้ทะเล ราคาบ้านยิ่งสูง

    ---
    *ML Deployment Project — Burapha University*
    """)
