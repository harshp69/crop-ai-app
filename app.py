import streamlit as st
import joblib
from scipy.optimize import linprog
import matplotlib.pyplot as plt

import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load ML model
model = joblib.load("crop_model.pkl")

# Crop mapping
crop_map = {
    0: "Bajra 🌱",
    1: "Rice 🌾",
    2: "Wheat 🌿",
    3: "Cotton 🧵"
}

# -------------------------
# PRICE PREDICTION FUNCTION
# -------------------------
def predict_price(df):
    df = df[['arrival_date', 'modal_price']].copy()

    df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
    df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')

    df = df.dropna()

    if len(df) < 3:
        return []

    df = df.sort_values('arrival_date')
    df['day'] = np.arange(len(df))

    X = df[['day']]
    y = df['modal_price']

    model_lr = LinearRegression()
    model_lr.fit(X, y)

    future_days = np.array([[len(df)+i] for i in range(7)])
    return model_lr.predict(future_days)

# Page setup
st.set_page_config(page_title="Smart Crop AI", layout="centered")

# -------------------------
# HEADER
# -------------------------
st.markdown("<h1 style='text-align: center; color: green;'>🌾 Smart Crop AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI + Optimization System</p>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# INPUTS
# -------------------------
col1, col2 = st.columns(2)

with col1:
    soil = st.slider("🌱 Soil Moisture", 0, 100, 50)
    rain = st.slider("🌧 Rainfall", 0, 300, 100)

with col2:
    temp = st.slider("🌡 Temperature", 0, 50, 25)
    land = st.number_input("🌍 Total Land (hectares)", value=100)

st.markdown("---")

# -------------------------
# SESSION STATE FIX
# -------------------------
if "show_crop" not in st.session_state:
    st.session_state.show_crop = False

if "show_optimize" not in st.session_state:
    st.session_state.show_optimize = False

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Predict Crop"):
        st.session_state.show_crop = True

with col2:
    if st.button("📊 Optimize Land"):
        st.session_state.show_optimize = True

# -------------------------
# CROP RESULT
# -------------------------
if st.session_state.show_crop:
    pred = model.predict([[soil, temp, rain]])[0]
    crop = crop_map.get(pred, "Unknown")
    st.success(f"🌾 Recommended Crop: {crop}")

# -------------------------
# OPTIMIZATION
# -------------------------
if st.session_state.show_optimize:

    pred = model.predict([[soil, temp, rain]])[0]

    profit = [20000, 30000, 25000, 35000]
    cost   = [8000, 12000, 10000, 15000]
    net_profit = [p - c for p, c in zip(profit, cost)]

    c = [-x for x in net_profit]

    water = [2, 5, 3, 4]
    demand = [land*0.5, land*0.6, land*0.5, land*0.4]

    A = [
        [1, 1, 1, 1],
        water
    ]

    b = [land, 300]

    bounds = [
        (land*0.1, demand[0]),
        (land*0.1, demand[1]),
        (land*0.1, demand[2]),
        (land*0.1, demand[3])
    ]

    c[pred] -= 2000

    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)

    bajra, rice, wheat, cotton = res.x

    st.success("💰 Profit Optimized Allocation:")

    st.write(f"🌱 Bajra: {round(bajra,2)} ha")
    st.write(f"🌾 Rice: {round(rice,2)} ha")
    st.write(f"🌿 Wheat: {round(wheat,2)} ha")
    st.write(f"🧵 Cotton: {round(cotton,2)} ha")

    total_profit = (
        bajra * net_profit[0] +
        rice * net_profit[1] +
        wheat * net_profit[2] +
        cotton * net_profit[3]
    )

    st.metric("💰 Estimated Profit", f"₹ {int(total_profit)}")

    crops = ["Bajra", "Rice", "Wheat", "Cotton"]
    values = [bajra, rice, wheat, cotton]

    fig, ax = plt.subplots()
    ax.bar(crops, values)
    ax.set_title("Optimal Allocation")

    st.pyplot(fig)

# -------------------------
# MANDI SECTION
# -------------------------
st.markdown("---")
st.header("📊 Live Mandi Price & Prediction")

API_KEY = "579b464db66ec23bdd0000018ebe52ae7ee2422652ff60fe57d186f1"

@st.cache_data
def load_data():
    url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={API_KEY}&format=json&limit=2000"
    data = requests.get(url).json()
    return pd.DataFrame(data['records'])

df = load_data().dropna()

states = sorted(df['state'].unique())
state = st.selectbox("🌍 Select State", states)

df_state = df[df['state'] == state]

mandis = sorted(df_state['market'].unique())
mandi = st.selectbox("🏙 Select Mandi", mandis)

df_mandi = df_state[df_state['market'] == mandi]

crops = sorted(df_mandi['commodity'].unique())
crop = st.selectbox("🌾 Select Crop", crops)

df_final = df_mandi[df_mandi['commodity'] == crop]

if df_final.empty:
    st.warning("No data available")
else:
    st.subheader("💰 Current Prices")
    st.dataframe(df_final[['commodity', 'min_price', 'max_price', 'modal_price']])

    df_final['arrival_date'] = pd.to_datetime(df_final['arrival_date'], errors='coerce')
    df_final['modal_price'] = pd.to_numeric(df_final['modal_price'], errors='coerce')
    df_final = df_final.dropna().sort_values('arrival_date')

    if len(df_final) > 1:
        st.subheader("📈 Price Trend")
        st.line_chart(df_final.set_index('arrival_date')['modal_price'])
    else:
        st.warning("Not enough data for trend")

    predictions = predict_price(df_final)

    st.subheader("🔮 Price Prediction (Next 7 Days)")

    if len(predictions) == 0:
        st.warning("Not enough data for prediction")
    else:
        for i, price in enumerate(predictions):
            st.metric(label=f"Day {i+1}", value=f"₹ {round(price,2)}")
