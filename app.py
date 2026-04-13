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

crop_map = {
    0: "Bajra 🌱",
    1: "Rice 🌾",
    2: "Wheat 🌿",
    3: "Cotton 🧵"
}

# -------------------------
# PRICE PREDICTION
# -------------------------
def predict_price(df):
    df = df[['arrival_date', 'modal_price']].copy()
    df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
    df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')
    df = df.dropna()

    if len(df) < 2:
        last_price = df['modal_price'].iloc[-1] if len(df)>0 else 1000
        return [last_price]*7

    df = df.sort_values('arrival_date')
    df['day'] = np.arange(len(df))

    X = df[['day']]
    y = df['modal_price']

    model_lr = LinearRegression()
    model_lr.fit(X, y)

    future_days = np.array([[len(df)+i] for i in range(7)])
    return model_lr.predict(future_days)

# -------------------------
# APP UI
# -------------------------
st.set_page_config(page_title="Smart Crop AI", layout="centered")

st.markdown("<h1 style='text-align:center;color:green;'>🌾 Smart Crop AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI + Optimization + Market Intelligence</p>", unsafe_allow_html=True)

st.markdown("---")

# INPUT
col1, col2 = st.columns(2)
with col1:
    soil = st.slider("🌱 Soil Moisture", 0, 100, 50)
    rain = st.slider("🌧 Rainfall", 0, 300, 100)

with col2:
    temp = st.slider("🌡 Temperature", 0, 50, 25)
    land = st.number_input("🌍 Total Land", value=100)

# SESSION
if "crop" not in st.session_state:
    st.session_state.crop = False
if "opt" not in st.session_state:
    st.session_state.opt = False

c1, c2 = st.columns(2)

with c1:
    if st.button("🚀 Predict Crop"):
        st.session_state.crop = True

with c2:
    if st.button("📊 Optimize Land"):
        st.session_state.opt = True

# CROP
if st.session_state.crop:
    pred = model.predict([[soil,temp,rain]])[0]
    st.success(f"🌾 Recommended Crop: {crop_map[pred]}")

# OPTIMIZATION
if st.session_state.opt:
    st.markdown("## 📊 Optimize Land")

    profit = [20000,30000,25000,35000]
    cost = [8000,12000,10000,15000]
    net = [p-c for p,c in zip(profit,cost)]

    c = [-x for x in net]
    water = [2,5,3,4]

    A = [[1,1,1,1], water]
    b = [land,300]

    bounds = [(0,land)]*4

    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)

    bajra,rice,wheat,cotton = res.x

    st.write(f"Bajra: {round(bajra,2)}")
    st.write(f"Rice: {round(rice,2)}")
    st.write(f"Wheat: {round(wheat,2)}")
    st.write(f"Cotton: {round(cotton,2)}")

# -------------------------
# MANDI DATA
# -------------------------
st.markdown("---")
st.header("📊 Live Mandi Price & Smart Analysis")

API_KEY = "579b464db66ec23bdd0000018ebe52ae7ee2422652ff60fe57d186f1"

@st.cache_data
def load_data():
    url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={API_KEY}&format=json&limit=2000"
    return pd.DataFrame(requests.get(url).json()['records'])

df = load_data().dropna()

state = st.selectbox("State", sorted(df['state'].unique()))
df_state = df[df['state']==state]

mandi = st.selectbox("Mandi", sorted(df_state['market'].unique()))
df_mandi = df_state[df_state['market']==mandi]

crop_option = st.selectbox("Crop", ["Rice","Wheat","Bajra","Cotton"])

mapping = {
    "Rice":["Rice","Paddy"],
    "Wheat":["Wheat"],
    "Bajra":["Bajra"],
    "Cotton":["Cotton"]
}

df_final = df_mandi[df_mandi['commodity'].str.contains('|'.join(mapping[crop_option]), case=False)]

if not df_final.empty:

    df_final['arrival_date'] = pd.to_datetime(df_final['arrival_date'], errors='coerce')
    df_final['modal_price'] = pd.to_numeric(df_final['modal_price'], errors='coerce')
    df_final = df_final.dropna().sort_values('arrival_date')

    st.subheader("📈 Price Trend")
    st.line_chart(df_final.set_index('arrival_date')['modal_price'])

    preds = predict_price(df_final)

    st.subheader("🔮 Prediction")
    for i,p in enumerate(preds):
        st.write(f"Day {i+1}: ₹ {round(p,2)}")

    # 🔥 TREND LOGIC
    if preds[-1] > preds[0]:
        st.success("📈 Price likely to INCREASE")
    else:
        st.error("📉 Price likely to DECREASE")

    # 💰 PROFIT LOGIC
    avg_price = df_final['modal_price'].mean()
    st.metric("💰 Avg Price", f"₹ {int(avg_price)}")

# -------------------------
# BEST CROP
# -------------------------
st.markdown("---")
st.header("🏆 Best Crop to Sell")

crop_prices = {}

for crop, keys in mapping.items():
    df_temp = df_mandi[df_mandi['commodity'].str.contains('|'.join(keys), case=False)]
    if not df_temp.empty:
        df_temp['modal_price'] = pd.to_numeric(df_temp['modal_price'], errors='coerce')
        crop_prices[crop] = df_temp['modal_price'].mean()

if crop_prices:
    best = max(crop_prices, key=crop_prices.get)
    st.success(f"🔥 Best Crop to Sell: {best}")
    st.write(crop_prices)

# -------------------------
# SMART RECOMMENDATION
# -------------------------
st.markdown("---")
st.header("🧠 Smart Recommendation")

if crop_prices:
    if best == "Cotton":
        st.success("Cotton high profit → Grow more cotton")
    elif best == "Rice":
        st.success("Rice stable demand → Safe crop")
    else:
        st.warning("Diversify crops for safety")
