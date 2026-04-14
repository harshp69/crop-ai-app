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
# PRICE PREDICTION FUNCTION
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
# UI
# -------------------------
st.set_page_config(page_title="Smart Crop AI", layout="centered")

st.markdown("<h1 style='text-align:center;color:green;'>🌾 Smart Crop AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI + Optimization System</p>", unsafe_allow_html=True)

st.markdown("---")

# INPUTS
col1, col2 = st.columns(2)

with col1:
    soil = st.slider("🌱 Soil Moisture", 0, 100, 50)
    rain = st.slider("🌧 Rainfall", 0, 300, 100)

with col2:
    temp = st.slider("🌡 Temperature", 0, 50, 25)
    land = st.number_input("🌍 Total Land (hectares)", value=100)

st.markdown("---")

# SESSION STATE
if "show_crop" not in st.session_state:
    st.session_state.show_crop = False
if "show_opt" not in st.session_state:
    st.session_state.show_opt = False

c1, c2 = st.columns(2)

with c1:
    if st.button("🚀 Predict Crop"):
        st.session_state.show_crop = True

with c2:
    if st.button("📊 Optimize Land"):
        st.session_state.show_opt = True

# -------------------------
# CROP RESULT
# -------------------------
if st.session_state.show_crop:
    pred = model.predict([[soil, temp, rain]])[0]
    st.success(f"🌾 Recommended Crop: {crop_map[pred]}")

# -------------------------
# OPTIMIZATION + GRAPH
# -------------------------
if st.session_state.show_opt:

    st.markdown("## 📊 Optimize Land")

    profit = [20000, 30000, 25000, 35000]
    cost   = [8000, 12000, 10000, 15000]
    net_profit = [p - c for p, c in zip(profit, cost)]

    c = [-x for x in net_profit]
    water = [2, 5, 3, 4]

    A = [[1,1,1,1], water]
    b = [land, 300]

    bounds = [(0, land)] * 4

    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)

    bajra, rice, wheat, cotton = res.x

    st.success("💰 Optimized Allocation")

    st.write(f"🌱 Bajra: {round(bajra,2)} ha")
    st.write(f"🌾 Rice: {round(rice,2)} ha")
    st.write(f"🌿 Wheat: {round(wheat,2)} ha")
    st.write(f"🧵 Cotton: {round(cotton,2)} ha")

    # GRAPH
    crops = ["Bajra", "Rice", "Wheat", "Cotton"]
    values = [bajra, rice, wheat, cotton]

    fig, ax = plt.subplots()
    ax.bar(crops, values)
    ax.set_title("Optimal Land Distribution")
    st.pyplot(fig)

# -------------------------
# MANDI SECTION (FINAL FIX)
# -------------------------
st.markdown("---")
st.header("📊 Live Mandi Price & Analysis")

API_KEY = "579b464db66ec23bdd0000018ebe52ae7ee2422652ff60fe57d186f1"

@st.cache_data
def load_data():
    url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={API_KEY}&format=json&limit=5000"
    data = requests.get(url).json()
    return pd.DataFrame(data['records'])

df = load_data()

# CLEAN DATA
df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')
df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
df = df.dropna(subset=['modal_price','arrival_date'])

# SELECT
state = st.selectbox("🌍 State", sorted(df['state'].dropna().unique()))
df_state = df[df['state'] == state]

mandi = st.selectbox("🏙 Mandi", sorted(df_state['market'].dropna().unique()))
df_mandi = df_state[df_state['market'] == mandi]

crop_option = st.selectbox("🌾 Crop", ["Rice","Wheat","Bajra","Cotton"])

mapping = {
    "Rice":["Rice","Paddy"],
    "Wheat":["Wheat"],
    "Bajra":["Bajra"],
    "Cotton":["Cotton"]
}

df_final = df_mandi[
    df_mandi['commodity'].str.contains('|'.join(mapping[crop_option]), case=False, na=False)
]

if df_final.empty:
    st.warning("No data found for this crop")
else:

    df_final = df_final.sort_values('arrival_date')

    # WEEKLY TREND
    df_weekly = df_final.set_index('arrival_date').resample('W').mean()
    df_weekly = df_weekly.dropna()

    st.subheader("📈 Price Trend (Weekly)")

    fig, ax = plt.subplots()
    ax.plot(df_weekly.index, df_weekly['modal_price'], marker='o')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.set_title("Weekly Price Trend")
    plt.xticks(rotation=45)

    st.pyplot(fig)

    # PREDICTION
    st.subheader("🔮 Future Price (Next 7 Days)")
    preds = predict_price(df_weekly.reset_index())

    for i, p in enumerate(preds):
        st.metric(f"Day {i+1}", f"₹ {round(p,2)}")
