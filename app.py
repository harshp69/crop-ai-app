import streamlit as st
import joblib
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# -------------------------
# LOAD MODEL
# -------------------------
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
    df['arrival_date'] = pd.to_datetime(df['arrival_date'])
    df = df.sort_values('arrival_date')

    df['day'] = np.arange(len(df))

    X = df[['day']]
    y = df['modal_price']

    model_lr = LinearRegression()
    model_lr.fit(X, y)

    future_days = np.arange(len(df), len(df)+7).reshape(-1,1)
    return model_lr.predict(future_days)

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Smart Crop AI", layout="centered")

st.title("🌾 Smart Crop AI")
st.write("AI + Optimization System")

st.markdown("---")

# INPUTS
col1, col2 = st.columns(2)

with col1:
    soil = st.slider("🌱 Soil Moisture", 0, 100, 50)
    rain = st.slider("🌧 Rainfall", 0, 300, 100)

with col2:
    temp = st.slider("🌡 Temperature", 0, 50, 25)
    land = st.number_input("🌍 Total Land (hectares)", value=100)

# BUTTONS
c1, c2 = st.columns(2)

with c1:
    predict_btn = st.button("🚀 Predict Crop")

with c2:
    optimize_btn = st.button("📊 Optimize Land")

# -------------------------
# CROP PREDICTION
# -------------------------
if predict_btn:
    pred = model.predict([[soil, temp, rain]])[0]
    st.success(f"🌾 Recommended Crop: {crop_map[pred]}")

# -------------------------
# OPTIMIZATION
# -------------------------
if optimize_btn:
    st.subheader("📊 Optimize Land")

    profit = [20000, 30000, 25000, 35000]
    cost   = [8000, 12000, 10000, 15000]
    net_profit = [p - c for p, c in zip(profit, cost)]

    c = [-x for x in net_profit]
    water = [2, 5, 3, 4]

    A = [[1,1,1,1], water]
    b = [land, 300]

    bounds = [(0, land)] * 4

    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)

    crops = ["Bajra", "Rice", "Wheat", "Cotton"]
    values = res.x

    for crop, val in zip(crops, values):
        st.write(f"{crop}: {round(val,2)} ha")

    fig, ax = plt.subplots()
    ax.bar(crops, values)
    ax.set_title("Optimal Land Distribution")
    st.pyplot(fig)

# -------------------------
# MANDI SECTION
# -------------------------
st.markdown("---")
st.header("📊 Live Mandi Price & Analysis")

API_KEY = "YOUR_API_KEY"

@st.cache_data
def load_data():
    url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={API_KEY}&format=json&limit=10000"
    data = requests.get(url).json()
    return pd.DataFrame(data['records'])

df = load_data()

# CLEAN
df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')
df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
df = df.dropna()

# NORMALIZE
df['state'] = df['state'].str.lower().str.strip()
df['market'] = df['market'].str.lower().str.strip()
df['commodity'] = df['commodity'].str.lower().str.strip()

# SELECT
state = st.selectbox("🌍 State", sorted(df['state'].unique()))
df_state = df[df['state'] == state]

mandi = st.selectbox("🏙 Mandi", sorted(df_state['market'].unique()))
df_mandi = df_state[df_state['market'] == mandi]

crop_option = st.selectbox("🌾 Crop", ["Rice","Wheat","Bajra","Cotton"])

mapping = {
    "Rice": ["rice","paddy"],
    "Wheat": ["wheat"],
    "Bajra": ["bajra"],
    "Cotton": ["cotton"]
}

df_final = df_mandi[
    df_mandi['commodity'].str.contains('|'.join(mapping[crop_option]), na=False)
]

# FALLBACK
if df_final.empty:
    st.warning("⚠️ Exact mandi data nahi mila, state level data dikha rahe hain")
    df_final = df[
        (df['state'] == state) &
        (df['commodity'].str.contains('|'.join(mapping[crop_option]), na=False))
    ]

df_final = df_final.sort_values('arrival_date')

# -------------------------
# SECTION 1: WEEKLY TREND
# -------------------------
st.markdown("---")
st.subheader("📈 Weekly Price Trend")

df_weekly = df_final.set_index('arrival_date').resample('W')['modal_price'].mean().dropna()

if len(df_weekly) > 1:
    fig, ax = plt.subplots()
    ax.plot(df_weekly.index, df_weekly.values, marker='o')
    ax.set_title("Weekly Price Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    plt.xticks(rotation=45)

    st.pyplot(fig)
else:
    st.warning("Not enough data")

# -------------------------
# SECTION 2: FUTURE PREDICTION
# -------------------------
st.markdown("---")
st.subheader("🔮 Future Price Prediction")

if len(df_weekly) > 2:
    trend_df = df_weekly.reset_index()
    preds = predict_price(trend_df)

    cols = st.columns(7)
    for i, p in enumerate(preds):
        cols[i].metric(f"Day {i+1}", f"₹ {round(p,2)}")
else:
    st.warning("Not enough data for prediction")
