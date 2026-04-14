import streamlit as st
import joblib
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Smart Crop AI", layout="centered")

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
    if len(df) < 3:
        return None

    df = df.copy()
    df['day'] = np.arange(len(df))

    model_lr = LinearRegression()
    model_lr.fit(df[['day']], df['modal_price'])

    future_days = np.arange(len(df), len(df)+7).reshape(-1,1)
    return model_lr.predict(future_days)

# -------------------------
# HEADER
# -------------------------
st.title("🌾 Smart Crop AI")
st.write("AI + Optimization System")

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

# -------------------------
# CROP PREDICTION
# -------------------------
if st.button("🚀 Predict Crop"):
    pred = model.predict([[soil, temp, rain]])[0]
    st.success(f"🌾 Recommended Crop: {crop_map[pred]}")

# -------------------------
# LAND OPTIMIZATION
# -------------------------
if st.button("📊 Optimize Land"):
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
st.header("📊 Latest Mandi Price & Analysis")

API_KEY = "579b464db66ec23bdd0000018ebe52ae7ee2422652ff60fe57d186f1"

@st.cache_data
def load_data():
    url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={API_KEY}&format=json&limit=10000"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        return pd.DataFrame(data.get('records', []))
    except:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("❌ Data load nahi ho raha")
    st.stop()

# CLEAN
df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')
df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
df = df.dropna(subset=['modal_price','arrival_date'])

# CLEAN TEXT
df['state_clean'] = df['state'].astype(str).str.strip().str.title()
df['market_clean'] = df['market'].astype(str).str.strip().str.title()
df['commodity_clean'] = df['commodity'].astype(str).str.strip().str.lower()

# -------------------------
# AUTO SELECT (IMPORTANT)
# -------------------------
latest_state = df['state_clean'].mode()[0]
df_state = df[df['state_clean'] == latest_state]

latest_mandi = df_state['market_clean'].mode()[0]

state = st.selectbox("🌍 State", sorted(df['state_clean'].unique()),
                     index=list(sorted(df['state_clean'].unique())).index(latest_state))

df_state = df[df['state_clean'] == state]

mandi = st.selectbox("🏙 Mandi", sorted(df_state['market_clean'].unique()),
                     index=list(sorted(df_state['market_clean'].unique())).index(latest_mandi))

df_mandi = df_state[df_state['market_clean'] == mandi]

crop_option = st.selectbox("🌾 Crop", ["Rice","Wheat","Bajra","Cotton"])

mapping = {
    "Rice": ["rice","paddy","basmati"],
    "Wheat": ["wheat"],
    "Bajra": ["bajra","pearl millet"],
    "Cotton": ["cotton"]
}

df_final = df_mandi[
    df_mandi['commodity_clean'].str.contains('|'.join(mapping[crop_option]), na=False)
]

# -------------------------
# FALLBACK
# -------------------------
if df_final.empty:
    st.warning("⚠️ Mandi data nahi mila → state data use kar rahe hain")
    df_final = df_state[
        df_state['commodity_clean'].str.contains('|'.join(mapping[crop_option]), na=False)
    ]

if df_final.empty:
    st.warning("⚠️ State data bhi nahi mila → all India data use kar rahe hain")
    df_final = df[
        df['commodity_clean'].str.contains('|'.join(mapping[crop_option]), na=False)
    ]

df_final = df_final.sort_values('arrival_date')

# -------------------------
# WEEKLY TREND
# -------------------------
st.markdown("---")
st.subheader("📈 Weekly Price Trend")

df_weekly = df_final.set_index('arrival_date')['modal_price'].resample('W').mean().dropna()

if len(df_weekly) > 1:
    fig, ax = plt.subplots()
    ax.plot(df_weekly.index, df_weekly.values, marker='o')
    ax.set_title("Weekly Price Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.warning("⚠️ Weekly data available nahi hai")

# -------------------------
# FUTURE PREDICTION
# -------------------------
st.markdown("---")
st.subheader("🔮 Future Price Prediction")

if len(df_weekly) > 2:
    trend_df = df_weekly.reset_index()
    trend_df.columns = ['arrival_date','modal_price']

    preds = predict_price(trend_df)

    if preds is not None:
        cols = st.columns(7)
        for i, p in enumerate(preds):
            cols[i].metric(f"Day {i+1}", f"₹ {round(p,2)}")
else:
    st.warning("⚠️ Prediction ke liye enough data nahi")
