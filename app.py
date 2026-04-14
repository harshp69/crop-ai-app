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
try:
    model = joblib.load("crop_model.pkl")
except:
    model = None

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
st.write("AI + Optimization + Mandi Intelligence")

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
    if model:
        pred = model.predict([[soil, temp, rain]])[0]
        st.success(f"🌾 Recommended Crop: {crop_map[pred]}")
    else:
        st.warning("Model file missing (crop_model.pkl)")

# -------------------------
# LAND OPTIMIZATION
# -------------------------
if st.button("📊 Optimize Land"):
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
    st.pyplot(fig)

# -------------------------
# MANDI DATA
# -------------------------
st.markdown("---")
st.header("📊 Live Mandi Price & Analysis")

API_KEY = "579b464db66ec23bdd0000018ebe52ae7ee2422652ff60fe57d186f1"

@st.cache_data
def load_data():
    url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={API_KEY}&format=json&limit=20000"

    try:
        res = requests.get(url, timeout=20)

        if res.status_code != 200:
            return pd.DataFrame()

        data = res.json()

        if 'records' not in data:
            return pd.DataFrame()

        return pd.DataFrame(data['records'])

    except:
        return pd.DataFrame()

df = load_data()

# -------------------------
# FALLBACK DATA
# -------------------------
if df.empty:
    st.warning("⚠️ Live API fail → sample data use ho raha hai")

    df = pd.DataFrame({
        "state": ["Maharashtra"]*30,
        "market": ["Nagpur"]*30,
        "commodity": ["wheat"]*30,
        "modal_price": np.random.randint(2000,3000,30),
        "arrival_date": pd.date_range(end=pd.Timestamp.today(), periods=30)
    })

# -------------------------
# CLEAN DATA
# -------------------------
df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')
df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')

df = df.dropna(subset=['modal_price','arrival_date'])

df['state_clean'] = df['state'].astype(str).str.strip().str.title()
df['market_clean'] = df['market'].astype(str).str.strip().str.title()
df['commodity_clean'] = df['commodity'].astype(str).str.lower()

# -------------------------
# SELECTORS
# -------------------------
state = st.selectbox("🌍 State", sorted(df['state_clean'].unique()))
df_state = df[df['state_clean'] == state]

mandi = st.selectbox("🏙 Mandi", sorted(df_state['market_clean'].unique()))
df_mandi = df_state[df_state['market_clean'] == mandi]

crop_option = st.selectbox("🌾 Crop", ["Rice","Wheat","Bajra","Cotton"])

mapping = {
    "Rice": ["rice","paddy","basmati"],
    "Wheat": ["wheat"],
    "Bajra": ["bajra","pearl millet"],
    "Cotton": ["cotton"]
}

keywords = mapping[crop_option]

df_final = df_mandi[
    df_mandi['commodity_clean'].apply(lambda x: any(k in x for k in keywords))
]

# fallback
if df_final.empty:
    df_final = df_state[
        df_state['commodity_clean'].apply(lambda x: any(k in x for k in keywords))
    ]

if df_final.empty:
    df_final = df[
        df['commodity_clean'].apply(lambda x: any(k in x for k in keywords))
    ]

# -------------------------
# LAST 60 DAYS FILTER
# -------------------------
latest_date = df_final['arrival_date'].max()
df_final = df_final[df_final['arrival_date'] >= latest_date - pd.Timedelta(days=60)]

df_final = df_final.sort_values('arrival_date')

# -------------------------
# WEEKLY TREND
# -------------------------
st.subheader("📈 Weekly Price Trend")

df_daily = df_final.groupby('arrival_date')['modal_price'].mean()

df_weekly = df_daily.rolling(7).mean().dropna()

if len(df_weekly) > 1:
    fig, ax = plt.subplots()
    ax.plot(df_weekly.index, df_weekly.values, marker='o')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.warning("⚠️ Trend data insufficient")

# -------------------------
# FUTURE PREDICTION
# -------------------------
st.subheader("🔮 Future Price Prediction")

if len(df_weekly) > 2:
    trend_df = df_weekly.reset_index()
else:
    trend_df = df_daily.reset_index()

trend_df.columns = ['arrival_date','modal_price']

preds = predict_price(trend_df)

if preds is not None:
    cols = st.columns(7)
    for i, p in enumerate(preds):
        cols[i].metric(f"Day {i+1}", f"₹ {round(p,2)}")
else:
    st.warning("⚠️ Prediction ke liye data kam hai")
