import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from bs4 import BeautifulSoup

st.set_page_config(page_title="Mandi AI", layout="centered")

st.title("🌾 Smart Mandi Price Analysis")

# -------------------------
# OPTION SELECT
# -------------------------
option = st.radio("Select Data Source", [
    "📡 API (data.gov)",
    "⚡ Smart Auto (Recommended)",
    "🌐 Live Scraping (Experimental)"
])

# -------------------------
# LOAD API DATA
# -------------------------
API_KEY = "579b464db66ec23bdd0000018ebe52ae7ee2422652ff60fe57d186f1"

@st.cache_data
def load_api():
    url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={API_KEY}&format=json&limit=10000"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        return pd.DataFrame(data.get('records', []))
    except:
        return pd.DataFrame()

# -------------------------
# SCRAPING (LIVE)
# -------------------------
@st.cache_data
def load_scrape():
    try:
        url = "https://agmarknet.gov.in/SearchCmmMkt.aspx"
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")

        tables = soup.find_all("table")
        data = []

        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cols = row.find_all("td")
                if len(cols) > 5:
                    data.append([c.text.strip() for c in cols])

        df = pd.DataFrame(data)
        return df
    except:
        return pd.DataFrame()

# -------------------------
# DATA LOAD LOGIC
# -------------------------
if option == "📡 API (data.gov)":
    df = load_api()

elif option == "⚡ Smart Auto (Recommended)":
    df = load_api()
    if df.empty:
        st.warning("API failed → switching to scraping")
        df = load_scrape()

elif option == "🌐 Live Scraping (Experimental)":
    df = load_scrape()

# -------------------------
# CHECK DATA
# -------------------------
if df.empty:
    st.error("❌ Data load nahi ho raha")
    st.stop()

# -------------------------
# CLEAN DATA (API TYPE)
# -------------------------
if 'modal_price' in df.columns:

    df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')
    df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')

    df = df.dropna(subset=['modal_price','arrival_date'])

    df['state'] = df['state'].astype(str).str.strip().str.title()
    df['market'] = df['market'].astype(str).str.strip().str.title()
    df['commodity'] = df['commodity'].astype(str).str.strip().str.lower()

    # SELECTORS
    state = st.selectbox("🌍 State", sorted(df['state'].unique()))
    df_state = df[df['state'] == state]

    mandi = st.selectbox("🏙 Mandi", sorted(df_state['market'].unique()))
    df_mandi = df_state[df_state['market'] == mandi]

    crop = st.selectbox("🌾 Crop", ["Rice","Wheat","Bajra","Cotton"])

    mapping = {
        "Rice": ["rice","paddy"],
        "Wheat": ["wheat"],
        "Bajra": ["bajra"],
        "Cotton": ["cotton"]
    }

    df_final = df_mandi[
        df_mandi['commodity'].str.contains('|'.join(mapping[crop]), na=False)
    ]

    # FALLBACK
    if df_final.empty:
        st.warning("Fallback to state data")
        df_final = df_state[
            df_state['commodity'].str.contains('|'.join(mapping[crop]), na=False)
        ]

    if df_final.empty:
        st.warning("Fallback to all India")
        df_final = df[
            df['commodity'].str.contains('|'.join(mapping[crop]), na=False)
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
        st.pyplot(fig)
    else:
        st.warning("Not enough data")

    # -------------------------
    # FUTURE PREDICTION
    # -------------------------
    st.markdown("---")
    st.subheader("🔮 Future Price Prediction")

    if len(df_weekly) > 2:
        df_temp = df_weekly.reset_index()
        df_temp.columns = ['date','price']

        df_temp['day'] = np.arange(len(df_temp))

        model = LinearRegression()
        model.fit(df_temp[['day']], df_temp['price'])

        future = np.arange(len(df_temp), len(df_temp)+7).reshape(-1,1)
        preds = model.predict(future)

        cols = st.columns(7)
        for i,p in enumerate(preds):
            cols[i].metric(f"Day {i+1}", f"₹ {round(p,2)}")

    else:
        st.warning("Not enough data")

else:
    st.info("Scraped data raw format (visualization coming soon)")
    st.dataframe(df.head(20))
