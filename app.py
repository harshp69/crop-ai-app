writefile app.py
import streamlit as st
import joblib
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# Load ML model
model = joblib.load("crop_model.pkl")

# Crop mapping
crop_map = {
    0: "Bajra 🌱",
    1: "Rice 🌾",
    2: "Wheat 🌿"
}

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
# ML PREDICTION
# -------------------------
if st.button("🚀 Predict Crop"):

    pred = model.predict([[soil, temp, rain]])[0]
    crop = crop_map.get(pred, "Unknown")

    st.success(f"🌾 Recommended Crop: {crop}")

st.markdown("---")

# -------------------------
# OPTIMIZATION (BALANCED)
# -------------------------
if st.button("📊 Optimize Land"):

    pred = model.predict([[soil, temp, rain]])[0]

    # Profit per hectare (₹)
    profit = [20000, 30000, 25000]

    # Cost per hectare (₹)
    cost = [8000, 12000, 10000]

    # Net profit
    net_profit = [p - c for p, c in zip(profit, cost)]

    # Convert to minimization
    c = [-x for x in net_profit]

    # Water usage (units)
    water = [2, 5, 3]

    # Max demand (hectares limit)
    demand = [land * 0.5, land * 0.6, land * 0.5]

    # Constraints
    A = [
        [1, 1, 1],     # total land
        water          # water constraint
    ]

    b = [
        land,
        300            # water limit
    ]

    # Bounds (realistic farming)
    bounds = [
        (land*0.1, demand[0]),
        (land*0.1, demand[1]),
        (land*0.1, demand[2])
    ]

    # ML boost
    c[pred] -= 2000

    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)

    x = res.x
    bajra, rice, wheat = x

    st.success("💰 Profit Optimized Allocation:")

    st.write(f"🌱 Bajra: {round(bajra,2)} ha")
    st.write(f"🌾 Rice: {round(rice,2)} ha")
    st.write(f"🌿 Wheat: {round(wheat,2)} ha")

    total_profit = (
        bajra * net_profit[0] +
        rice * net_profit[1] +
        wheat * net_profit[2]
    )

    st.metric("💰 Estimated Profit", f"₹ {int(total_profit)}")

    # Bar chart
    import matplotlib.pyplot as plt
    crops = ["Bajra", "Rice", "Wheat"]
    values = [bajra, rice, wheat]

    fig, ax = plt.subplots()
    ax.bar(crops, values)
    ax.set_title("Optimal Allocation")

    st.pyplot(fig)
