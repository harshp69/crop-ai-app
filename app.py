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

st.set_page_config(page_title="Smart Crop AI", layout="centered")

st.markdown("<h1 style='text-align: center; color: green;'>🌾 Smart Crop AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI + Optimization System</p>", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    soil = st.slider("🌱 Soil Moisture", 0, 100, 50)
    rain = st.slider("🌧 Rainfall", 0, 300, 100)

with col2:
    temp = st.slider("🌡 Temperature", 0, 50, 25)
    land = st.number_input("🌍 Total Land (hectares)", value=100)

st.markdown("---")

if st.button("🚀 Predict Crop"):
    pred = model.predict([[soil, temp, rain]])[0]
    crop = crop_map.get(pred, "Unknown")
    st.success(f"🌾 Recommended Crop: {crop}")

st.markdown("---")

st.markdown("## 💰 Smart Land Optimization")

if st.button("📊 Optimize Land"):
    pred = model.predict([[soil, temp, rain]])[0]

    profits = [-2000, -3000, -2500]
    profits[pred] -= 1000

    A = [
        [1, 1, 1],
        [2, 5, 3]
    ]

    b = [land, 300]

    bounds = [
        (land * 0.2, land),
        (land * 0.2, land),
        (land * 0.2, land)
    ]

    res = linprog(profits, A_ub=A, b_ub=b, bounds=bounds)

    x = res.x

    st.success("Balanced Land Allocation:")

    st.write(f"🌱 Bajra: {round(x[0],2)} ha")
    st.write(f"🌾 Rice: {round(x[1],2)} ha")
    st.write(f"🌿 Wheat: {round(x[2],2)} ha")

    fig, ax = plt.subplots()
    ax.bar(["Bajra", "Rice", "Wheat"], x)
    st.pyplot(fig)

st.markdown("---")
st.caption("🚀 AI + ML + Optimization System")