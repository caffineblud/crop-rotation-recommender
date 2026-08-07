"""
AI-Based Crop Rotation Recommendation System
Streamlit demo app (run locally: streamlit run app.py)
"""

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Crop Rotation Recommender", page_icon="🌾", layout="centered")

st.title("🌾 AI-Based Crop Rotation Recommendation System")
st.markdown(
    "Enter your soil and climate data, plus the crop you last grew, "
    "to get a recommended next crop for healthier rotation."
)

# ---- Load model (train this in the notebook first, save with joblib) ----
MODEL_PATH = "../notebooks/crop_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# ---- Crop rotation domain rules (keep in sync with notebook) ----
crop_metadata = {
    "rice": {"family": "cereal", "nitrogen_effect": "depleting"},
    "maize": {"family": "cereal", "nitrogen_effect": "depleting"},
    "chickpea": {"family": "legume", "nitrogen_effect": "fixing"},
    "kidneybeans": {"family": "legume", "nitrogen_effect": "fixing"},
    "pigeonpeas": {"family": "legume", "nitrogen_effect": "fixing"},
    "mothbeans": {"family": "legume", "nitrogen_effect": "fixing"},
    "mungbean": {"family": "legume", "nitrogen_effect": "fixing"},
    "blackgram": {"family": "legume", "nitrogen_effect": "fixing"},
    "lentil": {"family": "legume", "nitrogen_effect": "fixing"},
    "cotton": {"family": "fiber", "nitrogen_effect": "depleting"},
    "jute": {"family": "fiber", "nitrogen_effect": "neutral"},
    # TODO: fill in the rest of your dataset's crop labels
}

st.header("1. Soil & Climate Inputs")
col1, col2 = st.columns(2)
with col1:
    N = st.number_input("Nitrogen (N)", 0, 200, 50)
    P = st.number_input("Phosphorus (P)", 0, 200, 50)
    K = st.number_input("Potassium (K)", 0, 200, 50)
    ph = st.slider("Soil pH", 0.0, 14.0, 6.5)
with col2:
    temperature = st.number_input("Temperature (°C)", -10.0, 55.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

st.header("2. Previous Crop")
previous_crop = st.selectbox("What did you grow last season?", list(crop_metadata.keys()))

if st.button("Get Recommendation"):
    if model is None:
        st.warning(
            "No trained model found yet. Train and save one in the notebook "
            "(joblib.dump(model, 'crop_model.pkl')) before running this app."
        )
    else:
        input_df = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
        )
        # Get top-k predicted crops with probabilities
        probs = model.predict_proba(input_df)[0]
        classes = model.classes_
        ranked = sorted(zip(classes, probs), key=lambda x: -x[1])

        prev_family = crop_metadata.get(previous_crop, {}).get("family")
        prev_n_effect = crop_metadata.get(previous_crop, {}).get("nitrogen_effect")

        st.subheader("Recommendations")
        shown = 0
        for crop, prob in ranked:
            meta = crop_metadata.get(crop, {})
            # Rotation rule: avoid same family as previous crop
            if meta.get("family") == prev_family:
                continue
            reason = ""
            if prev_n_effect == "depleting" and meta.get("nitrogen_effect") == "fixing":
                reason = " — good nitrogen-fixing choice after a depleting crop"
            st.write(f"**{crop.title()}** — model confidence: {prob:.1%}{reason}")
            shown += 1
            if shown >= 3:
                break

st.markdown("---")
st.caption("University project prototype — not for real-world farming decisions.")
