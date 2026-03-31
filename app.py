import streamlit as st
import pandas as pd
import numpy as np
import pickle
st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight:bold;
    color:#00FFAA;
}
</style>
""", unsafe_allow_html=True)

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Fraud Detection", layout="wide")

# -------------------- LOAD MODEL --------------------
model = pickle.load(open("model.pkl", "rb"))

# -------------------- SIDEBAR --------------------
st.sidebar.title("💳 Fraud Detection App")
st.sidebar.write("ML-based Credit Card Fraud Detection")
page = st.sidebar.radio("Navigation", ["Single Prediction", "Batch Prediction"])

# -------------------- TITLE --------------------
st.title("💳 Credit Card Fraud Detection System")

# -------------------- SINGLE PREDICTION --------------------
if page == "Single Prediction":

    st.subheader("🔍 Enter Transaction Details")

    cols = st.columns(3)
    input_data = []

    # Create 30 inputs in grid
    for i in range(30):
        with cols[i % 3]:
            val = st.number_input(f"Feature {i + 1}", value=0.0)
            input_data.append(val)

    if st.button("🚀 Check Transaction"):

        input_array = np.array(input_data).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        prob = model.predict_proba(input_array)[0][1]

        st.subheader("📊 Result")

        if prediction == 1:
            st.error(f"🚨 Fraud Detected! Probability: {prob:.2f}")
        else:
            st.success(f"✅ Legit Transaction. Probability: {prob:.2f}")

# -------------------- BATCH PREDICTION --------------------
elif page == "Batch Prediction":

    st.subheader("📂 Upload CSV File")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)

        st.write("📊 Data Preview")
        st.dataframe(df.head())

        if st.button("🔍 Predict All"):
            predictions = model.predict(df)
            probabilities = model.predict_proba(df)[:, 1]

            df["Prediction"] = predictions
            df["Fraud_Probability"] = probabilities

            st.success("✅ Prediction Completed")

            st.dataframe(df)

            # Download button
            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("🚀 Built with Streamlit | ML Project")