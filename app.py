# app.py (FINAL PROFESSIONAL VERSION)

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Fraud Detection", layout="wide")

# -------------------------
# STYLE
# -------------------------
st.markdown("""
<style>
.main { background-color: #0E1117; }
h1, h2, h3 { color: #00FFD1; }
.stButton>button {
    background: linear-gradient(90deg, #00FFD1, #00BFFF);
    color: black;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# TITLE
# -------------------------
st.title("💳 Credit Card Fraud Detection System")
st.caption("Real-time fraud detection using Machine Learning")

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join("data", "creditcard.csv"))

df = load_data()

# Sidebar
st.sidebar.header("⚙️ Controls")
sample_size = st.sidebar.slider("Sample Size", 10000, 100000, 50000)
df = df.sample(sample_size, random_state=42)

# -------------------------
# DATASET (EXPANDER)
# -------------------------
with st.expander("📊 View Dataset Preview"):
    st.dataframe(df.head(10))

# -------------------------
# CLASS DISTRIBUTION
# -------------------------
st.subheader("📉 Fraud vs Normal Transactions")

fig, ax = plt.subplots()
df['Class'].value_counts().plot(kind='bar', ax=ax)
ax.set_title("Class Distribution")
st.pyplot(fig)

# -------------------------
# AUTO-FILL BUTTONS
# -------------------------
st.subheader("🎯 Load Sample Data")

if st.button("✅ Use Normal Transaction"):
    sample = df[df['Class'] == 0].sample(1).iloc[0]
    st.session_state["input_data"] = sample.drop("Class").values.tolist()
    st.session_state["actual_class"] = 0
    st.success("Loaded Normal Transaction")

elif st.button("🚨 Use Fraud Transaction"):
    sample = df[df['Class'] == 1].sample(1).iloc[0]
    st.session_state["input_data"] = sample.drop("Class").values.tolist()
    st.session_state["actual_class"] = 1
    st.error("Loaded Fraud Transaction")

elif st.button("🎲 Random Transaction"):
    sample = df.sample(1).iloc[0]
    st.session_state["input_data"] = sample.drop("Class").values.tolist()
    st.session_state["actual_class"] = sample["Class"]
    st.info(f"Actual Class: {sample['Class']}")

# -------------------------
# INPUT SECTION
# -------------------------
st.subheader("🧪 Test a Transaction")

X = df.drop("Class", axis=1)

input_data = st.session_state.get("input_data", [0.0]*len(X.columns))

cols = st.columns(3)

for i, col in enumerate(X.columns):
    input_data[i] = cols[i % 3].number_input(
        col,
        value=float(input_data[i]),
        format="%.5f"
    )

st.session_state["input_data"] = input_data

st.markdown("---")

# -------------------------
# PREDICTION
# -------------------------
if st.button("🔍 Predict Fraud"):

    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # Confidence bar
    st.progress(float(prob))

    # Prediction output
    if prediction == 1:
        st.error(f"🚨 FRAUD DETECTED (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Legit Transaction (Confidence: {1-prob:.2f})")

    # -------------------------
    # ACTUAL vs PREDICTED
    # -------------------------
    actual = st.session_state.get("actual_class", None)

    if actual is not None:
        st.markdown("### 🔎 Actual vs Predicted")

        col1, col2 = st.columns(2)

        with col1:
            if actual == 1:
                st.error("Actual: 🚨 Fraud")
            else:
                st.success("Actual: ✅ Legit")

        with col2:
            if prediction == 1:
                st.error("Predicted: 🚨 Fraud")
            else:
                st.success("Predicted: ✅ Legit")

    # -------------------------
    # FEATURE IMPORTANCE
    # -------------------------
    st.subheader("📊 Feature Importance")

    importances = model.feature_importances_
    feature_names = X.columns

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    fig2, ax2 = plt.subplots()
    ax2.barh(feat_df["Feature"], feat_df["Importance"])
    ax2.invert_yaxis()
    ax2.set_title("Top 10 Important Features")

    st.pyplot(fig2)