# ==========================================================
# 🧠 UPI INTELLIGENCE LAB (OFFLINE AI VERSION)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="UPI Intelligence Lab", layout="wide")

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\project\upi_transactions_2024.csv")

df = load_data()
df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

# Add duplicate flag
df['duplicate_flag'] = df['transaction_id'].duplicated(keep=False).astype(int)

# ==========================================================
# COMPUTE SMART RISK SCORE (AI-LIKE LOGIC)
# ==========================================================
df['risk_score'] = (
    df['fraud_flag'] * 50
    + (df['amount_(inr)'] / df['amount_(inr)'].max() * 30)
    + (df['hour_of_day'].apply(lambda x: 1 if x < 6 or x > 22 else 0) * 10)
    + df['duplicate_flag'] * 10
)

# ==========================================================
# HEADER
# ==========================================================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #141E30, #243B55);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.4);
    margin-bottom: 30px;">
    <h1 style='color:white; font-size:38px;'>🧠 UPI INTELLIGENCE LAB</h1>
    <p style='color:#dcdcdc; font-size:18px;'>
    Offline AI Engine | Risk Predictor | Smart Insights | Fraud Intelligence
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# SECTION 1 — OFFLINE AI CHATBOT
# ==========================================================
st.subheader("💬 AI Insight Chatbot (Offline Smart Engine)")

query = st.text_input("Ask about transactions, fraud, risk patterns...")

if st.button("Generate Insight"):
    if query:
        query_lower = query.lower()

        if "fraud rate" in query_lower:
            fraud_rate = df['fraud_flag'].mean() * 100
            st.success(f"Current fraud rate is {fraud_rate:.2f}%.")

        elif "highest risk" in query_lower:
            highest = df.sort_values("risk_score", ascending=False).head(1)
            st.dataframe(highest)

        elif "average amount" in query_lower:
            avg_amt = df['amount_(inr)'].mean()
            st.success(f"Average transaction amount is ₹{avg_amt:,.2f}")

        elif "duplicate" in query_lower:
            dup = df['duplicate_flag'].sum()
            st.success(f"There are {dup} duplicate transactions detected.")

        else:
            st.info("Insight: High-value transactions during late night hours increase fraud probability.")
    else:
        st.warning("Please enter a question.")

st.divider()

# ==========================================================
# SECTION 2 — DIRECT UPI ENTRY (SIMULATION)
# ==========================================================
st.subheader("🔐 Direct UPI Transaction Entry (Simulation)")

with st.form("upi_form"):
    sender = st.text_input("Sender UPI ID")
    receiver = st.text_input("Receiver UPI ID")
    amount = st.number_input("Amount (INR)", min_value=1.0)
    hour = st.slider("Transaction Hour", 0, 23)
    submit = st.form_submit_button("Submit Transaction")

if submit:
    simulated_risk = (
        amount / df['amount_(inr)'].max() * 30
        + (10 if hour < 6 or hour > 22 else 0)
    )

    st.success("Transaction Sent to Gateway (Simulated)")
    st.info(f"Predicted Risk Score: {simulated_risk:.2f}")

    if simulated_risk > 50:
        st.error("⚠️ High Risk Transaction")
    else:
        st.success("✅ Low Risk Transaction")

st.divider()

# ==========================================================
# SECTION 3 — SMART RULE-BASED RISK PREDICTOR
# ==========================================================
st.subheader("🤖 AI Risk Predictor (Rule-Based Intelligence)")

amount_input = st.number_input("Test Amount")
hour_input = st.slider("Test Hour", 0, 23)

if st.button("Predict Risk Level"):
    risk_value = (
        amount_input / df['amount_(inr)'].max() * 30
        + (10 if hour_input < 6 or hour_input > 22 else 0)
    )

    if risk_value >= 60:
        st.error("High Fraud Risk")
    elif risk_value >= 30:
        st.warning("Moderate Risk")
    else:
        st.success("Low Risk")

st.divider()

# ==========================================================
# SECTION 4 — SMART SUGGESTION ENGINE
# ==========================================================
st.subheader("💡 Smart Risk Suggestions")

fraud_rate = df['fraud_flag'].mean() * 100
avg_risk = df['risk_score'].mean()

if fraud_rate > 10:
    st.warning("Fraud rate is elevated. Suggest multi-factor authentication.")

if avg_risk > 50:
    st.warning("Average risk high. Enable transaction velocity monitoring.")

if df['duplicate_flag'].sum() > 0:
    st.info("Duplicate transactions detected. Enable duplicate blocking system.")

st.divider()

# ==========================================================
# SECTION 5 — EXTERNAL FILE IMPORT
# ==========================================================
st.subheader("📂 Import External Transaction File")

uploaded_file = st.file_uploader("Upload CSV File")

if uploaded_file:
    new_df = pd.read_csv(uploaded_file)
    st.success("New Data Uploaded Successfully")
    st.dataframe(new_df.head())

st.divider()

# ==========================================================
# SECTION 6 — BEHAVIORAL FINGERPRINT DETECTOR
# ==========================================================
st.subheader("🧬 Behavioral Pattern Detection")

behavior_variance = df.groupby('sender_bank')['hour_of_day'].std().sort_values(ascending=False)

st.write("Banks with inconsistent transaction timing:")
st.dataframe(behavior_variance.head(5))

st.divider()

# ==========================================================
# SECTION 7 — FRAUD HEAT INTELLIGENCE MAP
# ==========================================================
st.subheader("🔥 Fraud Density Heat Intelligence")

heat_data = df.groupby(['day_of_week','hour_of_day'])['fraud_flag'].mean().unstack()

fig = px.imshow(
    heat_data,
    title="Fraud Density Heatmap",
    color_continuous_scale="Reds"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ==========================================================
# SECTION 8 — SYSTEM PERFORMANCE METRIC
# ==========================================================
st.subheader("📊 System Intelligence Score")

intelligence_score = 100 - fraud_rate
st.metric("System Safety Score", f"{intelligence_score:.2f}/100")