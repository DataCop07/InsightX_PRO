import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="UPI Transactions Analysis Mode", layout="wide")

# ===========================
# Load CSV
# ===========================
try:
    df = pd.read_csv(r"C:\Users\Ram\Desktop\project\data\upi_transactions_2024.csv")
except FileNotFoundError:
    st.error("File upi_transactions_2024.csv not found! Please check the path.")
    st.stop()

# ===========================
# Normalize Columns
# ===========================
df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

# ===========================
# Premium Analysis Header
# ===========================
st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 2px 6px 25px rgba(0,0,0,0.25);
        margin-bottom: 25px;">
        <h1 style='color:white; margin:0; font-size:40px;'>
            📈 TRANSACTION ANALYSIS MODE
        </h1>
        <p style='color:white; font-size:18px; margin-top:10px;'>
            Volume Trends | Behavioral Patterns | Financial Performance Insights
        </p>
    </div>
""", unsafe_allow_html=True)

# ===========================
# Data Cleaning
# ===========================
df['amount_(inr)'] = pd.to_numeric(df['amount_(inr)'], errors='coerce').fillna(0)
df['transaction_status'] = df['transaction_status'].astype(str).str.strip().str.lower()
df['fraud_flag'] = pd.to_numeric(df['fraud_flag'], errors='coerce').fillna(0).astype(int)
df['transaction_id'] = df['transaction_id'].astype(str).str.strip()

df = df[df['transaction_id'] != ""]

# ===========================
# Risk Score
# ===========================
if 'risk_score' not in df.columns:
    df['risk_score'] = (
        df['fraud_flag'] * 50 +
        (df['amount_(inr)'] / df['amount_(inr)'].max() * 30) +
        (df['hour_of_day'].apply(lambda x: 1 if x < 6 or x > 22 else 0) * 10) +
        (df['device_type'].astype(str).str.lower().apply(lambda x: 10 if x not in ['android','ios'] else 0)) +
        (df['network_type'].astype(str).str.lower().apply(lambda x: 10 if x not in ['wifi','4g','5g'] else 0))
    )

# ===========================
# Sidebar Filters
# ===========================
st.sidebar.header("🔹 Filters")

transaction_id_filter = st.sidebar.text_input("Search Transaction ID")

filters = {
    "transaction_type": st.sidebar.multiselect("Transaction Type", df['transaction_type'].unique()),
    "merchant_category": st.sidebar.multiselect("Merchant Category", df['merchant_category'].dropna().unique()),
    "transaction_status": st.sidebar.multiselect("Transaction Status", df['transaction_status'].unique()),
    "sender_state": st.sidebar.multiselect("Sender State", df['sender_state'].dropna().unique()),
    "sender_bank": st.sidebar.multiselect("Sender Bank", df['sender_bank'].dropna().unique()),
    "receiver_bank": st.sidebar.multiselect("Receiver Bank", df['receiver_bank'].dropna().unique()),
    "device_type": st.sidebar.multiselect("Device Type", df['device_type'].dropna().unique()),
    "fraud_flag": st.sidebar.multiselect("Fraud Flag", df['fraud_flag'].unique()),
}

min_amount = int(df['amount_(inr)'].min())
max_amount = int(df['amount_(inr)'].max())
amount_range = st.sidebar.slider("Amount Range (INR)", min_amount, max_amount, (min_amount, max_amount))

risk_range = st.sidebar.slider("Risk Score Range", 0, 100, (0, 100))

# ===========================
# Apply Filters
# ===========================
df_filtered = df.copy()

if transaction_id_filter:
    df_filtered = df_filtered[df_filtered['transaction_id'].str.contains(transaction_id_filter)]

for key, values in filters.items():
    if values:
        df_filtered = df_filtered[df_filtered[key].isin(values)]

df_filtered = df_filtered[
    (df_filtered['amount_(inr)'] >= amount_range[0]) &
    (df_filtered['amount_(inr)'] <= amount_range[1])
]

df_filtered = df_filtered[
    (df_filtered['risk_score'] >= risk_range[0]) &
    (df_filtered['risk_score'] <= risk_range[1])
]

# ===========================
# Smart Duplicate Detection
# ===========================
smart_duplicates = df_filtered.duplicated(
    subset=['sender_bank','receiver_bank','amount_(inr)','hour_of_day'],
    keep=False
)

smart_duplicate_count = smart_duplicates.sum()

# ===========================
# KPIs
# ===========================
st.markdown("## **Analysis Mode: Key Insights**")

total_transactions = df_filtered.shape[0]
total_amount = df_filtered['amount_(inr)'].sum()
failed_transactions = df_filtered[df_filtered['transaction_status'] == 'failed'].shape[0]
high_risk_transactions = df_filtered[df_filtered['risk_score'] >= 70].shape[0]

# Row 1
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Total Amount (INR)", f"₹ {total_amount:,.2f}")
col3.metric("Failed Transactions", f"{failed_transactions:,}")

# Row 2
col4, col5 = st.columns(2)
col4.metric("Smart Duplicate Transactions", f"{smart_duplicate_count:,}")
col5.metric("High-Risk Transactions", f"{high_risk_transactions:,}")

# ===========================
# Executive Summary
# ===========================
st.markdown("## 📊 Executive Summary")

fraud_rate = (df_filtered['fraud_flag'].mean() * 100) if total_transactions > 0 else 0
failure_rate = (failed_transactions / total_transactions * 100) if total_transactions > 0 else 0

summary_text = f"""
• Total analyzed transactions: {total_transactions:,}

• Total transaction value: ₹ {total_amount:,.2f}

• Failure rate: {failure_rate:.2f}% of total transactions.

• Fraud rate: {fraud_rate:.2f}% of transactions flagged as fraud.

• High-risk transactions (Risk ≥ 70): {high_risk_transactions:,}

• Potential smart duplicates detected: {smart_duplicate_count:,}

Overall, transaction behavior appears {'stable' if fraud_rate < 5 else 'moderately risky' if fraud_rate < 15 else 'high risk'} 
based on fraud and risk score distribution.
"""

st.info(summary_text)

# ===========================
# Charts
# ===========================
st.markdown("## **Analysis Mode: Charts**")

fig_type = px.pie(df_filtered, names='transaction_type', title="Transaction Type Distribution")
st.plotly_chart(fig_type, use_container_width=True)

fig_device = px.pie(df_filtered, names='device_type', title="Device Type Distribution")
st.plotly_chart(fig_device, use_container_width=True)

df_filtered['risk_bin'] = pd.cut(
    df_filtered['risk_score'],
    bins=[0,20,40,60,80,100],
    labels=['Low','Medium-Low','Medium','Medium-High','High']
)

fig_amount = px.histogram(
    df_filtered,
    x='amount_(inr)',
    nbins=50,
    color='risk_bin',
    title="Transaction Amount Distribution by Risk"
)
st.plotly_chart(fig_amount, use_container_width=True)

# ===========================
# Risk vs Day of Week Heatmap
# ===========================

heatmap_data = df_filtered.groupby(
    ['day_of_week', 'transaction_type']
)['risk_score'].mean().reset_index()

fig_heat = px.density_heatmap(
    heatmap_data,
    x='day_of_week',
    y='transaction_type',
    z='risk_score',
    color_continuous_scale='RdYlGn_r',
    title="Average Risk Score: Day vs Transaction Type"
)

st.plotly_chart(fig_heat, use_container_width=True)

# ===========================
# Export
# ===========================
high_risk_df = df_filtered[df_filtered['risk_score'] >= 70]

st.download_button(
    label="Export Filtered Data as CSV",
    data=df_filtered.to_csv(index=False).encode('utf-8'),
    file_name='filtered_transactions_analysis.csv',
    mime='text/csv'
)

st.download_button(
    label="Export High-Risk Transactions CSV",
    data=high_risk_df.to_csv(index=False).encode('utf-8'),
    file_name='high_risk_transactions_analysis.csv',
    mime='text/csv'
)