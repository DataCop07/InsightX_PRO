import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="UPI Risk Control", layout="wide")

# ===========================
# Load CSV
# ===========================
try:
    df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\project\upi_transactions_2024.csv")
except FileNotFoundError:
    st.error("File upi_transactions_2024.csv not found! Please check the path or upload the file.")
    st.stop()

# ===========================
# Normalize column names
# ===========================
df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

# =====================================================
# PREMIUM HEADER
# =====================================================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #141E30, #243B55);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.4);
    margin-bottom: 25px;">
    <h2 style='color:white; margin:0; font-size:32px;'>
        🛡 RISK CONTROL & FRAUD MONITORING
    </h2>
    <p style='color:#dcdcdc; font-size:16px; margin-top:8px;'>
        Risk Indicators | Fraud Signals | Safety Intelligence
    </p>
</div>
""", unsafe_allow_html=True)

# ===========================
# Premium Risk Intelligence Header
# ===========================


# ===========================
# Add duplicate flag
# ===========================
df['duplicate_flag'] = df['transaction_id'].duplicated(keep=False).astype(int)

# ===========================
# Compute Risk Score
# ===========================
df['risk_score'] = (
    df['fraud_flag'] * 50
    + (df['amount_(inr)'] / df['amount_(inr)'].max() * 30)
    + (df['hour_of_day'].apply(lambda x: 1 if x < 6 or x > 22 else 0) * 10)
    + (df['device_type'].apply(lambda x: 10 if x.lower() not in ['android', 'ios'] else 0))
    + (df['network_type'].apply(lambda x: 10 if x.lower() not in ['wifi', '4g', '5g'] else 0))
    + df['duplicate_flag'] * 10
)

# ===========================
# Sidebar Filters
# ===========================
st.sidebar.markdown("## **Risk Control Filters**")

txn_filter = st.sidebar.multiselect(
    "Transaction Type",
    df['transaction_type'].unique(),
    default=df['transaction_type'].unique()
)

merchant_filter = st.sidebar.multiselect(
    "Merchant Category",
    df['merchant_category'].dropna().unique(),
    default=df['merchant_category'].dropna().unique()
)

sender_bank_filter = st.sidebar.multiselect(
    "Sender Bank",
    df['sender_bank'].dropna().unique(),
    default=df['sender_bank'].dropna().unique()
)

receiver_bank_filter = st.sidebar.multiselect(
    "Receiver Bank",
    df['receiver_bank'].dropna().unique(),
    default=df['receiver_bank'].dropna().unique()
)

sender_age_filter = st.sidebar.multiselect(
    "Sender Age Group",
    df['sender_age_group'].dropna().unique(),
    default=df['sender_age_group'].dropna().unique()
)

receiver_age_filter = st.sidebar.multiselect(
    "Receiver Age Group",
    df['receiver_age_group'].dropna().unique(),
    default=df['receiver_age_group'].dropna().unique()
)

device_filter = st.sidebar.multiselect(
    "Device Type",
    df['device_type'].dropna().unique(),
    default=df['device_type'].dropna().unique()
)

network_filter = st.sidebar.multiselect(
    "Network Type",
    df['network_type'].dropna().unique(),
    default=df['network_type'].dropna().unique()
)

status_filter = st.sidebar.multiselect(
    "Transaction Status",
    df['transaction_status'].dropna().unique(),
    default=df['transaction_status'].dropna().unique()
)

day_filter = st.sidebar.multiselect(
    "Day of Week",
    df['day_of_week'].dropna().unique(),
    default=df['day_of_week'].dropna().unique()
)

weekend_filter = st.sidebar.multiselect(
    "Is Weekend",
    df['is_weekend'].dropna().unique(),
    default=df['is_weekend'].dropna().unique()
)

high_value = st.sidebar.slider(
    "High-Value Threshold (INR)",
    0,
    int(df['amount_(inr)'].max()),
    10000,
    step=1000
)

txn_id_search = st.sidebar.text_input("Search by Transaction ID")

apply_filter = st.sidebar.button("Apply Filters")
clear_filter = st.sidebar.button("Clear All")

# ===========================
# Filter DataFrame
# ===========================
if apply_filter:
    filtered_df = df[
        (df['transaction_type'].isin(txn_filter)) &
        (df['merchant_category'].isin(merchant_filter)) &
        (df['sender_bank'].isin(sender_bank_filter)) &
        (df['receiver_bank'].isin(receiver_bank_filter)) &
        (df['sender_age_group'].isin(sender_age_filter)) &
        (df['receiver_age_group'].isin(receiver_age_filter)) &
        (df['device_type'].isin(device_filter)) &
        (df['network_type'].isin(network_filter)) &
        (df['transaction_status'].isin(status_filter)) &
        (df['day_of_week'].isin(day_filter)) &
        (df['is_weekend'].isin(weekend_filter)) &
        (df['amount_(inr)'] >= high_value)
    ]

    if txn_id_search:
        filtered_df = filtered_df[
            filtered_df['transaction_id'].astype(str).str.contains(txn_id_search)
        ]

elif clear_filter:
    filtered_df = df.copy()
else:
    filtered_df = df.copy()

# ===========================
# Risk Control KPIs
# ===========================
st.markdown("## **6. Risk Control Section**")

total_flagged = filtered_df[filtered_df['fraud_flag'] == 1].shape[0]
total_duplicates = filtered_df[filtered_df['duplicate_flag'] == 1].shape[0]
high_risk_count = filtered_df[filtered_df['risk_score'] >= 70].shape[0]
avg_risk_score = filtered_df['risk_score'].mean()
total_risk = filtered_df[filtered_df['risk_score'] >= 50].shape[0]

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Flagged Transactions", total_flagged)
col2.metric("Duplicate Transactions", total_duplicates)
col3.metric("High-Risk Transactions (≥70)", high_risk_count)
col4.metric("Avg Risk Score", f"{avg_risk_score:.2f}")
col5.metric("% Transactions Risky (≥50)", f"{total_risk/filtered_df.shape[0]*100:.2f}%")

# ===========================
# Top Risk Drivers
# ===========================
st.subheader("Top Risk Drivers")

top_flagged_states = (
    filtered_df.groupby('sender_state')['fraud_flag']
    .mean()
    .sort_values(ascending=False)
    .head(5) * 100
)
st.dataframe(top_flagged_states)

top_merchants_risk = (
    filtered_df.groupby('merchant_category')['risk_score']
    .mean()
    .sort_values(ascending=False)
    .head(5)
)
st.dataframe(top_merchants_risk)

device_risk = filtered_df.groupby('device_type')['risk_score'].mean()

fig_device = px.bar(
    device_risk,
    x=device_risk.index,
    y=device_risk.values,
    title="Average Risk Score by Device Type",
    color=device_risk.values,
    color_continuous_scale="RdYlGn_r"
)

st.plotly_chart(fig_device, use_container_width=True)

txn_risk = filtered_df.groupby('transaction_type')['risk_score'].mean()

fig_tnx = px.bar(
    txn_risk,
    x=txn_risk.index,
    y=txn_risk.values,
    title="Average Risk Score by Transaction Type",
    color=txn_risk.values,
    color_continuous_scale="RdYlGn_r"
)

st.plotly_chart(fig_tnx, use_container_width=True)

# ===========================
# High-Risk Transactions Table
# ===========================
st.subheader("High-Risk Transactions (Risk Score ≥70)")

high_risk_df = filtered_df[filtered_df['risk_score'] >= 70]

st.dataframe(
    high_risk_df[
        [
            'transaction_id',
            'amount_(inr)',
            'sender_state',
            'sender_bank',
            'receiver_bank',
            'sender_age_group',
            'receiver_age_group',
            'transaction_type',
            'merchant_category',
            'device_type',
            'network_type',
            'fraud_flag',
            'duplicate_flag',
            'risk_score'
        ]
    ]
)

# ===========================
# Risk Trend Line
# ===========================
st.subheader("Daily Risk Trend")

daily_risk = filtered_df.groupby('day_of_week')['risk_score'].mean()

fig_trend = px.line(
    daily_risk,
    x=daily_risk.index,
    y=daily_risk.values,
    title="Average Risk Score by Day of Week",
    markers=True
)

st.plotly_chart(fig_trend, use_container_width=True)

# ===========================
# Export Option
# ===========================
st.download_button(
    label="Export High-Risk Transactions CSV",
    data=high_risk_df.to_csv(index=False).encode('utf-8'),
    file_name='high_risk_transactions.csv',
    mime='text/csv'
)