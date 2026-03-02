import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="UPI Transactions Dashboard", layout="wide")

# ===========================
# Load CSV
# ===========================
try:
    df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\project\upi_transactions_2024.csv")
except FileNotFoundError:
    st.error("File not found! Please check the dataset path.")
    st.stop()

# ===========================
# Normalize Column Names
# ===========================
df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

# ===========================
# Floating Gradient Header Card
# ===========================
st.markdown("""
    <div style="
        background: linear-gradient(135deg, #00c6ff, #0072ff);
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 2px 4px 20px rgba(0,0,0,0.2);
        margin-bottom: 20px;">
        <h1 style='color:white; margin:0;'>🔎 GLOBAL FILTER</h1>
        <p style='color:white; margin:0;'>Interactive Transaction Filtering & Dynamic Analysis</p>
    </div>
""", unsafe_allow_html=True)

# ===========================
# Sidebar Filters
# ===========================
st.sidebar.header("🔍 Filter Controls")

txn_id_input = st.sidebar.text_input("Search Transaction ID")

transaction_types = st.sidebar.multiselect("Transaction Type", options=df['transaction_type'].unique())
merchant_categories = st.sidebar.multiselect("Merchant Category", options=df['merchant_category'].unique())
transaction_statuses = st.sidebar.multiselect("Transaction Status", options=df['transaction_status'].unique())
sender_states = st.sidebar.multiselect("Sender State", options=df['sender_state'].unique())
device_types = st.sidebar.multiselect("Device Type", options=df['device_type'].unique())
days_of_week = st.sidebar.multiselect("Day of Week", options=df['day_of_week'].unique())

min_amount = int(df['amount_(inr)'].min())
max_amount = int(df['amount_(inr)'].max())
amount_range = st.sidebar.slider("Amount (INR)", min_value=min_amount, max_value=max_amount, value=(min_amount, max_amount))

apply_btn = st.sidebar.button("Apply Filter")

# ===========================
# Filter Logic
# ===========================
filtered_df = df.copy()

if apply_btn:
    if txn_id_input.strip() != "":
        filtered_df = filtered_df[filtered_df['transaction_id'].astype(str) == txn_id_input.strip()]
    if transaction_types:
        filtered_df = filtered_df[filtered_df['transaction_type'].isin(transaction_types)]
    if merchant_categories:
        filtered_df = filtered_df[filtered_df['merchant_category'].isin(merchant_categories)]
    if transaction_statuses:
        filtered_df = filtered_df[filtered_df['transaction_status'].isin(transaction_statuses)]
    if sender_states:
        filtered_df = filtered_df[filtered_df['sender_state'].isin(sender_states)]
    if device_types:
        filtered_df = filtered_df[filtered_df['device_type'].isin(device_types)]
    if days_of_week:
        filtered_df = filtered_df[filtered_df['day_of_week'].isin(days_of_week)]

    filtered_df = filtered_df[
        (filtered_df['amount_(inr)'] >= amount_range[0]) &
        (filtered_df['amount_(inr)'] <= amount_range[1])
    ]

# ===========================
# Show Filtered Data (Longer Window)
# ===========================
st.markdown("## 📋 Filtered Transactions")
st.dataframe(
    filtered_df.reset_index(drop=True),
    use_container_width=True,
    height=600
)

# ===========================
# Summary Metrics
# ===========================
st.markdown("## 📊 Summary Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", filtered_df['transaction_id'].nunique())
col2.metric("Total Amount (INR)", f"{filtered_df['amount_(inr)'].sum():,.0f}")
col3.metric("Unique States", filtered_df['sender_state'].nunique())
col4.metric("Unique Devices", filtered_df['device_type'].nunique())

# ===========================
# Pie Charts
# ===========================
fig1 = px.pie(filtered_df, names='transaction_type', title="Transaction Type Distribution")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.pie(filtered_df, names='device_type', title="Device Type Distribution")
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.pie(filtered_df, names='merchant_category', title="Merchant Category Distribution")
st.plotly_chart(fig3, use_container_width=True)

# ===========================
# NEW: Graph Analysis Based on Filter (Added)
# ===========================
st.markdown("## 📈 Filter-Based Trend Analysis")

# Bar Chart - Transactions by Day
day_chart = px.bar(
    filtered_df.groupby('day_of_week').size().reset_index(name='count'),
    x='day_of_week',
    y='count',
    title="Transactions by Day (Filtered)"
)
st.plotly_chart(day_chart, use_container_width=True)

# Line Chart - Hourly Trend
hour_chart = px.line(
    filtered_df.groupby('hour_of_day').size().reset_index(name='count'),
    x='hour_of_day',
    y='count',
    title="Hourly Transaction Trend (Filtered)"
)
st.plotly_chart(hour_chart, use_container_width=True)

# Executive Insight
st.markdown("### 🔎 Key Insight")
st.markdown(f"""
- Filtered dataset contains **{filtered_df.shape[0]} transactions**.
- Total transaction value is **₹{filtered_df['amount_(inr)'].sum():,.0f}**.
- Most active transaction day: **{filtered_df['day_of_week'].mode()[0] if not filtered_df.empty else 'N/A'}**.
- Peak hour in filtered data: **{filtered_df['hour_of_day'].mode()[0] if not filtered_df.empty else 'N/A'}**.

This analysis dynamically adapts to applied filters, helping identify behavior patterns and business insights.
""")