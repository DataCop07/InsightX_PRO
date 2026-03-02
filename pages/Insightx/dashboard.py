import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="UPI Transactions Dashboard", layout="wide")

# ===========================
# Load CSV
# ===========================
try:
    df = pd.read_csv(r"C:\Users\Ram\Desktop\project\data\upi_transactions_2024.csv")
except FileNotFoundError:
    st.error("File upi_transactions_2024.csv not found! Please check the path.")
    st.stop()

# ===========================
# Normalize Column Names
# ===========================
df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

# ===========================
# Premium Dashboard Header
# ===========================
st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1f4037, #99f2c8);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 2px 6px 25px rgba(0,0,0,0.25);
        margin-bottom: 25px;">
        <h1 style='color:white; margin:0; font-size:42px;'>
            📊 UPI TRANSACTION DASHBOARD
        </h1>
        <p style='color:white; font-size:18px; margin-top:10px;'>
            Real-Time Financial Analytics | Risk Monitoring | Behavioral Intelligence
        </p>
    </div>
""", unsafe_allow_html=True)
# ===========================
# Clean Text Columns
# ===========================
text_cols = [
    'transaction_status', 'transaction_type', 'sender_state',
    'sender_bank', 'receiver_bank', 'device_type',
    'merchant_category', 'sender_age_group', 'receiver_age_group'
]

for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace("", np.nan)

df['transaction_status'] = df['transaction_status'].str.lower()

# Ensure amount numeric
df['amount_(inr)'] = pd.to_numeric(df['amount_(inr)'], errors='coerce')

# ===========================
# Gradient Card Function
# ===========================
def gradient_card(title, value, color1="#00d2ff", color2="#3a7bd5"):
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color1}, {color2});
            border-radius: 15px;
            padding: 20px;
            color: white;
            text-align: center;
            box-shadow: 2px 2px 15px rgba(0,0,0,0.2);
            margin-bottom: 10px;">
            <h4 style='margin:0;'>{title}</h4>
            <h2 style='margin:0;'>{value}</h2>
        </div>
        """, unsafe_allow_html=True)

# ===========================
# 1. Basic Structure Overview
# ===========================
st.markdown("## **1. Basic Structure Overview**")

col1, col2, col3, col4 = st.columns(4)
gradient_card("Unique States", df['sender_state'].nunique(), "#00c6ff", "#0072ff")
gradient_card("Unique Banks", pd.concat([df['sender_bank'], df['receiver_bank']]).nunique(), "#00ffd5", "#00cfff")
gradient_card("Transaction Types", df['transaction_type'].nunique(), "#ff9a9e", "#fecfef")
gradient_card("Total Transactions", df['transaction_id'].nunique(), "#f6d365", "#fda085")

# ===========================
# 2. Transaction Health
# ===========================
st.markdown("## **2. Transaction Health**")

success_count = df[df['transaction_status'].str.contains('success', na=False)].shape[0]
fail_count = df[df['transaction_status'].str.contains('fail', na=False)].shape[0]
review_count = df[df['fraud_flag'] == 1].shape[0]

total_amount = df['amount_(inr)'].sum()
avg_amount = df['amount_(inr)'].mean()
median_amount = df['amount_(inr)'].median()
max_amount = df['amount_(inr)'].max()

top1_count = max(1, int(0.01 * df.shape[0]))
top1_amount = df['amount_(inr)'].nlargest(top1_count).sum()

col1, col2, col3, col4 = st.columns(4)
gradient_card("Success Rate %", f"{(success_count/df.shape[0])*100:.2f}%", "#43e97b", "#38f9d7")
gradient_card("Failure Rate %", f"{(fail_count/df.shape[0])*100:.2f}%", "#ff6a00", "#ee0979")
gradient_card("Flagged for Review", review_count, "#ffb347", "#ffcc33")
gradient_card("Total Amount (INR)", f"{total_amount:,.0f}", "#36d1dc", "#5b86e5")

col1, col2, col3, col4 = st.columns(4)
gradient_card("Avg Transaction (INR)", f"{avg_amount:,.2f}", "#11998e", "#38ef7d")
gradient_card("Median Transaction (INR)", f"{median_amount:,.2f}", "#fc4a1a", "#f7b733")
gradient_card("Highest Transaction (INR)", f"{max_amount:,.0f}", "#f7971e", "#ffd200")
gradient_card("Top 1% Transactions Value (INR)", f"{top1_amount:,.0f}", "#00c6ff", "#0072ff")

# ===========================
# 3. Volume Summary
# ===========================
st.markdown("## **3. Volume Summary**")

total_volume = df.shape[0]
total_monetary = df['amount_(inr)'].sum()
avg_per_day = df.groupby('day_of_week').size().mean()


col1, col2, col3, col4, col5 = st.columns(5)
gradient_card("Total Transaction Volume", total_volume, "#ff5f6d", "#ffc371")
gradient_card("Total Monetary (INR)", f"{total_monetary:,.0f}", "#36d1dc", "#5b86e5")
gradient_card("Avg Transactions per Day", f"{avg_per_day:.0f}", "#43e97b", "#38f9d7")


# ===========================
# 4. Quick Ecosystem Overview
# ===========================
st.markdown("## **4. Quick Ecosystem Overview**")

fig1 = px.pie(df, names='transaction_type', title="% by Transaction Type")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.pie(df, names='device_type', title="% by Device Type")
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.pie(df, names='merchant_category', title="% by Merchant Category")
st.plotly_chart(fig3, use_container_width=True)

# ===========================
# ===========================
# 5. Time & Behavior Intelligence
# ===========================
st.markdown("## **5. Time & Behavior Intelligence**")

# Peak & Low Activity
peak_hour = df.groupby('hour_of_day').size().idxmax()
low_hour = df.groupby('hour_of_day').size().idxmin()

peak_day = df.groupby('day_of_week').size().idxmax()
low_day = df.groupby('day_of_week').size().idxmin()

# Weekend vs Weekday
weekend_days = ['Saturday', 'Sunday']
weekend_volume = df[df['day_of_week'].isin(weekend_days)].shape[0]
weekday_volume = df[~df['day_of_week'].isin(weekend_days)].shape[0]

weekend_percent = (weekend_volume / df.shape[0]) * 100
weekday_percent = (weekday_volume / df.shape[0]) * 100

col1, col2, col3, col4 = st.columns(4)
gradient_card("Peak Hour", peak_hour, "#00c6ff", "#0072ff")
gradient_card("Lowest Activity Hour", low_hour, "#ff6a00", "#ee0979")
gradient_card("Most Active Day", peak_day, "#43e97b", "#38f9d7")
gradient_card("Least Active Day", low_day, "#f7971e", "#ffd200")

col1, col2 = st.columns(2)
gradient_card("Weekend Transaction %", f"{weekend_percent:.2f}%", "#36d1dc", "#5b86e5")
gradient_card("Weekday Transaction %", f"{weekday_percent:.2f}%", "#ff5f6d", "#ffc371")

# Hourly Trend Chart
fig_hour = px.bar(
    df.groupby('hour_of_day').size().reset_index(name='count'),
    x='hour_of_day',
    y='count',
    title="Transactions by Hour"
)
st.plotly_chart(fig_hour, use_container_width=True)

# Day Trend Chart
fig_day = px.bar(
    df.groupby('day_of_week').size().reset_index(name='count'),
    x='day_of_week',
    y='count',
    title="Transactions by Day of Week"
)
st.plotly_chart(fig_day, use_container_width=True)

# Executive Insight
st.markdown("### 🔎 Key Insight")
st.markdown(f"""
- Peak transaction hour: **{peak_hour}**
- Most active day: **{peak_day}**
- Weekend transactions account for **{weekend_percent:.2f}%** of total volume.
- Weekdays dominate with **{weekday_percent:.2f}%** of total activity.

This indicates user behavior is strongly influenced by daily routines and work-week cycles.
""")