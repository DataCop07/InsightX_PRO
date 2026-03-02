import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Query Panel", layout="wide")

# ===========================
# Load CSV
# ===========================
try:
    df = pd.read_csv(r"C:\Users\Ram\Desktop\project\data\upi_transactions_2024.csv")
except FileNotFoundError:
    st.error("Dataset not found! Please check the path.")
    st.stop()

# Normalize column names
df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

# Clean numeric + categorical columns
if "amount_(inr)" in df.columns:
    df["amount_(inr)"] = pd.to_numeric(df["amount_(inr)"], errors="coerce")

if "transaction_status" in df.columns:
    df["transaction_status"] = df["transaction_status"].astype(str).str.strip().str.lower()

if "device_type" in df.columns:
    df["device_type"] = df["device_type"].astype(str).str.strip().str.title()

if "network_type" in df.columns:
    df["network_type"] = df["network_type"].astype(str).str.strip().str.title()

if "fraud_flag" in df.columns:
    df["fraud_flag"] = pd.to_numeric(df["fraud_flag"], errors="coerce").fillna(0)

# ===========================
# Header
# ===========================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #141E30, #243B55);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.4);
    margin-bottom: 25px;">
    <h1 style='color:white; margin:0; font-size:40px;'>
        🔍 ADVANCED QUERY ANALYSIS
    </h1>
    <p style='color:#dcdcdc; font-size:18px; margin-top:10px;'>
        Deep Data Exploration | Custom Insights | Strategic Decision Support
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("## 🧠 Query Panel")

# ---------------------------
# CATEGORY 1 – Descriptive
# ---------------------------
with st.expander("Descriptive: 📊 Transaction Overview", expanded=False):

    if st.button("Avg amount by transaction type"):
        result = df.groupby('transaction_type')['amount_(inr)'].mean().round(2)
        st.dataframe(result)
        fig = px.bar(result, x=result.index, y=result.values,
                     title="Average Amount by Transaction Type")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Success vs Failure vs Flagged rate"):
        total = len(df)
        success = (df['transaction_status'] == 'success').sum()
        failure = df['transaction_status'].isin(['failure','failed']).sum()
        flagged = (df['fraud_flag'] == 1).sum()

        st.write(f"✅ Success: {success}")
        st.write(f"❌ Failure: {failure}")
        st.write(f"🚩 Flagged: {flagged}")

        fig = px.pie(values=[success, failure, flagged],
                     names=['Success','Failure','Flagged'],
                     title="Transaction Status Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("State-wise transaction volume"):
        state_vol = df.groupby('sender_state').size().sort_values(ascending=False)
        st.dataframe(state_vol)
        fig = px.bar(state_vol, x=state_vol.index, y=state_vol.values,
                     title="Transaction Volume by State")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Total transaction value"):
        total_val = df['amount_(inr)'].sum()
        st.write(f"✅ Total Transaction Value: ₹{total_val:,.0f}")

# ---------------------------
# CATEGORY 2 – Comparative
# ---------------------------
with st.expander("Comparative: ⚖ Performance Comparison", expanded=False):

    if st.button("Failure rate: Android vs iOS"):
        rates = df[df['device_type'].isin(['Android','Ios','IOS'])] \
            .groupby('device_type')['transaction_status'] \
            .apply(lambda x: (x.isin(['failure','failed'])).mean()*100)
        st.dataframe(rates)
        fig = px.bar(rates, x=rates.index, y=rates.values,
                     title="Failure Rate by Device Type")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Network type success rate"):
        rates = df.groupby('network_type')['transaction_status'] \
            .apply(lambda x: (x=='success').mean()*100)
        st.dataframe(rates)
        fig = px.bar(rates, x=rates.index, y=rates.values,
                     title="Success Rate by Network Type")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Bank-wise failure rate"):
        rates = df.groupby('sender_bank')['transaction_status'] \
            .apply(lambda x: (x.isin(['failure','failed'])).mean()*100)
        st.dataframe(rates.sort_values())
        fig = px.bar(rates, x=rates.index, y=rates.values,
                     title="Failure Rate by Bank")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Weekend vs Weekday performance"):
        weekend_rates = df.groupby('is_weekend')['transaction_status'] \
            .apply(lambda x: (x=='success').mean()*100)
        st.dataframe(weekend_rates)
        fig = px.bar(weekend_rates, x=weekend_rates.index,
                     y=weekend_rates.values,
                     title="Weekend vs Weekday Success Rate")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Avg transaction by age group"):
        result = df.groupby('sender_age_group')['amount_(inr)'].mean()
        st.dataframe(result)
        fig = px.bar(result, x=result.index, y=result.values,
                     title="Average Transaction by Age Group")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# CATEGORY 3 – Temporal
# ---------------------------
with st.expander("Temporal: ⏳ Time-Based Analysis", expanded=False):

    if st.button("Peak transaction hours"):
        peak = df.groupby('hour_of_day').size().idxmax()
        st.write(f"✅ Peak Hour: {peak}")

    if st.button("Day of week with highest volume"):
        peak_day = df.groupby('day_of_week').size().idxmax()
        st.write(f"✅ Peak Day: {peak_day}")

    if st.button("Success variation by hour"):
        result = df.groupby('hour_of_day')['transaction_status'] \
            .apply(lambda x: (x=='success').mean()*100)
        st.dataframe(result)
        fig = px.line(result, x=result.index, y=result.values,
                      title="Success Rate by Hour")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Flagged transactions by hour"):
        result = df[df['fraud_flag']==1].groupby('hour_of_day').size()
        st.dataframe(result)

    if st.button("Hourly distribution P2M transactions"):
        result = df[df['transaction_type']=='P2M'].groupby('hour_of_day').size()
        st.dataframe(result)

# ---------------------------
# CATEGORY 4 – Segmentation
# ---------------------------
with st.expander("Segmentation: 👥 User & Ecosystem Segmentation", expanded=False):

    if st.button("Age group with most P2P"):
        result = df[df['transaction_type']=='P2P'] \
            .groupby('sender_age_group').size().sort_values(ascending=False)
        st.dataframe(result)

    if st.button("State with highest flagged ratio"):
        result = df.groupby('sender_state')['fraud_flag'].mean().sort_values(ascending=False)
        st.dataframe(result)

    if st.button("Device type in high-value transactions"):
        threshold = df['amount_(inr)'].quantile(0.9)
        result = df[df['amount_(inr)']>=threshold].groupby('device_type').size()
        st.dataframe(result)

    if st.button("Merchant category drives highest value"):
        result = df.groupby('merchant_category')['amount_(inr)'] \
            .sum().sort_values(ascending=False)
        st.dataframe(result)

    if st.button("Bank-wise transaction distribution"):
        result = df.groupby('sender_bank').size()
        st.dataframe(result)

# ---------------------------
# CATEGORY 5 – Correlation
# ---------------------------
with st.expander("Correlation: 🔍 Pattern & Correlation Analysis", expanded=False):

    if st.button("Network vs Failure rate"):
        result = df.groupby('network_type')['transaction_status'] \
            .apply(lambda x: (x.isin(['failure','failed'])).mean()*100)
        st.dataframe(result)
        fig = px.bar(result, x=result.index, y=result.values,
                     title="Failure Rate by Network")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("High-value transactions flagged?"):
        threshold = df['amount_(inr)'].quantile(0.9)
        flagged = df[(df['fraud_flag']==1) & (df['amount_(inr)']>=threshold)]
        st.write(f"✅ {flagged.shape[0]} high-value transactions flagged")

    if st.button("States with unusually high failure"):
        result = df.groupby('sender_state')['transaction_status'] \
            .apply(lambda x: (x.isin(['failure','failed'])).mean()*100) \
            .sort_values(ascending=False)
        st.dataframe(result)
        fig = px.bar(result, x=result.index, y=result.values,
                     title="Failure Rate by State")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Device type vs success"):
        result = df.groupby('device_type')['transaction_status'] \
            .apply(lambda x: (x=='success').mean()*100)
        st.dataframe(result)
        fig = px.bar(result, x=result.index, y=result.values,
                     title="Success Rate by Device Type")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Transaction amount by age group"):
        result = df.groupby('sender_age_group')['amount_(inr)'].mean()
        st.dataframe(result)

# ---------------------------
# CATEGORY 6 – Risk
# ---------------------------
with st.expander("Risk Analysis: ⚠ Risk & Anomaly Indicators", expanded=False):

    threshold = df['amount_(inr)'].quantile(0.9)

    if st.button("High-value flagged percentage"):
        high = df[df['amount_(inr)']>=threshold]
        total_high = len(high)
        if total_high > 0:
            flagged = high[high['fraud_flag']==1]
            percent = len(flagged)/total_high*100
            st.write(f"✅ {len(flagged)}/{total_high} flagged ({percent:.2f}%)")
        else:
            st.write("No high-value transactions found.")

    if st.button("Transaction type highest flagged ratio"):
        result = df.groupby('transaction_type')['fraud_flag'].mean().sort_values(ascending=False)
        st.dataframe(result)

    if st.button("Duplicate amount in short window"):
        dup = df[df.duplicated(subset=['amount_(inr)','hour_of_day'], keep=False)]
        if dup.empty:
            st.write("No duplicates found.")
        else:
            st.dataframe(dup)

    if st.button("Bank with highest flagged concentration"):
        result = df.groupby('sender_bank')['fraud_flag'].mean().sort_values(ascending=False)
        st.dataframe(result)

    if st.button("Flagged transactions by state/hour"):
        result = df[df['fraud_flag']==1].groupby(['sender_state','hour_of_day']).size()
        st.dataframe(result)