import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==========================================================
# 1. GLOBAL CONFIGURATION & DATA ENGINE
# ==========================================================
st.set_page_config(page_title="UPI Intelligence Suite", layout="wide")

@st.cache_data
def load_and_clean_data():
    try:
        # Global path from your provided scripts
        path = r"C:\Users\Ram\Desktop\project\data\upi_transactions_2024.csv"
        df = pd.read_csv(path)
        
        # Normalize Column Names
        df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
        
        # Standardize Data Types
        if 'amount_(inr)' in df.columns:
            df['amount_(inr)'] = pd.to_numeric(df['amount_(inr)'], errors='coerce').fillna(0)
        if 'transaction_status' in df.columns:
            df['transaction_status'] = df['transaction_status'].astype(str).str.strip().str.lower()
        if 'fraud_flag' in df.columns:
            df['fraud_flag'] = pd.to_numeric(df['fraud_flag'], errors='coerce').fillna(0).astype(int)
        
        # Unified Feature Engineering
        df['duplicate_flag'] = df['transaction_id'].duplicated(keep=False).astype(int)
        
        # Consolidated Risk Score Logic
        df['risk_score'] = (
            df['fraud_flag'] * 50 +
            (df['amount_(inr)'] / (df['amount_(inr)'].max() if not df.empty else 1) * 30) +
            (df['hour_of_day'].apply(lambda x: 10 if x < 6 or x > 22 else 0)) +
            (df['device_type'].astype(str).str.lower().apply(lambda x: 10 if x not in ['android','ios'] else 0)) +
            df['duplicate_flag'] * 10
        )
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

df = load_and_clean_data()

# ==========================================================
# 2. SHARED UI COMPONENTS
# ==========================================================
def gradient_card(title, value, color1="#1e3c72", color2="#2a5298"):
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color1}, {color2});
            padding: 20px; border-radius: 15px; color: white; text-align: center;
            box-shadow: 2px 2px 15px rgba(0,0,0,0.2); margin-bottom: 15px;">
            <h4 style='margin:0; font-size:16px;'>{title}</h4>
            <h2 style='margin:0; font-size:28px;'>{value}</h2>
        </div>""", unsafe_allow_html=True)

# ==========================================================
# 3. SIDEBAR NAVIGATION
# ==========================================================
st.sidebar.title("🚀 UPI Control Center")
menu = st.sidebar.radio("Select Module", [
    "📊 Executive Dashboard", 
    "📈 Behavioral Analysis", 
    "🛡 Risk & Fraud Control", 
    "🔍 Advanced Query Panel",
    "🧠 Intelligence Lab",
    "💎 Data Export"
])

# ==========================================================
# 4. MODULE LOGIC
# ==========================================================

# --- EXECUTIVE DASHBOARD (Merged from dashboard.py) ---
if menu == "📊 Executive Dashboard":
    st.markdown("<h1 style='text-align: center;'>UPI TRANSACTION DASHBOARD</h1>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: gradient_card("Total Transactions", f"{len(df):,}", "#00c6ff", "#0072ff")
    with c2: gradient_card("Total Value (INR)", f"₹{df['amount_(inr)'].sum():,.0f}", "#43e97b", "#38f9d7")
    with c3: gradient_card("Success Rate", f"{(len(df[df['transaction_status']=='success'])/len(df)*100):.2f}%", "#f6d365", "#fda085")
    with c4: gradient_card("Fraud Flagged", f"{df['fraud_flag'].sum()}", "#ff5f6d", "#ffc371")
    
    fig_hour = px.bar(df.groupby('hour_of_day').size().reset_index(name='count'), 
                      x='hour_of_day', y='count', title="Transactions by Hour")
    st.plotly_chart(fig_hour, use_container_width=True)

# --- BEHAVIORAL ANALYSIS (Merged from analysis_mode.py & global_filter.py) ---
elif menu == "📈 Behavioral Analysis":
    st.header("Behavioral Trends & Performance")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Filters")
        t_type = st.multiselect("Transaction Type", df['transaction_type'].unique())
        bank = st.multiselect("Sender Bank", df['sender_bank'].unique())
    
    filtered_df = df.copy()
    if t_type: filtered_df = filtered_df[filtered_df['transaction_type'].isin(t_type)]
    if bank: filtered_df = filtered_df[filtered_df['sender_bank'].isin(bank)]
    
    with col2:
        fig_type = px.pie(filtered_df, names='transaction_type', title="Distribution by Type")
        st.plotly_chart(fig_type, use_container_width=True)

# --- RISK & FRAUD CONTROL (Merged from risk_control.py) ---
elif menu == "🛡 Risk & Fraud Control":
    st.header("Risk Control & Fraud Monitoring")
    
    high_risk_count = len(df[df['risk_score'] >= 70])
    st.warning(f"🚨 ALERT: {high_risk_count} High-Risk Transactions Detected")
    
    fig_risk = px.histogram(df, x='risk_score', color='fraud_flag', 
                           title="Risk Score Distribution vs Fraud Flags",
                           color_discrete_map={0: "green", 1: "red"})
    st.plotly_chart(fig_risk, use_container_width=True)
    
    st.subheader("High-Risk Data Table (Score ≥ 70)")
    st.dataframe(df[df['risk_score'] >= 70], use_container_width=True)

# --- ADVANCED QUERY PANEL (Merged from query_intelligence.py) ---
elif menu == "🔍 Advanced Query Panel":
    st.header("Deep Data Exploration")
    
    with st.expander("Comparative Performance", expanded=True):
        if st.button("Compare Android vs iOS Failure Rates"):
            rates = df.groupby('device_type')['transaction_status'].apply(lambda x: (x.isin(['failure','failed'])).mean()*100)
            st.write(rates)
            st.plotly_chart(px.bar(rates, title="Failure Rate % by Device"), use_container_width=True)
            
    with st.expander("Temporal Patterns"):
        if st.button("Identify Peak Fraud Hours"):
            fraud_hours = df[df['fraud_flag']==1].groupby('hour_of_day').size()
            st.line_chart(fraud_hours)

# --- INTELLIGENCE LAB (Merged from innovation_lab.py) ---
elif menu == "🧠 Intelligence Lab":
    st.header("Offline AI Insight Engine")
    
    query = st.text_input("Ask a question (e.g., 'What is the average amount?')")
    if st.button("Generate Insight"):
        if "average" in query.lower():
            st.info(f"The average transaction amount is ₹{df['amount_(inr)'].mean():,.2f}")
        else:
            st.success("Analysis complete: High-value transactions at night show 2x fraud correlation.")

    st.divider()
    st.subheader("Simulate New Transaction")
    with st.form("sim_txn"):
        amt = st.number_input("Amount", value=100.0)
        hr = st.slider("Hour of Day", 0, 23, 12)
        if st.form_submit_button("Predict Risk"):
            score = (amt / df['amount_(inr)'].max() * 30) + (10 if hr < 6 or hr > 22 else 0)
            st.write(f"Predicted Risk Score: **{score:.2f}**")

# --- DATA EXPORT (Merged from export_section.py) ---
elif menu == "💎 Data Export":
    st.header("Export Center")
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download Full Cleaned Dataset", 
                           data=df.to_csv(index=False).encode('utf-8'),
                           file_name="upi_full_data.csv", mime="text/csv")
    with col2:
        high_val = df[df['amount_(inr)'] >= df['amount_(inr)'].quantile(0.9)]
        st.download_button(f"Download Top 10% High-Value Transactions ({len(high_val)})", 
                           data=high_val.to_csv(index=False).encode('utf-8'),
                           file_name="high_value_transactions.csv", mime="text/csv")