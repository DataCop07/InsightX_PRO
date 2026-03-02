import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Export Section", layout="wide")

# ===========================
# Load CSV
# ===========================
try:
    df = pd.read_csv(r"upi_transactions_2024.csv")
except FileNotFoundError:
    st.error("Dataset not found! Please check the path.")
    st.stop()

# ===========================
# Normalize Column Names
# ===========================
df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

# ===========================
# Premium Export Header
# ===========================
st.markdown("""
    <div style="
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 2px 6px 25px rgba(0,0,0,0.25);
        margin-bottom: 25px;">
        <h1 style='color:white; margin:0; font-size:40px;'>
            💎 DATA EXPORT CENTER
        </h1>
        <p style='color:white; font-size:18px; margin-top:10px;'>
            Download Filtered Data | High-Value Transactions | Risk Reports
        </p>
    </div>
""", unsafe_allow_html=True)

# ===========================
# Auto Detect Amount Column
# ===========================
amount_column = None
for col in df.columns:
    if "amount" in col:
        amount_column = col
        break

if amount_column:
    df[amount_column] = pd.to_numeric(df[amount_column], errors="coerce")

# ===========================
# Section Header
# ===========================
st.markdown("## 💎 Export Section")

# ===========================
# 1️⃣ Full Dataset
# ===========================
st.subheader("1️⃣ Export Full Dataset")
st.download_button(
    label="Download Full Dataset CSV",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name="upi_transactions_full.csv",
    mime='text/csv'
)

# ===========================
# 2️⃣ High-Value Transactions (Top 10%)
# ===========================
st.subheader("2️⃣ Export High-Value Transactions (Top 10%)")

if amount_column is None:
    st.error("No amount column detected in dataset.")
else:
    high_value_df = df[df[amount_column] >= df[amount_column].quantile(0.9)]

    st.write(f"Number of High-Value Transactions: {high_value_df.shape[0]}")
    st.dataframe(high_value_df)

    st.download_button(
        label="Download High-Value Transactions CSV",
        data=high_value_df.to_csv(index=False).encode('utf-8'),
        file_name="upi_transactions_high_value.csv",
        mime='text/csv'
    )

# ===========================
# 3️⃣ Flagged Transactions
# ===========================
st.subheader("3️⃣ Export Flagged Transactions")

if "fraud_flag" not in df.columns:
    st.warning("Column 'fraud_flag' not found in dataset.")
else:
    flagged_df = df[df["fraud_flag"] == 1]

    st.write(f"Number of Flagged Transactions: {flagged_df.shape[0]}")
    st.dataframe(flagged_df)

    st.download_button(
        label="Download Flagged Transactions CSV",
        data=flagged_df.to_csv(index=False).encode('utf-8'),
        file_name="upi_transactions_flagged.csv",
        mime='text/csv'
    )

# ===========================
# 4️⃣ Duplicate Transactions (Smart Check)
# ===========================
st.subheader("4️⃣ Export Duplicate Transactions")

if "transaction_id" not in df.columns:
    st.warning("Column 'transaction_id' not found in dataset.")
else:
    duplicate_df = df[df["transaction_id"].duplicated(keep=False)]

    st.write(f"Number of Duplicate Transactions: {duplicate_df.shape[0]}")
    st.dataframe(duplicate_df)

    st.download_button(
        label="Download Duplicate Transactions CSV",
        data=duplicate_df.to_csv(index=False).encode('utf-8'),
        file_name="upi_transactions_duplicates.csv",
        mime='text/csv'
    )
