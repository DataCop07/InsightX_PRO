import streamlit as st
import pandas as pd
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="InsightX",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD IMAGE ----------------
landing_image = Image.open("landing_image.jpg")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* Animated Gradient Background */
.main { background: linear-gradient(-45deg, #E0F7FA, #F8FBFF, #E3F2FD, #F1F8FF);
       background-size: 400% 400%; animation: gradient 12s ease infinite; }
@keyframes gradient { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }

/* Logo */
.logo-container { position: relative; display: inline-block; }
.logo { font-size: 64px; font-weight: 800; background: linear-gradient(90deg,#00B4D8,#0077B6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.logo-container::after { content: ""; position: absolute; left: 0; bottom: -8px; width: 0%; height: 4px; background: linear-gradient(90deg,#00B4D8,#0077B6); animation: underline 2s ease forwards; }
@keyframes underline { to { width: 100%; } }

/* Badge */
.badge { display: inline-block; background: #00B4D8; color: white; font-size: 12px; padding: 6px 12px; border-radius: 20px; margin-left: 12px; vertical-align: middle; }

/* Tagline */
.tagline { font-size: 20px; color: #475569; margin-top: 15px; }

/* Glass Hero */
.glass { backdrop-filter: blur(12px); background: rgba(255,255,255,0.7); padding: 60px; border-radius: 25px; box-shadow: 0px 10px 40px rgba(0,0,0,0.1); text-align: center; }

/* Divider */
.divider { margin: 50px auto; width: 80%; height: 2px; background: linear-gradient(to right, transparent, #cbd5e1, transparent); }

/* KPI Cards */
.kpi { background: white; padding: 30px; border-radius: 20px; box-shadow: 0px 8px 30px rgba(0,0,0,0.08); transition: 0.3s; text-align: center; }
.kpi:hover { transform: translateY(-6px); }

/* Buttons */
.stButton>button { background: linear-gradient(90deg,#00B4D8,#0096C7); color: white; border-radius: 30px; height: 50px; font-weight: 600; border: none; }

/* Footer */
.footer { text-align:center; padding:30px; margin-top:60px; border-top:1px solid #e5e7eb; color:#64748B; font-size:14px; }
section[data-testid="stSidebar"] { background: white; border-right: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("🚀 InsightX Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Home", "About", "Contact", "Login"]
)

# ---------------- GLOBAL DATASET ----------------
if 'df' not in st.session_state:
    st.session_state.df = None

# ---------------- HOME PAGE ----------------
if page == "Home":

    # 🔥 Landing Image (Top Banner)
    st.image(landing_image, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("""
        <div class="logo-container">
            <span class="logo">InsightX</span>
            <span class="badge">Powered By Advanced Analytics</span>
        </div>
        <div class="tagline">
        Executive Financial Intelligence Platform
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    def card(title, value):
        st.markdown(f"""
        <div class="kpi">
            <h4 style="color:#64748B;">{title}</h4>
            <h2>{value}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col1: card("Transactions", "250000")
    with col2: card("Revenue", "₹32.7 Cr")
    with col3: card("Success Rate", "95.05%")
    with col4: card("Risk Alerts", "4")

    st.markdown("<br><br>", unsafe_allow_html=True)

    if st.button("🚀 Leadership Launch Console"):
        if st.session_state.df is None:
            st.warning("Please upload the dataset first!")
        else:
            st.success("Launching Query Intelligence...")
            st.experimental_rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.subheader("📁 Upload Dataset (CSV)")
    file = st.file_uploader("Upload CSV File", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.session_state.df = df
        st.success("Dataset uploaded successfully!")
        st.dataframe(df.head())

# ---------------- LOGIN PAGE ----------------
elif page == "Login":
    st.title("🔑 Login to InsightX")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.success("Login successful!")
        else:
            st.error("Invalid credentials!")

# ---------------- ABOUT ----------------
elif page == "About":
    st.title("👤 About InsightX")
    st.write("""
    InsightX is a leadership analytics platform built to convert financial
    transaction data into strategic executive intelligence dashboards.
    """)

# ---------------- CONTACT ----------------
elif page == "Contact":
    st.title("📞 Contact Us")
    st.write("Email: support@insightx.ai")
    st.write("Phone: +91 98765 43210")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
© 2026 InsightX | Enterprise Financial Intelligence Platform
</div>
""", unsafe_allow_html=True)