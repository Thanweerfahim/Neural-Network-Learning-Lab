
import streamlit as st


def apply_styles():

    css = """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* GLOBAL */

    html, body, [data-testid="stApp"] {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background: #f5f8fb !important;
        color: #193746 !important;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4 {
        color: #153d48 !important;
        font-weight: 700 !important;
    }

    /* SIDEBAR */

    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e1eaee !important;
    }

    [data-testid="stSidebar"] > div {
        background: #ffffff !important;
    }

    [data-testid="stSidebar"] p {
        color: #244653 !important;
    }

    [data-testid="stSidebar"] button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        min-height: 42px !important;
        transition: 0.2s ease !important;
    }

    [data-testid="stSidebar"] button[kind="secondary"] {
        background: #f5f9fb !important;
        border: 1px solid #e5edf0 !important;
        color: #244653 !important;
    }

    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: #e8f5f3 !important;
        border-color: #c8e8e2 !important;
        color: #087f72 !important;
    }

    [data-testid="stSidebar"] button[kind="primary"] {
        background: #0d9488 !important;
        border: 1px solid #0d9488 !important;
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] button[kind="primary"]:hover {
        background: #0f766e !important;
        border-color: #0f766e !important;
    }

    /* HIDE BUILT-IN PAGE NAVIGATION */

    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* HEADER */

    .app-header {
        background: linear-gradient(135deg, #ffffff, #eaf8f6);
        border: 1px solid #dcefed;
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(15,70,80,0.04);
    }

    .app-header-badge {
        display: inline-block;
        background: #d9f5ef;
        color: #087f72 !important;
        padding: 6px 12px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .app-header-title {
        font-size: 30px;
        font-weight: 800;
        color: #123c46 !important;
        margin-bottom: 8px;
    }

    .app-header-subtitle {
        font-size: 14px;
        color: #607983 !important;
        line-height: 1.7;
    }

    /* CARDS */

    .custom-card {
        background: #ffffff !important;
        border: 1px solid #e5edf0;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 3px 14px rgba(20,55,70,0.04);
        transition: all 0.2s ease;
    }

    .custom-card:hover {
        border-color: #b7e5df;
        box-shadow: 0 6px 22px rgba(20,100,100,0.07);
        transform: translateY(-2px);
    }

    .card-title {
        font-size: 14px;
        font-weight: 600;
        color: #64808a !important;
        margin-bottom: 10px;
    }

    .card-value {
        font-size: 30px;
        font-weight: 800;
        color: #153d48 !important;
    }

    .card-description {
        font-size: 12px;
        color: #8297a0 !important;
        margin-top: 8px;
    }

    /* SECTIONS */

    .section-container {
        background: #ffffff;
        border: 1px solid #e6eef1;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 3px 12px rgba(20,55,70,0.03);
    }

    .section-heading {
        font-size: 20px;
        font-weight: 700;
        color: #153d48 !important;
        margin-bottom: 6px;
    }

    .section-description {
        font-size: 13px;
        color: #78909a !important;
        margin-bottom: 18px;
    }

    /* METRICS */

    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e5edf0 !important;
        border-radius: 16px !important;
        padding: 18px 20px !important;
        box-shadow: 0 3px 12px rgba(20,55,70,0.03);
    }

    [data-testid="stMetricLabel"] {
        color: #66818b !important;
    }

    [data-testid="stMetricValue"] {
        color: #153d48 !important;
        font-weight: 800 !important;
    }

    /* BUTTONS */

    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid #d6e8e8 !important;
        background: #ffffff !important;
        color: #17616b !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
    }

    .stButton > button:hover {
        background: #e9f8f6 !important;
        border-color: #70c8bc !important;
        color: #087f72 !important;
    }

    .stButton > button[kind="primary"] {
        background: #0d9488 !important;
        color: #ffffff !important;
        border: none !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #0f766e !important;
        color: #ffffff !important;
    }

    /* INPUTS */

    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {
        border-radius: 10px !important;
        color: #193746 !important;
        background: #ffffff !important;
    }

    [data-baseweb="select"] {
        background: #ffffff !important;
    }

    /* TABS */

    [data-testid="stTabs"] button {
        color: #52717b !important;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #0d9488 !important;
    }

    /* DATA TABLES */

    [data-testid="stDataFrame"] {
        border: 1px solid #e5edf0 !important;
        border-radius: 12px !important;
        overflow: hidden;
    }

    /* ALERTS */

    [data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    /* EXPANDERS */

    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e5edf0 !important;
        border-radius: 12px !important;
    }

    /* INFO BOX */

    .info-box {
        background: #effaf8;
        border-left: 4px solid #0d9488;
        border-radius: 10px;
        padding: 16px;
        margin: 12px 0;
    }

    .info-box-title {
        color: #12655e !important;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .info-box-text {
        color: #54747b !important;
        font-size: 13px;
        line-height: 1.6;
    }

    /* FOOTER */

    .app-footer {
        text-align: center;
        padding: 25px 10px;
        color: #94a9b0 !important;
        font-size: 12px;
        border-top: 1px solid #e5edf0;
        margin-top: 35px;
    }

    /* RESPONSIVE */

    @media (max-width: 768px) {

        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .app-header {
            padding: 20px;
        }

        .app-header-title {
            font-size: 23px;
        }

        .custom-card {
            padding: 16px;
        }

        .section-container {
            padding: 16px;
        }

        .card-value {
            font-size: 24px;
        }
    }

    </style>
    """

    st.markdown(css, unsafe_allow_html=True)
