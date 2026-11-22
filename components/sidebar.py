
import streamlit as st


PAGES = [
    ("🏠 Overview", "Overview"),
    ("⚙️ Training Lab", "Training Lab"),
    ("📊 Data Explorer", "Data Explorer"),
    ("🧠 Network Visualizer", "Network Visualizer"),
    ("🔮 Prediction Studio", "Prediction Studio"),
    ("🧪 Experiments", "Experiments"),
    ("📈 Model Insights", "Model Insights"),
]


def change_page(page_name):
    st.session_state["current_page"] = page_name


def render_sidebar():

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Overview"

    with st.sidebar:

        # BRAND
        st.html("""
        <div style="padding:15px 5px 20px 5px;">

            <div style="
                width:48px;
                height:48px;
                background:#0d9488;
                border-radius:14px;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:25px;
                margin-bottom:12px;
            ">
                🧠
            </div>

            <div style="
                font-size:22px;
                font-weight:800;
                color:#153d48;
            ">
                Neural Lab
            </div>

            <div style="
                font-size:12px;
                color:#78909a;
                margin-top:5px;
            ">
                Neural Network Learning Platform
            </div>

        </div>
        """)

        # NAVIGATION TITLE
        st.html("""
        <div style="
            color:#78909a;
            font-size:11px;
            font-weight:800;
            letter-spacing:1.5px;
            margin:10px 0;
        ">
            NAVIGATION
        </div>
        """)

        # NAVIGATION BUTTONS
        for label, page_name in PAGES:

            is_selected = (
                st.session_state["current_page"] == page_name
            )

            st.button(
                label,
                key=f"nav_{page_name}",
                on_click=change_page,
                args=(page_name,),
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            )

        st.divider()

        # PROJECT INFORMATION
        st.html("""
        <div style="
            background:#e9f8f5;
            border:1px solid #d5efea;
            border-radius:14px;
            padding:15px;
        ">

            <div style="
                font-size:13px;
                font-weight:700;
                color:#12655e;
                margin-bottom:8px;
            ">
                💡 Learning Project
            </div>

            <div style="
                font-size:12px;
                color:#58777e;
                line-height:1.6;
            ">
                Explore neural networks, train models,
                understand learning behaviour, and make
                predictions using sensor data.
            </div>

        </div>
        """)

    return st.session_state["current_page"]
