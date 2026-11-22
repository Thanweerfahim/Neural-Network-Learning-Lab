
import streamlit as st
from pathlib import Path

# =========================================================
# IMPORT UI COMPONENTS
# =========================================================

from components.styles import apply_styles
from components.sidebar import render_sidebar

# =========================================================
# IMPORT DATASET FUNCTIONS
# =========================================================

from core.dataset import prepare_dataset

# =========================================================
# IMPORT ALL APPLICATION PAGES
# =========================================================

from pages.overview import render_overview
from pages.data_explorer import render_data_explorer
from pages.network_visualizer import render_network_visualizer
from pages.training_lab import render_training_lab
from pages.prediction_studio import render_prediction_studio
from pages.experiments import render_experiments
from pages.model_insights import render_model_insights


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Neural Network Learning Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# APPLY GLOBAL DESIGN
# =========================================================

apply_styles()


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"

DEFAULT_DATASET = DATA_DIR / "sensor_dataset.csv"


# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

def initialize_session_state():

    defaults = {

        "nn_model": None,

        "training_history": None,

        "training_results": None,

        "training_config": None,

        "dataset": None,

        "dataset_loaded": False,

        "dataset_error": None,

        "test_evaluation_results": None,

        "test_evaluation_threshold": 0.5,

        "last_prediction": None,

    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


initialize_session_state()


# =========================================================
# LOAD DATASET
# =========================================================

def load_default_dataset():

    if st.session_state.dataset_loaded:

        return st.session_state.dataset

    if not DEFAULT_DATASET.exists():

        st.session_state.dataset_error = (
            f"Dataset not found: {DEFAULT_DATASET}"
        )

        return None

    try:

        dataset = prepare_dataset(
            str(DEFAULT_DATASET),
            random_state=42,
        )

        st.session_state.dataset = dataset

        st.session_state.dataset_loaded = True

        st.session_state.dataset_error = None

        return dataset

    except Exception as error:

        st.session_state.dataset_error = str(
            error
        )

        return None


# =========================================================
# UPLOAD DATASET
# =========================================================

def render_dataset_upload():

    st.sidebar.markdown(
        "### 📂 Dataset"
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload sensor CSV",

        type=["csv"],

        help="Upload a CSV file containing the sensor features and target columns.",
    )

    if uploaded_file is not None:

        try:

            uploaded_dataset = prepare_dataset(
                uploaded_file,
                random_state=42,
            )

            st.session_state.dataset = (
                uploaded_dataset
            )

            st.session_state.dataset_loaded = True

            st.session_state.dataset_error = None

            st.sidebar.success(
                "Dataset uploaded successfully!"
            )

        except Exception as error:

            st.sidebar.error(
                f"Dataset error: {error}"
            )

    return st.session_state.dataset


# =========================================================
# DATASET STATUS
# =========================================================

def render_dataset_status(dataset):

    if dataset is None:

        st.error(
            "The dataset could not be loaded."
        )

        st.markdown(
            """
            Please ensure your CSV file is located at:

            `data/sensor_dataset.csv`

            Or upload a CSV using the sidebar.
            """
        )

        if st.session_state.dataset_error:

            with st.expander(
                "Technical details"
            ):

                st.code(
                    st.session_state.dataset_error
                )

        return

    raw_data = dataset.get(
        "raw_data"
    )

    if raw_data is not None:

        st.sidebar.success(
            f"Dataset ready • {len(raw_data)} samples"
        )


# =========================================================
# FOOTER
# =========================================================

def render_footer():

    st.html("""
    <div style="
        text-align:center;
        padding:25px 10px 15px 10px;
        margin-top:35px;
        border-top:1px solid #e2e8f0;
        color:#78909a;
    ">

        <div style="
            font-size:16px;
            font-weight:700;
            color:#0d9488;
            margin-bottom:6px;
        ">
            🧠 Neural Network Learning Lab
        </div>

        <div style="
            font-size:12px;
        ">
            Interactive Machine Learning Education
            &nbsp;•&nbsp; 3–5–4–2 Neural Network
        </div>

    </div>
    """)


# =========================================================
# PAGE ROUTER
# =========================================================

def render_selected_page(
    page,
    dataset,
):

    model = st.session_state.get(
        "nn_model"
    )

    training_results = st.session_state.get(
        "training_results"
    )

    # -----------------------------------------------------
    # OVERVIEW
    # -----------------------------------------------------

    if page == "Overview":

        render_overview(
            dataset=dataset,

            training_results=training_results,

            model=model,
        )

    # -----------------------------------------------------
    # DATA EXPLORER
    # -----------------------------------------------------

    elif page == "Data Explorer":

        render_data_explorer(
            dataset=dataset
        )

    # -----------------------------------------------------
    # NETWORK VISUALIZER
    # -----------------------------------------------------

    elif page == "Network Visualizer":

        render_network_visualizer(
            model=model,

            dataset=dataset,
        )

    # -----------------------------------------------------
    # TRAINING LAB
    # -----------------------------------------------------

    elif page == "Training Lab":

        render_training_lab(
            dataset=dataset,

            model=model,

            training_results=training_results,
        )

    # -----------------------------------------------------
    # PREDICTION STUDIO
    # -----------------------------------------------------

    elif page == "Prediction Studio":

        render_prediction_studio(
            dataset=dataset,

            model=model,
        )

    # -----------------------------------------------------
    # EXPERIMENTS
    # -----------------------------------------------------

    elif page == "Experiments":

        render_experiments(
            dataset=dataset
        )

    # -----------------------------------------------------
    # MODEL INSIGHTS
    # -----------------------------------------------------

    elif page == "Model Insights":

        render_model_insights(
            dataset=dataset,

            model=model,
        )

    else:

        st.warning(
            "Page not found."
        )


# =========================================================
# MAIN APPLICATION
# =========================================================

def main():

    # Load default dataset
    dataset = load_default_dataset()

    # Dataset upload interface
    uploaded_dataset = render_dataset_upload()

    if uploaded_dataset is not None:

        dataset = uploaded_dataset

    # Dataset status
    render_dataset_status(
        dataset
    )

    # Navigation
    selected_page = render_sidebar()

    # Main page content
    render_selected_page(
        selected_page,

        dataset,
    )

    # Footer
    render_footer()


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    main()
