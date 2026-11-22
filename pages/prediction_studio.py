import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.dataset import FEATURE_COLUMNS


# =========================================================
# PAGE HEADER
# =========================================================

def render_header():

    st.markdown(
        """
        <div class="app-header">

            <div class="app-header-badge">
                MODEL PREDICTION
            </div>

            <div class="app-header-title">
                Prediction Studio 🔮
            </div>

            <div class="app-header-subtitle">
                Enter sensor measurements and use your
                trained neural network to predict
                possible machine alerts.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# SENSOR INPUTS
# =========================================================

def render_sensor_inputs():

    st.markdown(
        "## 🛠️ Sensor Measurements"
    )

    st.markdown(
        "Adjust the sensor values below to simulate "
        "different machine conditions."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        temperature = st.number_input(
            "🌡️ Temperature (°C)",
            min_value=-50.0,
            max_value=200.0,
            value=25.0,
            step=0.5,
        )

    with col2:

        vibration = st.number_input(
            "〰️ Vibration (mm/s)",
            min_value=0.0,
            max_value=100.0,
            value=2.0,
            step=0.1,
        )

    with col3:

        pressure = st.number_input(
            "💨 Pressure (kPa)",
            min_value=0.0,
            max_value=2000.0,
            value=101.3,
            step=1.0,
        )

    # -----------------------------------------------------
    # Sliders
    # -----------------------------------------------------

    st.markdown(
        "### Fine Adjustment"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        temperature_slider = st.slider(
            "Temperature",
            -50.0,
            200.0,
            float(temperature),
            0.5,
            key="temp_slider",
        )

    with col2:

        vibration_slider = st.slider(
            "Vibration",
            0.0,
            100.0,
            float(vibration),
            0.1,
            key="vibration_slider",
        )

    with col3:

        pressure_slider = st.slider(
            "Pressure",
            0.0,
            2000.0,
            float(pressure),
            1.0,
            key="pressure_slider",
        )

    # Use slider values as final values
    values = {
        "temperature_c": temperature_slider,
        "vibration_mm_s": vibration_slider,
        "pressure_kpa": pressure_slider,
    }

    return values


# =========================================================
# STANDARDIZATION
# =========================================================

def standardize_manual_input(
    values,
    dataset,
):

    # Get original training data
    X_train_raw = dataset.get(
        "X_train"
    )

    if X_train_raw is None:

        raise ValueError(
            "Training features are unavailable."
        )

    # Training-only mean and standard deviation
    mean = X_train_raw.mean(
        axis=0
    ).to_numpy(
        dtype=np.float64
    )

    std = X_train_raw.std(
        axis=0,
        ddof=0,
    ).to_numpy(
        dtype=np.float64
    )

    # Avoid division by zero
    std[std == 0] = 1.0

    manual_array = np.array([
        values[column]
        for column in FEATURE_COLUMNS
    ], dtype=np.float64)

    scaled = (
        manual_array - mean
    ) / std

    return scaled.reshape(
        1, -1
    )


# =========================================================
# PROBABILITY GAUGES
# =========================================================

def create_probability_gauge(
    probability,
    title,
    color,
):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",

            value=probability * 100,

            number={
                "suffix": "%",
                "font": {
                    "size": 38,
                },
            },

            title={
                "text": title,
            },

            gauge={
                "axis": {
                    "range": [
                        0,
                        100,
                    ],
                },

                "bar": {
                    "color": color,
                },

                "bgcolor": "white",

                "borderwidth": 0,

                "steps": [
                    {
                        "range": [
                            0,
                            50,
                        ],
                        "color": "#e8f5f1",
                    },

                    {
                        "range": [
                            50,
                            100,
                        ],
                        "color": "#fff0e6",
                    },
                ],

                "threshold": {
                    "line": {
                        "color": "#374151",
                        "width": 4,
                    },

                    "thickness": 0.8,

                    "value": 50,
                },
            },
        )
    )

    fig.update_layout(
        height=300,

        margin={
            "l": 20,
            "r": 20,
            "t": 50,
            "b": 20,
        },

        paper_bgcolor="white",
    )

    return fig


# =========================================================
# PREDICTION RESULTS
# =========================================================

def render_prediction_results(
    probabilities,
    threshold,
):

    overheat_probability = float(
        probabilities[0, 0]
    )

    fault_probability = float(
        probabilities[0, 1]
    )

    overheat_prediction = (
        overheat_probability >= threshold
    )

    fault_prediction = (
        fault_probability >= threshold
    )

    # -----------------------------------------------------
    # Probability cards
    # -----------------------------------------------------

    st.markdown(
        "## 🎯 Prediction Results"
    )

    col1, col2 = st.columns(2)

    with col1:

        fig1 = create_probability_gauge(
            overheat_probability,
            "Overheat Alert",
            "#f59e0b",
        )

        st.plotly_chart(
            fig1,
            use_container_width=True,
        )

    with col2:

        fig2 = create_probability_gauge(
            fault_probability,
            "Fault Alert",
            "#ef4444",
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
        )

   
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.dataset import FEATURE_COLUMNS


# =========================================================
# PAGE HEADER
# =========================================================

def render_header():

    st.html("""
    <div class="app-header">
        <div class="app-header-badge">MODEL PREDICTION</div>
        <div class="app-header-title">Prediction Studio 🔮</div>
        <div class="app-header-subtitle">
            Enter sensor measurements and use your trained
            neural network to predict possible machine alerts.
        </div>
    </div>
    """)


# =========================================================
# SENSOR INPUTS
# =========================================================

def render_sensor_inputs():

    st.markdown("## 🛠️ Sensor Measurements")

    st.write(
        "Adjust the sensor values below to simulate "
        "different machine conditions."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        temperature = st.slider(
            "🌡️ Temperature (°C)",
            min_value=-50.0,
            max_value=200.0,
            value=25.0,
            step=0.5,
            key="prediction_temperature"
        )

    with col2:

        vibration = st.slider(
            "〰️ Vibration (mm/s)",
            min_value=0.0,
            max_value=100.0,
            value=2.0,
            step=0.1,
            key="prediction_vibration"
        )

    with col3:

        pressure = st.slider(
            "💨 Pressure (kPa)",
            min_value=0.0,
            max_value=2000.0,
            value=101.3,
            step=1.0,
            key="prediction_pressure"
        )

    # Display current values
    st.markdown("### Current Measurements")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Temperature",
        f"{temperature:.1f} °C"
    )

    metric2.metric(
        "Vibration",
        f"{vibration:.1f} mm/s"
    )

    metric3.metric(
        "Pressure",
        f"{pressure:.1f} kPa"
    )

    values = {
        "temperature_c": temperature,
        "vibration_mm_s": vibration,
        "pressure_kpa": pressure
    }

    return values


# =========================================================
# GET RAW TRAINING FEATURES
# =========================================================

def get_raw_training_features(dataset):

    X_train_raw = dataset.get("X_train")

    if X_train_raw is None:

        raise ValueError(
            "Raw training features X_train are unavailable. "
            "The dataset must contain original training values."
        )

    # Convert to a writable NumPy array
    if isinstance(X_train_raw, pd.DataFrame):

        X = X_train_raw.copy().to_numpy(
            dtype=np.float64
        )

    else:

        X = np.array(
            X_train_raw,
            dtype=np.float64,
            copy=True
        )

    if X.ndim != 2:

        raise ValueError(
            "Training features must be a 2-dimensional array."
        )

    if X.shape[1] != len(FEATURE_COLUMNS):

        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} features, "
            f"but found {X.shape[1]}."
        )

    if not np.isfinite(X).all():

        raise ValueError(
            "Training data contains NaN or infinite values."
        )

    return X


# =========================================================
# STANDARDIZATION
# =========================================================

def standardize_manual_input(
    values,
    dataset
):

    # -----------------------------------------------------
    # Prepare raw training data
    # -----------------------------------------------------

    X_train_raw = get_raw_training_features(
        dataset
    )

    # -----------------------------------------------------
    # Calculate training mean and standard deviation
    # -----------------------------------------------------

    mean = np.mean(
        X_train_raw,
        axis=0
    ).copy()

    std = np.std(
        X_train_raw,
        axis=0,
        ddof=0
    ).copy()

    # Writable copy
    std = np.array(
        std,
        dtype=np.float64,
        copy=True
    )

    # Avoid zero standard deviation
    std[std == 0] = 1.0

    # -----------------------------------------------------
    # Prepare manual input
    # -----------------------------------------------------

    manual_array = np.array(
        [
            values[column]
            for column in FEATURE_COLUMNS
        ],
        dtype=np.float64
    ).copy()

    # -----------------------------------------------------
    # Standardize
    # -----------------------------------------------------

    scaled = (
        manual_array - mean
    ) / std

    # Return writable array
    scaled = np.array(
        scaled,
        dtype=np.float64,
        copy=True
    )

    return scaled.reshape(1, -1)


# =========================================================
# PROBABILITY GAUGE
# =========================================================

def create_probability_gauge(
    probability,
    title,
    color
):

    probability = float(
        np.clip(probability, 0.0, 1.0)
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",

            value=probability * 100,

            number={
                "suffix": "%",
                "font": {
                    "size": 38
                }
            },

            title={
                "text": title
            },

            gauge={
                "axis": {
                    "range": [0, 100]
                },

                "bar": {
                    "color": color
                },

                "bgcolor": "white",

                "borderwidth": 0,

                "steps": [
                    {
                        "range": [0, 50],
                        "color": "#E8F5F1"
                    },
                    {
                        "range": [50, 100],
                        "color": "#FFF0E6"
                    }
                ],

                "threshold": {
                    "line": {
                        "color": "#374151",
                        "width": 4
                    },

                    "thickness": 0.8,

                    "value": 50
                }
            }
        )
    )

    fig.update_layout(
        height=300,

        margin={
            "l": 20,
            "r": 20,
            "t": 50,
            "b": 20
        },

        paper_bgcolor="white"
    )

    return fig


# =========================================================
# PREDICTION RESULTS
# =========================================================

def render_prediction_results(
    probabilities,
    threshold
):

    # Make writable, safe array
    probabilities = np.array(
        probabilities,
        dtype=np.float64,
        copy=True
    )

    if probabilities.ndim == 1:

        probabilities = probabilities.reshape(
            1, -1
        )

    if probabilities.shape[1] < 2:

        raise ValueError(
            "The model must produce two output probabilities."
        )

    overheat_probability = float(
        probabilities[0, 0]
    )

    fault_probability = float(
        probabilities[0, 1]
    )

    overheat_prediction = (
        overheat_probability >= threshold
    )

    fault_prediction = (
        fault_probability >= threshold
    )

    # -----------------------------------------------------
    # Probability gauges
    # -----------------------------------------------------

    st.markdown("## 🎯 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        fig1 = create_probability_gauge(
            overheat_probability,
            "Overheat Alert",
            "#F59E0B"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with col2:

        fig2 = create_probability_gauge(
            fault_probability,
            "Fault Alert",
            "#EF4444"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # -----------------------------------------------------
    # Alert status
    # -----------------------------------------------------

    st.markdown("## 🚦 Machine Alert Status")

    col1, col2 = st.columns(2)

    with col1:

        if overheat_prediction:

            st.error("🔥 OVERHEAT ALERT")

            st.write(
                f"Probability: "
                f"{overheat_probability:.2%}"
            )

            st.write(
                "The prediction meets or exceeds "
                "the selected threshold."
            )

        else:

            st.success("✅ NO OVERHEAT ALERT")

            st.write(
                f"Probability: "
                f"{overheat_probability:.2%}"
            )

            st.write(
                "The prediction is below "
                "the selected threshold."
            )

    with col2:

        if fault_prediction:

            st.error("⚠️ FAULT ALERT")

            st.write(
                f"Probability: "
                f"{fault_probability:.2%}"
            )

            st.write(
                "The prediction meets or exceeds "
                "the selected threshold."
            )

        else:

            st.success("✅ NO FAULT ALERT")

            st.write(
                f"Probability: "
                f"{fault_probability:.2%}"
            )

            st.write(
                "The prediction is below "
                "the selected threshold."
            )

    # -----------------------------------------------------
    # Results table
    # -----------------------------------------------------

    results_df = pd.DataFrame({

        "Output": [
            "Overheat Alert",
            "Fault Alert"
        ],

        "Probability": [
            overheat_probability,
            fault_probability
        ],

        "Threshold": [
            threshold,
            threshold
        ],

        "Predicted Class": [
            int(overheat_prediction),
            int(fault_prediction)
        ],

        "Decision": [
            "ALERT" if overheat_prediction else "NORMAL",
            "ALERT" if fault_prediction else "NORMAL"
        ]
    })

    st.markdown("### Prediction Summary")

    display_df = results_df.copy()

    display_df["Probability"] = (
        display_df["Probability"].map(
            lambda x: f"{x:.2%}"
        )
    )

    display_df["Threshold"] = (
        display_df["Threshold"].map(
            lambda x: f"{x:.2%}"
        )
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    return results_df


# =========================================================
# INPUT SUMMARY
# =========================================================

def render_input_summary(
    values,
    scaled_values
):

    st.markdown("## 📊 Input Summary")

    scaled_values = np.array(
        scaled_values,
        dtype=np.float64,
        copy=True
    ).reshape(-1)

    input_df = pd.DataFrame({

        "Feature": [
            "Temperature",
            "Vibration",
            "Pressure"
        ],

        "Measurement": [
            "°C",
            "mm/s",
            "kPa"
        ],

        "Raw Value": [
            values["temperature_c"],
            values["vibration_mm_s"],
            values["pressure_kpa"]
        ],

        "Standardized Value": scaled_values
    })

    st.dataframe(
        input_df.round(5),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# MAIN PAGE
# =========================================================

def render_prediction_studio(
    dataset=None,
    model=None
):

    render_header()

    # -----------------------------------------------------
    # Retrieve model
    # -----------------------------------------------------

    if model is None:

        model = st.session_state.get(
            "nn_model"
        )

    if model is None:

        st.warning(
            "No trained model is available. "
            "Go to Training Lab and train a model first."
        )

        st.info(
            "After training, return here to make predictions."
        )

        return

    # -----------------------------------------------------
    # Retrieve dataset
    # -----------------------------------------------------

    if dataset is None:

        dataset = st.session_state.get(
            "dataset"
        )

    if dataset is None:

        st.error(
            "Dataset is not available. "
            "Please load your dataset first."
        )

        return

    # -----------------------------------------------------
    # Prediction settings
    # -----------------------------------------------------

    st.markdown("## Prediction Settings")

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,

        help=(
            "If the predicted probability is equal to or "
            "greater than this threshold, the output "
            "will be classified as 1."
        ),

        key="prediction_threshold"
    )

    st.caption(
        f"Current threshold: {threshold:.2f}"
    )

    st.divider()

    # -----------------------------------------------------
    # Sensor inputs
    # -----------------------------------------------------

    values = render_sensor_inputs()

    st.divider()

    # -----------------------------------------------------
    # Prediction button
    # -----------------------------------------------------

    predict_button = st.button(
        "🔮 Run Prediction",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        try:

            # Standardize input
            scaled_values = standardize_manual_input(
                values,
                dataset
            )

            # Make sure input is writable
            scaled_values = np.array(
                scaled_values,
                dtype=np.float64,
                copy=True
            )

            # Predict
            probabilities = model.predict_proba(
                scaled_values
            )

            # Make output writable
            probabilities = np.array(
                probabilities,
                dtype=np.float64,
                copy=True
            )

            # Validate prediction
            if not np.isfinite(probabilities).all():

                raise ValueError(
                    "Model produced invalid probabilities."
                )

            # Store results
            st.session_state["last_prediction"] = (
                probabilities
            )

            st.session_state["last_prediction_inputs"] = (
                values.copy()
            )

            st.session_state["last_prediction_scaled"] = (
                scaled_values.copy()
            )

            st.session_state["last_prediction_threshold"] = (
                threshold
            )

            st.session_state["prediction_success"] = True

        except Exception as e:

            st.session_state["prediction_success"] = False

            st.error(
                f"Prediction failed: {e}"
            )

            st.exception(e)

            return

    # -----------------------------------------------------
    # Retrieve previous successful prediction
    # -----------------------------------------------------

    probabilities = st.session_state.get(
        "last_prediction"
    )

    if probabilities is None:

        st.info(
            "Adjust the sensor measurements and "
            "click Run Prediction."
        )

        return

    # -----------------------------------------------------
    # Previous prediction values
    # -----------------------------------------------------

    previous_values = st.session_state.get(
        "last_prediction_inputs",
        values
    )

    previous_scaled = st.session_state.get(
        "last_prediction_scaled"
    )

    previous_threshold = st.session_state.get(
        "last_prediction_threshold",
        threshold
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    st.divider()

    render_prediction_results(
        probabilities,
        previous_threshold
    )

    st.divider()

    render_input_summary(
        previous_values,
        previous_scaled
    )

    st.divider()

    # -----------------------------------------------------
    # Explanation
    # -----------------------------------------------------

    st.html("""
    <div class="info-box">
        <div class="info-box-title">
            🧠 How the prediction works
        </div>

        <div class="info-box-text">
            Your sensor measurements are standardized
            using the mean and standard deviation of
            the training dataset.

            The standardized measurements are passed
            through the trained 3–5–4–2 neural network.

            The two output neurons produce probabilities
            for Overheat Alert and Fault Alert.

            The decision threshold converts these
            probabilities into predicted classes.
        </div>
    </div>
    """)

    st.caption(
        "Educational machine-learning simulation using "
        "a synthetic dataset. This is not a real "
        "industrial safety system."
    )
