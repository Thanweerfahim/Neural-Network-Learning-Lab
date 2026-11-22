
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from core.neural_network import NeuralNetwork
from core.training import (
    train_model,
    get_training_history_dataframe,
    interpret_learning,
)


# =========================================================
# HEADER
# =========================================================

def render_header():

    st.html("""
    <div class="app-header">
        <h1>🧠 Training Lab</h1>
        <p>Train, monitor, and understand your neural network.</p>
    </div>
    """)


# =========================================================
# CONFIGURATION
# =========================================================

def render_training_controls():

    st.markdown("## 🔧 Configure Your Experiment")
    st.markdown("### ⚙️ Training Configuration")

    col1, col2 = st.columns(2)

    with col1:

        learning_rate = st.select_slider(
            "Learning Rate",
            options=[
                0.001,
                0.005,
                0.01,
                0.02,
                0.05,
                0.1
            ],
            value=0.01
        )

        epochs = st.slider(
            "Number of Epochs",
            min_value=20,
            max_value=500,
            value=100,
            step=10
        )

    with col2:

        activation = st.selectbox(
            "Hidden Layer Activation",
            ["ReLU", "Sigmoid"]
        )

        batch_size = st.selectbox(
            "Batch Size",
            [16, 32, 64],
            index=1
        )

    st.html("""
    <div class="info-box">
        <strong>💡 Configuration Guide</strong><br>
        A smaller learning rate provides more gradual learning.
        More epochs allow the model to learn longer.
        Batch size controls the number of samples used for
        each weight update.
    </div>
    """)

    return learning_rate, epochs, activation, batch_size


# =========================================================
# DATA PREPARATION
# =========================================================

def get_training_data(dataset):

    required_keys = [
        "X_train_scaled",
        "y_train",
        "X_val_scaled",
        "y_val"
    ]

    missing = [
        key for key in required_keys
        if key not in dataset
    ]

    if missing:

        st.error(
            "Missing dataset variables: "
            + ", ".join(missing)
        )

        return None

    try:

        X_train = np.asarray(
            dataset["X_train_scaled"],
            dtype=np.float64
        )

        y_train = np.asarray(
            dataset["y_train"],
            dtype=np.float64
        )

        X_val = np.asarray(
            dataset["X_val_scaled"],
            dtype=np.float64
        )

        y_val = np.asarray(
            dataset["y_val"],
            dtype=np.float64
        )

        return X_train, y_train, X_val, y_val

    except Exception as e:

        st.error(
            f"Dataset preparation error: {e}"
        )

        return None


# =========================================================
# METRIC HELPERS
# =========================================================

def normalize_metric_name(name):

    name = str(name).lower().strip()

    name = name.replace("-", "_")
    name = name.replace(" ", "_")

    return name


def get_metric_value(data, possible_names):

    if not isinstance(data, dict):
        return None

    normalized = {
        normalize_metric_name(k): v
        for k, v in data.items()
    }

    for name in possible_names:

        name = normalize_metric_name(name)

        if name in normalized:

            value = normalized[name]

            try:
                return float(value)
            except (TypeError, ValueError):
                return None

    # Fuzzy matching
    for key, value in normalized.items():

        for name in possible_names:

            name = normalize_metric_name(name)

            if name in key:

                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

    return None


# =========================================================
# HISTORY NORMALIZATION
# =========================================================

def normalize_history(training_output):

    if training_output is None:

        return pd.DataFrame()

    # -----------------------------------------------------
    # Extract wrapper dictionary
    # -----------------------------------------------------

    wrapper = (
        training_output
        if isinstance(training_output, dict)
        else {}
    )

    raw_history = wrapper.get(
        "history",
        training_output
    )

    # -----------------------------------------------------
    # Convert raw history into DataFrame
    # -----------------------------------------------------

    if isinstance(raw_history, pd.DataFrame):

        df = raw_history.copy()

    elif isinstance(raw_history, dict):

        try:

            df = pd.DataFrame(raw_history)

        except Exception:

            df = pd.DataFrame()

    elif isinstance(raw_history, list):

        try:

            df = pd.DataFrame(raw_history)

        except Exception:

            df = pd.DataFrame()

    else:

        df = pd.DataFrame()

    # -----------------------------------------------------
    # Normalize columns
    # -----------------------------------------------------

    if not df.empty:

        df.columns = [
            normalize_metric_name(c)
            for c in df.columns
        ]

    # -----------------------------------------------------
    # Find and rename metrics
    # -----------------------------------------------------

    metric_aliases = {

        "train_loss": [
            "train_loss",
            "training_loss",
            "loss",
            "trainloss",
            "trainingloss",
            "training_error"
        ],

        "val_loss": [
            "val_loss",
            "validation_loss",
            "valid_loss",
            "eval_loss",
            "validationloss"
        ],

        "train_accuracy": [
            "train_accuracy",
            "training_accuracy",
            "train_acc",
            "accuracy",
            "acc"
        ],

        "val_accuracy": [
            "val_accuracy",
            "validation_accuracy",
            "valid_accuracy",
            "val_acc",
            "validation_acc"
        ]
    }

    for standard_name, aliases in metric_aliases.items():

        if standard_name in df.columns:
            continue

        for alias in aliases:

            if alias in df.columns:

                df.rename(
                    columns={
                        alias: standard_name
                    },
                    inplace=True
                )

                break

    # -----------------------------------------------------
    # Additional fuzzy metric detection
    # -----------------------------------------------------

    for col in list(df.columns):

        name = str(col).lower()

        if "loss" in name:

            if (
                "val" in name
                or "valid" in name
            ):

                if "val_loss" not in df.columns:

                    df.rename(
                        columns={
                            col: "val_loss"
                        },
                        inplace=True
                    )

            else:

                if "train_loss" not in df.columns:

                    df.rename(
                        columns={
                            col: "train_loss"
                        },
                        inplace=True
                    )

        elif (
            "accuracy" in name
            or "acc" in name
        ):

            if (
                "val" in name
                or "valid" in name
            ):

                if "val_accuracy" not in df.columns:

                    df.rename(
                        columns={
                            col: "val_accuracy"
                        },
                        inplace=True
                    )

            else:

                if "train_accuracy" not in df.columns:

                    df.rename(
                        columns={
                            col: "train_accuracy"
                        },
                        inplace=True
                    )

    # -----------------------------------------------------
    # Epoch column
    # -----------------------------------------------------

    if not df.empty:

        if "epoch" not in df.columns:

            df.insert(
                0,
                "epoch",
                range(1, len(df) + 1)
            )

    # -----------------------------------------------------
    # Convert metrics to numeric
    # -----------------------------------------------------

    numeric_columns = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # -----------------------------------------------------
    # Extract final training metrics
    # -----------------------------------------------------

    final_train = wrapper.get(
        "final_train",
        {}
    )

    final_val = wrapper.get(
        "final_val",
        {}
    )

    train_loss = get_metric_value(
        final_train,
        [
            "loss",
            "train_loss",
            "training_loss"
        ]
    )

    train_accuracy = get_metric_value(
        final_train,
        [
            "accuracy",
            "acc",
            "train_accuracy"
        ]
    )

    val_loss = get_metric_value(
        final_val,
        [
            "loss",
            "val_loss",
            "validation_loss"
        ]
    )

    val_accuracy = get_metric_value(
        final_val,
        [
            "accuracy",
            "acc",
            "val_accuracy",
            "validation_accuracy"
        ]
    )

    # -----------------------------------------------------
    # Add final metrics if available
    # -----------------------------------------------------

    final_values = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    }

    # If history is empty, create a row from final metrics
    if df.empty:

        if any(
            value is not None
            for value in final_values.values()
        ):

            df = pd.DataFrame(
                [final_values]
            )

            df.insert(
                0,
                "epoch",
                wrapper.get(
                    "epochs_completed",
                    1
                )
            )

    else:

        # Update the last row with final metrics
        for key, value in final_values.items():

            if value is not None:

                if key not in df.columns:

                    df[key] = np.nan

                df.loc[
                    df.index[-1],
                    key
                ] = value

    # -----------------------------------------------------
    # If accuracy is stored as 0-100, convert to 0-1
    # -----------------------------------------------------

    for col in [
        "train_accuracy",
        "val_accuracy"
    ]:

        if col in df.columns:

            maximum = df[col].max()

            if (
                pd.notna(maximum)
                and maximum > 1
                and maximum <= 100
            ):

                df[col] = df[col] / 100.0

    return df


# =========================================================
# PERFORMANCE METRICS
# =========================================================

def render_metrics(history_df):

    st.markdown("### 📊 Training Performance")

    if history_df.empty:

        st.warning(
            "Training metrics are not available."
        )

        return

    latest = history_df.iloc[-1]

    metric_items = [
        ("Training Loss", "train_loss", False),
        ("Validation Loss", "val_loss", False),
        ("Training Accuracy", "train_accuracy", True),
        ("Validation Accuracy", "val_accuracy", True)
    ]

    available = [
        item for item in metric_items
        if item[1] in history_df.columns
    ]

    if not available:

        st.warning(
            "No recognized metrics found."
        )

        st.write(
            "Columns:",
            list(history_df.columns)
        )

        return

    columns = st.columns(
        len(available)
    )

    for col, (label, key, is_accuracy) in zip(
        columns,
        available
    ):

        value = latest.get(
            key,
            np.nan
        )

        if pd.isna(value):

            text_value = "N/A"

        elif is_accuracy:

            text_value = f"{float(value):.2%}"

        else:

            text_value = f"{float(value):.4f}"

        col.metric(
            label,
            text_value
        )


# =========================================================
# LOSS CHART
# =========================================================

def render_loss_chart(history_df):

    st.markdown("### 📉 Learning Curve")

    if history_df.empty:

        st.info(
            "No training history available."
        )

        return

    fig = go.Figure()

    if "train_loss" in history_df.columns:

        fig.add_trace(
            go.Scatter(
                x=history_df["epoch"],
                y=history_df["train_loss"],
                mode="lines",
                name="Training Loss",
                line=dict(
                    color="#0D9488",
                    width=3
                )
            )
        )

    if "val_loss" in history_df.columns:

        fig.add_trace(
            go.Scatter(
                x=history_df["epoch"],
                y=history_df["val_loss"],
                mode="lines",
                name="Validation Loss",
                line=dict(
                    color="#F59E0B",
                    width=3
                )
            )
        )

    if len(fig.data) == 0:

        st.info(
            "Loss values are not available."
        )

        return

    fig.update_layout(
        template="plotly_white",
        height=400,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# ACCURACY CHART
# =========================================================

def render_accuracy_chart(history_df):

    st.markdown("### 📈 Accuracy Progress")

    if history_df.empty:

        st.info(
            "No training history available."
        )

        return

    fig = go.Figure()

    if "train_accuracy" in history_df.columns:

        fig.add_trace(
            go.Scatter(
                x=history_df["epoch"],
                y=history_df["train_accuracy"],
                mode="lines",
                name="Training Accuracy",
                line=dict(
                    color="#0D9488",
                    width=3
                )
            )
        )

    if "val_accuracy" in history_df.columns:

        fig.add_trace(
            go.Scatter(
                x=history_df["epoch"],
                y=history_df["val_accuracy"],
                mode="lines",
                name="Validation Accuracy",
                line=dict(
                    color="#8B5CF6",
                    width=3
                )
            )
        )

    if len(fig.data) == 0:

        st.info(
            "Accuracy values are not available."
        )

        return

    fig.update_layout(
        template="plotly_white",
        height=400,
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        hovermode="x unified"
    )

    fig.update_yaxes(
        range=[0, 1],
        tickformat=".0%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# INTERPRETATION
# =========================================================

def render_interpretation(history_df):

    st.markdown("### 🧠 Learning Interpretation")

    if history_df.empty:

        st.info(
            "Training history is unavailable."
        )

        return

    if (
        "train_loss" not in history_df.columns
        or "val_loss" not in history_df.columns
    ):

        st.info(
            "Loss-based interpretation is unavailable."
        )

        return

    latest = history_df.iloc[-1]

    train_loss = latest["train_loss"]
    val_loss = latest["val_loss"]

    if pd.isna(train_loss) or pd.isna(val_loss):

        st.info(
            "Loss values are unavailable."
        )

        return

    train_loss = float(train_loss)
    val_loss = float(val_loss)

    difference = val_loss - train_loss

    # -----------------------------------------------------
    # Existing interpretation helper
    # -----------------------------------------------------

    try:

        result = interpret_learning(
            train_loss,
            val_loss
        )

        if result:

            st.info(
                str(result)
            )

    except Exception:

        pass

    # -----------------------------------------------------
    # Interpretation
    # -----------------------------------------------------

    if difference > 0.10:

        st.warning(
            "⚠️ Possible overfitting: validation loss is "
            "higher than training loss."
        )

    elif difference < -0.10:

        st.info(
            "Validation loss is lower than training loss. "
            "This may happen when training data is more difficult."
        )

    else:

        st.success(
            "Training and validation losses are relatively close."
        )

    # Accuracy interpretation
    if (
        "train_accuracy" in history_df.columns
        and "val_accuracy" in history_df.columns
    ):

        train_acc = latest["train_accuracy"]
        val_acc = latest["val_accuracy"]

        if (
            pd.notna(train_acc)
            and pd.notna(val_acc)
        ):

            if train_acc - val_acc > 0.10:

                st.warning(
                    "Training accuracy is much higher than "
                    "validation accuracy. Possible overfitting."
                )

            elif val_acc >= 0.80:

                st.success(
                    f"Validation accuracy is good: {val_acc:.2%}"
                )

            else:

                st.info(
                    f"Validation accuracy: {val_acc:.2%}. "
                    "Try adjusting the experiment configuration."
                )


# =========================================================
# HISTORY TABLE
# =========================================================

def render_history_table(history_df):

    st.markdown("### 📋 Training History")

    if history_df.empty:

        st.warning(
            "Training history is empty."
        )

        return

    display_df = history_df.copy()

    numeric_columns = display_df.select_dtypes(
        include=[np.number]
    ).columns

    display_df[numeric_columns] = (
        display_df[numeric_columns].round(4)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Training History",
        data=csv_data,
        file_name="training_history.csv",
        mime="text/csv"
    )


# =========================================================
# TRAINING STATUS
# =========================================================

def render_training_status():

    st.markdown("### Training Status")

    history = st.session_state.get(
        "training_history"
    )

    if history is None:

        st.warning(
            "Model has not been trained yet."
        )

        return

    st.success(
        "✅ Model training completed successfully."
    )

    config = st.session_state.get(
        "training_config",
        {}
    )

    if config:

        st.markdown(
            "#### Training Configuration"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Learning Rate",
            config.get("learning_rate", "N/A")
        )

        col2.metric(
            "Epochs",
            config.get("epochs", "N/A")
        )

        col3.metric(
            "Activation",
            config.get("activation", "N/A")
        )

        col4.metric(
            "Batch Size",
            config.get("batch_size", "N/A")
        )


# =========================================================
# MAIN PAGE
# =========================================================

def render_training_lab(
    dataset=None,
    model=None,
    training_results=None
):

    render_header()

    # Retrieve dataset from session
    if dataset is None:

        dataset = st.session_state.get(
            "dataset"
        )

    if dataset is None:

        st.warning(
            "Please upload and prepare a dataset before training."
        )

        return

    # Prepare training data
    data = get_training_data(
        dataset
    )

    if data is None:
        return

    X_train, y_train, X_val, y_val = data

    # Configuration
    learning_rate, epochs, activation, batch_size = (
        render_training_controls()
    )

    st.divider()

    # -----------------------------------------------------
    # Buttons
    # -----------------------------------------------------

    col1, col2 = st.columns(
        [3, 1]
    )

    with col1:

        train_button = st.button(
            "🚀 Start Training",
            type="primary",
            use_container_width=True
        )

    with col2:

        reset_button = st.button(
            "🔄 Reset",
            use_container_width=True
        )

    # -----------------------------------------------------
    # Reset
    # -----------------------------------------------------

    if reset_button:

        reset_keys = [
            "nn_model",
            "training_history",
            "training_results",
            "training_config",
            "test_evaluation_results",
            "last_prediction"
        ]

        for key in reset_keys:

            st.session_state.pop(
                key,
                None
            )

        st.rerun()

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    if train_button:

        progress = st.progress(0)

        status = st.empty()

        try:

            status.info(
                "Initializing neural network..."
            )

            new_model = NeuralNetwork(
                activation=activation,
                learning_rate=learning_rate,
                seed=42
            )

            progress.progress(10)

            status.info(
                "Training neural network..."
            )

            # Run actual training
            output = train_model(
                model=new_model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=epochs,
                batch_size=batch_size,
                seed=42
            )

            progress.progress(85)

            # Store the full returned output
            st.session_state["training_history"] = output

            st.session_state["nn_model"] = new_model

            st.session_state["training_results"] = (
                training_results
                if training_results is not None
                else {}
            )

            st.session_state["training_config"] = {
                "learning_rate": learning_rate,
                "epochs": epochs,
                "activation": activation,
                "batch_size": batch_size
            }

            progress.progress(100)

            status.success(
                "Training completed successfully!"
            )

            st.success(
                "Your neural network has been trained successfully."
            )

            st.balloons()

        except Exception as e:

            progress.empty()

            status.empty()

            st.error(
                f"Training failed: {e}"
            )

            st.exception(e)

            return

    # -----------------------------------------------------
    # Get saved training history
    # -----------------------------------------------------

    saved_history = st.session_state.get(
        "training_history"
    )

    if saved_history is None:

        st.info(
            "Configure your experiment and click "
            "**Start Training** to begin."
        )

        return

    # -----------------------------------------------------
    # Convert history
    # -----------------------------------------------------

    history_df = normalize_history(
        saved_history
    )

    # -----------------------------------------------------
    # Display raw history diagnostics if needed
    # -----------------------------------------------------

    if history_df.empty:

        st.error(
            "Training completed, but the history could not be converted."
        )

        st.markdown("#### Debug Information")

        st.write(
            "Returned data type:",
            type(saved_history).__name__
        )

        st.write(
            "Returned training data:",
            saved_history
        )

        return

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    st.divider()

    render_metrics(
        history_df
    )

    st.divider()

    render_loss_chart(
        history_df
    )

    st.divider()

    render_accuracy_chart(
        history_df
    )

    st.divider()

    render_interpretation(
        history_df
    )

    st.divider()

    render_history_table(
        history_df
    )

    st.divider()

    render_training_status()
