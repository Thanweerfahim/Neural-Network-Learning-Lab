
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.neural_network import NeuralNetwork
from core.training import train_model, evaluate_model


# =========================================================
# PAGE HEADER
# =========================================================

def render_header():

    st.html("""
    <div class="app-header">
        <div class="app-header-badge">EXPERIMENTAL ANALYSIS</div>
        <div class="app-header-title">Experiments Lab 🧪</div>
        <div class="app-header-subtitle">
            Compare hyperparameters and investigate how different
            configurations influence neural network performance.
        </div>
    </div>
    """)


# =========================================================
# DATA ACCESS
# =========================================================

def get_data(dataset):

    if dataset is None:
        return None

    required_keys = [
        "X_train_scaled",
        "y_train",
        "X_val_scaled",
        "y_val",
        "X_test_scaled",
        "y_test",
    ]

    for key in required_keys:

        if dataset.get(key) is None:
            st.error(f"Dataset is missing: {key}")
            return None

    try:

        arrays = [
            np.array(dataset["X_train_scaled"], dtype=np.float64, copy=True),
            np.array(dataset["y_train"], dtype=np.float64, copy=True),
            np.array(dataset["X_val_scaled"], dtype=np.float64, copy=True),
            np.array(dataset["y_val"], dtype=np.float64, copy=True),
            np.array(dataset["X_test_scaled"], dtype=np.float64, copy=True),
            np.array(dataset["y_test"], dtype=np.float64, copy=True),
        ]

        for array in arrays:

            if not np.isfinite(array).all():
                st.error("Dataset contains NaN or infinite values.")
                return None

        return tuple(arrays)

    except Exception as e:

        st.error(f"Unable to prepare dataset: {e}")
        return None


# =========================================================
# METRIC HELPERS
# =========================================================

def get_metric(metrics, possible_names, default=np.nan):

    if metrics is None:
        return default

    if isinstance(metrics, pd.DataFrame):

        if metrics.empty:
            return default

        metrics = metrics.iloc[-1].to_dict()

    if isinstance(metrics, pd.Series):
        metrics = metrics.to_dict()

    if not isinstance(metrics, dict):
        return default

    for name in possible_names:

        if name in metrics:

            try:
                value = float(metrics[name])

                if np.isfinite(value):
                    return value

            except (TypeError, ValueError):
                pass

    return default


def normalize_evaluation(result):

    if isinstance(result, pd.DataFrame):

        if result.empty:
            return {}

        result = result.iloc[-1].to_dict()

    elif isinstance(result, pd.Series):

        result = result.to_dict()

    if isinstance(result, dict):
        return result

    return {}


def get_loss(metrics):

    return get_metric(
        metrics,
        [
            "loss",
            "test_loss",
            "train_loss",
            "val_loss",
            "binary_crossentropy",
            "cross_entropy",
        ]
    )


def get_accuracy(metrics):

    return get_metric(
        metrics,
        [
            "accuracy",
            "acc",
            "binary_accuracy",
            "test_accuracy",
            "train_accuracy",
            "val_accuracy",
        ]
    )


def get_exact_match(metrics):

    return get_metric(
        metrics,
        [
            "exact_match_accuracy",
            "exact_accuracy",
            "exact_match",
        ],
        default=np.nan,
    )


# =========================================================
# EXPERIMENT CONFIGURATION
# =========================================================

def render_experiment_controls():

    st.markdown("## Experiment Configuration")

    st.markdown(
        "Choose the configurations you want to compare."
    )

    col1, col2 = st.columns(2)

    with col1:

        learning_rates = st.multiselect(
            "Learning Rates",
            options=[0.001, 0.01, 0.1],
            default=[0.001, 0.01, 0.1],
        )

        activations = st.multiselect(
            "Activation Functions",
            options=["ReLU", "Sigmoid"],
            default=["ReLU", "Sigmoid"],
        )

    with col2:

        epoch_options = st.multiselect(
            "Epoch Configurations",
            options=[20, 50, 100, 200, 300],
            default=[20, 100, 300],
        )

        batch_size = st.selectbox(
            "Batch Size",
            options=[16, 32, 64],
            index=1,
        )

    st.html("""
    <div class="info-box">
        <div class="info-box-title">Fair Experiment Design</div>
        <div class="info-box-text">
            Every experiment uses the same training and validation split,
            the same random seed, and the same batch size.
            Only the selected configuration changes.
        </div>
    </div>
    """)

    return (
        learning_rates,
        activations,
        epoch_options,
        batch_size,
    )


# =========================================================
# CONFIGURATION GENERATION
# =========================================================

def create_configurations(
    learning_rates,
    activations,
    epoch_options,
):

    configurations = []

    for lr in learning_rates:

        for activation in activations:

            for epochs in epoch_options:

                configurations.append({
                    "learning_rate": lr,
                    "activation": activation,
                    "epochs": epochs,
                })

    return configurations


# =========================================================
# TRAINING HISTORY NORMALIZATION
# =========================================================

def extract_history(training_output):

    """
    Handles different training return formats.

    Supported:
        - DataFrame
        - List of dictionaries
        - Dictionary containing history
        - Tuple containing history
    """

    history = None

    if isinstance(training_output, pd.DataFrame):

        history = training_output.copy()

    elif isinstance(training_output, list):

        history = pd.DataFrame(training_output)

    elif isinstance(training_output, tuple):

        if len(training_output) > 0:

            first_item = training_output[0]

            if isinstance(first_item, pd.DataFrame):
                history = first_item.copy()

            elif isinstance(first_item, list):
                history = pd.DataFrame(first_item)

            elif isinstance(first_item, dict):
                history = pd.DataFrame(first_item)

    elif isinstance(training_output, dict):

        history = training_output.get("history")

        if isinstance(history, pd.DataFrame):
            history = history.copy()

        elif isinstance(history, list):
            history = pd.DataFrame(history)

        elif isinstance(history, dict):

            try:
                history = pd.DataFrame(history)

            except Exception:
                history = None

    if history is None:

        return pd.DataFrame()

    if not isinstance(history, pd.DataFrame):

        try:
            history = pd.DataFrame(history)

        except Exception:
            return pd.DataFrame()

    return history


# =========================================================
# RUN EXPERIMENTS
# =========================================================

def run_experiments(
    configurations,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size,
):

    results = []

    total = len(configurations)

    if total == 0:
        return pd.DataFrame()

    progress = st.progress(
        0,
        text="Preparing experiments...",
    )

    status = st.empty()

    for index, config in enumerate(configurations):

        status.info(
            f"Experiment {index + 1} of {total}: "
            f"LR={config['learning_rate']} | "
            f"Activation={config['activation']} | "
            f"Epochs={config['epochs']}"
        )

        # -------------------------------------------------
        # Create fresh neural network
        # -------------------------------------------------

        model = NeuralNetwork(
            activation=config["activation"],
            learning_rate=config["learning_rate"],
            seed=42,
        )

        # -------------------------------------------------
        # Train model
        # -------------------------------------------------

        training_output = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=config["epochs"],
            batch_size=batch_size,
            seed=42,
        )

        # -------------------------------------------------
        # Extract history if available
        # -------------------------------------------------

        history_df = extract_history(
            training_output
        )

        # -------------------------------------------------
        # Evaluate model
        # -------------------------------------------------

        train_evaluation = normalize_evaluation(
            evaluate_model(
                model,
                X_train,
                y_train,
            )
        )

        val_evaluation = normalize_evaluation(
            evaluate_model(
                model,
                X_val,
                y_val,
            )
        )

        test_evaluation = normalize_evaluation(
            evaluate_model(
                model,
                X_test,
                y_test,
            )
        )

        # -------------------------------------------------
        # Get final metrics
        # -------------------------------------------------

        train_loss = get_loss(
            train_evaluation
        )

        train_accuracy = get_accuracy(
            train_evaluation
        )

        val_loss = get_loss(
            val_evaluation
        )

        val_accuracy = get_accuracy(
            val_evaluation
        )

        test_loss = get_loss(
            test_evaluation
        )

        test_accuracy = get_accuracy(
            test_evaluation
        )

        exact_match_accuracy = get_exact_match(
            test_evaluation
        )

        # -------------------------------------------------
        # Fallback: use training history if needed
        # -------------------------------------------------

        if not history_df.empty:

            last_row = history_df.iloc[-1]

            if not np.isfinite(train_loss):

                train_loss = get_metric(
                    last_row,
                    ["train_loss", "loss"],
                )

            if not np.isfinite(train_accuracy):

                train_accuracy = get_metric(
                    last_row,
                    [
                        "train_accuracy",
                        "train_acc",
                        "accuracy",
                    ],
                )

            if not np.isfinite(val_loss):

                val_loss = get_metric(
                    last_row,
                    [
                        "val_loss",
                        "validation_loss",
                    ],
                )

            if not np.isfinite(val_accuracy):

                val_accuracy = get_metric(
                    last_row,
                    [
                        "val_accuracy",
                        "validation_accuracy",
                        "val_acc",
                    ],
                )

        # -------------------------------------------------
        # Store experiment result
        # -------------------------------------------------

        result = {
            "Experiment": index + 1,

            "Learning Rate": config["learning_rate"],

            "Activation": config["activation"],

            "Epochs": config["epochs"],

            "Batch Size": batch_size,

            "Train Loss": train_loss,

            "Validation Loss": val_loss,

            "Train Accuracy": train_accuracy,

            "Validation Accuracy": val_accuracy,

            "Test Loss": test_loss,

            "Test Accuracy": test_accuracy,

            "Exact Match Accuracy": exact_match_accuracy,
        }

        results.append(result)

        # -------------------------------------------------
        # Update progress
        # -------------------------------------------------

        progress.progress(
            int(((index + 1) / total) * 100),
            text=f"Completed {index + 1} of {total} experiments",
        )

    status.success(
        "All experiments completed successfully!"
    )

    progress.empty()

    return pd.DataFrame(results)


# =========================================================
# RESULTS TABLE
# =========================================================

def render_results_table(results_df):

    st.markdown("## 📋 Experiment Results")

    display_df = results_df.copy()

    percent_columns = [
        "Train Accuracy",
        "Validation Accuracy",
        "Test Accuracy",
        "Exact Match Accuracy",
    ]

    for column in percent_columns:

        if column in display_df.columns:

            display_df[column] = (
                display_df[column] * 100
            )

    numeric_columns = display_df.select_dtypes(
        include="number"
    ).columns

    display_df[numeric_columns] = (
        display_df[numeric_columns].round(4)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = results_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Experiment Results",
        data=csv_data,
        file_name="experiment_results.csv",
        mime="text/csv",
    )


# =========================================================
# LOSS COMPARISON
# =========================================================

def render_loss_comparison(results_df):

    st.markdown("## 📉 Loss Comparison")

    fig = go.Figure()

    for activation in results_df["Activation"].unique():

        subset = results_df[
            results_df["Activation"] == activation
        ]

        labels = [
            f"LR {lr} · {epochs} epochs"
            for lr, epochs in zip(
                subset["Learning Rate"],
                subset["Epochs"],
            )
        ]

        fig.add_trace(
            go.Bar(
                x=labels,
                y=subset["Validation Loss"],
                name=activation,
            )
        )

    fig.update_layout(
        title="Validation Loss Across Experiments",
        xaxis_title="Configuration",
        yaxis_title="Validation Loss",
        barmode="group",
        template="plotly_white",
        height=500,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =========================================================
# ACCURACY COMPARISON
# =========================================================

def render_accuracy_comparison(results_df):

    st.markdown("## 📈 Accuracy Comparison")

    fig = px.bar(
        results_df,
        x="Experiment",
        y=[
            "Train Accuracy",
            "Validation Accuracy",
            "Test Accuracy",
        ],
        barmode="group",
        color_discrete_sequence=[
            "#0d9488",
            "#f59e0b",
            "#6366f1",
        ],
        hover_data=[
            "Learning Rate",
            "Activation",
            "Epochs",
        ],
    )

    fig.update_layout(
        title="Accuracy Comparison by Experiment",
        xaxis_title="Experiment",
        yaxis_title="Accuracy",
        yaxis={"range": [0, 1]},
        template="plotly_white",
        height=450,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =========================================================
# LEARNING RATE ANALYSIS
# =========================================================

def render_learning_rate_analysis(results_df):

    st.markdown("## ⚡ Learning Rate Analysis")

    fig = px.line(
        results_df,
        x="Learning Rate",
        y="Validation Loss",
        color="Activation",
        symbol="Epochs",
        markers=True,
    )

    fig.update_layout(
        title="Effect of Learning Rate on Validation Loss",
        xaxis_title="Learning Rate",
        yaxis_title="Validation Loss",
        xaxis_type="log",
        template="plotly_white",
        height=450,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown("""
    **How to interpret this graph**

    - A smaller learning rate may learn slowly.
    - A moderate learning rate may provide stable improvement.
    - A high learning rate may cause unstable learning.
    - Lower validation loss generally indicates better performance
      on unseen validation data.

    Compare your actual results rather than assuming a particular
    learning rate is always best.
    """)


# =========================================================
# BEST EXPERIMENT
# =========================================================

def render_best_experiment(results_df):

    st.markdown("## 🏆 Best Experiment")

    valid_results = results_df.dropna(
        subset=["Validation Loss"]
    )

    if valid_results.empty:

        st.warning(
            "Validation loss was unavailable. "
            "Cannot determine the best experiment."
        )

        return

    best_index = valid_results[
        "Validation Loss"
    ].idxmin()

    best = results_df.loc[best_index]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Learning Rate",
        best["Learning Rate"],
    )

    col2.metric(
        "Activation",
        best["Activation"],
    )

    col3.metric(
        "Epochs",
        int(best["Epochs"]),
    )

    col4.metric(
        "Validation Loss",
        f"{best['Validation Loss']:.5f}",
    )

    st.success(
        "The configuration above achieved the lowest "
        "validation loss among the experiments that were run."
    )

    st.caption(
        "This is the best configuration in this experiment, "
        "not necessarily the best possible neural network."
    )


# =========================================================
# INTERPRETATION
# =========================================================

def render_interpretation(results_df):

    st.markdown("## 🧠 Experiment Interpretation")

    valid_results = results_df.dropna(
        subset=["Validation Loss"]
    )

    if valid_results.empty:

        st.warning(
            "Unable to interpret results because "
            "validation loss is unavailable."
        )

        return

    best = valid_results.loc[
        valid_results["Validation Loss"].idxmin()
    ]

    accuracy_results = results_df.dropna(
        subset=["Test Accuracy"]
    )

    if not accuracy_results.empty:

        best_accuracy = accuracy_results.loc[
            accuracy_results["Test Accuracy"].idxmax()
        ]

        highest_test_accuracy = (
            best_accuracy["Test Accuracy"] * 100
        )

    else:

        highest_test_accuracy = None

    highest_lr = results_df[
        "Learning Rate"
    ].max()

    lowest_lr = results_df[
        "Learning Rate"
    ].min()

    st.markdown("### Results Summary")

    st.write(
        f"""
        **Lowest validation loss:** {best['Validation Loss']:.5f}

        **Learning rate:** {best['Learning Rate']}

        **Activation function:** {best['Activation']}

        **Epochs:** {int(best['Epochs'])}

        **Learning rates compared:** {lowest_lr} to {highest_lr}
        """
    )

    if highest_test_accuracy is not None:

        st.write(
            f"**Highest test accuracy:** "
            f"{highest_test_accuracy:.2f}%"
        )

    st.markdown("### Questions to Consider")

    st.markdown("""
    1. Did increasing the learning rate improve or worsen validation loss?
    2. Did ReLU outperform Sigmoid?
    3. Did increasing the number of epochs improve validation performance?
    4. Did training accuracy improve while validation accuracy stopped improving?
    5. Which configuration gives the best balance between training
       and validation performance?

    Use the actual values shown in your experiment table to answer these.
    """)

    st.warning(
        "Use validation results to select a configuration. "
        "Test results should be treated as final evaluation and "
        "should not be used to repeatedly choose hyperparameters."
    )


# =========================================================
# MAIN PAGE
# =========================================================

def render_experiments(dataset=None):

    render_header()

    # -----------------------------------------------------
    # Get dataset
    # -----------------------------------------------------

    data = get_data(dataset)

    if data is None:

        st.warning(
            "Load and prepare the dataset before running experiments."
        )

        return

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = data

    # -----------------------------------------------------
    # Controls
    # -----------------------------------------------------

    (
        learning_rates,
        activations,
        epoch_options,
        batch_size,
    ) = render_experiment_controls()

    configurations = create_configurations(
        learning_rates,
        activations,
        epoch_options,
    )

    # -----------------------------------------------------
    # Experiment plan
    # -----------------------------------------------------

    st.markdown("## Experiment Plan")

    st.metric(
        "Total Experiments",
        len(configurations),
    )

    if configurations:

        plan_df = pd.DataFrame(
            configurations
        )

        st.dataframe(
            plan_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.warning(
            "Select at least one learning rate, activation "
            "function, and epoch setting."
        )

        return

    if len(configurations) > 18:

        st.warning(
            "You selected many configurations. "
            "This may take some time to complete."
        )

    # -----------------------------------------------------
    # Run experiments
    # -----------------------------------------------------

    run_button = st.button(
        "🧪 Run All Experiments",
        type="primary",
        use_container_width=True,
    )

    if run_button:

        try:

            results_df = run_experiments(
                configurations=configurations,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                batch_size=batch_size,
            )

            if results_df.empty:

                st.error(
                    "No experiment results were generated."
                )

            else:

                st.session_state[
                    "experiment_results"
                ] = results_df

                st.session_state[
                    "experiment_configurations"
                ] = configurations

                st.success(
                    "Experiments finished successfully!"
                )

        except Exception as e:

            st.error(
                f"Experiment failed: {e}"
            )

            st.exception(e)

    # -----------------------------------------------------
    # Display stored results
    # -----------------------------------------------------

    results_df = st.session_state.get(
        "experiment_results"
    )

    if results_df is None:

        st.info(
            "Select configurations and click "
            "'Run All Experiments' to begin."
        )

        return

    st.divider()

    render_results_table(
        results_df
    )

    render_best_experiment(
        results_df
    )

    render_loss_comparison(
        results_df
    )

    render_accuracy_comparison(
        results_df
    )

    render_learning_rate_analysis(
        results_df
    )

    render_interpretation(
        results_df
    )
