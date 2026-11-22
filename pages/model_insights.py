
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
)


# =========================================================
# PAGE HEADER
# =========================================================

def render_header():

    st.html("""
    <div class="app-header">
        <div class="app-header-badge">MODEL EVALUATION</div>
        <div class="app-header-title">Model Insights 📊</div>
        <div class="app-header-subtitle">
            Evaluate generalization performance using unseen test data,
            inspect classification errors, and understand model strengths
            and weaknesses.
        </div>
    </div>
    """)


# =========================================================
# GET TEST DATA
# =========================================================

def get_test_data(dataset):

    if dataset is None:
        return None

    X_test = dataset.get("X_test_scaled")
    y_test = dataset.get("y_test")
    X_test_raw = dataset.get("X_test")

    if X_test is None or y_test is None:

        st.error("Test dataset is unavailable.")
        return None

    try:

        X_test = np.array(
            X_test,
            dtype=np.float64,
            copy=True,
        )

        y_test = np.array(
            y_test,
            dtype=np.float64,
            copy=True,
        )

        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        if not np.isfinite(X_test).all():
            st.error("Test inputs contain invalid values.")
            return None

        if not np.isfinite(y_test).all():
            st.error("Test labels contain invalid values.")
            return None

        if X_test.shape[0] != y_test.shape[0]:
            st.error(
                "The number of test inputs and labels does not match."
            )
            return None

        if y_test.shape[1] != 2:
            st.error(
                "Expected exactly 2 output targets: "
                "Overheat Alert and Fault Alert."
            )
            return None

        return X_test, y_test, X_test_raw

    except Exception as e:

        st.error(f"Unable to prepare test data: {e}")
        return None


# =========================================================
# EVALUATION
# =========================================================

def evaluate_on_test(
    model,
    X_test,
    y_test,
    threshold=0.5,
):

    # Use writable copies to avoid read-only array errors.
    X_test = np.array(
        X_test,
        dtype=np.float64,
        copy=True,
    )

    y_test = np.array(
        y_test,
        dtype=np.int64,
        copy=True,
    )

    probabilities = np.array(
        model.predict_proba(X_test),
        dtype=np.float64,
        copy=True,
    )

    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, 1)

    if probabilities.shape != y_test.shape:

        raise ValueError(
            f"Prediction shape {probabilities.shape} "
            f"does not match target shape {y_test.shape}."
        )

    if not np.isfinite(probabilities).all():

        raise ValueError(
            "Model produced invalid prediction probabilities."
        )

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # -----------------------------------------------------
    # Binary cross-entropy
    # -----------------------------------------------------

    epsilon = 1e-12

    clipped_probabilities = np.clip(
        probabilities,
        epsilon,
        1 - epsilon,
    )

    loss = -np.mean(
        y_test * np.log(clipped_probabilities)
        +
        (1 - y_test) * np.log(
            1 - clipped_probabilities
        )
    )

    # -----------------------------------------------------
    # Element-wise accuracy
    # -----------------------------------------------------

    element_accuracy = accuracy_score(
        y_test.flatten(),
        predictions.flatten(),
    )

    # -----------------------------------------------------
    # Exact match accuracy
    # -----------------------------------------------------

    exact_match_accuracy = np.mean(
        np.all(
            predictions == y_test,
            axis=1,
        )
    )

    # -----------------------------------------------------
    # Per-target metrics
    # -----------------------------------------------------

    target_names = [
        "Overheat Alert",
        "Fault Alert",
    ]

    target_results = {}

    for index, target_name in enumerate(target_names):

        y_true = y_test[:, index]
        y_pred = predictions[:, index]

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Alert"],
            zero_division=0,
            output_dict=True,
        )

        target_results[target_name] = {

            "accuracy": accuracy_score(
                y_true,
                y_pred,
            ),

            "precision": precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

            "recall": recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

            "f1": f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

            "confusion_matrix": cm,

            "report": report,
        }

    return {

        "loss": float(loss),

        "element_accuracy": float(
            element_accuracy
        ),

        "exact_match_accuracy": float(
            exact_match_accuracy
        ),

        "probabilities": probabilities,

        "predictions": predictions,

        "target_results": target_results,

        "threshold": float(threshold),
    }


# =========================================================
# METRIC CARDS
# =========================================================

def render_metric_cards(results):

    st.markdown("## 🎯 Overall Test Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Test Loss",
        f"{results['loss']:.4f}",
    )

    col2.metric(
        "Element Accuracy",
        f"{results['element_accuracy'] * 100:.2f}%",
    )

    col3.metric(
        "Exact Match Accuracy",
        f"{results['exact_match_accuracy'] * 100:.2f}%",
    )

    col4.metric(
        "Test Samples",
        len(results["predictions"]),
    )


# =========================================================
# PER-TARGET METRICS
# =========================================================

def render_target_metrics(results):

    st.markdown("## 📌 Per-Target Performance")

    target_results = results["target_results"]

    col1, col2 = st.columns(2)

    targets = list(target_results.keys())

    for index, column in enumerate([col1, col2]):

        target_name = targets[index]

        metrics = target_results[target_name]

        with column:

            st.markdown(f"### {target_name}")

            st.metric(
                "Accuracy",
                f"{metrics['accuracy'] * 100:.2f}%",
            )

            st.metric(
                "Precision",
                f"{metrics['precision'] * 100:.2f}%",
            )

            st.metric(
                "Recall",
                f"{metrics['recall'] * 100:.2f}%",
            )

            st.metric(
                "F1-Score",
                f"{metrics['f1'] * 100:.2f}%",
            )


# =========================================================
# CONFUSION MATRICES
# =========================================================

def render_confusion_matrices(results):

    st.markdown("## 🔍 Confusion Matrices")

    target_results = results["target_results"]

    col1, col2 = st.columns(2)

    targets = list(target_results.keys())

    for index, column in enumerate([col1, col2]):

        target_name = targets[index]

        cm = target_results[target_name]["confusion_matrix"]

        with column:

            st.markdown(f"### {target_name}")

            fig = go.Figure(
                go.Heatmap(
                    z=cm,

                    x=[
                        "Predicted Normal",
                        "Predicted Alert",
                    ],

                    y=[
                        "Actual Normal",
                        "Actual Alert",
                    ],

                    text=cm,

                    texttemplate="%{text}",

                    colorscale=[
                        [0, "#e6f5f2"],
                        [1, "#0d9488"],
                    ],

                    showscale=False,
                )
            )

            fig.update_layout(
                template="plotly_white",
                height=400,
                xaxis_title="Predicted Label",
                yaxis_title="Actual Label",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            tn, fp, fn, tp = cm.ravel()

            st.markdown(
                f"""
                **True Negative:** {tn}  
                **False Positive:** {fp}  
                **False Negative:** {fn}  
                **True Positive:** {tp}
                """
            )


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

def render_classification_reports(results):

    st.markdown("## 📋 Classification Reports")

    target_results = results["target_results"]

    for target_name, metrics in target_results.items():

        with st.expander(
            f"{target_name} — Detailed Report",
            expanded=False,
        ):

            report_df = pd.DataFrame(
                metrics["report"]
            ).transpose()

            st.dataframe(
                report_df.round(4),
                use_container_width=True,
            )


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

def render_actual_vs_predicted(
    results,
    y_test,
):

    st.markdown("## 🔄 Actual vs Predicted")

    probabilities = results["probabilities"]
    predictions = results["predictions"]

    comparison_df = pd.DataFrame({

        "Actual Overheat": y_test[:, 0],

        "Predicted Overheat": predictions[:, 0],

        "Overheat Probability": probabilities[:, 0],

        "Actual Fault": y_test[:, 1],

        "Predicted Fault": predictions[:, 1],

        "Fault Probability": probabilities[:, 1],
    })

    comparison_df["Correct Overheat"] = (
        comparison_df["Actual Overheat"]
        ==
        comparison_df["Predicted Overheat"]
    )

    comparison_df["Correct Fault"] = (
        comparison_df["Actual Fault"]
        ==
        comparison_df["Predicted Fault"]
    )

    comparison_df["Both Correct"] = (
        comparison_df["Correct Overheat"]
        &
        comparison_df["Correct Fault"]
    )

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = comparison_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Test Predictions",
        data=csv_data,
        file_name="test_predictions.csv",
        mime="text/csv",
    )


# =========================================================
# ERROR ANALYSIS
# =========================================================

def render_error_analysis(
    results,
    y_test,
    X_test_raw,
):

    st.markdown("## 🚨 Error Analysis")

    predictions = results["predictions"]
    probabilities = results["probabilities"]

    incorrect = predictions != y_test

    total_errors = int(incorrect.sum())

    total_predictions = y_test.size

    error_rate = (
        total_errors / total_predictions
        if total_predictions > 0
        else 0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Incorrect Output Predictions",
        total_errors,
    )

    col2.metric(
        "Total Output Predictions",
        total_predictions,
    )

    col3.metric(
        "Output Error Rate",
        f"{error_rate * 100:.2f}%",
    )

    # -----------------------------------------------------
    # Incorrect records
    # -----------------------------------------------------

    incorrect_rows = np.where(
        np.any(incorrect, axis=1)
    )[0]

    if len(incorrect_rows) == 0:

        st.success(
            "All test samples were classified "
            "correctly for both outputs."
        )

        return

    st.markdown("### Incorrectly Predicted Samples")

    error_df = pd.DataFrame({

        "Test Sample": incorrect_rows,

        "Actual Overheat": y_test[incorrect_rows, 0],

        "Predicted Overheat": predictions[incorrect_rows, 0],

        "Overheat Probability": probabilities[incorrect_rows, 0],

        "Actual Fault": y_test[incorrect_rows, 1],

        "Predicted Fault": predictions[incorrect_rows, 1],

        "Fault Probability": probabilities[incorrect_rows, 1],
    })

    # -----------------------------------------------------
    # Add original sensor features
    # -----------------------------------------------------

    if X_test_raw is not None:

        try:

            raw_df = pd.DataFrame(
                X_test_raw
            )

            raw_df = raw_df.iloc[
                incorrect_rows
            ].reset_index(drop=True)

            error_df = pd.concat(
                [
                    raw_df,
                    error_df.reset_index(drop=True),
                ],
                axis=1,
            )

        except Exception:

            st.caption(
                "Original sensor values could not be added "
                "to the error table."
            )

    st.dataframe(
        error_df.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download Error Analysis",
        data=error_df.to_csv(index=False).encode("utf-8"),
        file_name="model_error_analysis.csv",
        mime="text/csv",
    )


# =========================================================
# INTERPRETATION
# =========================================================

def render_interpretation(results):

    st.markdown("## 🧠 Performance Interpretation")

    accuracy = results["element_accuracy"]
    loss = results["loss"]
    exact_match = results["exact_match_accuracy"]

    if accuracy >= 0.90:

        st.success(
            "The model achieved strong overall test accuracy. "
            "However, inspect per-target recall and confusion "
            "matrices to ensure alert cases are detected."
        )

    elif accuracy >= 0.75:

        st.info(
            "The model achieved reasonably good test accuracy. "
            "Further tuning may improve generalization."
        )

    else:

        st.warning(
            "Test accuracy is relatively low. Consider "
            "experimenting with learning rate, activation "
            "function, epochs, or model initialization."
        )

    st.markdown(
        f"""
        **Test loss:** {loss:.4f}

        **Element-wise accuracy:** {accuracy * 100:.2f}%

        **Exact-match accuracy:** {exact_match * 100:.2f}%
        """
    )

    st.markdown("""
    ### What these metrics mean

    - **Accuracy:** Percentage of individual output labels predicted correctly.
    - **Precision:** Among predicted alerts, how many were actually alerts?
    - **Recall:** Among actual alerts, how many did the model detect?
    - **F1-score:** Balance between precision and recall.
    - **Confusion matrix:** Shows correct and incorrect predictions for each class.
    - **Exact-match accuracy:** Percentage of samples for which both outputs
      were predicted correctly.
    """)

    # -----------------------------------------------------
    # Alert recall warning
    # -----------------------------------------------------

    fault_metrics = results["target_results"]["Fault Alert"]

    if fault_metrics["recall"] < 0.50:

        st.warning(
            f"Fault Alert recall is only "
            f"{fault_metrics['recall'] * 100:.2f}%. "
            "The model is missing many actual fault cases. "
            "For a fault-detection system, this is an important "
            "weakness even if overall accuracy looks acceptable."
        )


# =========================================================
# MAIN PAGE
# =========================================================

def render_model_insights(
    dataset=None,
    model=None,
):

    render_header()

    # -----------------------------------------------------
    # Get model
    # -----------------------------------------------------

    if model is None:

        model = st.session_state.get(
            "nn_model"
        )

    if model is None:

        st.warning(
            "No trained model is available."
        )

        st.info(
            "Go to Training Lab, train your model, "
            "and then return here."
        )

        return

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    test_data = get_test_data(
        dataset
    )

    if test_data is None:
        return

    (
        X_test,
        y_test,
        X_test_raw,
    ) = test_data

    # -----------------------------------------------------
    # Evaluation settings
    # -----------------------------------------------------

    st.markdown("## Evaluation Settings")

    threshold = st.slider(
        "Classification Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        key="insights_threshold",
    )

    st.caption(
        "A probability equal to or above the threshold "
        "is classified as 1."
    )

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    evaluate_button = st.button(
        "🔬 Evaluate Model on Test Data",
        type="primary",
        use_container_width=True,
    )

    if evaluate_button:

        try:

            with st.spinner(
                "Evaluating model on unseen test samples..."
            ):

                results = evaluate_on_test(
                    model,
                    X_test,
                    y_test,
                    threshold,
                )

                st.session_state[
                    "test_evaluation_results"
                ] = results

                st.session_state[
                    "test_evaluation_threshold"
                ] = threshold

            st.success(
                "Test evaluation completed!"
            )

        except Exception as e:

            st.error(
                f"Evaluation failed: {e}"
            )

            st.exception(e)

            return

    # -----------------------------------------------------
    # Get saved results
    # -----------------------------------------------------

    saved_results = st.session_state.get(
        "test_evaluation_results"
    )

    if saved_results is None:

        st.info(
            "Click Evaluate Model on Test Data "
            "to begin the evaluation."
        )

        return

    saved_threshold = saved_results.get(
        "threshold",
        0.5,
    )

    st.caption(
        f"Results were calculated using threshold "
        f"{saved_threshold:.2f}."
    )

    st.divider()

    # -----------------------------------------------------
    # Render all results
    # -----------------------------------------------------

    render_metric_cards(saved_results)

    render_target_metrics(saved_results)

    render_confusion_matrices(saved_results)

    render_classification_reports(saved_results)

    render_actual_vs_predicted(
        saved_results,
        y_test,
    )

    render_error_analysis(
        saved_results,
        y_test,
        X_test_raw,
    )

    render_interpretation(saved_results)

    # -----------------------------------------------------
    # Disclaimer
    # -----------------------------------------------------

    st.html("""
    <div class="info-box">
        <div class="info-box-title">Important</div>
        <div class="info-box-text">
            This project uses synthetic sensor data for educational
            purposes. Its predictions are not suitable for real
            industrial safety decisions.
        </div>
    </div>
    """)
