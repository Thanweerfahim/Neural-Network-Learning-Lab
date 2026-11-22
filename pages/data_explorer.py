import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# CONSTANTS
# =========================================================

FEATURE_COLUMNS = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_kpa",
]

TARGET_COLUMNS = [
    "overheat_alert",
    "fault_alert",
]

FEATURE_LABELS = {
    "temperature_c": "Temperature (°C)",
    "vibration_mm_s": "Vibration (mm/s)",
    "pressure_kpa": "Pressure (kPa)",
}

TARGET_LABELS = {
    "overheat_alert": "Overheat Alert",
    "fault_alert": "Fault Alert",
}


# =========================================================
# PAGE HEADER
# =========================================================

def render_header():

    st.html("""
    <div style="
        background: linear-gradient(135deg, #e8f8f5, #f4fffc);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #cceee6;
        margin-bottom: 20px;
    ">

        <div style="
            color: #0f766e;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 2px;
            margin-bottom: 10px;
        ">
            DATA ANALYTICS
        </div>

        <h1 style="
            color: #134e4a;
            margin: 0;
            font-size: 30px;
        ">
            Data Explorer
        </h1>

        <p style="
            color: #52716d;
            font-size: 15px;
            margin-top: 10px;
            margin-bottom: 0;
        ">
            Explore sensor data, distributions, relationships,
            statistical summaries, and dataset preparation.
        </p>

    </div>
    """)


# =========================================================
# DATASET EXTRACTION
# =========================================================

def get_dataframe(dataset):

    if dataset is None:
        return None

    if isinstance(dataset, pd.DataFrame):
        return dataset.copy()

    if isinstance(dataset, dict):

        possible_keys = [
            "df",
            "dataframe",
            "original_df",
            "raw_df",
            "raw_data",
            "data",
            "dataset",
        ]

        for key in possible_keys:

            value = dataset.get(key)

            if isinstance(value, pd.DataFrame):
                return value.copy()

    return None


# =========================================================
# METRIC CARD
# =========================================================

def render_metrics(df, dataset=None):

    total_samples = len(df)

    input_features = len(
        [col for col in FEATURE_COLUMNS if col in df.columns]
    )

    target_features = len(
        [col for col in TARGET_COLUMNS if col in df.columns]
    )

    missing_values = int(df.isnull().sum().sum())

    train_size = 0
    validation_size = 0
    test_size = 0

    if isinstance(dataset, dict):

        if "X_train" in dataset:
            train_size = len(dataset["X_train"])

        elif "X_train_scaled" in dataset:
            train_size = len(dataset["X_train_scaled"])

        if "X_val" in dataset:
            validation_size = len(dataset["X_val"])

        elif "X_val_scaled" in dataset:
            validation_size = len(dataset["X_val_scaled"])

        if "X_test" in dataset:
            test_size = len(dataset["X_test"])

        elif "X_test_scaled" in dataset:
            test_size = len(dataset["X_test_scaled"])

    st.html(f"""
    <div style="
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 10px;
    ">

        <div style="
            background: white;
            border: 1px solid #dceeea;
            border-radius: 14px;
            padding: 18px;
        ">
            <div style="color:#64748b;font-size:13px;">
                Total Samples
            </div>
            <div style="
                color:#0f766e;
                font-size:27px;
                font-weight:700;
                margin-top:5px;
            ">
                {total_samples:,}
            </div>
        </div>

        <div style="
            background: white;
            border: 1px solid #dceeea;
            border-radius: 14px;
            padding: 18px;
        ">
            <div style="color:#64748b;font-size:13px;">
                Input Features
            </div>
            <div style="
                color:#0f766e;
                font-size:27px;
                font-weight:700;
                margin-top:5px;
            ">
                {input_features}
            </div>
        </div>

        <div style="
            background: white;
            border: 1px solid #dceeea;
            border-radius: 14px;
            padding: 18px;
        ">
            <div style="color:#64748b;font-size:13px;">
                Target Outputs
            </div>
            <div style="
                color:#0f766e;
                font-size:27px;
                font-weight:700;
                margin-top:5px;
            ">
                {target_features}
            </div>
        </div>

        <div style="
            background: white;
            border: 1px solid #dceeea;
            border-radius: 14px;
            padding: 18px;
        ">
            <div style="color:#64748b;font-size:13px;">
                Missing Values
            </div>
            <div style="
                color:{'#dc2626' if missing_values > 0 else '#0f766e'};
                font-size:27px;
                font-weight:700;
                margin-top:5px;
            ">
                {missing_values:,}
            </div>
        </div>

    </div>
    """)

    # Dataset splitting information

    if train_size + validation_size + test_size > 0:

        st.markdown("### Dataset Split")

        split_cols = st.columns(3)

        with split_cols[0]:
            st.metric("Training Samples", f"{train_size:,}")

        with split_cols[1]:
            st.metric("Validation Samples", f"{validation_size:,}")

        with split_cols[2]:
            st.metric("Testing Samples", f"{test_size:,}")


# =========================================================
# DATA PREVIEW
# =========================================================

def render_data_preview(df):

    st.markdown("## 📋 Dataset Preview")

    tab1, tab2, tab3 = st.tabs([
        "First Rows",
        "Last Rows",
        "Complete Dataset",
    ])

    with tab1:

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True,
        )

    with tab2:

        st.dataframe(
            df.tail(10),
            use_container_width=True,
            hide_index=True,
        )

    with tab3:

        st.dataframe(
            df,
            use_container_width=True,
            height=400,
        )

    st.caption(
        f"Dataset contains {len(df):,} rows and {len(df.columns)} columns."
    )


# =========================================================
# STATISTICAL SUMMARY
# =========================================================

def render_feature_statistics(df):

    st.markdown("## 📊 Statistical Summary")

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:

        st.warning("No numerical columns found in the dataset.")

        return

    summary = numeric_df.describe().T

    summary["missing"] = numeric_df.isnull().sum()

    summary["unique"] = numeric_df.nunique()

    summary = summary.round(3)

    st.dataframe(
        summary,
        use_container_width=True,
    )

    st.markdown("### Statistical Information")

    st.html("""
    <div style="
        background:#f0fdfa;
        border:1px solid #ccfbf1;
        border-radius:12px;
        padding:16px;
        color:#134e4a;
    ">
        <b>Mean:</b> Average value of a feature.<br><br>

        <b>Standard Deviation:</b> Measures how spread out
        the values are from the mean.<br><br>

        <b>Minimum and Maximum:</b> Lowest and highest
        values in the dataset.<br><br>

        <b>Quartiles:</b> Divide the data into four equal
        parts and help understand the distribution.
    </div>
    """)


# =========================================================
# FEATURE DISTRIBUTIONS
# =========================================================

def render_feature_distributions(df):

    st.markdown("## 📈 Feature Distributions")

    available_features = [
        col for col in FEATURE_COLUMNS
        if col in df.columns
    ]

    if not available_features:

        st.warning("No input feature columns found.")

        return

    selected_feature = st.selectbox(
        "Select a feature to analyze",
        available_features,
        format_func=lambda x: FEATURE_LABELS.get(x, x),
        key="data_explorer_hist_feature",
    )

    feature_df = df[[selected_feature]].dropna()

    fig = px.histogram(
        feature_df,
        x=selected_feature,
        nbins=30,
        marginal="box",
        title=f"Distribution of {FEATURE_LABELS.get(selected_feature, selected_feature)}",
        color_discrete_sequence=["#0d9488"],
    )

    fig.update_layout(
        template="plotly_white",
        height=450,
        bargap=0.05,
        xaxis_title=FEATURE_LABELS.get(selected_feature, selected_feature),
        yaxis_title="Frequency",
    )

    fig.update_traces(
        marker_line_color="white",
        marker_line_width=1,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # Compare multiple features

    st.markdown("### Feature Comparison")

    selected_features = st.multiselect(
        "Select features to compare",
        available_features,
        default=available_features,
        format_func=lambda x: FEATURE_LABELS.get(x, x),
        key="data_explorer_box_features",
    )

    if selected_features:

        comparison_df = df[selected_features].copy()

        fig = go.Figure()

        for feature in selected_features:

            fig.add_trace(
                go.Box(
                    y=comparison_df[feature],
                    name=FEATURE_LABELS.get(feature, feature),
                    boxpoints="outliers",
                )
            )

        fig.update_layout(
            title="Feature Range Comparison",
            template="plotly_white",
            height=450,
            yaxis_title="Value",
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.caption(
            "Box plots help identify median values, spread, "
            "and potential outliers."
        )


# =========================================================
# TARGET DISTRIBUTIONS
# =========================================================

def render_target_distribution(df):

    st.markdown("## 🎯 Target Distribution")

    available_targets = [
        col for col in TARGET_COLUMNS
        if col in df.columns
    ]

    if not available_targets:

        st.warning("Target columns are missing.")

        return

    cols = st.columns(len(available_targets))

    for index, target in enumerate(available_targets):

        target_counts = (
            df[target]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        target_counts.columns = ["Target", "Count"]

        target_counts["Target"] = target_counts["Target"].astype(str)

        with cols[index]:

            fig = px.bar(
                target_counts,
                x="Target",
                y="Count",
                color="Target",
                title=TARGET_LABELS.get(target, target),
                color_discrete_map={
                    "0": "#0d9488",
                    "1": "#f97316",
                },
                text="Count",
            )

            fig.update_layout(
                template="plotly_white",
                height=350,
                showlegend=False,
                xaxis_title="Class",
                yaxis_title="Number of Samples",
            )

            fig.update_traces(
                textposition="outside",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            total = target_counts["Count"].sum()

            if total > 0:

                positive_count = int(
                    df[target].sum()
                )

                negative_count = total - positive_count

                st.write(
                    f"**Normal (0):** {negative_count:,} "
                    f"({negative_count / total * 100:.1f}%)"
                )

                st.write(
                    f"**Alert (1):** {positive_count:,} "
                    f"({positive_count / total * 100:.1f}%)"
                )


# =========================================================
# ALERT PATTERNS
# =========================================================

def render_alert_patterns(df):

    st.markdown("## 🚨 Combined Alert Patterns")

    if not all(col in df.columns for col in TARGET_COLUMNS):

        st.warning("Both target columns are required.")

        return

    overheat = df["overheat_alert"]

    fault = df["fault_alert"]

    patterns = np.select(
        [
            (overheat == 0) & (fault == 0),
            (overheat == 1) & (fault == 0),
            (overheat == 0) & (fault == 1),
            (overheat == 1) & (fault == 1),
        ],
        [
            "No Alerts",
            "Overheat Only",
            "Fault Only",
            "Both Alerts",
        ],
        default="Unknown",
    )

    pattern_counts = pd.Series(patterns).value_counts()

    preferred_order = [
        "No Alerts",
        "Overheat Only",
        "Fault Only",
        "Both Alerts",
    ]

    pattern_counts = (
        pattern_counts
        .reindex(preferred_order, fill_value=0)
    )

    pattern_df = pattern_counts.reset_index()

    pattern_df.columns = ["Pattern", "Count"]

    fig = px.pie(
        pattern_df,
        names="Pattern",
        values="Count",
        hole=0.45,
        title="Distribution of Combined Alert Patterns",
        color="Pattern",
        color_discrete_map={
            "No Alerts": "#0d9488",
            "Overheat Only": "#f59e0b",
            "Fault Only": "#3b82f6",
            "Both Alerts": "#ef4444",
        },
    )

    fig.update_layout(
        template="plotly_white",
        height=450,
    )

    fig.update_traces(
        textinfo="label+percent",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown("### Pattern Summary")

    cols = st.columns(4)

    for index, pattern in enumerate(preferred_order):

        count = int(pattern_counts[pattern])

        percentage = (
            count / len(df) * 100
            if len(df) > 0
            else 0
        )

        with cols[index]:

            st.metric(
                pattern,
                f"{count:,}",
                f"{percentage:.1f}%",
            )


# =========================================================
# CORRELATION ANALYSIS
# =========================================================

def render_correlation(df):

    st.markdown("## 🔗 Correlation Analysis")

    numeric_columns = [
        col for col in FEATURE_COLUMNS + TARGET_COLUMNS
        if col in df.columns
    ]

    if len(numeric_columns) < 2:

        st.warning("Not enough numerical columns for correlation analysis.")

        return

    correlation_matrix = df[numeric_columns].corr()

    fig = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature and Target Correlation Matrix",
    )

    fig.update_layout(
        template="plotly_white",
        height=550,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown("### How to Interpret Correlation")

    st.html("""
    <div style="
        background:#f0fdfa;
        border:1px solid #ccfbf1;
        padding:16px;
        border-radius:12px;
        color:#134e4a;
    ">
        <b>+1:</b> Strong positive relationship.<br><br>

        <b>0:</b> Little or no linear relationship.<br><br>

        <b>-1:</b> Strong negative relationship.<br><br>

        Correlation indicates a relationship between variables.
        It does not prove that one variable causes another.
    </div>
    """)


# =========================================================
# DATASET SPLIT VISUALIZATION
# =========================================================

def render_data_split(dataset):

    st.markdown("## 🗂️ Dataset Split Analysis")

    if not isinstance(dataset, dict):

        st.info("Dataset split information is not available.")

        return

    split_sizes = {}

    possible_keys = {
        "Training": ["X_train", "X_train_scaled"],
        "Validation": ["X_val", "X_val_scaled"],
        "Testing": ["X_test", "X_test_scaled"],
    }

    for split_name, keys in possible_keys.items():

        for key in keys:

            if key in dataset:

                split_sizes[split_name] = len(dataset[key])

                break

    if not split_sizes or sum(split_sizes.values()) == 0:

        st.info("Dataset split information is not available.")

        return

    split_df = pd.DataFrame({
        "Dataset": list(split_sizes.keys()),
        "Samples": list(split_sizes.values()),
    })

    total = split_df["Samples"].sum()

    split_df["Percentage"] = (
        split_df["Samples"] / total * 100
    )

    fig = px.pie(
        split_df,
        names="Dataset",
        values="Samples",
        hole=0.45,
        title="Training, Validation and Testing Distribution",
        color="Dataset",
        color_discrete_map={
            "Training": "#0d9488",
            "Validation": "#f59e0b",
            "Testing": "#3b82f6",
        },
    )

    fig.update_layout(
        template="plotly_white",
        height=420,
    )

    fig.update_traces(
        textinfo="label+percent",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.dataframe(
        split_df.style.format({
            "Samples": "{:,.0f}",
            "Percentage": "{:.2f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# STANDARDIZATION ANALYSIS
# =========================================================

def render_standardization(df, dataset):

    st.markdown("## ⚖️ Feature Standardization")

    available_features = [
        col for col in FEATURE_COLUMNS
        if col in df.columns
    ]

    if not available_features:

        st.warning("Feature columns are not available.")

        return

    original = df[available_features].copy()

    original_mean = original.mean()

    original_std = original.std()

    st.markdown("### Original Feature Statistics")

    original_stats = pd.DataFrame({
        "Feature": [
            FEATURE_LABELS.get(col, col)
            for col in available_features
        ],
        "Mean": original_mean.values,
        "Standard Deviation": original_std.values,
    })

    st.dataframe(
        original_stats.style.format({
            "Mean": "{:.4f}",
            "Standard Deviation": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Look for standardized training data

    scaled_data = None

    if isinstance(dataset, dict):

        if "X_train_scaled" in dataset:

            scaled_data = dataset["X_train_scaled"]

        elif "X_train" in dataset:

            scaled_data = dataset["X_train"]

    if scaled_data is not None:

        scaled_data = np.asarray(scaled_data)

        if scaled_data.ndim == 2 and scaled_data.shape[1] >= len(available_features):

            scaled_df = pd.DataFrame(
                scaled_data[:, :len(available_features)],
                columns=available_features,
            )

            st.markdown("### Standardized Training Data")

            scaled_stats = pd.DataFrame({
                "Feature": [
                    FEATURE_LABELS.get(col, col)
                    for col in available_features
                ],
                "Mean After Scaling": scaled_df.mean().values,
                "Standard Deviation After Scaling": scaled_df.std().values,
            })

            st.dataframe(
                scaled_stats.style.format({
                    "Mean After Scaling": "{:.4f}",
                    "Standard Deviation After Scaling": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # Compare distributions

            comparison_df = pd.DataFrame({
                "Feature": available_features,
                "Original Mean": original_mean.values,
                "Original Std": original_std.values,
                "Scaled Mean": scaled_df.mean().values,
                "Scaled Std": scaled_df.std().values,
            })

            st.markdown("### Before and After Standardization")

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=[
                        FEATURE_LABELS.get(col, col)
                        for col in available_features
                    ],
                    y=comparison_df["Original Mean"],
                    name="Original Mean",
                    marker_color="#94a3b8",
                )
            )

            fig.add_trace(
                go.Bar(
                    x=[
                        FEATURE_LABELS.get(col, col)
                        for col in available_features
                    ],
                    y=comparison_df["Scaled Mean"],
                    name="Scaled Mean",
                    marker_color="#0d9488",
                )
            )

            fig.update_layout(
                barmode="group",
                template="plotly_white",
                height=400,
                title="Original Mean vs Standardized Mean",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    st.html("""
    <div style="
        background:#f0fdfa;
        border:1px solid #ccfbf1;
        padding:16px;
        border-radius:12px;
        color:#134e4a;
    ">
        <h4 style="margin-top:0;color:#0f766e;">
            Why Standardization is Important
        </h4>

        Standardization transforms features to a similar scale.
        This prevents features with large numerical values from
        dominating the learning process.

        <br><br>

        <b>Formula:</b>

        <br><br>

        Standardized Value = (Value − Mean) / Standard Deviation

        <br><br>

        Standardized features normally have a mean close to 0
        and a standard deviation close to 1.
    </div>
    """)


# =========================================================
# MAIN DATA EXPLORER PAGE
# =========================================================

def render_data_explorer(dataset=None):

    render_header()

    # Extract raw dataframe

    df = get_dataframe(dataset)

    if df is None:

        st.warning(
            "Dataset is not available. Please upload or load "
            "the dataset from the sidebar first."
        )

        return

    if df.empty:

        st.warning("The dataset is empty.")

        return

    # -----------------------------------------------------
    # OVERVIEW
    # -----------------------------------------------------

    render_metrics(df, dataset)

    st.divider()

    # -----------------------------------------------------
    # PREVIEW
    # -----------------------------------------------------

    render_data_preview(df)

    st.divider()

    # -----------------------------------------------------
    # STATISTICS
    # -----------------------------------------------------

    render_feature_statistics(df)

    st.divider()

    # -----------------------------------------------------
    # DISTRIBUTIONS
    # -----------------------------------------------------

    render_feature_distributions(df)

    st.divider()

    # -----------------------------------------------------
    # TARGET ANALYSIS
    # -----------------------------------------------------

    render_target_distribution(df)

    st.divider()

    # -----------------------------------------------------
    # ALERT PATTERNS
    # -----------------------------------------------------

    render_alert_patterns(df)

    st.divider()

    # -----------------------------------------------------
    # CORRELATION
    # -----------------------------------------------------

    render_correlation(df)

    st.divider()

    # -----------------------------------------------------
    # SPLIT ANALYSIS
    # -----------------------------------------------------

    render_data_split(dataset)

    st.divider()

    # -----------------------------------------------------
    # STANDARDIZATION
    # -----------------------------------------------------

    render_standardization(df, dataset)