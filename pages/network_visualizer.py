import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# =========================================================
# ARCHITECTURE
# =========================================================

LAYER_SIZES = [3, 5, 4, 2]

LAYER_NAMES = [
    "Input Layer",
    "Hidden Layer 1",
    "Hidden Layer 2",
    "Output Layer",
]

NEURON_LABELS = [
    ["Temperature", "Vibration", "Pressure"],
    ["H1-1", "H1-2", "H1-3", "H1-4", "H1-5"],
    ["H2-1", "H2-2", "H2-3", "H2-4"],
    ["Overheat", "Fault"],
]

FEATURE_LABELS = [
    "Temperature (°C)",
    "Vibration (mm/s)",
    "Pressure (kPa)",
]

OUTPUT_LABELS = [
    "Overheat Alert",
    "Fault Alert",
]


# =========================================================
# HEADER
# =========================================================

def render_header():

    st.html("""
    <div style="
        background:linear-gradient(135deg,#e8f8f5,#f8fffe);
        border:1px solid #cceee6;
        border-radius:20px;
        padding:30px;
        margin-bottom:20px;
    ">

        <div style="
            color:#0f766e;
            font-size:12px;
            font-weight:700;
            letter-spacing:2px;
            margin-bottom:10px;
        ">
            MODEL ARCHITECTURE
        </div>

        <h1 style="
            color:#134e4a;
            font-size:30px;
            margin:0;
        ">
            Network Visualizer 🧠
        </h1>

        <p style="
            color:#52716d;
            font-size:15px;
            margin-top:10px;
            margin-bottom:0;
        ">
            Explore the neural network architecture, neuron activations,
            weights, biases, and forward propagation.
        </p>

    </div>
    """)


# =========================================================
# HELPER: CONVERT ARRAY
# =========================================================

def to_numpy(data):

    if data is None:
        return None

    if isinstance(data, pd.DataFrame):
        return data.to_numpy(dtype=np.float64)

    if isinstance(data, pd.Series):
        return data.to_numpy(dtype=np.float64)

    return np.asarray(data, dtype=np.float64)


# =========================================================
# HELPER: EXTRACT DATASET SAMPLE
# =========================================================

def get_sample(dataset):

    if dataset is None or not isinstance(dataset, dict):
        return None, None, None

    X_test_scaled = dataset.get("X_test_scaled")
    X_test_raw = dataset.get("X_test")
    y_test = dataset.get("y_test")

    if X_test_scaled is None:
        return None, None, None

    X_scaled = to_numpy(X_test_scaled)
    X_raw = to_numpy(X_test_raw)
    y = to_numpy(y_test)

    if X_scaled.ndim == 1:
        X_scaled = X_scaled.reshape(1, -1)

    if X_raw is not None and X_raw.ndim == 1:
        X_raw = X_raw.reshape(1, -1)

    if y is not None and y.ndim == 1:
        y = y.reshape(1, -1)

    if len(X_scaled) == 0:
        return None, None, None

    st.markdown("### 🧪 Test Sample Selection")

    selected_index = st.slider(
        "Select a test observation",
        min_value=0,
        max_value=len(X_scaled) - 1,
        value=0,
        step=1,
        key="network_sample_slider",
    )

    sample_scaled = X_scaled[selected_index:selected_index + 1]

    if X_raw is not None and len(X_raw) > selected_index:
        sample_raw = X_raw[selected_index:selected_index + 1]
    else:
        sample_raw = sample_scaled.copy()

    if y is not None and len(y) > selected_index:
        sample_targets = y[selected_index:selected_index + 1]
    else:
        sample_targets = None

    return sample_scaled, sample_raw, sample_targets


# =========================================================
# NETWORK DIAGRAM
# =========================================================

def create_network_diagram(model, trace):

    fig = go.Figure()

    layer_x = [0, 1, 2, 3]

    neuron_positions = {}

    # -----------------------------------------------------
    # Neuron positions
    # -----------------------------------------------------

    for layer_index, neuron_count in enumerate(LAYER_SIZES):

        y_positions = np.linspace(
            1.0,
            -1.0,
            neuron_count,
        )

        neuron_positions[layer_index] = [
            (layer_x[layer_index], y)
            for y in y_positions
        ]

    # -----------------------------------------------------
    # Weight matrices
    # -----------------------------------------------------

    weights = [
        model.W1,
        model.W2,
        model.W3,
    ]

    # -----------------------------------------------------
    # Draw connections
    # -----------------------------------------------------

    for layer_index, W in enumerate(weights):

        source_positions = neuron_positions[layer_index]
        target_positions = neuron_positions[layer_index + 1]

        for source_index in range(W.shape[0]):

            for target_index in range(W.shape[1]):

                weight = float(
                    W[source_index, target_index]
                )

                x0, y0 = source_positions[source_index]
                x1, y1 = target_positions[target_index]

                if weight >= 0:
                    line_color = "rgba(13,148,136,0.40)"
                else:
                    line_color = "rgba(239,68,68,0.40)"

                line_width = 0.7 + min(abs(weight) * 1.5, 3.0)

                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line={
                            "color": line_color,
                            "width": line_width,
                        },
                        hovertemplate=(
                            f"<b>Connection</b><br>"
                            f"Layer: {layer_index + 1}<br>"
                            f"From Neuron: {source_index + 1}<br>"
                            f"To Neuron: {target_index + 1}<br>"
                            f"Weight: {weight:.6f}"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )

    # -----------------------------------------------------
    # Activations
    # -----------------------------------------------------

    layer_values = [
        trace["input"][0],
        trace["a1"][0],
        trace["a2"][0],
        trace["a3"][0],
    ]

    # -----------------------------------------------------
    # Draw neurons
    # -----------------------------------------------------

    for layer_index, positions in neuron_positions.items():

        x_values = [p[0] for p in positions]
        y_values = [p[1] for p in positions]

        node_values = layer_values[layer_index]

        node_labels = NEURON_LABELS[layer_index]

        hover_text = []

        for neuron_index in range(len(positions)):

            hover_text.append(
                f"<b>{node_labels[neuron_index]}</b><br>"
                f"Layer: {LAYER_NAMES[layer_index]}<br>"
                f"Neuron: {neuron_index + 1}<br>"
                f"Activation: {float(node_values[neuron_index]):.6f}"
            )

        # Output neuron colors

        if layer_index == 3:

            node_colors = [
                "#f59e0b",
                "#ef4444",
            ]

        else:

            node_colors = [
                "#0d9488"
            ] * len(positions)

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers+text",
                text=node_labels,
                textposition="middle right",
                textfont={
                    "size": 11,
                    "color": "#365965",
                },
                marker={
                    "size": 32,
                    "color": node_colors,
                    "line": {
                        "width": 2,
                        "color": "white",
                    },
                },
                hovertext=hover_text,
                hoverinfo="text",
                showlegend=False,
            )
        )

    # -----------------------------------------------------
    # Layer labels
    # -----------------------------------------------------

    for layer_index, x in enumerate(layer_x):

        fig.add_annotation(
            x=x,
            y=1.35,
            text=(
                f"<b>{LAYER_NAMES[layer_index]}</b>"
                f"<br>{LAYER_SIZES[layer_index]} neurons"
            ),
            showarrow=False,
            font={
                "size": 13,
                "color": "#164b55",
            },
        )

    # -----------------------------------------------------
    # Layout
    # -----------------------------------------------------

    fig.update_layout(
        title={
            "text": "Fully Connected Neural Network · 3–5–4–2",
            "x": 0.5,
        },
        height=650,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={
            "l": 30,
            "r": 100,
            "t": 100,
            "b": 30,
        },
        xaxis={
            "visible": False,
            "range": [-0.3, 3.5],
        },
        yaxis={
            "visible": False,
            "range": [-1.5, 1.6],
        },
        hovermode="closest",
    )

    return fig


# =========================================================
# FORWARD PROPAGATION
# =========================================================

def render_forward_propagation(
    model,
    sample_scaled,
    sample_raw,
    sample_targets=None,
):

    st.markdown("## ⚡ Forward Propagation")

    # Forward pass

    probabilities = model.forward(sample_scaled)

    probabilities = np.asarray(probabilities)

    trace = model.get_forward_trace()

    trace["input"] = sample_scaled

    # -----------------------------------------------------
    # Probability cards
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Overheat Probability",
            f"{probabilities[0, 0] * 100:.2f}%",
        )

    with col2:

        st.metric(
            "Fault Probability",
            f"{probabilities[0, 1] * 100:.2f}%",
        )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    predicted = (
        probabilities >= 0.5
    ).astype(int)

    if sample_targets is not None:

        actual_values = [
            int(sample_targets[0, 0]),
            int(sample_targets[0, 1]),
        ]

    else:

        actual_values = ["Unavailable", "Unavailable"]

    result_table = pd.DataFrame({
        "Target": OUTPUT_LABELS,
        "Actual": actual_values,
        "Probability": [
            float(probabilities[0, 0]),
            float(probabilities[0, 1]),
        ],
        "Predicted": [
            int(predicted[0, 0]),
            int(predicted[0, 1]),
        ],
    })

    st.markdown("### Prediction Results")

    st.dataframe(
        result_table,
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------
    # Input values
    # -----------------------------------------------------

    input_table = pd.DataFrame({
        "Feature": FEATURE_LABELS,
        "Raw Input": sample_raw[0],
        "Scaled Input": sample_scaled[0],
    })

    st.markdown("### Input Values")

    st.dataframe(
        input_table.round(5),
        use_container_width=True,
        hide_index=True,
    )

    return trace, probabilities


# =========================================================
# ACTIVATION VALUES
# =========================================================

def render_activation_values(trace):

    st.markdown("## 🔢 Neuron Activation Values")

    tabs = st.tabs([
        "Input",
        "Hidden Layer 1",
        "Hidden Layer 2",
        "Output",
    ])

    values = [
        trace["input"][0],
        trace["a1"][0],
        trace["a2"][0],
        trace["a3"][0],
    ]

    for layer_index, tab in enumerate(tabs):

        with tab:

            neuron_names = NEURON_LABELS[layer_index]

            layer_values = values[layer_index]

            activation_table = pd.DataFrame({
                "Neuron": neuron_names,
                "Activation": layer_values,
            })

            st.dataframe(
                activation_table.round(6),
                use_container_width=True,
                hide_index=True,
            )

            fig = go.Figure(
                go.Bar(
                    x=neuron_names,
                    y=layer_values,
                    marker_color="#0d9488",
                )
            )

            fig.update_layout(
                title=f"{LAYER_NAMES[layer_index]} Activations",
                template="plotly_white",
                height=350,
                xaxis_title="Neuron",
                yaxis_title="Activation Value",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


# =========================================================
# WEIGHTS AND BIASES
# =========================================================

def render_parameters(model):

    st.markdown("## ⚙️ Weights and Biases")

    parameters = model.get_parameters()

    tabs = st.tabs([
        "Layer 1",
        "Layer 2",
        "Output Layer",
    ])

    layer_info = [
        ("W1", "b1"),
        ("W2", "b2"),
        ("W3", "b3"),
    ]

    layer_names = [
        "Input → Hidden Layer 1",
        "Hidden Layer 1 → Hidden Layer 2",
        "Hidden Layer 2 → Output",
    ]

    for i, tab in enumerate(tabs):

        with tab:

            weight_key, bias_key = layer_info[i]

            weights = np.asarray(parameters[weight_key])

            biases = np.asarray(parameters[bias_key])

            st.markdown(f"### {layer_names[i]}")

            st.caption(
                f"Weight matrix shape: {weights.shape}"
            )

            # Weight matrix

            st.markdown("#### Weight Matrix")

            weight_df = pd.DataFrame(weights)

            st.dataframe(
                weight_df.round(5),
                use_container_width=True,
            )

            # Biases

            st.markdown("#### Bias Values")

            biases = biases.reshape(-1)

            bias_df = pd.DataFrame({
                "Neuron": [
                    f"Neuron {j + 1}"
                    for j in range(len(biases))
                ],
                "Bias": biases,
            })

            st.dataframe(
                bias_df.round(5),
                use_container_width=True,
                hide_index=True,
            )

            # Heatmap

            st.markdown("#### Weight Heatmap")

            fig = go.Figure(
                go.Heatmap(
                    z=weights,
                    text=np.round(weights, 3),
                    texttemplate="%{text}",
                    colorscale="RdBu",
                    zmid=0,
                    colorbar={
                        "title": "Weight",
                    },
                )
            )

            fig.update_layout(
                title=f"{weight_key} Weight Heatmap",
                template="plotly_white",
                height=400,
                xaxis_title="Destination Neuron",
                yaxis_title="Source Neuron",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


# =========================================================
# ACTIVATION FUNCTION EXPLANATION
# =========================================================

def render_activation_explanation(model):

    st.markdown("## 📚 Activation Functions")

    activation_name = getattr(
        model,
        "activation_name",
        "Unknown",
    )

    # -----------------------------------------------------
    # ReLU
    # -----------------------------------------------------

    if activation_name == "ReLU":

        st.html("""
        <div style="
            background:#ecfdf5;
            border:1px solid #a7f3d0;
            border-radius:12px;
            padding:16px;
        ">
            <h3 style="color:#047857;margin-top:0;">
                Current Activation: ReLU
            </h3>
            <p style="color:#065f46;">
                ReLU returns zero for negative inputs and
                returns the original value for positive inputs.
            </p>
        </div>
        """)

        st.latex(r"f(x)= \max(0,x)")

        st.markdown("""
        **Advantages**

        - Computationally simple.
        - Helps reduce the vanishing gradient problem for positive activations.
        - Often effective for hidden layers.

        **Limitation**

        - Neurons can become inactive if they consistently receive negative inputs.
        """)

    # -----------------------------------------------------
    # Sigmoid
    # -----------------------------------------------------

    elif activation_name == "Sigmoid":

        st.html("""
        <div style="
            background:#eff6ff;
            border:1px solid #bfdbfe;
            border-radius:12px;
            padding:16px;
        ">
            <h3 style="color:#1d4ed8;margin-top:0;">
                Current Activation: Sigmoid
            </h3>
            <p style="color:#1e40af;">
                Sigmoid converts input values into a range
                between zero and one.
            </p>
        </div>
        """)

        st.latex(r"f(x)= \frac{1}{1+e^{-x}}")

        st.markdown("""
        **Advantages**

        - Smooth output.
        - Produces values between zero and one.

        **Limitation**

        - Gradients can become very small for extremely
          positive or negative inputs.
        """)

    else:

        st.info(
            f"Current activation function: {activation_name}"
        )

    # -----------------------------------------------------
    # Output activation
    # -----------------------------------------------------

    st.markdown("### Output Layer")

    st.html("""
    <div style="
        background:#f0fdfa;
        border:1px solid #ccfbf1;
        padding:16px;
        border-radius:12px;
        color:#134e4a;
    ">
        The output layer uses sigmoid independently for
        the two target variables.

        <br><br>

        <b>Overheat Alert</b> and <b>Fault Alert</b>
        are separate predictions.

        <br><br>

        Therefore, the network can predict both alerts
        at the same time, neither alert, or either
        individual alert.
    </div>
    """)

    st.markdown("### Possible Predictions")

    prediction_table = pd.DataFrame({
        "Overheat": [0, 0, 1, 1],
        "Fault": [0, 1, 0, 1],
        "Meaning": [
            "No alerts",
            "Fault only",
            "Overheat only",
            "Both alerts",
        ],
    })

    st.dataframe(
        prediction_table,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# MAIN PAGE
# =========================================================

def render_network_visualizer(
    model=None,
    dataset=None,
):

    render_header()

    # -----------------------------------------------------
    # Model check
    # -----------------------------------------------------

    if model is None:

        st.html("""
        <div style="
            background:#fff7ed;
            border:1px solid #fed7aa;
            border-radius:12px;
            padding:18px;
            color:#9a3412;
        ">
            <h4 style="margin-top:0;">
                Model Not Initialized
            </h4>
            The neural network model is not available yet.
            Train or initialize the model using the Training Lab.
        </div>
        """)

        return

    # -----------------------------------------------------
    # Architecture summary
    # -----------------------------------------------------

    st.html("""
    <div style="
        background:#f0fdfa;
        border:1px solid #ccfbf1;
        border-radius:14px;
        padding:18px;
        margin-bottom:15px;
        color:#134e4a;
    ">
        <b>Architecture:</b> 3 → 5 → 4 → 2

        <br><br>

        <b>Input:</b> Temperature, Vibration, Pressure

        <br>

        <b>Hidden Layers:</b> 5 neurons and 4 neurons

        <br>

        <b>Output:</b> Overheat Alert and Fault Alert

        <br><br>

        Every neuron in one layer is connected to
        every neuron in the next layer.
    </div>
    """)

    # -----------------------------------------------------
    # Sample selection
    # -----------------------------------------------------

    sample_scaled, sample_raw, sample_targets = get_sample(dataset)

    # If dataset unavailable, use a demonstration input

    if sample_scaled is None:

        st.info(
            "Dataset test samples are unavailable. "
            "A zero-valued input is used for visualization."
        )

        sample_scaled = np.zeros((1, 3))

        sample_raw = np.zeros((1, 3))

        sample_targets = None

    # -----------------------------------------------------
    # Forward propagation
    # -----------------------------------------------------

    trace, probabilities = render_forward_propagation(
        model,
        sample_scaled,
        sample_raw,
        sample_targets,
    )

    st.divider()

    # -----------------------------------------------------
    # Network diagram
    # -----------------------------------------------------

    st.markdown("## 🌐 Network Diagram")

    fig = create_network_diagram(
        model,
        trace,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.html("""
    <div style="
        display:flex;
        gap:18px;
        flex-wrap:wrap;
        background:#f8fafc;
        border:1px solid #e2e8f0;
        border-radius:12px;
        padding:14px;
    ">
        <div style="color:#0f766e;">
            <b>● Positive Weights</b>
        </div>

        <div style="color:#dc2626;">
            <b>● Negative Weights</b>
        </div>

        <div style="color:#64748b;">
            <b>Hover over nodes and connections to see details.</b>
        </div>
    </div>
    """)

    st.divider()

    # -----------------------------------------------------
    # Activations
    # -----------------------------------------------------

    render_activation_values(trace)

    st.divider()

    # -----------------------------------------------------
    # Parameters
    # -----------------------------------------------------

    render_parameters(model)

    st.divider()

    # -----------------------------------------------------
    # Activation explanations
    # -----------------------------------------------------

    render_activation_explanation(model)