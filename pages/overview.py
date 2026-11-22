
import streamlit as st


def render_overview(dataset=None, training_results=None, model=None):

    # =====================================================
    # HEADER
    # =====================================================

    st.html("""
    <div class="app-header">
        <div class="app-header-badge">
            NEURAL NETWORK LEARNING LAB
        </div>

        <div class="app-header-title">
            Welcome to Neural Network Lab 🧠
        </div>

        <div class="app-header-subtitle">
            Explore, train, visualize, and understand a
            neural network designed for industrial
            sensor fault detection.
        </div>
    </div>
    """)

    # =====================================================
    # STATUS
    # =====================================================

    dataset_loaded = dataset is not None
    model_trained = training_results is not None

    dataset_status = (
        "Dataset Loaded"
        if dataset_loaded
        else "Awaiting Dataset"
    )

    training_status = (
        "Training Complete"
        if model_trained
        else "Not Trained"
    )

    # =====================================================
    # METRICS
    # =====================================================

    if dataset_loaded:

        raw_data = dataset.get("raw_data")

        if raw_data is not None:
            rows = len(raw_data)
            columns = len(raw_data.columns)
        else:
            rows = 0
            columns = 0

        train_rows = len(dataset.get("X_train", []))
        val_rows = len(dataset.get("X_val", []))
        test_rows = len(dataset.get("X_test", []))

    else:

        rows = 0
        columns = 0
        train_rows = 0
        val_rows = 0
        test_rows = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Dataset Samples",
            value=rows,
        )

    with col2:
        st.metric(
            label="Input Features",
            value=3,
        )

    with col3:
        st.metric(
            label="Output Targets",
            value=2,
        )

    with col4:
        st.metric(
            label="Network Architecture",
            value="3–5–4–2",
        )

    # =====================================================
    # DATASET & MODEL STATUS
    # =====================================================

    st.markdown("## 📌 Project Status")

    status_col1, status_col2 = st.columns(2)

    with status_col1:

        st.html(f"""
        <div class="custom-card">
            <div class="card-title">
                📊 Dataset Status
            </div>

            <div class="card-value" style="font-size:22px;">
                {dataset_status}
            </div>

            <div class="card-description">
                {
                    "Your dataset is ready for exploration."
                    if dataset_loaded
                    else "Load your sensor CSV dataset to begin."
                }
            </div>
        </div>
        """)

    with status_col2:

        st.html(f"""
        <div class="custom-card">
            <div class="card-title">
                ⚙️ Model Status
            </div>

            <div class="card-value" style="font-size:22px;">
                {training_status}
            </div>

            <div class="card-description">
                {
                    "Your model has completed training."
                    if model_trained
                    else "Configure the model and start training."
                }
            </div>
        </div>
        """)

    st.markdown("")

    # =====================================================
    # DATASET DISTRIBUTION
    # =====================================================

    st.markdown("## 📂 Dataset Distribution")

    split_col1, split_col2, split_col3 = st.columns(3)

    with split_col1:

        st.html(f"""
        <div class="custom-card">
            <div class="card-title">
                🟢 Training Set
            </div>

            <div class="card-value">
                {train_rows}
            </div>

            <div class="card-description">
                70% of dataset
            </div>
        </div>
        """)

    with split_col2:

        st.html(f"""
        <div class="custom-card">
            <div class="card-title">
                🟡 Validation Set
            </div>

            <div class="card-value">
                {val_rows}
            </div>

            <div class="card-description">
                15% of dataset
            </div>
        </div>
        """)

    with split_col3:

        st.html(f"""
        <div class="custom-card">
            <div class="card-title">
                🔵 Testing Set
            </div>

            <div class="card-value">
                {test_rows}
            </div>

            <div class="card-description">
                15% of dataset
            </div>
        </div>
        """)

    # =====================================================
    # NETWORK ARCHITECTURE
    # =====================================================

    st.markdown("## 🧠 Neural Network Architecture")

    st.html("""
    <div class="section-container">
        <div class="section-heading">
            Fully Connected Neural Network
        </div>

        <div class="section-description">
            Three input neurons, two hidden layers,
            and two independent sigmoid outputs.
        </div>
    </div>
    """)

    architecture_col1, architecture_col2 = st.columns(
        [1.3, 1]
    )

    with architecture_col1:

        st.html("""
        <div style="
            background:linear-gradient(135deg,#ffffff,#f0faf9);
            border:1px solid #dcefed;
            border-radius:18px;
            padding:25px;
            text-align:center;
        ">

            <div style="
                display:flex;
                justify-content:space-around;
                align-items:center;
                flex-wrap:wrap;
                gap:15px;
            ">

                <div>
                    <div style="
                        font-size:12px;
                        color:#66818b;
                        font-weight:600;
                    ">
                        INPUT
                    </div>

                    <div style="
                        font-size:25px;
                        font-weight:800;
                        color:#0d9488;
                    ">
                        3
                    </div>

                    <div style="
                        font-size:11px;
                        color:#78909a;
                    ">
                        Features
                    </div>
                </div>

                <div style="font-size:25px;color:#a0bec3;">
                    →
                </div>

                <div>
                    <div style="
                        font-size:12px;
                        color:#66818b;
                        font-weight:600;
                    ">
                        HIDDEN 1
                    </div>

                    <div style="
                        font-size:25px;
                        font-weight:800;
                        color:#0d9488;
                    ">
                        5
                    </div>

                    <div style="
                        font-size:11px;
                        color:#78909a;
                    ">
                        Neurons
                    </div>
                </div>

                <div style="font-size:25px;color:#a0bec3;">
                    →
                </div>

                <div>
                    <div style="
                        font-size:12px;
                        color:#66818b;
                        font-weight:600;
                    ">
                        HIDDEN 2
                    </div>

                    <div style="
                        font-size:25px;
                        font-weight:800;
                        color:#0d9488;
                    ">
                        4
                    </div>

                    <div style="
                        font-size:11px;
                        color:#78909a;
                    ">
                        Neurons
                    </div>
                </div>

                <div style="font-size:25px;color:#a0bec3;">
                    →
                </div>

                <div>
                    <div style="
                        font-size:12px;
                        color:#66818b;
                        font-weight:600;
                    ">
                        OUTPUT
                    </div>

                    <div style="
                        font-size:25px;
                        font-weight:800;
                        color:#0d9488;
                    ">
                        2
                    </div>

                    <div style="
                        font-size:11px;
                        color:#78909a;
                    ">
                        Alerts
                    </div>
                </div>

            </div>
        </div>
        """)

    with architecture_col2:

        st.html("""
        <div class="custom-card">

            <div class="card-title">
                INPUT FEATURES
            </div>

            <div style="
                color:#153d48;
                font-size:13px;
                line-height:2;
            ">
                🌡️ Temperature<br>
                📳 Vibration<br>
                💨 Pressure
            </div>

            <hr>

            <div class="card-title">
                OUTPUT PREDICTIONS
            </div>

            <div style="
                color:#153d48;
                font-size:13px;
                line-height:2;
            ">
                🔥 Overheat Alert<br>
                ⚠️ Fault Alert
            </div>

        </div>
        """)

    # =====================================================
    # LEARNING OBJECTIVES
    # =====================================================

    st.markdown("## 🎓 Learning Objectives")

    objective_col1, objective_col2, objective_col3 = (
        st.columns(3)
    )

    with objective_col1:

        st.html("""
        <div class="custom-card">

            <div style="font-size:26px;">
                🔬
            </div>

            <div style="
                font-size:16px;
                font-weight:700;
                color:#153d48;
                margin-top:10px;
            ">
                Understand
            </div>

            <div class="card-description">
                Explore weights, biases, activation
                functions and forward propagation.
            </div>

        </div>
        """)

    with objective_col2:

        st.html("""
        <div class="custom-card">

            <div style="font-size:26px;">
                📉
            </div>

            <div style="
                font-size:16px;
                font-weight:700;
                color:#153d48;
                margin-top:10px;
            ">
                Train
            </div>

            <div class="card-description">
                Observe how loss changes during
                learning and compare settings.
            </div>

        </div>
        """)

    with objective_col3:

        st.html("""
        <div class="custom-card">

            <div style="font-size:26px;">
                🔮
            </div>

            <div style="
                font-size:16px;
                font-weight:700;
                color:#153d48;
                margin-top:10px;
            ">
                Predict
            </div>

            <div class="card-description">
                Test sensor values and understand
                the resulting alert probabilities.
            </div>

        </div>
        """)

    # =====================================================
    # DATASET MESSAGE
    # =====================================================

    if dataset_loaded:

        st.success(
            "Dataset successfully loaded and prepared. "
            "You can explore the data and begin training."
        )

    else:

        st.info(
            "Your dataset has not been loaded into the "
            "application yet. We will connect it in app.py."
        )

    # =====================================================
    # FOOTER
    # =====================================================

    st.html("""
    <div class="app-footer">
        Neural Network Learning Lab
        · Interactive Machine Learning Education
    </div>
    """)
