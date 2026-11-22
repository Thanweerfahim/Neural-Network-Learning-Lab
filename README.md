# Neural Network Learning Lab

An interactive machine learning educational application built with Streamlit to demonstrate neural network training, experimentation, prediction, and model evaluation using industrial sensor data.

## Project Overview

Neural Network Learning Lab allows users to explore how a neural network learns from sensor data. Users can configure training parameters, train the model, visualize training performance, conduct experiments, make predictions, and evaluate model performance.

The application uses a neural network architecture of **3–5–4–2**.

## Neural Network Architecture

- Input Layer: 3 neurons
- Hidden Layer 1: 5 neurons
- Hidden Layer 2: 4 neurons
- Output Layer: 2 neurons

### Input Features

- Temperature
- Vibration
- Pressure

### Output Predictions

- Overheat Alert
- Fault Alert

## Features

- Interactive dashboard
- Dataset upload using CSV files
- Dataset exploration and visualization
- Neural network architecture visualization
- Adjustable learning rate
- Adjustable number of epochs
- Adjustable batch size
- Activation function selection
- Model training and performance visualization
- Prediction using custom sensor values
- Experiment comparison
- Model evaluation and performance metrics

## Technologies Used

- Python
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Plotly
- Scikit-learn

## Project Structure

```text
Neural-Network-Learning-Lab/
│
├── app.py
├── README.md
├── requirements.txt
│
├── components/
│   ├── styles.py
│   └── sidebar.py
│
├── core/
│   ├── dataset.py
│   ├── neural_network.py
│   └── training.py
│
├── pages/
│   ├── overview.py
│   ├── data_explorer.py
│   ├── network_visualizer.py
│   ├── training_lab.py
│   ├── prediction_studio.py
│   ├── experiments.py
│   └── model_insights.py
│
└── data/
    └── sensor_dataset.csv
```

## Dataset

The application uses a CSV dataset containing industrial sensor measurements and target labels.

The input features include:

- Temperature
- Vibration
- Pressure

The target variables include:

- Overheat Alert
- Fault Alert

The dataset is prepared and processed before training. The application separates the data into training, validation, and testing sets and applies feature scaling where required.

The dataset can be loaded from the default data folder or uploaded through the dashboard sidebar.

## Installation

### 1. Install Python

Install Python 3.10 or a compatible version.

### 2. Clone or Download the Project

Download the project source code and open the project folder in your terminal or VS Code.

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Run the following command from the project directory:

```bash
streamlit run app.py
```

The dashboard will open in your web browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

## How to Use the Application

### 1. Overview

View project information, dataset information, and model status.

### 2. Data Explorer

Explore sensor data, distributions, relationships, and target variables.

### 3. Network Visualizer

Visualize the neural network structure and understand the connections between layers.

### 4. Training Lab

Configure the learning rate, epochs, batch size, and activation function. Train the neural network and view the training performance.

### 5. Prediction Studio

Enter sensor measurements to obtain predictions for overheat and fault alerts.

### 6. Experiments

Compare different neural network configurations and analyze their performance.

### 7. Model Insights

Evaluate the trained model using performance metrics, confusion matrices, and prediction analysis.

## Experimentation

Experiments can be conducted by changing:

- Learning rate
- Activation function
- Number of epochs
- Batch size

The results can be compared to understand how different configurations affect model learning and performance.

## Project Objective

The objective of this project is to provide an interactive learning environment for understanding neural networks, model training, parameter tuning, prediction, and evaluation using industrial sensor data.

## Author

Neural Network Learning Lab — Interactive Machine Learning Education
