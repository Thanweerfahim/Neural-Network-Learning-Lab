import numpy as np


class NeuralNetwork:
    """
    Fully connected neural network:

        Input:  3 neurons
        Hidden: 5 neurons
        Hidden: 4 neurons
        Output: 2 neurons

    Architecture: 3-5-4-2

    Supports:
        - ReLU activation
        - Sigmoid activation
        - Forward propagation
        - Backpropagation
        - Binary cross entropy
        - SGD
    """

    def __init__(
        self,
        activation="ReLU",
        learning_rate=0.01,
        seed=42,
    ):

        self.input_size = 3
        self.hidden1_size = 5
        self.hidden2_size = 4
        self.output_size = 2

        self.activation_name = activation
        self.learning_rate = learning_rate

        self.seed = seed

        self.rng = np.random.default_rng(seed)

        # He initialization for ReLU, Xavier for sigmoid
        if activation == "ReLU":

            self.W1 = self.rng.normal(
                0,
                np.sqrt(2 / self.input_size),
                (self.input_size, self.hidden1_size),
            )

            self.W2 = self.rng.normal(
                0,
                np.sqrt(2 / self.hidden1_size),
                (self.hidden1_size, self.hidden2_size),
            )

            self.W3 = self.rng.normal(
                0,
                np.sqrt(2 / self.hidden2_size),
                (self.hidden2_size, self.output_size),
            )

        else:

            self.W1 = self.rng.normal(
                0,
                np.sqrt(1 / self.input_size),
                (self.input_size, self.hidden1_size),
            )

            self.W2 = self.rng.normal(
                0,
                np.sqrt(1 / self.hidden1_size),
                (self.hidden1_size, self.hidden2_size),
            )

            self.W3 = self.rng.normal(
                0,
                np.sqrt(1 / self.hidden2_size),
                (self.hidden2_size, self.output_size),
            )

        # Biases
        self.b1 = np.zeros((1, self.hidden1_size))
        self.b2 = np.zeros((1, self.hidden2_size))
        self.b3 = np.zeros((1, self.output_size))

    # =====================================================
    # ACTIVATION FUNCTIONS
    # =====================================================

    @staticmethod
    def sigmoid(x):

        x = np.clip(x, -500, 500)

        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):

        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):

        return (x > 0).astype(float)

    @staticmethod
    def sigmoid_derivative_from_activation(a):

        return a * (1 - a)

    def activate(self, z):

        if self.activation_name == "ReLU":

            return self.relu(z)

        return self.sigmoid(z)

    def activation_derivative(self, z, a):

        if self.activation_name == "ReLU":

            return self.relu_derivative(z)

        return self.sigmoid_derivative_from_activation(a)

    # =====================================================
    # FORWARD PROPAGATION
    # =====================================================

    def forward(self, X):

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        if X.ndim == 1:

            X = X.reshape(1, -1)

        # Layer 1
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.activate(self.z1)

        # Layer 2
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.activate(self.z2)

        # Output layer
        self.z3 = self.a2 @ self.W3 + self.b3

        # Independent sigmoid outputs
        self.a3 = self.sigmoid(self.z3)

        return self.a3

    # =====================================================
    # BINARY CROSS-ENTROPY LOSS
    # =====================================================

    @staticmethod
    def binary_cross_entropy(
        y_true,
        y_pred,
    ):

        epsilon = 1e-12

        y_pred = np.clip(
            y_pred,
            epsilon,
            1 - epsilon,
        )

        loss = -np.mean(
            y_true * np.log(y_pred)
            +
            (1 - y_true) * np.log(1 - y_pred)
        )

        return float(loss)

    # =====================================================
    # BACKPROPAGATION
    # =====================================================

    def backward(
        self,
        X,
        y,
    ):

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        y = np.asarray(
            y,
            dtype=np.float64,
        )

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if y.ndim == 1:
            y = y.reshape(1, -1)

        batch_size = X.shape[0]

        # Output layer gradient
        dZ3 = self.a3 - y

        dW3 = (
            self.a2.T @ dZ3
        ) / batch_size

        db3 = np.mean(
            dZ3,
            axis=0,
            keepdims=True,
        )

        # Hidden layer 2 gradient
        dA2 = dZ3 @ self.W3.T

        dZ2 = (
            dA2
            *
            self.activation_derivative(
                self.z2,
                self.a2,
            )
        )

        dW2 = (
            self.a1.T @ dZ2
        ) / batch_size

        db2 = np.mean(
            dZ2,
            axis=0,
            keepdims=True,
        )

        # Hidden layer 1 gradient
        dA1 = dZ2 @ self.W2.T

        dZ1 = (
            dA1
            *
            self.activation_derivative(
                self.z1,
                self.a1,
            )
        )

        dW1 = (
            X.T @ dZ1
        ) / batch_size

        db1 = np.mean(
            dZ1,
            axis=0,
            keepdims=True,
        )

        # SGD updates
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    # =====================================================
    # TRAINING STEP
    # =====================================================

    def train_step(
        self,
        X,
        y,
    ):

        predictions = self.forward(X)

        loss = self.binary_cross_entropy(
            y,
            predictions,
        )

        self.backward(
            X,
            y,
        )

        return loss

    # =====================================================
    # PREDICTION
    # =====================================================

    def predict_proba(self, X):

        return self.forward(X)

    def predict(
        self,
        X,
        threshold=0.5,
    ):

        probabilities = self.predict_proba(X)

        return (
            probabilities >= threshold
        ).astype(int)

    # =====================================================
    # NETWORK PARAMETERS
    # =====================================================

    def get_parameters(self):

        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),

            "W2": self.W2.copy(),
            "b2": self.b2.copy(),

            "W3": self.W3.copy(),
            "b3": self.b3.copy(),
        }

    def get_forward_trace(self):

        return {
            "z1": self.z1.copy(),
            "a1": self.a1.copy(),

            "z2": self.z2.copy(),
            "a2": self.a2.copy(),

            "z3": self.z3.copy(),
            "a3": self.a3.copy(),
        }