import numpy as np

from core.neural_network import NeuralNetwork


# =========================================================
# MODEL EVALUATION
# =========================================================

def evaluate_model(model, X, y):
    """
    Evaluate model on a dataset.

    Returns:
        Loss
        Element-wise accuracy
        Exact-match accuracy
        Predictions
        Probabilities
    """

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if y.ndim == 1:
        y = y.reshape(1, -1)

    probabilities = model.predict_proba(X)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    loss = model.binary_cross_entropy(
        y,
        probabilities,
    )

    # Accuracy across individual output labels
    element_accuracy = np.mean(
        predictions == y
    )

    # Both outputs must be correct for a sample
    # to count as an exact match
    exact_match_accuracy = np.mean(
        np.all(
            predictions == y,
            axis=1,
        )
    )

    return {
        "loss": float(loss),

        "element_accuracy": float(
            element_accuracy
        ),

        "exact_match_accuracy": float(
            exact_match_accuracy
        ),

        "predictions": predictions,

        "probabilities": probabilities,
    }


# =========================================================
# TRAINING
# =========================================================

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=100,
    batch_size=32,
    seed=42,
):
    """
    Train a neural network using mini-batch SGD.

    Records actual training and validation loss
    after each epoch.

    Parameters
    ----------
    model:
        NeuralNetwork instance

    X_train:
        Training features

    y_train:
        Training targets

    X_val:
        Validation features

    y_val:
        Validation targets

    epochs:
        Number of training epochs

    batch_size:
        Number of samples in each batch

    seed:
        Random seed for reproducible shuffling
    """

    X_train = np.asarray(
        X_train,
        dtype=np.float64,
    )

    y_train = np.asarray(
        y_train,
        dtype=np.float64,
    )

    X_val = np.asarray(
        X_val,
        dtype=np.float64,
    )

    y_val = np.asarray(
        y_val,
        dtype=np.float64,
    )

    # Handle one-dimensional input
    if X_train.ndim == 1:
        X_train = X_train.reshape(1, -1)

    if y_train.ndim == 1:
        y_train = y_train.reshape(1, -1)

    if X_val.ndim == 1:
        X_val = X_val.reshape(1, -1)

    if y_val.ndim == 1:
        y_val = y_val.reshape(1, -1)

    # Reproducible shuffling
    rng = np.random.default_rng(seed)

    history = {
        "epoch": [],

        "train_loss": [],
        "val_loss": [],

        "train_accuracy": [],
        "val_accuracy": [],

        "train_exact_accuracy": [],
        "val_exact_accuracy": [],
    }

    n_samples = len(X_train)

    # =====================================================
    # EPOCH LOOP
    # =====================================================

    for epoch in range(1, epochs + 1):

        # Shuffle training observations
        indices = rng.permutation(
            n_samples
        )

        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # =================================================
        # MINI-BATCH TRAINING
        # =================================================

        for start in range(
            0,
            n_samples,
            batch_size,
        ):

            end = start + batch_size

            X_batch = X_shuffled[
                start:end
            ]

            y_batch = y_shuffled[
                start:end
            ]

            # Forward propagation,
            # calculate loss,
            # backpropagation,
            # update weights
            model.train_step(
                X_batch,
                y_batch,
            )

        # =================================================
        # EVALUATE AFTER EACH EPOCH
        # =================================================

        train_results = evaluate_model(
            model,
            X_train,
            y_train,
        )

        val_results = evaluate_model(
            model,
            X_val,
            y_val,
        )

        # Store actual values
        history["epoch"].append(epoch)

        history["train_loss"].append(
            train_results["loss"]
        )

        history["val_loss"].append(
            val_results["loss"]
        )

        history["train_accuracy"].append(
            train_results["element_accuracy"]
        )

        history["val_accuracy"].append(
            val_results["element_accuracy"]
        )

        history["train_exact_accuracy"].append(
            train_results["exact_match_accuracy"]
        )

        history["val_exact_accuracy"].append(
            val_results["exact_match_accuracy"]
        )

    # =====================================================
    # FINAL RESULTS
    # =====================================================

    final_train_results = evaluate_model(
        model,
        X_train,
        y_train,
    )

    final_val_results = evaluate_model(
        model,
        X_val,
        y_val,
    )

    results = {
        "history": history,

        "final_train": final_train_results,

        "final_val": final_val_results,

        "epochs_completed": epochs,

        "batch_size": batch_size,

        "learning_rate": model.learning_rate,

        "activation": model.activation_name,
    }

    return results


# =========================================================
# TRAINING HISTORY HELPERS
# =========================================================

def get_training_history_dataframe(
    history,
):
    """
    Convert training history to a DataFrame
    for display and plotting.
    """

    import pandas as pd

    return pd.DataFrame(history)


# =========================================================
# LOSS INTERPRETATION
# =========================================================

def interpret_learning(
    train_loss,
    val_loss,
):
    """
    Provide a simple interpretation of the
    training and validation losses.
    """

    if len(train_loss) < 2 or len(val_loss) < 2:

        return "Not enough training history to interpret."

    final_train = train_loss[-1]
    final_val = val_loss[-1]

    initial_train = train_loss[0]
    initial_val = val_loss[0]

    train_improved = (
        final_train < initial_train
    )

    val_improved = (
        final_val < initial_val
    )

    gap = final_val - final_train

    if train_improved and val_improved:

        if gap > 0.15:

            return (
                "Training loss decreased and validation loss "
                "also improved. However, the difference "
                "between them suggests the model may be "
                "starting to overfit."
            )

        return (
            "Both training and validation losses decreased. "
            "The model appears to be learning and "
            "generalizing reasonably well."
        )

    if train_improved and not val_improved:

        return (
            "Training loss improved, but validation loss "
            "did not improve. This may indicate overfitting "
            "or that the model is not generalizing well."
        )

    if not train_improved and not val_improved:

        return (
            "Neither loss improved significantly. "
            "The model may need a different learning rate, "
            "activation function, or more training."
        )

    return (
        "Validation loss improved. Continue monitoring "
        "training and validation behaviour."
    )