"""
model_builder.py
----------------
This file is responsible for CREATING the ANN (Artificial Neural Network) model.

It contains:
- A helper function to build a baseline ANN using Keras (TensorFlow backend).
- This model will be used in:
      train.py                -> for baseline training
      tune_hyperparameters.py -> for hyperparameter tuning (later)

We keep the model-building logic here so it is NOT duplicated in multiple files.
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# We import the TrainParams dataclass from config.py
# so we can use its values for hidden units, activation, learning rate, etc.
from config import TrainParams


def build_ann_model(
    input_dim: int,
    num_classes: int,
    train_params: Optional[TrainParams] = None,
) -> tf.keras.Model:
    """
    Build and compile a simple fully-connected ANN for multi-class classification.

    Args:
        input_dim (int):
            Number of input features (columns in X).
            Example: if each sample has 100 features, input_dim = 100.

        num_classes (int):
            Number of output classes.
            Example: 26 for A-Z alphabets, or however many labels your dataset has.

        train_params (TrainParams, optional):
            Dataclass instance containing:
                - hidden_units
                - hidden_layers
                - activation
                - output_activation
                - learning_rate
            If None, we create a default TrainParams() inside the function.

    Returns:
        model (tf.keras.Model):
            A compiled Keras model ready for training.
    """

    # ------------------------------------------------------------------
    # 1. Use default training parameters if none are provided
    # ------------------------------------------------------------------
    # If the caller does not pass train_params, we set a default one.
    if train_params is None:
        train_params = TrainParams()
        # This uses the default values defined in config.py

    # ------------------------------------------------------------------
    # 2. Create a Sequential model
    # ------------------------------------------------------------------
    # We use tf.keras.Sequential:
    #   - Layers are added in order from input to output.
    #   - Very simple to read and understand for feed-forward networks.
    model = models.Sequential()

    # ------------------------------------------------------------------
    # 3. Input layer + first hidden layer
    # ------------------------------------------------------------------
    # The first Dense layer needs input_dim (number of features).
    #
    # hidden_units: how many neurons in the hidden layer.
    # activation  : non-linear activation function, usually "relu" for ANN.
    #
    # Example:
    # Dense(64, input_dim=100, activation="relu")
    #
    # NOTE:
    #   We use "input_shape=(input_dim,)" instead of "input_dim" (both work).
    model.add(
        layers.Dense(
            units=train_params.hidden_units,      # neurons in hidden layer
            activation=train_params.activation,   # e.g., "relu"
            input_shape=(input_dim,),             # shape of one input sample
            name="hidden_layer_1"
        )
    )

    # ------------------------------------------------------------------
    # 4. Additional hidden layers (if hidden_layers > 1)
    # ------------------------------------------------------------------
    # If train_params.hidden_layers is greater than 1, we add more Dense layers.
    # For example, if hidden_layers = 3:
    #   -> hidden_layer_1 (we already created)
    #   -> hidden_layer_2
    #   -> hidden_layer_3
    #
    # We keep the same number of units & activation for simplicity.
    for i in range(2, train_params.hidden_layers + 1):
        model.add(
            layers.Dense(
                units=train_params.hidden_units,
                activation=train_params.activation,
                name=f"hidden_layer_{i}"
            )
        )

    # ------------------------------------------------------------------
    # 5. Output layer
    # ------------------------------------------------------------------
    # For multi-class classification, our output layer size = num_classes.
    # Activation should be "softmax" (defined in train_params.output_activation).
    #
    # softmax:
    #   - Produces a probability distribution over all classes.
    #   - Each output node corresponds to one class.
    model.add(
        layers.Dense(
            units=num_classes,
            activation=train_params.output_activation,  # usually "softmax"
            name="output_layer"
        )
    )

    # ------------------------------------------------------------------
    # 6. Choose optimizer
    # ------------------------------------------------------------------
    # We use Adam optimizer, which is a popular choice for ANNs.
    # learning_rate: how big a step we take in each gradient update.
    optimizer = optimizers.Adam(learning_rate=train_params.learning_rate)

    # ------------------------------------------------------------------
    # 7. Compile the model
    # ------------------------------------------------------------------
    # We must specify:
    #   - loss function
    #   - optimizer
    #   - metrics to monitor
    #
    # For multi-class classification with one-hot encoded labels:
    #   loss = "categorical_crossentropy"
    #
    # Metrics:
    #   - "accuracy" to monitor how often predictions are correct.
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    # Now the ANN is fully defined and compiled.
    return model


# ----------------------------------------------------------------------
# Optional: Test build_ann_model quickly by running this file directly.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    """
    This part only runs when you execute:
        python model_builder.py

    It will:
    1. Build a dummy model with arbitrary input_dim and num_classes
    2. Print the model summary

    This is just for a quick manual test that the architecture is valid.
    """

    # Example: Suppose we have 100 input features and 10 classes.
    example_input_dim = 100
    example_num_classes = 10

    # Build a model using default TrainParams
    test_model = build_ann_model(
        input_dim=example_input_dim,
        num_classes=example_num_classes,
        train_params=None  # this will use default TrainParams()
    )

    # Print model architecture
    test_model.summary()


