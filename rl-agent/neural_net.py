"""Create the neural network and handles all the interactions with it."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import is_sequence_of_type


class ANET(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int],
        hidden_units: list[int],
        output_shape: tuple[int, int],
        activation_function: str,
    ) -> None:
        """Init a ANET

        Args:
            input_shape (tuple[int, int]): Shape of input params/features
            hidden_units (list[int]): Number of units (neurons) per layer
            output_shape (tuple[int, int]): Shape of max output params (largest action space)
            activation_function (str): Activation function to use: linear, sigmoid, yanh, RELU
        """
        # Inherit from nn.Module
        super(ANET, self).__init__()

        # Check params
        is_sequence_of_type("input_shape", input_shape, tuple, int, min=1, max=2)
        is_sequence_of_type("hidden_units", hidden_units, list, int, min=1)
        is_sequence_of_type("output_shape", output_shape, tuple, int, min=1, max=2)

        # Set params
        self.activation_function = activation_function

        # Define input layer
        self.input_layer = nn.Linear(np.multiply.reduce(input_shape), hidden_units[0])

        # Define hidden layers
        self.hidden_layers = nn.ModuleList()
        if len(hidden_units) == 1:
            self.hidden_layers.append(nn.Linear(hidden_units[0], hidden_units[0]))
        else:
            for i, j in zip(hidden_units, hidden_units[1:]):
                self.hidden_layers.append(nn.Linear(i, j))

        # Define output layer
        self.output_layer = nn.Linear(
            hidden_units[-1], np.multiply.reduce(output_shape)
        )

    def predict(self, state) -> any:
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


def state_to_tensor() -> torch.Tensor:
    """Converts a state to a tensor

    Returns:
        torch.Tensor: game state as a tensor
    """
    raise NotImplementedError


def scale_output() -> np.ndarray:
    """The output layer must have a fixed number of outputs
    even though there might not be so many avaiable actions.
    From any given state, there is a reduced number of legal actions.
    This function will try to scale the distribution to only represent legal actions.

    Illegal moves will be set to zero.
    Remaining moves will be re-nomralized to a sum of 1.

    Returns:
        np.ndarray: Probability distribution for the legal actions
    """
    raise NotImplementedError
