"""Create the neural network and handles all the interactions with it."""
import torch
import numpy as np


class ANET(torch.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden_layers: int,
        n_output: int,
        n_units: int,
        activation_function: str,
    ) -> None:
        """Init a ANET

        Args:
            n_input (int): Number of max input params/features (largest game space)
            n_hidden_layers (int): Number of hidden layers
            n_output (int): Number of max output params (largest action space)
            n_units (int): Number of units (neurons) per layer
            activation_function (str): Activation function to use: linear, sigmoid, yanh, RELU
        """
        pass

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
