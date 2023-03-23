"""Create the neural network and handles all the interactions with it."""
import numpy as np
import torch
import torch.nn as nn
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
            activation_function (str): Activation function to use: linear, sigmoid, tanh, RELU
        """
        # Inherit from nn.Module
        super(ANET, self).__init__()

        # Check params
        is_sequence_of_type("input_shape", input_shape, tuple, int, min=1, max=2)
        is_sequence_of_type("hidden_units", hidden_units, list, int, min=1)
        is_sequence_of_type("output_shape", output_shape, tuple, int, min=1, max=2)

        # Set params
        self.activation_function = set_activation_class(activation_function)

        # Add layers
        modules = []

        # input layer
        modules.append(nn.Linear(np.multiply.reduce(input_shape), hidden_units[0]))

        # hidden layers
        if len(hidden_units) == 1:
            modules.append(nn.Linear(hidden_units[0], hidden_units[0]))
            if self.activation_function is not None:
                modules.append(self.activation_function)
        else:
            for i, j in zip(hidden_units, hidden_units[1:]):
                modules.append(nn.Linear(i, j))
                if self.activation_function is not None:
                    modules.append(self.activation_function)

        # output layer
        modules.append(nn.Linear(hidden_units[-1], np.multiply.reduce(output_shape)))
        modules.append(nn.Softmax())

        # Define the model
        self.layers = nn.Sequential(*modules)

    def predict(self, state) -> any:
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


def state_to_tensor(state) -> torch.Tensor:
    """Converts a state to a tensor.
    Flatten the tensor - in case of ndarray of 2+d
    Convert it to floats: PyTorch parameters expect float32 input

    Returns:
        torch.Tensor: game state as a tensor
    """
    return torch.from_numpy(state).flatten().float()


def scale_output(x) -> np.ndarray:
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


def set_activation_class(activation_function: str) -> callable:
    """Set the activation function for the layers: linear, sigmoid, tanh, or RELU

    Args:
        activation_function (str): The activation function

    Returns:
        callable: The functional interface for the selected activation function
    """
    match activation_function.lower():
        case "linear":
            return None
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()
        case "RELU":
            return nn.ReLU()
        case _:
            raise ValueError(
                f"{activation_function} is not supported, please use a supported activation function: linear, sigmoid, tanh, or RELU"
            )
