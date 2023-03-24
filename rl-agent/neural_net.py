"""Create the neural network and handles all the interactions with it."""
import numpy as np
import torch
import torch.nn as nn
from helpers import is_sequence_of_type
from state_manager import State


class ANET(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int],
        hidden_layers: list[int],
        output_shape: tuple[int, int],
        activation_function: str,
    ) -> None:
        """Init the neural network.

        Args:
            input_shape (tuple[int, int]): Shape of input params/features
            hidden_layers (list[int]): Number of units (neurons) per layer
            output_shape (tuple[int, int]): Shape of max output params (largest action space)
            activation_function (str): Activation function to use: linear, sigmoid, tanh, relu
        """
        # Inherit from nn.Module
        super(ANET, self).__init__()

        # Check params
        is_sequence_of_type("input_shape", input_shape, tuple, int, min=1, max=2)
        is_sequence_of_type("hidden_units", hidden_layers, list, int, min=1)
        is_sequence_of_type("output_shape", output_shape, tuple, int, min=1, max=2)

        # Set params
        self.activation_function = set_activation_class(activation_function)

        # Add layers
        modules = []

        # input layer
        modules.append(nn.Linear(np.multiply.reduce(input_shape), hidden_layers[0]))

        # hidden layers
        if len(hidden_layers) == 1:
            modules.append(nn.Linear(hidden_layers[0], hidden_layers[0]))
            if self.activation_function is not None:
                modules.append(self.activation_function)
        else:
            for i, j in zip(hidden_layers, hidden_layers[1:]):
                modules.append(nn.Linear(i, j))
                if self.activation_function is not None:
                    modules.append(self.activation_function)

        # output layer
        modules.append(nn.Linear(hidden_layers[-1], np.multiply.reduce(output_shape)))
        modules.append(nn.Softmax())

        # Define the model
        self.layers = nn.Sequential(*modules)

        # Define the loss function
        self.loss_function = nn.CrossEntropyLoss()

    def predict(self, state: State) -> any:
        prediction = self.forward(state.get_state())
        return scale_prediction(
            prediction, state.get_legal_actions(), state.get_all_actions()
        )

    def forward(self, x: np.ndarray):
        """Do a forward pass in the network to predict a state

        Args:
            x (np.ndarray): game state

        Returns:
            np.ndarray: predicted values as a 1d np.ndarray
        """
        return tensor_to_np(self.layers(np_to_tensor(x)))

    def update(self, batch):
        raise NotImplementedError


def np_to_tensor(x: State) -> torch.Tensor:
    """Converts a state to a tensor.
    Flatten the tensor - in case of ndarray of 2+d
    Convert it to floats: PyTorch parameters expect float32 input

    Args:
        state (np.ndarray): game state

    Returns:
        torch.Tensor: game state as a tensor
    """
    return torch.from_numpy(x).flatten().float()


def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a numpy representation

    Args:
        tensor (torch.Tensor): _description_

    Returns:
        np.ndarray: _description_
    """
    return tensor.detach().numpy()


def scale_prediction(
    x: np.ndarray, legal_actions: np.ndarray, all_actions: np.ndarray
) -> np.ndarray:
    """The output layer must have a fixed number of outputs
    even though there might not be so many avaiable actions.
    From any given state, there is a reduced number of legal actions.
    This function will try to scale the distribution to only represent legal actions.

    Illegal moves will be set to zero.
    Remaining moves will be re-nomralized to a sum of 1.

    Args:
        x (np.ndarray): Probability distribution for all possible actions
        legal_actions (np.ndarray): list of legal actions

    Returns:
        np.ndarray:  Probability distribution for the legal actions
    """
    # Set predictions of illegal actions to 0
    illegal_actions = np.isin(all_actions, legal_actions).astype(int)  # mask of 0 and 1
    x = np.multiply(x, illegal_actions)

    # Normalize
    x = x / np.linalg.norm(x)
    return x


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
