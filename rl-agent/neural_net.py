"""Create the neural network and handles all the interactions with it."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from helpers import is_sequence_of_type, is_int
from state_manager import State


class ANET(nn.Module):
    """Class for the neural network"""

    def __init__(
        self,
        input_shape: tuple[int, int],
        hidden_layers: list[int],
        output_lenght: int,
        activation_function: str,
        learning_rate: float,
    ) -> None:
        """Init the neural network.

        Args:
            input_shape (tuple[int, int]): Shape of input params/features
            hidden_layers (list[int]): Number of units (neurons) per layer
            output_lenght (int): Max output params (largest action space)
            activation_function (str): Activation function to use: linear, sigmoid, tanh, relu
            learning_rate (float): The rate of which the network will learn
        """
        # Inherit from nn.Module
        super(ANET, self).__init__()

        # Check params
        is_sequence_of_type("input_shape", input_shape, tuple, int, min=1, max=2)
        is_sequence_of_type("hidden_units", hidden_layers, list, int, min=1)
        is_int("output_lenght", output_lenght, min=1)

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
        modules.append(nn.Linear(hidden_layers[-1], output_lenght))
        modules.append(nn.Softmax())

        # Define the model
        self.layers = nn.Sequential(*modules)

        # Define the loss function
        self.loss_function = nn.CrossEntropyLoss()

        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Init weights and biases
        self.apply(init_weights_uniform)

    def predict(self, state: State) -> np.ndarray:
        """Given a state, return the action probabilities for all actions in the game

        Args:
            state (State): game state

        Returns:
            np.ndarray: action probabilities
        """
        return scale_prediction(
            self.forward(state.current_state),
            state.legal_actions,
            state.actions,
        )

    def forward(self, x: np.ndarray):
        """Do a forward pass in the network and return predictions.
        2d games will be converted to 1d array

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
        tensor (torch.Tensor): Action probabilities from the network

    Returns:
        np.ndarray: action probabilities
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
    x = x / np.sum(x)
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
        case "relu":
            return nn.ReLU()
        case _:
            raise ValueError(
                f"{activation_function} is not supported, please use a supported activation function: linear, sigmoid, tanh, or RELU"
            )


def init_weights_uniform(model):
    """Takes in a model and applies unifor rule to initialize weights and biases
    Source: https://stackoverflow.com/a/55546528

    Args:
        model (nn.Module): pytorch model
    """
    if isinstance(model, nn.Linear):
        n = model.in_features
        y = 1.0 / np.sqrt(n)
        model.weight.data.uniform_(-y, y)
        model.bias.data.fill_(0)
