"""This is the behavior policy: the actor network.

This network is always in the perspective of player 1, 
meaning that for player 2 we must minimize instead of maximize"""
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

        # input layer + 1 to account for player info
        modules.append(nn.Linear(np.multiply.reduce(input_shape) + 1, hidden_layers[0]))

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
        input = np_to_tensor(state.current_state, state.current_player)
        pred = tensor_to_np(self.forward(input))
        return scale_prediction(
            pred,
            state.legal_actions,
            state.actions,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do a forward pass in the network and return predictions.

        Args:
            x (torch.Tensor): game state

        Returns:
            torch.Tensor: predicted values
        """
        return self.layers(x)

    def train(self, batch: list[tuple[State, np.ndarray]]):
        """Sample the replay buffer and do updates (gradient decent)

        Args:
            batch (list[tuple[np.ndarray, np.ndarray]]): List of replay buffers
        """
        # Get target values for samples
        input: "list[np.ndarray]" = []
        for entry in batch:
            input.append(
                transform_state(entry[0].current_state, entry[0].current_player)
            )

        # Extract input values and taget values from batch
        target = [
            entry[1] for entry in batch
        ]  # TODO: Probably do some q value stuff here?

        # Convert to tensors
        input = torch.FloatTensor(np.array(input))
        target = torch.FloatTensor(np.array(target))

        # Do forward pass
        input = self.forward(input)

        # Calc loss
        loss = self.loss_function(input, target)
        print(f"loss: {loss.item()}")

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def transform_state(state: np.ndarray, player: int) -> np.ndarray:
    """This method will transform the state space, and include the current player

    Args:
        state (np.ndarray): Current state space
        player (int): playerid (PID)

    Returns:
        np.ndarray: the transformed state
    """
    state = state.flatten()

    # Add player info
    if player == 1:
        player_arr = np.zeros((1))
    else:
        player_arr = np.ones((1))
    return np.concatenate((state, player_arr))


def np_to_tensor(state: np.ndarray, player: int) -> torch.Tensor:
    """Converts a state to a tensor.
    Flatten the tensor - in case of ndarray of 2+d
    Convert it to floats: PyTorch parameters expect float32 input

    Args:
        state (np.ndarray): game state

    Returns:
        torch.Tensor: game state as a tensor
    """
    return torch.from_numpy(transform_state(state, player)).float()


def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a numpy representation

    Args:
        tensor (torch.Tensor): Action probabilities from the network

    Returns:
        np.ndarray: action probabilities
    """
    return tensor.detach().numpy()


def scale_prediction(
    x: np.ndarray,
    legal_actions: np.ndarray,
    all_actions: np.ndarray,
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
        all_actions (np.ndarray): All actions

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
