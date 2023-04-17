"""This is the behavior policy: the actor network.

This network is always in the perspective of player 1, 
meaning that for player 2 we must minimize instead of maximize"""
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from state_manager import State
from rbuf import RBUF, Replay


class ANET(nn.Module):
    """Class for the neural network"""

    def __init__(
        self,
        input_shape: "list[int]",
        output_lenght: int,
        hidden_layers: list[int] = [128, 128, 128],
        activation_function: str = "relu",
        learning_rate: float = 0.001,
        batch_size: int = 32,
        discount_factor: int = 1,  # assumed to be 1
        gradient_steps: int = 1,
        max_grad_norm: float = 10,
        device: Union[torch.device, str] = "auto",
        save_replays: bool = False,
    ) -> None:
        """Init the neural network.

        Args:
            input_shape (list[int]): Shape of input params/features
            output_lenght (int):  Max output params (largest action space)
            hidden_layers (list[int], optional): Number of units (neurons) per layer. Defaults to [128, 128, 128].
            activation_function (str, optional): Activation function to use: linear, sigmoid, tanh, relu. Defaults to "relu".
            learning_rate (float, optional): The rate of which the network will learn (0,1]. Defaults to 0.001.
            batch_size (int, optional): Minibatch size for each gradient update. Defaults to 32.
            discount_factor (int, optional): Reward importance [0,1]. Defaults to 1.
            max_grad_norm (float, optional): Max value for gradient clipping. Defaults to 10.
            device (Union[torch.device, str], optional): Device the network should use. Defaults to "auto" meaning that it will use GPU if available.
            save_replays (bool): If we want to resume training at a later date, this must be set to True. Defaults to False.
        """
        # Inherit from nn.Module
        super(ANET, self).__init__()

        # Set params
        self.activation_function = set_activation_class(activation_function)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.gradient_steps = gradient_steps
        self.max_grad_norm = max_grad_norm
        self.device = get_device(device)
        self.save_replays = save_replays
        self.replays: RBUF = RBUF()

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

        # Set device
        self.to(self.device)

        # Set training mode
        self.set_training_mode(True)

    def predict(self, state: State) -> np.ndarray:
        """Given a state, return the action probabilities for all actions in the game

        Args:
            state (State): game state

        Returns:
            int: index of the action to perform
        """
        input = np_to_tensor(state.current_state, state.current_player).to(self.device)
        output = tensor_to_np(self.forward(input))
        scaled_output = scale_prediction(
            output,
            state.legal_actions,
        )
        return np.argmax(scaled_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do a forward pass in the network and return predictions.

        Args:
            x (torch.Tensor): game state

        Returns:
            torch.Tensor: predicted values
        """
        return self.layers(x)

    def update(self):
        """Sample the replay buffer and do updates"""
        self.set_training_mode(True)

        # Get batch
        samples: "list[Replay]" = self.replays.sample(self.batch_size)

        # Prepare holders
        input: "list[np.ndarray]" = []
        target: "list[np.ndarray]" = []

        # Iterate over the samples
        for sample in samples:
            # Get states
            input.append(transform_state(sample.state, sample.player))

            # Get target values
            target.append(sample.target_value * self.discount_factor)

        # Convert to tensors
        input = torch.FloatTensor(np.array(input)).to(self.device)
        target = torch.FloatTensor(np.array(target)).to(self.device)

        # Do forward pass
        input = self.forward(input)

        # Calc loss
        loss = self.loss_function(input, target)
        print(f"loss: {loss.item()}")

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def load(self, filepath: str, continue_training=False):
        """Tries to load a saved model

        Args:
            name (str): name of the model
        """
        # Load checkpoint
        print(f"Loading {filepath}")
        checkpoint = torch.load(filepath)

        # Update model
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.replays = checkpoint["replays"]

        # Handle training.
        if continue_training is True and self.save_replays is not True:
            raise ValueError(
                f"You cannot continue training a model that did not save replays!"
            )

        if continue_training:
            self.set_training_mode(True)
        else:
            self.set_training_mode(False)

    def save(self, filepath: str):
        """Saves the current model

        Args:
            name (str): name of the model
        """
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "replays": self.replays,
            },
            filepath,
        )

    def add_replay(self, state: State, action_probabilities: np.ndarray):
        """Add a replay to the replay buffer

        Args:
            state (State): Current state
            action_probabilities (np.ndarray): target values
        """
        self.replays.add(state, action_probabilities)

    def set_training_mode(self, mode: bool):
        """Set the model to either training mode or evaluation mode

        Args:
            mode (bool): If True, set it to training mode. Else set it to evaluation mode.
        """
        self.train(mode)


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
    return tensor.cpu().data.numpy()


def scale_prediction(
    x: np.ndarray,
    legal_actions: np.ndarray,
) -> np.ndarray:
    """The output layer must have a fixed number of outputs
    even though there might not be so many avaiable actions.
    From any given state, there is a reduced number of legal actions.
    This function will try to scale the distribution to only represent legal actions.

    Illegal moves will be set to zero.
    Remaining moves will be re-nomralized to a sum of 1.

    Args:
        x (np.ndarray): Probability distribution for all possible actions
        legal_actions (np.ndarray): mask of legal actions (binary array)

    Returns:
        np.ndarray:  Probability distribution for the legal actions
    """
    x = np.multiply(x, legal_actions)

    # Normalize
    x_sum = np.sum(x)
    if x_sum != 0:
        # in the few cases where the model gives zero chance of winning to legal actions, return the mask
        x = x / x_sum
    else:
        x = legal_actions
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


def get_device(device: Union[torch.device, str]) -> torch.device:
    """Get the pytorch device

    Args:
        device (Union[torch.device, str]): A pytorch device, or "auto", "cuda", "cpu"

    Returns:
        torch.device: Device to use
    """

    # Auto, set cuda
    if device == "auto":
        device = "cuda"

    # Set device
    if not torch.cuda.is_available() or device == "cpu":
        return torch.device("cpu")
    return torch.device(device)
