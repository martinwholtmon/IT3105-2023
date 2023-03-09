import torch


class ANET:
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
            n_input (int): Number of input params
            n_hidden_layers (int): Number of hidden layers
            n_output (int): Number of output params
            n_units (int): Number of units (neurons) per layer
            activation_function (str): Activation function to use: linear, sigmoid, yanh, RELU
        """
        pass

    def get_action(self, state) -> any:
        raise NotImplementedError

    def update(self):
        raise NotImplementedError
