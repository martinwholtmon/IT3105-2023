"""This module will handle all the interaction with the games
It is essentially the gym environment for the games
"""
from abc import ABC, abstractmethod
import numpy as np


class State(ABC):
    """Abstract class for the game state

    Args:
        ABC (ABC): Helper class that provides a standard way to create an ABC using inheritance
    """

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return the current game state"""

    @abstractmethod
    def perform_action(self, action):
        """Perform an action in the state"""

    @abstractmethod
    def sample(self) -> any:
        """Return a random legal action"""

    @abstractmethod
    def get_legal_actions(self) -> list[any]:
        """Generate a list of legal actions for the current player

        Returns:
            list[any]: List of legal actions
        """

    @abstractmethod
    def get_all_actions(self) -> list[any]:
        """Return the list of all actions

        Returns:
            list[any]: List of actions
        """

    @abstractmethod
    def is_terminated(self) -> bool:
        """Check if the game is finished

        Returns:
            bool: Game is finished
        """

    @abstractmethod
    def get_reward(self, player) -> float:
        """Get the reward

        Returns:
            float: the reward
        """

    @abstractmethod
    def reset(self, seed):
        """Resets the game"""


class Env:
    """The game environment where players can perform steps"""

    def __init__(self, state: State) -> None:
        self.state: State = state
        self.current_player = 1
        self.n_players = 2

    def step(self, action) -> tuple[State, float, bool]:
        """Perform a step in the game

        Args:
            action (any): Action to perform

        Returns:
            tuple[State, float, bool]: new_state, reward, is_terminated
        """
        self.state.perform_action(action)
        reward = self.state.get_reward(self.current_player)

        # Update player and return
        self.current_player = (self.current_player % self.n_players) + 1
        return self.state, reward, self.state.is_terminated()

    def reset(self, seed: int = None) -> State:
        """Reset the game to an initial game state. If nessesary, introduces randomness.

        Args:
            seed (int, optional): If you want consistent randomness. Defaults to None.

        Returns:
            State: The new state
        """
        self.current_player = 1
        self.state.reset(seed)
        return self.state
