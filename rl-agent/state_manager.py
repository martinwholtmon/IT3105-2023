"""This module will handle all the interaction with the games
It is essentially the gym environment for the games
"""
from abc import ABC, abstractmethod
import random
import numpy as np


class State(ABC):
    """Abstract class for the game state

    Args:
        ABC (ABC): Helper class that provides a standard way to create an ABC using inheritance
    """

    @abstractmethod
    def __init__(self) -> None:
        self.current_state: np.ndarray = None
        self.current_player = 1
        self.n_players = 2
        self.current_state: np.ndarray = None
        self.actions: "list[any]" = []
        self.legal_actions: "list[any]" = []

    @abstractmethod
    def perform_action(self, action):
        """Perform an action in the state"""

    @abstractmethod
    def is_terminated(self) -> bool:
        """Check if the game is finished

        Returns:
            bool: Game is finished
        """

    @abstractmethod
    def clone(self):
        """Clone/dereference the game state"""

    @abstractmethod
    def _create_init_state(self):
        """Creates the initial state."""

    @abstractmethod
    def _generate_actions(self) -> list[any]:
        """Generates the legal actions for the current state"""

    @abstractmethod
    def _update_legal_actions(self, action=None):
        """Updates the legal actions"""

    def sample(self) -> any:
        """Return a random legal action"""
        return random.choice(self.legal_actions)

    def reset(self, seed):
        """Resets the game"""
        self.current_player = 1
        self._create_init_state()
        self.legal_actions = self.actions.copy()

    def next_state(self, action):
        """Clones the current game state, and returns the next game state"""
        next_state = self.clone()
        next_state.perform_action(action)
        return next_state

    def get_reward(self) -> float:
        """Get the reward

        Returns:
            float: the reward
        """
        if self.is_terminated():
            if self.current_player == 1:
                return 1
            return -1
        return 0

    def _next_player(self):
        """Change the current_player in the state to the next player"""
        if not self.is_terminated():
            self.current_player = (self.current_player % self.n_players) + 1


class Env:
    """The game environment where players can perform steps"""

    def __init__(self, state: State) -> None:
        self.state: State = state

    def step(self, action) -> tuple[State, float, bool]:
        """Perform a step in the game

        Args:
            action (any): Action to perform

        Returns:
            tuple[State, float, bool]: new_state, reward, is_terminated
        """
        self.state.perform_action(action)
        reward = self.state.get_reward()
        return self.state, reward, self.state.is_terminated()

    def reset(self, seed: int = None) -> State:
        """Reset the game to an initial game state. If nessesary, introduces randomness.

        Args:
            seed (int, optional): If you want consistent randomness. Defaults to None.

        Returns:
            State: The new state
        """
        self.state.reset(seed)
        return self.state
