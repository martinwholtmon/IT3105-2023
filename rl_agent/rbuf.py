"""Module for the replay buffer"""
import random
from dataclasses import dataclass
import numpy as np
from rl_agent.state_manager import State


@dataclass
class Replay:
    """Class for a replay in the RBUF"""

    state: np.ndarray
    player: int
    target_value: np.ndarray


class RBUF:
    def __init__(self) -> None:
        self.replays: "list[Replay]" = []

    def add(self, state: State, action_probabilities: np.ndarray):
        """Add a replay to the replay buffer

        Args:
            state (State): a game state
            action_probabilities (np.ndarray): action probabilities
        """
        self.replays.append(
            Replay(state.current_state, state.current_player, action_probabilities)
        )

    def sample(self, size) -> "list[Replay]":
        """Sample a random amount of samples from the replay buffer

        Args:
            size (int): number of samples

        Returns:
            list[Replay]: list of samples
        """
        if size >= len(self.replays):
            return self.replays
        return random.sample(self.replays, size)

    def clear(self):
        """Clear the replay buffer"""
        self.replays.clear()
