"""Define the game state for Nim
"""
import random
import numpy as np
from state_manager import State


class Nim(State):
    """Game of Nim:
    NIM is actually a variety of games involving pieces (or stones) on a nondescript board
    that players alternatively remove, with the player removing the last piece being the winner,
    or loser, depending on the game variant. Typical constraints of the game include a minimum
    and maximum number of pieces that can be removed at any one time, a partitioning of the pieces
    into heaps such that players must extract from a single heap on a given turn, and boards with
    distinct geometry that affects where pieces reside, how they can be grouped during removal, etc.

    To keep it simple, we do not care about grouping,
    just assume that all pieces are grouped together.
    """

    def __init__(self, N: int, K: int) -> None:
        """Initialize the game

        Args:
            N (int): Number of initial pieces
            K (int): Maximum number of pieces that a player is allowed to remove at once
        """
        super().__init__()
        self.initial_pieces = N
        self.max_remove_pieces = K
        self.current_state = np.array([self.initial_pieces])
        self.actions = [
            i for i in range(1, min(self.max_remove_pieces, self.initial_pieces) + 1)
        ]

    def perform_action(self, action):
        """Perform an action in the state"""
        self.current_state[0] -= action
        self.next_player()

    def sample(self) -> any:
        """Return a random legal action"""
        return random.choice(self.get_legal_actions())

    def get_legal_actions(self) -> list[any]:
        """Generate a list of legal actions

        Returns:
            list[any]: List of legal actions
        """
        return [
            i for i in range(1, min(self.max_remove_pieces, self.current_state[0]) + 1)
        ]

    def is_terminated(self) -> bool:
        """Check if the game is finished

        Returns:
            bool: Game is finished
        """
        return self.current_state[0] == 0

    def reset(self, seed):
        """Resets the game"""
        self.current_state[0] = self.initial_pieces
        self.current_player = 1
