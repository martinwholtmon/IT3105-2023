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
        self.actions = list(
            range(1, min(self.max_remove_pieces, self.initial_pieces) + 1)
        )
        self.update_legal_actions()

    def perform_action(self, action):
        """Perform an action in the state"""
        self.current_state[0] -= action
        self.next_player()
        self.update_legal_actions()

    def sample(self) -> any:
        """Return a random legal action"""
        return random.choice(self.legal_actions)

    def update_legal_actions(self) -> list[any]:
        """Updates the list of legal actions"""
        self.legal_actions = [
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
        self.update_legal_actions()

    def clone(self):
        """Clone/dereference the game state"""
        new_state = Nim(self.initial_pieces, self.max_remove_pieces)
        new_state.current_state = self.current_state.copy()  # Dereference
        new_state.actions = self.actions  # Reference the list as its constant
        new_state.legal_actions = self.legal_actions.copy()
        new_state.current_player = self.current_player
        return new_state

    def next_state(self, action):
        """Clones the current game state, and returns the next game state"""
        next_state = self.clone()
        next_state.perform_action(action)
        return next_state
