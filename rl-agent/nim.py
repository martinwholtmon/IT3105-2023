"""Define the game state for Nim
"""
import random
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
        self.initial_pieces = N
        self.max_remove_pieces = K
        self.current_state = self.initial_pieces

    def get_state(self):
        """Return the current game state"""
        return self.current_state

    def perform_action(self, action):
        """Perform an action in the state"""
        self.current_state -= action

    def sample(self, player) -> any:
        """Return a random legal action for the player

        Args:
            player (int): The current player
        """
        random.choice(self.get_legal_actions(player))

    def get_legal_actions(self, player) -> list[any]:
        """Generate a list of legal actions for the current player

        Returns:
            list[any]: List of legal actions
        """
        return [
            i for i in range(1, min(self.max_remove_pieces, self.current_state) + 1)
        ]

    def is_terminated(self) -> bool:
        """Check if the game is finished

        Returns:
            bool: Game is finished
        """
        return self.current_state == 0

    def get_reward(self, player) -> float:
        """Get the reward

        Returns:
            float: the reward
        """
        if self.is_terminated():
            return 1
        return 0

    def reset(self, seed):
        """Resets the game"""
        self.current_state = self.initial_pieces
