import random


class Nim:
    """Game of Nim:
    NIM is actually a variety of games involving pieces (or stones) on a nondescript board that players alternatively remove,
    with the player removing the last piece being the winner, or loser, depending on the game variant.
    Typical constraints of the game include a minimum and maximum number of pieces that can be removed at any one time,
    a partitioning of the pieces into heaps such that players must extract from a single heap on a given turn,
    and boards with distinct geometry that affects where pieces reside, how they can be grouped during removal, etc.

    To keep it simple, we do not care about grouping, just assume that all pieces are grouped togehter.
    """

    def __init__(self, N: int, K: int) -> None:
        """Initialize the game

        Args:
            N (int): Number of initial pieces
            K (int): Maximum number of pieces that a player is allowed to remove at once
        """
        self.initial_pieces = N
        self.max_remove_pieces = K
        self.n_players = 2
        self.current_player = 1
        self.current_state = self.initial_pieces

    def step(self, action) -> tuple[int, int, bool]:
        """Perform a step in the game

        Returns:
            tuple[int, int, bool]: current_state, reward, is_terminated
        """
        # Perform the action
        self.current_state -= action

        # Check for reward
        reward = self.get_reward()

        # Update current player
        self.current_player = (self.current_player % self.n_players) + 1

        # return
        return (
            self.current_state,
            reward,
            self.is_terminated(),
        )

    def sample(self) -> int:
        """Return a random legal action

        Returns:
            int: action to perform
        """
        return random.choice(self.get_legal_actions())

    def get_legal_actions(self) -> list[int]:
        """Generate a list of legal actions. For this game, it would be a list of possible pieces to remove

        Returns:
            list[int]: Legal actions
        """
        return [
            i for i in range(1, min(self.max_remove_pieces, self.current_state) + 1)
        ]

    def is_terminated(self) -> bool:
        """If the game is finished

        Returns:
            bool: Game is finished
        """
        return self.current_state == 0

    def get_reward(self) -> int:
        """Get the reward

        Returns:
            int: Reward
        """
        if self.is_terminated():
            return 1
        return 0
