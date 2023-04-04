"""Define the game state for Nim
"""
import random
import math
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

    def __init__(self, N: int, K: int, heaps: int, deterministic: bool = True) -> None:
        """Initialize the game

        Args:
            N (int): Number of initial pieces
            K (int): Maximum number of pieces that a player is allowed to remove at once
            heaps (int): Number of heaps
            deterministic (bool): how the stones are split among the heap
        """
        super().__init__()
        self.initial_pieces = N
        self.max_remove_pieces = K
        self.heaps = heaps
        self.deterministic = deterministic

        # Update state and action space
        self._create_init_state()
        self.actions = self._generate_actions()
        self._update_legal_actions()
        self._reset_legal_actions()

    def perform_action(self, action):
        """Perform an action in the state"""
        # Get heap and stones to remove
        heap, stones = action

        # Update state
        self.current_state[heap] -= stones
        self._next_player()
        self._update_legal_actions()

    def sample(self) -> any:
        """Return a random legal action"""
        return random.choice(self.legal_actions)

    def is_terminated(self) -> bool:
        """Check if the game is finished

        Returns:
            bool: Game is finished
        """
        return np.sum(self.current_state) == 0

    def reset(self, seed):
        """Resets the game"""
        self.current_player = 1
        self._create_init_state()
        self._reset_legal_actions()

    def clone(self):
        """Clone/dereference the game state"""
        new_state = Nim(self.initial_pieces, self.max_remove_pieces, self.heaps)
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

    def _create_init_state(self):
        """Creates the initial state.
        [
            [..],
            [..],
            [..],
        ]
        """
        stones_left = self.initial_pieces
        state = []

        # Devide the stones deterministic
        if self.deterministic:
            half_point = math.floor(stones_left / self.heaps)
            difference = math.floor(self.heaps / 2)

            # Populate the heaps, but not the last one
            for _ in range(self.heaps - 1):
                stones = half_point + difference
                state.append(stones)
                stones_left -= stones
                difference -= 1

            # Fill the last heap with remaining stones
            state.append(stones_left)
        else:
            heap_id = 0
            # while stones_left > 0:
            #     stones =
            #     state
            #     stones_left -= stones

        self.current_state = np.array(state)

    def _generate_actions(self) -> list[any]:
        """Generates the actions for the current state"""
        actions = []
        for i, heap in enumerate(self.current_state):
            if heap > 0:
                heap_actions = list(range(1, min(self.max_remove_pieces, heap) + 1))
                for action in heap_actions:
                    actions.append((i, action))
        return actions

    def _update_legal_actions(self):
        """Updates the legal actions dependent on values in the heaps"""
        self.legal_actions = self._generate_actions()

    def _reset_legal_actions(self):
        """Resets the list of legal actions"""
        self.legal_actions = self._generate_actions()
