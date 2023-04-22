"""Define the game state for Hex
"""
from __future__ import annotations
import copy
import numpy as np
from rl_agent.state_manager import State


class _Cell:
    def __init__(
        self, pos: int, owner: int = 0, neighbors: list[tuple[int, int]] = None
    ) -> None:
        self.pos = pos
        self.owner = owner
        if neighbors is None:
            self.neighbors: list[tuple[int, int]] = []
        else:
            self.neighbors = neighbors

    def clone(self) -> _Cell:
        """Clone a cell

        Returns:
            _Cell: Cloned cell
        """
        new_cell = copy.copy(self)
        new_cell.neighbors = self.neighbors.copy()
        return new_cell


class Hex(State):
    """The game of Hex:
    The game of hex is a board of diamond shape where each player own the oposite side
    of the diamond (top sides), where the goal is to connect to their diagonal side first
    (connected chain of pieces between the two sides).
        Player 1: North-west -> south-east
        Player 2: North-east -> south-west

    Legal moves are all the positions that are available. Once a pieces has been placed, it cannot be removed.


    Representation:
        This game will have two representations:
            current_state: 1d array with ints (0-2) dependent on game state and ownership.
            This state is fed into the neural network.
            game_state: This will be a 2d array containing objects (_Cell)
            for each position on the board. Includes neighbor relationships etc..
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_state: list[list[_Cell]] = []
        self.size = size
        self.player_one_goal = [i * self.size for i in range(self.size)]
        self.player_two_goal = list(range(self.size))

        # Update state and action space
        self._create_init_state()
        self.actions = self._generate_actions()
        self.legal_actions = np.ones(len(self.actions), dtype=np.int8)

    def perform_action(self, action: int):
        """Perform an action in the state"""
        y, x = self.actions[action]

        # Update cell
        cell: _Cell = self.game_state[y][x]
        cell.owner = self.current_player

        # Update simple board
        self.current_state[cell.pos] = self.current_player

        # check if terminated
        self.terminated = self.is_terminated()

        # Update legal actions
        self._update_legal_actions(action)
        self._next_player()

    def is_terminated(self) -> bool:
        """Check if the game is finished.

        When reached diagonal side, check the ownership towards ownerships side.
        If a path is found, return true.

        Returns:
            bool: Game is finished
        """
        # Check states for p1 (top -> bottom)
        for point in [row[-1] for row in self.game_state]:
            if point.owner == 1:
                if complete_path(
                    self.game_state,
                    point,
                    player=1,
                    goal_pos=self.player_one_goal,
                    visited=set(),
                ):
                    return True

        # Check states for p2 (left -> right)
        for point in self.game_state[-1]:
            if point.owner == 2:
                if complete_path(
                    self.game_state,
                    point,
                    player=2,
                    goal_pos=self.player_two_goal,
                    visited=set(),
                ):
                    return True
        return False

    def clone(self):
        """Clone/dereference the game state"""
        # Create shallow copy
        new_state = copy.copy(self)

        # Create the game space
        new_state.game_state = []
        for y in range(self.size):
            new_state.game_state.append([])
            for cell in self.game_state[y]:
                new_state.game_state[y].append(cell.clone())

        # Update attributes that needs dereferencing
        # (int are immutable, lists/numpy objects are not)
        new_state.current_state = self.current_state.copy()
        new_state.legal_actions = self.legal_actions.copy()
        return new_state

    def render(self):
        """Render the game of Hex as a Diamond"""
        for y in self.game_state:
            print(" ".join(str(c.owner) for c in y))
        print()

        # Top rectangle + middle
        for i in range(self.size):
            vals = [str(self.game_state[i - j][j].owner) for j in range(i + 1)]
            print(" " * (self.size - i - 1) + " ".join(vals))

        # Bottom rectangle
        for i in range(self.size - 2, -1, -1):
            vals = [
                str(self.game_state[self.size - 1 - j][self.size - 1 - i + j].owner)
                for j in range(i + 1)
            ]
            print(" " * (self.size - i - 1) + " ".join(vals))

    def _create_init_state(self):
        """Creates the initial state.

        The game state will be represented in a 2d array in a square form
        even though the game has a diamond shape. In the cases where we want
        to print/display the board, we will just rotate the it 45 degrees to the right.

        So, the first player owns top and bottom, while player two own left and right.

        Meaning that:
            - top left of the board will be the top position
            - bottom right will be the bottom
            - top right will be right
            - bottom left will left
        """
        # Create the simplified state
        self.current_state = np.zeros((self.size**2,), dtype=int)

        # Generate all cells and add to game state
        index = 0
        self.game_state.clear()
        for y in range(self.size):
            self.game_state.append([])
            for x in range(self.size):
                self.game_state[y].append(_Cell(pos=index))
                index += 1

        # Assign neighbors for each cell
        for y in range(self.size):
            for x in range(self.size):
                cell = self.game_state[y][x]
                neighboring_points = find_neighboring_points(
                    x=x, y=y, max=self.size - 1
                )
                for px, py in neighboring_points:
                    cell.neighbors.append((py, px))

    def _generate_actions(self) -> list[any]:
        """Generates the legal actions for the initial state"""
        actions = []
        for y in range(self.size):
            for x in range(self.size):
                actions.append((y, x))
        return actions

    def _update_legal_actions(self, action):
        """Updates the legal actions dependent on values in the heaps"""
        self.legal_actions[action] = 0


def find_neighboring_points(x: int, y: int, max: int) -> list[tuple[int, int]]:
    """Given a point, it will find evert neighboring point

    Args:
        x (int): x-axis
        y (int): y-axis
        max (int): y-axis

    Returns:
        list[tuple[int, int]]: tuples of neighboring points
    """
    points = []

    # Up
    if y > 0:
        points.append((x, y - 1))
        # up right
        if x < max:
            points.append((x + 1, y - 1))

    # left
    if x > 0:
        points.append((x - 1, y))

    # right
    if x < max:
        points.append((x + 1, y))

    # down
    if y < max:
        points.append((x, y + 1))
        # down left
        if x > 0:
            points.append((x - 1, y + 1))
    return points


def complete_path(
    game_state: list[list[_Cell]],
    cell: _Cell,
    player: int,
    goal_pos: list[int],
    visited: set[int],
) -> bool:
    """This method paths towards the start position to see if we have a complete path.
    To make sure we are not looping, we cannot visit a node/cell twice

    Args:
        game_state (list[list[_Cell]]): The current game state
        cell (_Cell): the current cell
        player (int): Player id.
            Player 1: Left -> right
            Player 2: Top -> bottom
    Returns:
        bool: Has complete path
    """
    # Reached base case
    if cell.pos in goal_pos:
        return True

    # Explore
    visted_list = visited.copy()
    for neighbor in cell.neighbors:
        py, px = neighbor
        cell = game_state[py][px]
        if cell.pos not in visted_list and cell.owner == player:
            visted_list.add(cell.pos)
            if complete_path(game_state, cell, player, goal_pos, visted_list):
                return True

    # Did not find path
    return False
