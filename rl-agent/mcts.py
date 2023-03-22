"""This module will contain the MCTS algorithm"""
from __future__ import annotations
import copy
from state_manager import State
from neural_net import ANET


class Node:
    """A node in the MCTS"""

    def __init__(self, state: State, parent: Node = None, action: any = None):
        self.state = state  # Current game state
        self.parent = parent  # Previous game state
        self.action = action
        self.children: "list[Node]" = []  # Child states by performing actions
        self.untried_actions: list = (
            state.get_legal_actions()
        )  # avaiable actions in the current game state
        self.value = 0  # Predicts who will win (wins)
        self.num_visits = 0  # Might be used to keep exploration high

    def is_leaf(self) -> bool:
        """Check if node is leaf node

        Returns:
            bool: If leaf node
        """
        return len(self.children) == 0

    def fully_expanded(self) -> bool:
        """Check if the node has expanded all its options/actions

        Returns:
            bool: No more actions left to explore
        """
        return len(self.untried_actions) == 0

    def select_child(self, exploration_factor) -> Node:
        """Use the tree policy the select a child node (next game state)

        Args:
            exploration_factor (float): How much the exploration we want

        Returns:
            Node: The child node
        """
        return NotImplementedError
    def expand(self) -> Node:
        """Expand child states

        Returns:
            Node: Child node for the next game state
        """
        action = self.untried_actions.pop()
        next_state = copy.deepcopy(self.state).perform_action(action)
        child_node = Node(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        """Update the value and visit

        Args:
            reward (int): the reward to add
        """
        self.value += reward
        self.num_visits += 1
