"""This module will contain the MCTS algorithm"""
from __future__ import annotations
import copy
import numpy as np
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
        # UCT formula
        if self.state.current_player == 1:  # player 1 -> Maximize
            scores = [
                (child.value / child.num_visits if child.num_visits else 0)
                + exploration_factor
                * np.sqrt(np.log(self.num_visits) / (1 + child.num_visits))
                for child in self.children
            ]
            return self.children[np.argmax(scores)]
        else:  # player 2 -> Minimize
            scores = [
                (child.value / child.num_visits if child.num_visits else 0)
                - exploration_factor
                * np.sqrt(np.log(self.num_visits) / (1 + child.num_visits))
                for child in self.children
            ]
            return self.children[np.argmin(scores)]

    def expand(self) -> Node:
        """Expand child states

        Returns:
            Node: Child node for the next game state
        """
        action = self.untried_actions.pop()
        next_state = self.state.next_state(action)
        child_node = Node(state=next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        """Update the value and visit

        Args:
            reward (int): the reward to add
        """
        self.value += reward
        self.num_visits += 1


def mcts(
    state: State, policy: ANET, simulations: int, exploration_factor: float
) -> tuple[any, np.ndarray]:
    """Run the MCTS algorithm with M simulations to select the best action to perform

    Args:
        state (State): The state space
        policy (ANET): The policy used to select actions
        simulations (int): Number of simulations
        exploration_factor (float):

    Returns:
        tuple[any, np.ndarray]:
            any: Action to perform
            np.ndarray: Action probabilities -> visit count normalized

    """
    # Take a deepcopy of the current state.
    # If it use a lot of memory, create own copy function for each game (implements State)
    root = Node(state)

    # Run MCTS
    for _ in range(simulations):
        node = root
        current_state = state.clone()

        # Select action
        node, current_state = tree_search(node, current_state, exploration_factor)

        # Expansion
        node, current_state = node_expansion(node, current_state)

        # Simulate action (expansion) and rollout
        reward = leaf_evaluation(current_state, policy)

        # Backpropagate results through tree
        backpropagation(node, reward)

    # Choose the best action -> action with most visits
    action_visits = [child.num_visits for child in root.children]
    action_visits = action_visits / np.sum(action_visits)  # Normalized
    return root.children[np.argmax(action_visits)].action, action_visits


def tree_search(node: Node, state: State, exploration_factor: float):
    """Traversing the tree from the root to a leaf node by using the tree policy

    Args:
        node (Node): Node in the MCTS tree
        state (State): Current game state
        exploration_factor (float): How explorative the selection will be

    Returns:
        tuple[Node, State]: leaf node, state that has action along the path applied
    """
    # Select best child and perform the action to current state
    while not node.is_leaf() and node.fully_expanded():
        node = node.select_child(exploration_factor)
        state.perform_action(node.action)
    return node, state


def node_expansion(node: Node, state: State):
    """Generating some or all child states of a parent state,
    and then connecting the tree node housing the parent state (a.k.a. parent node)
    to the nodes housing the child states (a.k.a. child nodes).

    Args:
        node (Node): Node in the MCTS tree
        state (State): Current game state

    Returns:
        tuple[Node, State]: new child node, state with action for child node performed
    """
    if not node.fully_expanded() and not state.is_terminated():
        node = node.expand()
        state.perform_action(node.action)
    return node, state


def leaf_evaluation(state: State, policy: ANET) -> float:
    """Estimating the value of a leaf node in the tree by doing
    a rollout simulation using the default policy from the leaf
    node's state to a final state.

    Args:
        state (State): final game state
        policy (ANET): The neural network used to get the actions probability distributions

    Returns:
        float: The reward
    """
    while not state.is_terminated():
        action_probabilities = policy.predict(state)
        action = np.random.choice(state.get_all_actions(), p=action_probabilities)
        state.perform_action(action)
    return state.get_reward()


def backpropagation(node: Node, reward: float):
    """Passing the evaluation of a final state back up the tree,
    updating relevant data at all nodes and edges on the path from
    the final state to the tree root.

    Args:
        node (Node): The leaf node
        reward (float): Reward after leaf evaluation
    """
    while node is not None:
        node.update(reward)
        node = node.parent
        reward -= reward  # negate the reward because it's a min-max kind of situation
