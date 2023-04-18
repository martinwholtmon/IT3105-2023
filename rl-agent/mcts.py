"""This module will contain the MCTS algorithm"""
from __future__ import annotations
import random
import time
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
        self.untried_actions: list = np.where(state.legal_actions == 1)[
            0
        ].tolist()  # legal actions
        self.value = 0  # Predicts who will win (wins)
        self.num_visits = 0  # Might be used to keep exploration high

        # Shuffle the list of untried actions
        random.shuffle(self.untried_actions)

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

    def select_child(self, exploration_bonus) -> Node:
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
                + exploration_bonus
                * np.sqrt(np.log(self.num_visits) / (1 + child.num_visits))
                for child in self.children
            ]
            return self.children[np.argmax(scores)]
        else:  # player 2 -> Minimize
            scores = [
                (child.value / child.num_visits if child.num_visits else 0)
                - exploration_bonus
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
    subtree: Node,
    state: State,
    policy: ANET,
    simulations: int,
    exploration_bonus: float,
    exploration_factor: float,
    timeout: float,
) -> tuple[any, np.ndarray]:
    """Run the MCTS algorithm with M simulations to select the best action to perform

    Args:
        state (State): The state space
        policy (ANET): The policy used to select actions
        simulations (int): Number of simulations
        exploration_bonus (float): Exploration bonus in tree policy
        exploration_factor (float): How explorative the MCTS is during rollouts
        timeout (float): How long the MCTS should run before returning a result

    Returns:
        np.ndarray: Action probabilities -> visit count normalized

    """
    # Take a deepcopy of the current state.
    if subtree is None:
        root = Node(state)
    else:
        root = subtree

    # Run MCTS
    start_time = time.perf_counter()
    for _ in range(simulations):
        # Reach time limit, stop mcts
        if (time.perf_counter() - start_time) > timeout:
            break

        node = root
        current_state = state.clone()

        # Select action
        node, current_state = tree_search(node, current_state, exploration_bonus)

        # Expansion
        node, current_state = node_expansion(node, current_state)

        # Simulate action (expansion) and rollout
        reward = leaf_evaluation(current_state, policy, exploration_factor)

        # Backpropagate results through tree
        backpropagation(node, reward)

    # Return action probabilities
    action_probabilities = np.zeros((len(state.actions),))

    # Input visit counts
    for child in root.children:
        action_probabilities[child.action] = child.num_visits

    # Normalize
    action_probabilities = action_probabilities / np.sum(action_probabilities)

    # find best action and retrieve the subtree
    action = np.argmax(action_probabilities)
    subtree = None
    for tree in root.children:
        if tree.action == action:
            subtree = tree
            break
    return action, action_probabilities, subtree


def tree_search(node: Node, state: State, exploration_bonus: float):
    """Traversing the tree from the root to a leaf node by using the tree policy

    Args:
        node (Node): Node in the MCTS tree
        state (State): Current game state
        exploration_bonus (float): How explorative the selection will be

    Returns:
        tuple[Node, State]: leaf node, state that has action along the path applied
    """
    # Select best child and perform the action to current state
    while not node.is_leaf() and node.fully_expanded():
        node = node.select_child(exploration_bonus)
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


def leaf_evaluation(state: State, policy: ANET, exploration_factor: float) -> float:
    """Estimating the value of a leaf node in the tree by doing
    a rollout simulation using the default policy from the leaf
    node's state to a final state.

    Args:
        state (State): final game state
        policy (ANET): The neural network used to get the actions probability distributions
        exploration_factor (float): The probability of selecting a random action instead of the predicted best one.
        "Epsilon-greedy" approch

    Returns:
        float: The reward
    """
    while not state.is_terminated():
        if random.uniform(0, 1) <= exploration_factor:
            # Explore
            action = state.sample()
        else:
            # Select best action
            action = policy.predict(state)
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
