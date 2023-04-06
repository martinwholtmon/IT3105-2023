"""Connects the MCTS algo with the deep learning module

Note: Subject to refactor as the policy is really the weights of the deep learning module
"""
import numpy as np
from mcts import mcts
from state_manager import State
from neural_net import ANET


class Policy:
    def __init__(
        self,
        neural_net: ANET,
        M: int,
        exploration_factor: float = 1,
        exploration_fraction: float = 0,
    ) -> None:
        """Initiate the MCTS policy

        Args:
            neural_net (ANET): the neural network to train and use for prediction
            M (int): Simulations of MCTS followed by a rollout
            exploration_factor (float, optional): How explorative the MCTS is. Defaults to 1.
            exploration_fraction (float, optional): fraction which the exploration rate is reduced each episode. Defaults to 0.
        """
        # Set params
        self.neural_net: ANET = neural_net
        self.M = M
        self.exploration_factor = exploration_factor
        self.exploration_fraction = exploration_fraction
        self.rbuf: "list[tuple[State, np.ndarray]]" = []
        self.subtree = None

    def update(self):
        """Update the target policy"""
        self.neural_net.train(self.rbuf)
        self.rbuf_clear()
        self.subtree = None

    def select_action(self, state: State, training_mode: bool = False) -> any:
        """Select the best action given policy

        Args:
            state (State): The current state
            training_mode (bool, optional): Perform MCTS, save target and state to RBUF. Defaults to False.

        Returns:
            any: Action to perform
        """
        if training_mode:
            action, action_probabilities, subtree = mcts(
                self.subtree,
                state,
                self.neural_net,
                self.M,
                self.exploration_factor,
                # TODO: self.exploration_fraction,
            )
            self._rbuf_add(state, action_probabilities)

            # Set subtree
            self.subtree = subtree

            return action
        else:
            return state.actions[self.neural_net.predict(state)]

    def _rbuf_add(self, state: State, action_probabilities: np.ndarray):
        """Add a replay to the replay buffer
        Args:
            state (State): a game state
            action_probabilities (np.ndarray): action probabilities
        """
        self.rbuf.append((state.clone(), action_probabilities))

    def rbuf_clear(self):
        """Clear the replay buffer"""
        self.rbuf.clear()
