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
        neural_net,
        M,
        # learning_rate,
        # discount_factor,
        exploration_factor,
    ) -> None:
        """Initiate the MCTS policy

        Args:
            M (int): Simulations of MCTS followed by a rollout
            learning_rate (float): Learning rate (0,1]
            discount_factor (float): Reward importance [0,1]
            exploration_factor (float): How explorative the MCTS is
        """
        # Check params
        if M < 1:
            raise ValueError("You must run at least one MCTS simulation")
        # if learning_rate <= 0 or learning_rate > 1:
        #     raise ValueError("Alpha (learning rate) must be in the interval (0,1]")
        # if discount_factor < 0 or discount_factor > 1:
        #     raise ValueError("Gamma (Reward importance) must be in the interval [0,1]")

        # Set params
        self.neural_net: ANET = neural_net
        self.M = M
        # self.learning_rate = learning_rate  # TODO: Remove if unused
        # self.discount_factor = discount_factor  # TODO: Remove if unused
        self.exploration_factor = exploration_factor
        self.rbuf: "list[tuple[State, np.ndarray]]" = []

    def update(self, state, action, next_state, reward):
        """Update the target policy

        Args:
            state (State): The game state
            action (any): action to be performed
            next_state (State): The next game state after performing an action
            reward (float): Reward after preforming the action
        """
        raise NotImplementedError

    def select_action(self, state: State, training_mode: bool = False) -> any:
        """Select the best action given policy

        Args:
            state (State): The current state
            training_mode (bool, optional): Perform MCTS, save target and state to RBUF. Defaults to False.

        Returns:
            any: Action to perform
        """
        if training_mode:
            action_probabilities = mcts(
                state, self.neural_net, self.M, self.exploration_factor
            )
            self._rbuf_add(state, action_probabilities)
        else:
            action_probabilities = self.neural_net.predict(state)
        return state.actions[np.argmax(action_probabilities)]

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
