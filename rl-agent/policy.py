"""Connects the MCTS algo with the deep learning module

Note: Subject to refactor as the policy is really the weights of the deep learning module
"""
from mcts import mcts


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
        self.neural_net = neural_net
        self.M = M
        # self.learning_rate = learning_rate  # TODO: Remove if unused
        # self.discount_factor = discount_factor  # TODO: Remove if unused
        self.exploration_factor = exploration_factor

    def update(self, state, action, next_state, reward):
        """Update the target policy

        Args:
            state (State): The game state
            action (any): action to be performed
            next_state (State): The next game state after performing an action
            reward (float): Reward after preforming the action
        """
        raise NotImplementedError

    def select_action(self, state) -> any:
        """Select the best action by performing MCTS

        Args:
            state (State): The state space

        Returns:
            any: action to perform
        """
        return mcts(state, self.neural_net, self.M, self.exploration_factor)
