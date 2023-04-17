"""Connects the MCTS algo with the deep learning module

Note: Subject to refactor as the policy is really the weights of the deep learning module
"""
from mcts import mcts
from state_manager import State
from neural_net import ANET
from helpers import build_model_path, save_config


class Policy:
    def __init__(
        self,
        neural_net: ANET,
        M: int = 100,
        exploration_factor: float = 1,
        exploration_fraction: float = 0,
    ) -> None:
        """Initiate the MCTS policy

        Args:
            neural_net (ANET): the neural network to train and use for prediction
            M (int): Simulations of MCTS followed by a rollout. Defaults to 100
            exploration_factor (float, optional): How explorative the MCTS is. Defaults to 1.
            exploration_fraction (float, optional): fraction which the exploration rate is reduced each episode. Defaults to 0.
        """
        # Set params
        self.neural_net: ANET = neural_net
        self.M = M
        self.exploration_factor = exploration_factor
        self.exploration_fraction = exploration_fraction
        self.subtree = None

    def update(self):
        """Update the target policy"""
        self.neural_net.update()
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

            # Save replay
            self.neural_net.add_replay(state, action_probabilities)

            # Set subtree
            self.subtree = subtree

            return action
        else:
            return self.neural_net.predict(state)

    def save(self, session_uuid, game_name, episode):
        """Invoke the save function on the neural network

        Args:
            game_name (str): name of the game
            episode (str): Episode number/final state
        """
        # Save model
        filepath = build_model_path(f"{game_name}_{episode}_{session_uuid}.pth")
        self.neural_net.save(filepath)

        # Save config
        custom_info = {"session_uuid": session_uuid, "episode_nr": episode}
        save_config(session_uuid, {"custom": custom_info})
