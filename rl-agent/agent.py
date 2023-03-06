"""The Reinforcement learning agent
"""
from state_manager import Env
from policy import Policy


class RLAgent:
    def __init__(self, env: Env, policy: Policy, episodes: int, epsilon: float):
        """Initialize the agent
        Args:
            env (Env): Gym environment
            policy (Policy): The RL policy
            episodes (int): Nr. of episodes/games to run
            epsilon (float): Exploration factor [0,1] Prevent overfitting
        """
        # Check parameters
        if episodes < 1:
            raise ValueError("You must run at least one episode")
        if epsilon < 0 or epsilon > 1:
            raise ValueError(
                "Epsilon (exploration factor) must be in the interval [0,1]"
            )

        # Set parameters
        self.env = env
        self.policy = policy
        self.episodes = episodes
        self.epsilon = epsilon
