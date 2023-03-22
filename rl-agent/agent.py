"""The Reinforcement learning agent
"""
import random
from state_manager import Env
from policy import Policy


class RLAgent:
    def __init__(
        self, env: Env, policy: Policy, episodes: int, epsilon: float, player_id: int
    ):
        """Initialize the agent
        Args:
            env (Env): Gym environment
            policy (Policy): The RL policy
            episodes (int): Nr. of episodes/games to run
            epsilon (float): Exploration factor [0,1] Prevent overfitting
            player_id (int): The id of the player
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
        self.player_id = player_id

    def train(self):
        """Train the agent"""
        for episode in range(1, self.episodes + 2):
            state = self.env.reset()
            terminated = False
            cumulative_rewards = 0
            episode_length = 0

            while not terminated:
                # Select action
                if random.uniform(0, 1) < self.epsilon:
                    # Explore: Random legal action
                    action = self.env.state.sample(self.player_id)
                else:
                    action = self.policy.select_action(state)

                # Perform action
                next_state, reward, terminated = self.env.step(action)

                # Update policy
                self.policy.update(state, action, next_state, reward)

                # Update score and state
                cumulative_rewards += reward
                episode_length += 1
                state = next_state

                # TODO: (opt) Adjusting epsilon: start high -> reduce

            print(
                f"Episode {episode}: reward={cumulative_rewards}, steps={episode_length}"
            )
