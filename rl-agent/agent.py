"""The Reinforcement learning agent
"""
import uuid
from state_manager import Env
from policy import Policy
from topp import TOPP


class RLAgent:
    def __init__(
        self,
        env: Env,
        policy: Policy,
        episodes: int,
        epsilon: float,
        save_interval: int,
    ):
        """Initialize the agent
        Args:
            env (Env): Gym environment
            policy (Policy): The RL policy
            episodes (int): Nr. of episodes/games to run
            epsilon (float): Exploration factor [0,1] Prevent overfitting
            save_interval (int): interval to save model
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
        self.save_interval = save_interval
        self.uuid = str(uuid.uuid4())
        self.game_name = env.state.__class__.__name__

    def train(self):
        """Train the agent"""
        print(f"Training session {self.uuid}")

        self.policy.rbuf_clear()

        for episode in range(1, self.episodes + 1):
            state = self.env.reset()
            terminated = False
            cumulative_rewards = 0
            episode_length = 0

            while not terminated:
                # Select action
                action = self.policy.select_action(state, training_mode=True)

                # Perform action
                # print(
                #     f"Epoch {episode_length}: State={state.current_state}, Player={state.current_player}, selected action={state.actions[action]}"
                # )
                next_state, reward, terminated = self.env.step(action)

                # Update score and state
                cumulative_rewards += reward
                episode_length += 1
                state = next_state

                # TODO: (opt) Adjusting epsilon: start high -> reduce

            # Episode is done, update
            self.policy.update()
            print(
                f"Episode {episode}: reward={cumulative_rewards}, steps={episode_length}"
            )

            # Save
            if episode % self.save_interval == 0:
                self.policy.save(self.uuid, self.game_name, str(episode))

        # Save final model
        self.policy.save(self.uuid, self.game_name, str(episode))

    def evaluate(self, games):
        """This will initialize the The Tournament of Progressive Policies (TOPP):
        Each saved model will compete in a round-robin fashion against the
        other models in a series of games. The TOPP winner will be saved.

        Args:
            games (int): number of games in one series
        """
        topp = TOPP(self.uuid, self.env)
        topp.play(games)
