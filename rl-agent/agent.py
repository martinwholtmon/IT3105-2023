"""The Reinforcement learning agent
"""
from state_manager import Env
from policy import Policy


class RLAgent:
    def __init__(
        self,
        env: Env,
        policy: Policy,
        episodes: int,
        save_interval: int,
        session_uuid: str = None,
        episode_nr: int = 0,
        render: bool = False,
    ):
        """Initialize the agent
        Args:
            env (Env): Gym environment
            policy (Policy): The RL policy
            episodes (int): Nr. of episodes/games to run
            save_interval (int): interval to save model
        """
        # Check parameters
        if episodes < 1:
            raise ValueError("You must run at least one episode")

        # Set parameters
        self.env = env
        self.policy = policy
        self.episodes = episodes
        self.save_interval = save_interval
        self.session_uuid = session_uuid
        self.episode_nr = int(episode_nr)
        self.render = render
        self.game_name = env.state.__class__.__name__

    def train(self) -> str:
        """Train the agent. In case we resume training,
        we can set the correct episode start

        Args:
            episode_start (int, optional): Episode to start at. Defaults to 1.
        """
        if self.episode_nr >= self.episodes:
            return self.session_uuid

        print(f"Training session {self.session_uuid}")
        for episode in range(self.episode_nr + 1, self.episodes + 1):
            state = self.env.reset()
            terminated = False
            cumulative_rewards = 0
            episode_length = 0

            while not terminated:
                # Select action
                action = self.policy.select_action(state, training_mode=True)

                # Perform action
                next_state, reward, terminated = self.env.step(action)
                if self.render:
                    print(
                        f"Epoch {episode_length}: State={state.current_state}, Player={state.current_player}, selected action={state.actions[action]}"
                    )
                    next_state.render()

                # Update score and state
                cumulative_rewards += reward
                episode_length += 1
                state = next_state

            # Episode is done, update
            print(
                f"Episode {episode}: reward={cumulative_rewards}, steps={episode_length}"
            )
            self.policy.update()
            print()

            # Save
            if episode % self.save_interval == 0:
                self.policy.save(self.session_uuid, self.game_name, str(episode))

        # Save final model
        self.policy.save(self.session_uuid, self.game_name, str(episode))
