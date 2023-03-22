from state_manager import Env
from games.nim import Nim
from agent import RLAgent
from policy import Policy


def main():
    # RL Params
    exploration_factor = 0.1

    # Load the game
    game = Nim(20, 4)

    # Define the environment
    env = Env(game)

    # Define the RL Policy
    policy = Policy()

    # Define the agent
    agent = RLAgent(env=env, policy=policy, episodes=10, epsilon=exploration_factor)


if __name__ == "__main__":
    main()
