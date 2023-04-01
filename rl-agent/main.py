import argparse
from state_manager import Env
from games.nim import Nim
from agent import RLAgent
from policy import Policy
from neural_net import ANET

# Define arguments
parser = argparse.ArgumentParser(description="description")
parser.add_argument("--render", action="store_true", help="render the environment")
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="interval between training status logs (default: 10)",
)
args = parser.parse_args()
# To use in code: if args.render:


def main():
    # Load the game
    game = Nim(12, 4)

    # Define the environment
    env = Env(game)

    # Define the neural network
    game_shape = env.state.current_state.shape
    action_shape = len(env.state.legal_actions)
    neural_network = ANET(
        input_shape=game_shape,
        hidden_layers=[64, 64, 64],
        output_lenght=action_shape,
        activation_function="relu",
        learning_rate=0.01,
    )

    # Define the RL Policy using MCTS
    policy = Policy(
        neural_net=neural_network,
        M=400,
        # learning_rate=1,
        # discount_factor=1,  # assumed to be 1
        exploration_factor=1,
    )

    # # Define the agent
    agent = RLAgent(env=env, policy=policy, episodes=200, epsilon=1)
    agent.train()


if __name__ == "__main__":
    main()
