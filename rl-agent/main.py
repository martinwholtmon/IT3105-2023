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
    game = Nim(20, 4)

    # Define the environment
    env = Env(game)

    # Define the neural network
    game_shape = env.state.get_state().shape
    action_shape = len(env.state.get_legal_actions())
    neural_network = ANET(
        input_shape=game_shape,
        hidden_layers=[10, 10],
        output_lenght=action_shape,
        activation_function="relu",
    )

    # Define the RL Policy using MCTS
    policy = Policy(
        neural_net=neural_network,
        M=500,
        # learning_rate=1,
        # discount_factor=1,
        exploration_factor=1,
    )

    # # Define the agent
    agent = RLAgent(env=env, policy=policy, episodes=10, epsilon=1)
    agent.train()


if __name__ == "__main__":
    main()
