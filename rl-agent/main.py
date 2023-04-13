import argparse
from state_manager import Env
from games.nim import Nim
from games.hex import Hex
from agent import RLAgent
from policy import Policy
from neural_net import ANET
from helpers import load_config

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
    # Load config
    config = load_config()

    # Load the game
    game_type = config["game_type"]
    game_params = config["game_params"]
    match game_type:
        case "Hex":
            game = Hex(**game_params)
        case "Nim":
            game = Nim(**game_params)
        case _:
            raise NotImplementedError(f"{game_type} is not supported!")

    # Define the environment
    env = Env(game)

    # Define the neural network
    game_shape = env.state.current_state.shape
    action_shape = len(env.state.legal_actions)
    neural_network = ANET(
        input_shape=game_shape,
        output_lenght=action_shape,
        **config["neural_network_params"],
    )

    # Define the RL Policy using MCTS
    policy = Policy(
        neural_net=neural_network,
        **config["policy_params"],
    )

    # # Define the agent
    agent = RLAgent(
        env=env,
        policy=policy,
        **config["agent_params"],
    )
    agent.train()
    agent.evaluate(**config["topp_params"])


if __name__ == "__main__":
    main()
