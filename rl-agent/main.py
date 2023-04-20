import argparse
import uuid
from agent import RLAgent
from policy import Policy
from topp import TOPP
from helpers import (
    load_config,
    load_env,
    load_net,
    get_latest_model_filename,
    save_config,
)


def argument_parser() -> argparse.Namespace:
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Description of your script")

    # Add an optional positional argument for par1 with default value None
    parser.add_argument("uuid", type=str, nargs="?", help="Session uuid", default=None)

    # Add two optional arguments for toggle1 and toggle2
    parser.add_argument("--topp", action="store_true", help="Run TOPP", default=False)
    parser.add_argument(
        "--render", action="store_true", help="Render the games played", default=False
    )

    # Parse the command-line arguments
    return parser.parse_args()


def train_agent(config: dict, session_uuid: str, render: bool):
    if (
        config["neural_network_params"]["save_replays"] is False
        and session_uuid is not None
    ):
        raise ValueError(
            "Cannot continue training a model that does not have saved replays"
        )

    # Define the environment
    env = load_env(config)

    # Define the neural network
    neural_network = load_net(env, config)

    # if we load old mode, retrieve latest one
    if session_uuid is not None:
        neural_network.load(get_latest_model_filename(session_uuid), True)
    else:
        session_uuid = str(uuid.uuid4())

    # Save the current config
    save_config(session_uuid)

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
        **config.get("custom", {}),
        render=render,
        session_uuid=session_uuid,
    )
    agent.train()
    print(f"Finished training, here is the session uuid: {session_uuid}")


def topp(config: dict, session_uuid: str, render: bool):
    # Run Topp
    topp = TOPP(uuid=session_uuid)
    topp.play(
        **config["topp_params"],
        render=render,
    )


def main():
    # get arguments
    args = argument_parser()

    # Load config
    config = load_config(args.uuid)

    # Train the agent
    if not args.topp:
        train_agent(config, args.uuid, args.render)

    # run TOPP
    if args.topp:
        topp(config, args.uuid, args.render)


if __name__ == "__main__":
    main()
