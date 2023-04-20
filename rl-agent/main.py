from agent import RLAgent
from policy import Policy
from topp import TOPP
from helpers import load_config, load_env, load_net, get_latest_model_filename


def train_agent(config: dict, session_uuid: str, render: bool):
    # Define the environment
    env = load_env(config)

    # Define the neural network
    neural_network = load_net(env, config)

    # if we load old mode, retrieve latest one
    if session_uuid is not None:
        neural_network.load(get_latest_model_filename(session_uuid), True)

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
    )
    agent.train()


def topp(config: dict, session_uuid: str, render: bool):
    # Run Topp
    topp = TOPP(uuid=session_uuid)
    topp.play(
        **config["topp_params"],
        render=render,
    )


def main():
    # Global params
    # If we want to load old session and continue training,
    # update this variable to the session UUID
    # TODO: Add this directly to config file
    session_uuid = None
    render = False

    # Load config
    config = load_config(session_uuid)

    # Train the agent
    if config["neural_network_params"]["save_replays"] is not False:
        train_agent(config, session_uuid, render)

    # run TOPP
    topp(config, session_uuid, render)


if __name__ == "__main__":
    main()
