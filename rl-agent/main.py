from agent import RLAgent
from policy import Policy
from helpers import load_config, load_env, load_net, get_latest_model_filename


def main():
    # If we want to load old session and continue training,
    # update this variable to the session UUID
    # TODO: Add this directly to config file
    session_uuid = None

    # Load config
    config = load_config(session_uuid)

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
    )
    agent.train()
    agent.evaluate(**config["topp_params"])


if __name__ == "__main__":
    main()
