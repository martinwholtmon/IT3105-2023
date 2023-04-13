from agent import RLAgent
from policy import Policy
from helpers import load_config, load_env, load_net


def main():
    # Load config
    config = load_config()

    # Define the environment
    env = load_env(config)

    # Define the neural network
    neural_network = load_net(env, config)

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
