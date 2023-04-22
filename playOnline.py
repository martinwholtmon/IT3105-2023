from oht.token_handler import get_token
from rl_agent import helpers


def main():
    token = get_token()
    config = helpers.load_config("76d0c05f-7ffa-4049-a1a1-277cf56a8c2d")
    env = helpers.load_env(config)
    net = helpers.load_net(env, config)

    # Load best net
    path_to_net = r"C:/Users/Martin/Documents/GIT/NTNU/IT3105-2023/models/Hex_10_76d0c05f-7ffa-4049-a1a1-277cf56a8c2d.pth"
    # net.load(path_to_net)


if __name__ == "__main__":
    main()
