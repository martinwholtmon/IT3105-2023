"""Helper functions"""
import json
from pathlib import Path
from state_manager import Env, State
from games.hex import Hex
from games.nim import Nim
from neural_net import ANET

TOPDIR = Path(__file__).parent.parent
BASEDIR_MODELS = TOPDIR / "models"


def build_model_path(filename) -> str:
    """Build the full path of a model given filename

    Args:
        filename (str): name of model in the form of game_iteration_uuid.pth

    Returns:
        str: full path of the model
    """
    filename = BASEDIR_MODELS / filename
    return str(filename)


def get_model_filenames(uuid: str) -> list[str]:
    """Given a uuid, retrieve all file names for the models

    Args:
        uuid (str): uuid for current iteration of models

    Returns:
        list[str]: a list of file names for the models
    """
    filenames = []
    for file in Path(BASEDIR_MODELS).glob("*"):
        if file.name.endswith(f"{uuid}.pth"):
            filenames.append(file.as_posix())
    return filenames


def get_latest_model_filename(uuid: str) -> str:
    """Will retrieve the filepath for the latest model in a training session

    Args:
        uuid (str): uuid for training session

    Returns:
        str: path to saved model
    """
    return get_model_filenames(uuid)[-1]

    """Will save the current settings to config

    Args:
        uuid (str): _description_
    """
    # Load config file
    config = load_config()

    # Save
    filename = BASEDIR_MODELS / f"{uuid}.json"
    with open(filename, "w") as fp:
        json.dump(config, fp)


def load_config(uuid: str = None) -> dict:
    """Load the config

    Args:
        uuid (str, optional): If an uuid is provided, it will load it from the models directory.
            Defaults to None, meaning it will use the current config.

    Returns:
        dict: config file
    """
    # Set dir and filename
    if uuid is None:
        dir = TOPDIR / "rl-agent"
        filename = "config.json"
    else:
        dir = BASEDIR_MODELS
        filename = f"{uuid}.json"

    # Load config
    with open(dir / filename, "r") as f:
        config = json.load(f)
    return config


def load_game(game_type, params) -> State:
    """Will load the game given config file

    Args:
        game_type (str): name of the game
        params (dict): params to init the game

    Raises:
        NotImplementedError: If the provided game does not exist

    Returns:
        State: initialized game
    """
    match game_type:
        case "Hex":
            return Hex(**params)
        case "Nim":
            return Nim(**params)
        case _:
            raise NotImplementedError(f"{game_type} is not supported!")


def load_env(config) -> Env:
    """Will load an environment with the selected game from config

    Args:
        config (dict): config file

    Returns:
        Env: Environment to play the game
    """
    game = load_game(config["game_type"], config["game_params"])
    return Env(game)


def load_net(env: Env, config: dict) -> ANET:
    """Will load a neural network

    Args:
        env (Env): Current environment. Used to set correct input/output size
        config (dict): config file

    Returns:
        ANET: Neural network
    """
    game_shape = env.state.current_state.shape
    action_shape = len(env.state.legal_actions)
    return ANET(
        input_shape=game_shape,
        output_lenght=action_shape,
        **config["neural_network_params"],
    )
