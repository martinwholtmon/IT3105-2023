"""Helper functions"""
import json
from pathlib import Path

TOPDIR = Path(__file__).parent.parent
BASEDIR_MODELS = TOPDIR / "models"


def get_model_path(filename) -> str:
    """Get the full path of a model given filename

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
        if file.stem.endswith(uuid):
            filenames.append(file.name)
    return filenames


def save_config(uuid: str):
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


if __name__ == "__main__":
    print(load_config())
