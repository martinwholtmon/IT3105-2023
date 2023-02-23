"""This module will handle all the interaction with the game of Hex. 
It is essentially the gym environment for the game of Hex
"""
import numpy as np


class ActionSpace:
    def __init__(self) -> None:
        space: list[int] = None

    def sample(self) -> int:
        raise NotImplementedError


class ObservationSpace:
    def __init__(self) -> None:
        space: State = None


class State:
    """Element of ObservationSpace representing the current game state"""

    pass


class Env:
    def __init__(self) -> None:
        action_space: ActionSpace = None
        observation_space: ObservationSpace = None

    def reset(seed: int = None) -> State:
        raise NotImplementedError

    def step(action: int) -> tuple[State, float, bool]:
        raise NotImplementedError


def get_environment() -> Env:
    raise NotImplementedError
