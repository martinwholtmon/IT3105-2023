import itertools
import random
from dataclasses import dataclass
from state_manager import Env
from neural_net import ANET
from helpers import get_model_filenames


@dataclass
class Model:
    """A model/ANET with current score etc"""

    model_name: str
    neural_net: ANET
    wins: int = 0


class TOPP:
    """Perform round-robin tournament between the saved neural networks"""

    def __init__(self, uuid: str, env: Env) -> None:
        """Initialize the TOPP

        Args:
            uuid (str): uuid of training session
        """
        self.env: Env = env
        self.models: "list[Model]" = load_models(uuid)

    def play(self, games: int):
        """Play the tournament

        Args:
            games (int): number of games to play per series
        """
        pairings = list(itertools.combinations(self.models, 2))

        for model1, model2 in pairings:
            print(
                f"Series of games between {model1.model_name} and {model2.model_name}"
            )
            for game in range(games):
                winning_model = self._execute_game(model1, model2)

                winning_model.wins += 1
                print(
                    f"Game {game}: {winning_model.model_name} won, score={winning_model.wins}"
                )

        for m in self.models:
            print(m.model_name, m.wins)

    def _execute_game(self, model1: Model, model2: Model) -> Model:
        """Will execute one game, randomly assign models to the starting player

        Args:
            model1 (Model): Neural Network
            model2 (Model): Neural Network

        Returns:
            Model: The winning model
        """
        # To have some randomness, randomly assign the models as either player 1 or 2
        p1, p2 = (model1, model2) if random.choice([True, False]) else (model2, model1)

        # Init the game
        state = self.env.reset()
        terminated = False
        cumulative_rewards = 0
        episode_length = 0

        while not terminated:
            # Select action
            action = (
                p1.neural_net.predict(state)
                if state.current_player == 1
                else p2.neural_net.predict(state)
            )

            # Perform action
            # print(
            #     f"Epoch {episode_length}: State={state.current_state}, Player={state.current_player}, selected action={state.actions[action]}"
            # )
            next_state, reward, terminated = self.env.step(action)

            # Update score and state
            cumulative_rewards += reward
            episode_length += 1
            state = next_state

        # Return winning model
        return p1 if cumulative_rewards == 1 else p2


def load_models(uuid: str) -> "list[Model]":
    """Given a UUID, it will load all the models

    Args:
        uuid (str): the uuid for the trained models

    Returns:
        dict: dict containing the models
    """
    models = []
    filenames = get_model_filenames(uuid)
    for filename in filenames:
        # TODO: Currently hardcoded, but must be loaded from file given the uuid (issue #52)
        net = ANET(
            input_shape=(9,),
            output_lenght=9,
            hidden_layers=[82, 82],
            activation_function="relu",
            learning_rate=0.001,
            batch_size=32,
            discount_factor=1,
            gradient_steps=1,
            max_grad_norm=1,
            device="cpu",
        )
        net.load(filename, False)
        models.append(Model(filename, net))
    return models
