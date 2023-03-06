class Policy:
    def __init__(self) -> None:
        pass

    def update(self, state, action, next_state, reward):
        raise NotImplementedError

    def select_action(self, state) -> any:
        raise NotImplementedError
