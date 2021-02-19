from collections import deque


class EarlyStopping:
    def __init__(self, n: int, maximize: bool = False) -> None:
        self.queue = deque([float("inf")], maxlen=n)
        self.maximize = maximize

    def __call__(self, value: float) -> bool:
        # negative value if maximizing
        value = -value if self.maximize else value
        # check if value is still minimizing
        if value <= max(self.queue):
            # append to queue
            self.queue.append(value)
            # no early stopping
            return False
        else:
            # early stop
            return True
