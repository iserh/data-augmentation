from collections import deque


class EarlyStopping:
    def __init__(self, n: int, maximize: bool = False, warm_up_steps: int = 10) -> None:
        self.queue = deque([float("inf")], maxlen=n)
        self.maximize = maximize
        self.warm_up_steps = warm_up_steps
        self.warm_up_cnt = 0

    def __call__(self, value: float) -> bool:
        # negative value if maximizing
        value = -value if self.maximize else value
        # check if value is still minimizing
        if value < max(self.queue) or self.warm_up_cnt < self.warm_up_steps:
            self.warm_up_cnt += 1 if self.warm_up_cnt < self.warm_up_steps else 0
            # append to queue
            self.queue.append(value)
            # no early stopping
            return False
        else:
            # early stop
            return True
