import numpy as np


class Result:
    def __init__(self, times: list[float]):
        times = np.array(times) / 1e6  # ns to ms
        self.mean = np.mean(times)
        self.std = np.std(times)

    def __repr__(self) -> str:
        return f"{self.mean:.5f}"
