import numpy as np


class Result:
    def __init__(self, label: str, times: list[float]):
        times = np.array(times) / 1e6  # ns to ms
        self.label = label
        self.n = len(times)
        self.mean = np.mean(times)
        self.std = np.std(times)

    def __repr__(self) -> str:
        return f"\033[37m({self.label})\033[0m {self.mean:.3f} ± {self.se():.3f} ms ({self.n} run/s)"

    def __float__(self):
        return self.mean

    def se(self):
        return float(self.std / (self.n ** 0.5))
