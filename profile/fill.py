import numpy as np
import torch

from perfprofiler import *


class TensorFill(TimingSuite):
    def __init__(self, n):
        self.ndarray = np.zeros(n)
        self.tensor_cpu = torch.zeros(n)
        self.tensor_mps = torch.zeros(n, device="mps")

    @measure_performance("NumPy")
    def run(self):
        self.ndarray.fill(5)

    @measure_performance("PyTorch CPU")
    def run(self):
        self.tensor_cpu.fill_(5)

    @measure_performance("PyTorch MPS")
    def run(self):
        self.tensor_mps.fill_(5)


if __name__ == "__main__":
    results = TensorFill.profile_each([1000, 10000, 100000, 1000000, 10000000])
    print(results)
