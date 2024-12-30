import numpy as np
import torch

from perfprofiler import *


class TensorFill(TimingSuite):
    def __init__(self, n):
        self.n = n

        self.ndarray = np.zeros(n, dtype="float32")
        self.tensor_cpu = torch.zeros(n, dtype=torch.float32)
        self.tensor_mps = torch.zeros(n, device="mps", dtype=torch.float32)

        self.tensor_mps.fill_(1)

    @measure_performance("NumPy")
    def run(self):
        self.ndarray.fill(5)

    @measure_performance("PyTorch CPU")
    def run(self):
        self.tensor_cpu.fill_(5)

    # @measure_performance("PyTorch MPS")
    # def run(self):
    #     self.tensor_mps.fill_(5)

    @measure_rust_performance("Chela CPU (Accelerate)", target="fill")
    def run(self, executable):
        return self.run_rust(executable, self.n)

    @measure_rust_performance("Chela CPU (Naïve)", target="fill_naive")
    def run(self, executable):
        return self.run_rust(executable, self.n)


if __name__ == "__main__":
    sizes = [128, 256, 499, 512, 1023, 1024, 2048]
    results = TensorFill.profile_each(sizes, n=50)
    plot_results(sizes, results)
