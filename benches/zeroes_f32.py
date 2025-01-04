import numpy as np
import torch

from perfprofiler import *

class TensorZeroes(TimingSuite):
    def __init__(self, n):
        self.tensor_cpu = None
        self.ndarray = None
        self.n = n

    @measure_performance("NumPy")
    def run(self):
        self.ndarray = np.zeros(self.n, dtype="float32")

    @measure_performance("PyTorch CPU")
    def run(self):
        self.tensor_cpu = torch.zeros(self.n, dtype=torch.float32)

    @measure_rust_performance("Chela CPU", target="zeroes_f32")
    def run(self, executable):
        return self.run_rust(executable, self.n)

if __name__ == "__main__":
    sizes = [2 ** n for n in range(9, 25)]
    results = TensorZeroes.profile_each(sizes, n=10)
    plot_results(sizes, results, "tensor.zeroes() CPU time vs length")