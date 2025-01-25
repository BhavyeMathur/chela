import numpy as np
import torch

from perfprofiler import *

class TensorMul(TimingSuite):
    def __init__(self, n):
        self.n = n

        self.ndarray1: np.ndarray = np.random.rand(n).astype(np.float32)
        self.ndarray2: np.ndarray = np.random.rand(n).astype(np.float32)

        self.tensor_cpu1 = torch.rand(n, dtype=torch.float32)
        self.tensor_cpu2 = torch.rand(n, dtype=torch.float32)

    @measure_performance("NumPy")
    def run(self):
        self.ndarray1 * self.ndarray2

    @measure_performance("PyTorch CPU")
    def run(self):
        self.tensor_cpu1 * self.tensor_cpu2

    @measure_rust_performance("Chela CPU", target="mul_f32")
    def run(self, executable):
        return self.run_rust(executable, self.n)

if __name__ == "__main__":
    sizes = [2 ** n for n in range(9, 25)]
    results = TensorMul.profile_each(sizes, n=10)
    plot_results(sizes, results, "tensor * CPU time vs length")