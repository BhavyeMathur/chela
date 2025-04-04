import numpy as np
import torch

from perfprofiler import *

class TensorSum(TimingSuite):
    def __init__(self, n):
        self.n = n
        self.ndarray: np.ndarray = np.random.rand(n, 2).astype(np.float32)
        self.tensor_cpu = torch.rand(n, 2, dtype=torch.float32)

    @measure_performance("NumPy")
    def run(self):
        self.ndarray.sum(axis=0)

    @measure_performance("PyTorch CPU")
    def run(self):
        self.tensor_cpu.sum(axis=0)

    @measure_rust_performance("Chela CPU", target="sum_f32_slice")
    def run(self, executable):
        return self.run_rust(executable, self.n)

if __name__ == "__main__":
    sizes = [2 ** n for n in range(9, 21)]
    results = TensorSum.profile_each(sizes, n=10)
    plot_results(sizes, results, "tensor.sum() CPU time vs length")
