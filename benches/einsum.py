import numpy as np
import torch

from perfprofiler import *

class TensorSum(TimingSuite):
    def __init__(self, n):
        self.n = n

        i = int(10)
        j = int(100)
        k = int(1000)
        m = int(50)

        self.ndarray_a: np.ndarray = np.random.rand(i, j).astype(np.float32)
        self.ndarray_b: np.ndarray = np.random.rand(k, m).astype(np.float32)

        # self.tensor_a_cpu = torch.rand((i, k), dtype=torch.float32)
        # self.tensor_b_cpu = torch.rand((j, k), dtype=torch.float32)

    @measure_performance("NumPy")
    def run(self):
        np.einsum("ij,km->im", self.ndarray_a, self.ndarray_b)

    # @measure_performance("PyTorch CPU")
    # def run(self):
    #     torch.einsum("ik,jk->ij", self.tensor_a_cpu, self.tensor_b_cpu)

    @measure_rust_performance("Chela CPU", target="einsum")
    def run(self, executable):
        return self.run_rust(executable, self.n)

if __name__ == "__main__":
    results = TensorSum.profile_each([0], n=10)
    print(results)
