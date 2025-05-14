import numpy as np
import torch

from perfprofiler import *

class TensorSum(TimingSuite):
    def __init__(self, n):
        self.n = n

        a = int(100)
        b = int(5)
        c = int(20)
        d = int(50)
        e = int(100)

        self.ndarray_a: np.ndarray = np.random.rand(a, b, c).astype(np.float32)
        self.ndarray_b: np.ndarray = np.random.rand(b, d).astype(np.float32)
        self.ndarray_c: np.ndarray = np.random.rand(c, e).astype(np.float32)

        self.tensor_a_cpu = torch.rand((a, b, c), dtype=torch.float32)
        self.tensor_b_cpu = torch.rand((b, d), dtype=torch.float32)
        self.tensor_c_cpu = torch.rand((c, e), dtype=torch.float32)

    @measure_performance("NumPy")
    def run(self):
        np.einsum("abc,bd,ce->ae", self.ndarray_a, self.ndarray_b, self.ndarray_c)

    @measure_performance("PyTorch CPU")
    def run(self):
        torch.einsum("abc,bd,ce->ae", self.tensor_a_cpu, self.tensor_b_cpu, self.tensor_c_cpu)

    @measure_rust_performance("Chela CPU", target="einsum")
    def run(self, executable):
        return self.run_rust(executable, self.n)

if __name__ == "__main__":
    results = TensorSum.profile_each([0], n=10)
    print(results)
