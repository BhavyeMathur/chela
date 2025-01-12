import numpy as np
import torch

from perfprofiler import *


class TensorFill(TimingSuite):
    def __init__(self, n):
        self.n = n

        self.ndarray: np.ndarray = np.zeros((n, 2), dtype="float32")
        self.ndarray_slice = self.ndarray[:, 0]

        self.tensor_cpu = torch.zeros((n, 2), dtype=torch.float32)
        self.tensor_cpu_slice = self.tensor_cpu[:, 0]

    @measure_performance("NumPy")
    def run(self):
        self.ndarray_slice.fill(5.0)

    @measure_performance("PyTorch CPU")
    def run(self):
        self.tensor_cpu_slice.fill_(5.0)

    @measure_rust_performance("Chela CPU", target="fill_f32_slice")
    def run(self, executable):
        return self.run_rust(executable, self.n)


if __name__ == "__main__":
    sizes = [2 ** n for n in range(9, 25)]
    results = TensorFill.profile_each(sizes, n=10)
    plot_results(sizes, results, "tensor.fill() CPU time vs length")
