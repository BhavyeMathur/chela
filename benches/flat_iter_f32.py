import numpy as np
import torch

from perfprofiler import *


class TensorFlatIter(TimingSuite):
    def __init__(self, n):
        self.n = n

        self.ndarray: np.ndarray = np.zeros(n, dtype="float32")
        self.tensor_cpu = torch.zeros(n, dtype=torch.float32)

    @measure_performance("NumPy")
    def run(self):
        for e in self.ndarray.flat:
            pass


    @measure_performance("PyTorch CPU")
    def run(self):
        for e in self.tensor_cpu.flatten():
            pass

    @measure_rust_performance("Chela CPU", target="flat_iter_f32")
    def run(self, executable):
        return self.run_rust(executable, self.n)


if __name__ == "__main__":
    sizes = [2 ** n for n in range(9, 25)]
    results = TensorFlatIter.profile_each(sizes, n=10)
    plot_results(sizes, results, "tensor.flat_iter() CPU time vs length")
