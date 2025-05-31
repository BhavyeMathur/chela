import torch
import numpy as np

from perfprofiler import *


TORCH_DTYPE = torch.float32


class TensorBackwardsTimingSuite(TimingSuite):
    ID: int

    @measure_rust_performance("Chela CPU", target="backwards")
    def run(self, executable):
        return run_rust(executable, self.ID)


class TensorBackwards0(TensorBackwardsTimingSuite):
    ID = 0
    name = "Add Backwards"

    def __init__(self):      
        self.tensor_a = torch.rand(1000, dtype=TORCH_DTYPE, requires_grad=True)
        self.tensor_b = torch.rand(1000, dtype=TORCH_DTYPE, requires_grad=True)
        self.tensor_c = torch.rand(1000, dtype=TORCH_DTYPE, requires_grad=True)

    @measure_performance("PyTorch CPU")
    def run(self):
        for _ in range(100):
            self.tensor_a.grad.zero()
            self.tensor_b.grad.zero()
            self.tensor_c.grad.zero()
            
            result = (self.tensor_a + self.tensor_b) / (self.tensor_c + 1)
            result.backward()



if __name__ == "__main__":
    results = profile_all([
        TensorBackwards0
    ], n=20)
    plot_barplot(results, "Tensor Operations Benchmark")
