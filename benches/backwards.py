import torch

from perfprofiler import *

TORCH_DTYPE = torch.float32


class TensorBackwardsTimingSuite(TimingSuite):
    ID: int

    @measure_rust_performance("Chela CPU", target="backwards")
    def run(self, executable):
        return run_rust(executable, self.ID)


class TensorBackwards0(TensorBackwardsTimingSuite):
    ID = 0
    name = "Arithmetic Backwards"

    def __init__(self):
        n = 1000
        self.tensor_a = torch.rand(n, dtype=TORCH_DTYPE, requires_grad=True)
        self.tensor_b = torch.rand(n, dtype=TORCH_DTYPE, requires_grad=True)
        self.tensor_c = torch.rand(n, dtype=TORCH_DTYPE, requires_grad=True)

        self.ones = torch.ones(n, dtype=TORCH_DTYPE)

    @measure_performance("PyTorch CPU")
    def run(self):
        for _ in range(1000):
            result = (self.tensor_a * self.tensor_b) / (self.tensor_c + 1)
            result.backward(self.ones)

            self.tensor_a.grad.zero_()
            self.tensor_b.grad.zero_()
            self.tensor_c.grad.zero_()


class TensorBackwards1(TensorBackwardsTimingSuite):
    ID = 1
    name = "Ax + b"

    def __init__(self):
        i = 1000
        j = 500
        self.x = torch.rand(j, dtype=TORCH_DTYPE)
        self.a = torch.rand(i, j, dtype=TORCH_DTYPE, requires_grad=True)
        self.b = torch.rand(i, dtype=TORCH_DTYPE, requires_grad=True)

        self.ones = torch.ones(i, dtype=TORCH_DTYPE)

    @measure_performance("PyTorch CPU")
    def run(self):
        for _ in range(100):
            result = (self.a @ self.x) + self.b
            result.backward(self.ones)

            self.a.grad.zero_()
            self.b.grad.zero_()


if __name__ == "__main__":
    results = profile_all([
        # TensorBackwards0,
        TensorBackwards1
    ], n=20)
    plot_barplot(results, "Autograd Backwards Benchmark", normalize="PyTorch CPU")
