import numpy as np
import torch

from perfprofiler import *

NUMPY_DTYPE = np.float32
TORCH_DTYPE = torch.float32

N = 10000000


class TensorReduceTimingSuite(TimingSuite):
    ID: int

    def __init__(self, shape: tuple[int, ...], method: str):
        self.ndarray: np.ndarray = np.random.rand(*shape).astype(NUMPY_DTYPE)
        self.tensor: torch.Tensor = torch.rand(*shape, dtype=TORCH_DTYPE)
        self.method = method

    @measure_performance("NumPy")
    def run(self):
        getattr(self.ndarray, self.method)()

    @measure_performance("PyTorch CPU")
    def run(self):
        self.tensor.sum()

    @measure_rust_performance("Chela CPU", target="reduce")
    def run(self, executable):
        return run_rust(executable, self.ID)



class TensorReduce0(TensorReduceTimingSuite):
    ID = 0
    name = "Sum"
    
    def __init__(self):
        super().__init__((N, ), "sum")


class TensorReduce1(TensorReduceTimingSuite):
    ID = 1
    name = "Product"

    def __init__(self):
        super().__init__((N, ), "prod")


class TensorReduce2(TensorReduceTimingSuite):
    ID = 2
    name = "Min"

    def __init__(self):
        super().__init__((N, ), "min")


class TensorReduce3(TensorReduceTimingSuite):
    ID = 3
    name = "Max"

    def __init__(self):
        super().__init__((N, ), "max")


class TensorReduce10(TensorReduceTimingSuite):
    ID = 10
    name = "Sum Slice"

    def __init__(self):
        super().__init__((N, 2), "sum")
        self.ndarray = self.ndarray[:, 0]
        self.tensor = self.tensor[:, 0]


class TensorReduce11(TensorReduceTimingSuite):
    ID = 11
    name = "Product Slice"

    def __init__(self):
        super().__init__((N, 2), "prod")
        self.ndarray = self.ndarray[:, 0]
        self.tensor = self.tensor[:, 0]


class TensorReduce12(TensorReduceTimingSuite):
    ID = 12
    name = "Min Slice"

    def __init__(self):
        super().__init__((N, 2), "min")
        self.ndarray = self.ndarray[:, 0]
        self.tensor = self.tensor[:, 0]


class TensorReduce13(TensorReduceTimingSuite):
    ID = 13
    name = "Max Slice"

    def __init__(self):
        super().__init__((N, 2), "max")
        self.ndarray = self.ndarray[:, 0]
        self.tensor = self.tensor[:, 0]


if __name__ == "__main__":
    results = profile_all([
        TensorReduce0,
        TensorReduce10,

        TensorReduce1,
        TensorReduce11,

        TensorReduce2,
        TensorReduce12,

        TensorReduce3,
        TensorReduce13,
    ], n=10)
    plot_barplot(results, "Tensor Reduction Benchmark")
