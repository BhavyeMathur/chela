import torch
import numpy as np

from perfprofiler import *

NUMPY_DTYPE = np.float32
TORCH_DTYPE = torch.float32

N = 4096


class TensorBinaryOps(TimingSuite):
    ID: int

    tensors: list
    ndarrays: list[np.ndarray]

    operation = "Subtraction"

    def __init__(self, shapes, slices=None):
        self.ndarrays = rand_ndarrays_with_shape(shapes, slices=slices, dtype=NUMPY_DTYPE)
        self.tensors = rand_tensors_with_shape(shapes, slices=slices, dtype=TORCH_DTYPE)

    @measure_rust_performance("Chela CPU", target="binary_ops")
    def run(self, executable):
        return run_rust(executable, self.ID)

    @measure_performance("PyTorch CPU")
    def run(self):
        _ = self.tensors[0] - self.tensors[1]

    @measure_performance("NumPy")
    def run(self):
        _ = self.ndarrays[0] - self.ndarrays[1]


class TensorBinaryOps0(TensorBinaryOps):
    ID = 0
    name = "[1], [1]"

    def __init__(self):
        super().__init__([(N, ), (N, )])


class TensorBinaryOps1(TensorBinaryOps):
    ID = 1
    name = "[1], [0]"

    def __init__(self):
        super().__init__([(N,), (1,)])


class TensorBinaryOps2(TensorBinaryOps):
    ID = 2
    name = "[n], [1]"

    def __init__(self):
        super().__init__([(N, 3), (N, )], slices=((Ellipsis, 0), ()))


class TensorBinaryOps3(TensorBinaryOps):
    ID = 2
    name = "[n], [0]"

    def __init__(self):
        super().__init__([(N, 3), (1, )], slices=((Ellipsis, 0), ()))


class TensorBinaryOps4(TensorBinaryOps):
    ID = 4
    name = "Non-Unif, [1]"

    def __init__(self):
        super().__init__([(N, 3), (N, 2)], slices=((Ellipsis, slice(0, 2)), ()))


class TensorBinaryOps5(TensorBinaryOps):
    ID = 5
    name = "Non-Unif, [0]"

    def __init__(self):
        super().__init__([(N, 3), (1,)], slices=((Ellipsis, slice(0, 2)), ()))


class TensorBinaryOps6(TensorBinaryOps):
    ID = 6
    name = "Non-Unif, [n]"

    def __init__(self):
        super().__init__([(N, 3), (N, 2, 2)], slices=((Ellipsis, slice(0, 2)), (Ellipsis, 0)))


class TensorBinaryOps7(TensorBinaryOps):
    ID = 7
    name = "Non-Unif, Non-Unif"

    def __init__(self):
        super().__init__([(N, 3), (N, 3)], slices=((Ellipsis, slice(0, 2)), (Ellipsis, slice(0, 2))))



if __name__ == "__main__":
    results = profile_all([
        TensorBinaryOps0,
        TensorBinaryOps1,
        TensorBinaryOps2,
        TensorBinaryOps3,
        TensorBinaryOps4,
        TensorBinaryOps5,
        TensorBinaryOps6,
        TensorBinaryOps7
    ], n=20)
    plot_barplot(results, f"{TensorBinaryOps.operation} Benchmark ({N=})", normalize="NumPy")
