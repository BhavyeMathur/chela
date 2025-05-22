import torch
import numpy as np

from perfprofiler import *


TORCH_DTYPE = torch.float32
NUMPY_DTYPE = np.float32


class TensorOpTimingSuite(TimingSuite):
    ID: int

    def __init__(self, einsum_string: str | list[str], dimensions: dict[str, int], shapes=None, slices=None):
        self.einsum_string = einsum_string

        shapes = self.tensor_dims_from_einsum_string(einsum_string, dimensions, shapes=shapes)
        self.ndarrays = rand_ndarrays_with_shape(shapes, slices=slices, dtype=NUMPY_DTYPE)
        self.tensors = rand_tensors_with_shape(shapes, slices=slices, dtype=TORCH_DTYPE)

    @classmethod
    def tensor_dims_from_einsum_string(cls, einsum_string: str, dimensions: dict[str, int], shapes: list[str] = None) \
            -> list[tuple[int, ...]]:
        einsum_string = einsum_string.split("->")[0]
        labels = einsum_string.split(",") if shapes is None else shapes
        return [tuple(dimensions[char] for char in label) for label in labels]

    @measure_rust_performance("Chela CPU", target="einsum")
    def run(self, executable):
        return run_rust(executable, self.ID)


class TensorOp1(TensorOpTimingSuite):
    ID = 1001
    name = "Dot Product"

    def __init__(self):
        super().__init__("i,i->",
                         {"i": 10000})

    @measure_performance("PyTorch CPU")
    def run(self):
        self.tensors[0].dot(self.tensors[1])

    @measure_performance("NumPy")
    def run(self):
        self.ndarrays[0].dot(self.ndarrays[1])


class TensorOp2(TensorOpTimingSuite):
    ID = 1002
    name = "Matrix-Vector"

    def __init__(self):
        super().__init__("ij,j->i",
                         {"i": 1000, "j": 500})

    @measure_performance("PyTorch CPU")
    def run(self):
        _ = self.tensors[0] @ self.tensors[1]

    @measure_performance("NumPy")
    def run(self):
        self.ndarrays[0] @ self.ndarrays[1]


class TensorOp3(TensorOpTimingSuite):
    ID = 1003
    name = "Matrix Mult"

    def __init__(self):
        super().__init__("ij,jk->ik",
                         {"i": 100, "j": 1000, "k": 500})

    @measure_performance("PyTorch CPU")
    def run(self):
        _ = self.tensors[0] @ self.tensors[1]

    @measure_performance("NumPy")
    def run(self):
        self.ndarrays[0] @ self.ndarrays[1]


class TensorOp4(TensorOpTimingSuite):
    ID = 12
    name = "Outer Product"

    def __init__(self):
        super().__init__("i,j->ij",
                         {"i": 100, "j": 1000})

    @measure_performance("PyTorch CPU")
    def run(self):
        torch.outer(self.tensors[0], self.tensors[1])

    @measure_performance("NumPy")
    def run(self):
        np.outer(self.ndarrays[0], self.ndarrays[1])


class TensorOp5(TensorOpTimingSuite):
    ID = 5
    name = "Batch Matrices"

    def __init__(self):
        super().__init__("bij,bjk->bik",
                         {"i": 100, "j": 50, "k": 100, "b": 64})

    @measure_performance("PyTorch CPU")
    def run(self):
        _ = self.tensors[0] @ self.tensors[1]

    @measure_performance("NumPy")
    def run(self):
        self.ndarrays[0] @ self.ndarrays[1]


class TensorOp6(TensorOpTimingSuite):
    ID = 1006
    name = "Trace"

    def __init__(self):
        super().__init__("ii->",
                         {"i": 1000})

    @measure_performance("PyTorch CPU")
    def run(self):
        self.tensors[0].trace()

    @measure_performance("NumPy")
    def run(self):
        self.ndarrays[0].trace()


class TensorOp8(TensorOpTimingSuite):
    ID = 8
    name = "Sum"

    def __init__(self):
        super().__init__("abc->",
                         {"a": 100, "b": 100, "c": 100})

    @measure_performance("PyTorch CPU")
    def run(self):
        self.tensors[0].sum()

    @measure_performance("NumPy")
    def run(self):
        self.ndarrays[0].sum()


class TensorOp9(TensorOpTimingSuite):
    ID = 1009
    name = "Diagonal"

    def __init__(self):
        super().__init__("ii->i",
                         {"i": 1000})

    @measure_performance("PyTorch CPU")
    def run(self):
        self.tensors[0].diagonal()

    @measure_performance("NumPy")
    def run(self):
        self.ndarrays[0].diagonal()


class TensorOp11(TensorOpTimingSuite):
    ID = 11
    name = "Hadamard Product"

    def __init__(self):
        super().__init__("ijk,ijk->ijk",
                         {"i": 100, "j": 100, "k": 100})

    @measure_performance("PyTorch CPU")
    def run(self):
        _ = self.tensors[0] * self.tensors[1]

    @measure_performance("NumPy")
    def run(self):
        self.ndarrays[0] * self.ndarrays[1]


class TensorOp13(TensorOpTimingSuite):
    ID = 13
    name = "Batch Dot Product"

    def __init__(self):
        super().__init__("bi,bi->b",
                         {"i": 1000, "b": 512})

    @measure_performance("PyTorch CPU")
    def run(self):
        if TORCH_DTYPE not in {torch.float32, torch.float64, torch.complex64, torch.complex128}:
            np.einsum("bi,bi->b", *self.ndarrays)
        else:
            torch.linalg.vecdot(self.tensors[0], self.tensors[1])

    @measure_performance("NumPy")
    def run(self):
        np.einsum("bi,bi->b", *self.ndarrays)


class TensorOp14(TensorOpTimingSuite):
    ID = 14
    name = "Star Contraction"

    def __init__(self):
        super().__init__("ij,ik,il->jkl",
                         {"i": 100, "j": 90, "k": 150, "l": 50})

    @measure_performance("PyTorch CPU")
    def run(self):
        torch.einsum("ij,ik,il->jkl", *self.tensors)

    @measure_performance("NumPy")
    def run(self):
        np.einsum("ij,ik,il->jkl", *self.ndarrays)


if __name__ == "__main__":
    results = profile_all([
        TensorOp1,  # dot product
        TensorOp13,  # batch dot product

        TensorOp2,  # matrix-vector
        TensorOp3,  # matrix mult
        TensorOp5,  # batch matrices

        TensorOp4,  # outer product
        TensorOp11,  # hadamard product
        TensorOp8,  # sum

        TensorOp6,  # trace
        TensorOp9,  # diagonal
        TensorOp14,  # star contraction
    ], n=20)
    plot_barplot(results, "Tensor Operations Benchmark")
