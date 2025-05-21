import numpy as np
import torch

from perfprofiler import *


class EinsumTimingSuite(TimingSuite):
    ID: int

    def __init__(self, einsum_string: str | list[str], dimensions: dict[str, int], shapes=None, slices=None):
        self.einsum_string = einsum_string

        shapes = self.tensor_dims_from_einsum_string(einsum_string, dimensions, shapes=shapes)
        self.ndarrays = rand_ndarrays_with_shape(shapes, slices=slices)
        self.tensors = rand_tensors_with_shape(shapes, slices=slices)

    @classmethod
    def tensor_dims_from_einsum_string(cls, einsum_string: str, dimensions: dict[str, int], shapes: list[str] = None) \
            -> list[tuple[int, ...]]:
        einsum_string = einsum_string.split("->")[0]
        labels = einsum_string.split(",") if shapes is None else shapes
        return [tuple(dimensions[char] for char in label) for label in labels]

    @measure_performance("PyTorch CPU")
    def run(self):
        torch.einsum(self.einsum_string, *self.tensors)

    @measure_performance("NumPy")
    def run(self):
        np.einsum(self.einsum_string, *self.ndarrays)

    @measure_rust_performance("Chela CPU", target="einsum")
    def run(self, executable):
        return run_rust(executable, self.ID)


class TensorEinsum1(EinsumTimingSuite):
    ID = 1
    name = "Dot Product"

    def __init__(self):
        super().__init__("i,i->",
                         {"i": 10000})


class TensorEinsum2(EinsumTimingSuite):
    ID = 2
    name = "Matrix-Vector"

    def __init__(self):
        super().__init__("ij,j->i",
                         {"i": 1000, "j": 500})


class TensorEinsum3(EinsumTimingSuite):
    ID = 3
    name = "Matrix Mult"

    def __init__(self):
        super().__init__("ij,jk->ik",
                         {"i": 100, "j": 1000, "k": 500})


class TensorEinsum4(EinsumTimingSuite):
    ID = 4
    name = "Outer Product"

    def __init__(self):
        super().__init__("ik,jk->ij",
                         {"i": 100, "j": 1000, "k": 500})


class TensorEinsum5(EinsumTimingSuite):
    ID = 5
    name = "Batch Matrices"

    def __init__(self):
        super().__init__("ijl,jkl->ikl",
                         {"i": 100, "j": 500, "k": 100, "l": 3})


class TensorEinsum6(EinsumTimingSuite):
    ID = 6
    name = "Trace"

    def __init__(self):
        super().__init__("ii->",
                         {"i": 1000})


class TensorEinsum7(EinsumTimingSuite):
    ID = 7
    name = "Broadcasting"

    def __init__(self):
        super().__init__("ij,kj->ikj",
                         {"i": 128, "j": 64, "k": 32})


class TensorEinsum8(EinsumTimingSuite):
    ID = 8
    name = "Sum"

    def __init__(self):
        super().__init__("abc->",
                         {"a": 100, "b": 100, "c": 100})


class TensorEinsum9(EinsumTimingSuite):
    ID = 9
    name = "Diagonal"

    def __init__(self):
        super().__init__("ii->i",
                         {"i": 1000})


class TensorEinsum10(EinsumTimingSuite):
    ID = 10
    name = "Reshape"

    def __init__(self):
        super().__init__("abcd->dcba",
                         {"a": 10, "b": 20, "c": 30, "d": 40})


class TensorEinsum11(EinsumTimingSuite):
    ID = 11
    name = "Hadamard Product"

    def __init__(self):
        super().__init__("ijk,ijk->ijk",
                         {"i": 100, "j": 100, "k": 100})


if __name__ == "__main__":
    results = profile_all([
        TensorEinsum1,  # dot product
        TensorEinsum2,  # matrix-vector
        TensorEinsum3,  # matrix mult
        TensorEinsum5,  # batch matrices

        TensorEinsum4,  # outer product
        TensorEinsum11,  # hadamard product

        TensorEinsum7,  # broadcasting
        TensorEinsum8,  # sum

        TensorEinsum6,  # trace
        TensorEinsum9,  # diagonal
        TensorEinsum10,  # reshape
    ], n=20)
    plot_barplot(results, "Einstein Summation Benchmark")
