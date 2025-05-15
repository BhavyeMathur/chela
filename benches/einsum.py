import numpy as np
import torch

from perfprofiler import *


class TensorEinsumBase(TimingSuite):
    ID: int

    def __init__(self, einsum_string: str, shapes: dict[str, int]):
        self.einsum_string = einsum_string

        shapes = self.tensor_dims_from_einsum_string(einsum_string, shapes)
        self.ndarrays = rand_ndarrays_with_shape(shapes)
        self.tensors = rand_tensors_with_shape(shapes)

    @classmethod
    def tensor_dims_from_einsum_string(cls, einsum_string: str, shapes: dict[str, int]) -> list[tuple[int, ...]]:
        einsum_string = einsum_string.split("->")[0]
        labels = einsum_string.split(",")
        return [tuple(shapes[char] for char in label) for label in labels]

    @measure_performance("PyTorch CPU")
    def run(self):
        torch.einsum(self.einsum_string, *self.tensors)

    @measure_performance("NumPy")
    def run(self):
        np.einsum(self.einsum_string, *self.ndarrays)

    @measure_rust_performance("Chela CPU", target="einsum")
    def run(self, executable):
        return run_rust(executable, self.ID)


class TensorEinsum1(TensorEinsumBase):
    ID = 1
    name = "Dot Product"

    def __init__(self):
        super().__init__("i,i->",
                         {"i": 10000})


class TensorEinsum2(TensorEinsumBase):
    ID = 2
    name = "Matrix-Vector"

    def __init__(self):
        super().__init__("ij,j->i",
                         {"i": 1000, "j": 500})


class TensorEinsum3(TensorEinsumBase):
    ID = 3
    name = "Matrix Mult"

    def __init__(self):
        super().__init__("ij,jk->ik",
                         {"i": 100, "j": 1000, "k": 500})


class TensorEinsum4(TensorEinsumBase):
    ID = 4
    name = "Outer Product"

    def __init__(self):
        super().__init__("ik,jk->ij",
                         {"i": 100, "j": 1000, "k": 500})


class TensorEinsum5(TensorEinsumBase):
    ID = 5
    name = "3 Operands"

    def __init__(self):
        super().__init__("abc,bd,de->ae",
                         {"a": 100, "b": 5, "c": 20, "d": 50, "e": 100})


class TensorEinsum6(TensorEinsumBase):
    ID = 6
    name = "4 Operands"

    def __init__(self):
        super().__init__("abc,bd,bc,ce->ae",
                         {"a": 100, "b": 5, "c": 20, "d": 50, "e": 100})


class TensorEinsum7(TensorEinsumBase):
    ID = 7
    name = "Broadcasting"

    def __init__(self):
        super().__init__("ij,kj->ikj",
                         {"i": 128, "j": 64, "k": 32})


class TensorEinsum8(TensorEinsumBase):
    ID = 8
    name = "Sum"

    def __init__(self):
        super().__init__("abc->",
                         {"a": 100, "b": 100, "c": 100})


class TensorEinsum9(TensorEinsumBase):
    ID = 9
    name = "Diagonal"

    def __init__(self):
        super().__init__("ii->i",
                         {"i": 4096})


class TensorEinsum10(TensorEinsumBase):
    ID = 10
    name = "Reshape"

    def __init__(self):
        super().__init__("abcd->dcba",
                         {"a": 10, "b": 20, "c": 30, "d": 40})


if __name__ == "__main__":
    results = profile_all([TensorEinsum1, TensorEinsum2, TensorEinsum3, TensorEinsum4, TensorEinsum5,
                           TensorEinsum6, TensorEinsum7, TensorEinsum8, TensorEinsum9, TensorEinsum10], n=10)
    plot_barplot(results, "Einstein Summation Benchmark")
