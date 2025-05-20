from perfprofiler import *
from einsum import TensorEinsumBase

I = 100
J = 500
K = 1000

C = 2


class EinsumOnSlices0(TensorEinsumBase):
    ID = 200
    name = "ij,jk->"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J, "k": K, "C": C},
                         shapes=["ij", "jkC"], slices=[tuple(), (slice(None), slice(None), 0)])


class EinsumOnSlices1(TensorEinsumBase):
    ID = 201
    name = "i->"

    def __init__(self):
        super().__init__(self.name, {"i": 10000, "C": C},
                         shapes=["iC"], slices=[(slice(None), 0)])


class Einsum3Operands0(TensorEinsumBase):
    ID = 202
    name = "i,j,k->"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J, "k": K})


class Einsum3Operands1(TensorEinsumBase):
    ID = 203
    name = "ij,j,k->"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J, "k": K})


if __name__ == "__main__":
    results = profile_all([
        EinsumOnSlices0,
        EinsumOnSlices1,
        Einsum3Operands0,
        Einsum3Operands1
    ], n=10)
    plot_barplot(results, "Einstein Summation Benchmark")
