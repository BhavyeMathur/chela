from perfprofiler import *
from einsum import TensorEinsumBase

I = 100
J = 50
K = 100


class Einsum2Matrices0(TensorEinsumBase):
    ID = 100
    name = "ij,jk->"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J, "k": K})


class Einsum2Matrices1(TensorEinsumBase):
    ID = 101
    name = "ij,jk->i"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J, "k": K})


class Einsum2Matrices2(TensorEinsumBase):
    ID = 107
    name = "ij,ki->i"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J, "k": K})


class Einsum2Matrices3(TensorEinsumBase):
    ID = 106
    name = "ij,ki->j"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J, "k": K})


class Einsum2Matrices4(TensorEinsumBase):
    ID = 102
    name = "ij,jk->ij"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J, "k": K})


class Einsum2Matrices5(TensorEinsumBase):
    ID = 104
    name = "ij,jk->ik"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J, "k": K})


class Einsum2Matrices6(TensorEinsumBase):
    ID = 105
    name = "ik,jk->ij"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J, "k": K})


class Einsum2Matrices7(TensorEinsumBase):
    ID = 103
    name = "ij,jk->ijk"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J, "k": K})


if __name__ == "__main__":
    results = profile_all([
        Einsum2Matrices0,
        Einsum2Matrices1,
        Einsum2Matrices2,
        Einsum2Matrices3,
        Einsum2Matrices4,
        Einsum2Matrices5,
        Einsum2Matrices6,
        Einsum2Matrices7,
    ], n=20)
    plot_barplot(results, "Einstein Summation Benchmark")
