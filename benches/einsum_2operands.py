from perfprofiler import *

from einsum import EinsumTimingSuite

I = 100
J = 500
K = 1000


class Einsum2Operands0(EinsumTimingSuite):
    ID = 100
    name = "ij,jk->"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J, "k": K})


class Einsum2Operands1(EinsumTimingSuite):
    ID = 101
    name = "ij,jk->i"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J, "k": K})


class Einsum2Operands2(EinsumTimingSuite):
    ID = 107
    name = "ij,ki->i"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J, "k": K})


class Einsum2Operands3(EinsumTimingSuite):
    ID = 106
    name = "ij,ki->j"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J, "k": K})


class Einsum2Operands4(EinsumTimingSuite):
    ID = 102
    name = "ij,jk->ij"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J, "k": K})


class Einsum2Operands5(EinsumTimingSuite):
    ID = 104
    name = "ij,jk->ik"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J, "k": K})


class Einsum2Operands6(EinsumTimingSuite):
    ID = 105
    name = "ik,jk->ij"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J, "k": K})


class Einsum2Operands7(EinsumTimingSuite):
    ID = 103
    name = "ij,jk->ijk"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J, "k": K})


class Einsum2Operands8(EinsumTimingSuite):
    ID = 108
    name = "ij,j->i"

    def __init__(self):
        super().__init__(self.name, {"i": I, "j": J})


if __name__ == "__main__":
    results = profile_all([
        Einsum2Operands0,
        Einsum2Operands1,
        Einsum2Operands2,
        Einsum2Operands3,
        Einsum2Operands4,
        Einsum2Operands5,
        Einsum2Operands6,
        Einsum2Operands7,
        Einsum2Operands8,
    ])
    plot_barplot(results, "Einstein Summation Benchmark")
