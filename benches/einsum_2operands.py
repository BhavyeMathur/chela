from perfprofiler import *
from einsum import TensorEinsumBase

U = 1000
V = 500


class Einsum2Operands0(TensorEinsumBase):
    ID = 200
    name = "uv,v->u"

    def __init__(self):
        super().__init__(self.name, {"u": U, "v": V})


if __name__ == "__main__":
    results = profile_all([
        Einsum2Operands0,
    ], n=20)
    plot_barplot(results, "Einstein Summation Benchmark")
