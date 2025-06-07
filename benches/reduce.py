from perfprofiler import *

N = 1000000


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
        getattr(self.tensor, self.method)()

    @measure_rust_performance("Redstone CPU", target="reduce")
    def run(self, executable):
        return run_rust(executable, self.ID, TRIALS, WARMUP)



class TensorReduce0(TensorReduceTimingSuite):
    ID = 0
    name = "Sum"
    
    def __init__(self):
        super().__init__((N, ), "sum")


class TensorReduce1(TensorReduceTimingSuite):
    ID = 1
    name = "Prod"

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
    name = "Sum Non-Contig"

    def __init__(self):
        super().__init__((N, 2), "sum")
        self.ndarray = self.ndarray[:, 0]
        self.tensor = self.tensor[:, 0]


class TensorReduce11(TensorReduceTimingSuite):
    ID = 11
    name = "Prod Non-Contig"

    def __init__(self):
        super().__init__((N, 2), "prod")
        self.ndarray = self.ndarray[:, 0]
        self.tensor = self.tensor[:, 0]


class TensorReduce12(TensorReduceTimingSuite):
    ID = 12
    name = "Min Non-Contig"

    def __init__(self):
        super().__init__((N, 2), "min")
        self.ndarray = self.ndarray[:, 0]
        self.tensor = self.tensor[:, 0]


class TensorReduce13(TensorReduceTimingSuite):
    ID = 13
    name = "Max Non-Contig"

    def __init__(self):
        super().__init__((N, 2), "max")
        self.ndarray = self.ndarray[:, 0]
        self.tensor = self.tensor[:, 0]


class TensorReduce20(TensorReduceTimingSuite):
    ID = 20
    name = "Sum Non-Unif"

    def __init__(self):
        super().__init__((N, 3), "sum")
        self.ndarray = self.ndarray[:, 0:2]
        self.tensor = self.tensor[:, 0:2]


class TensorReduce21(TensorReduceTimingSuite):
    ID = 21
    name = "Prod Non-Unif"

    def __init__(self):
        super().__init__((N, 3), "prod")
        self.ndarray = self.ndarray[:, 0:2]
        self.tensor = self.tensor[:, 0:2]


class TensorReduce22(TensorReduceTimingSuite):
    ID = 22
    name = "Min Non-Unif"

    def __init__(self):
        super().__init__((N, 3), "min")
        self.ndarray = self.ndarray[:, 0:2]
        self.tensor = self.tensor[:, 0:2]


class TensorReduce23(TensorReduceTimingSuite):
    ID = 23
    name = "Max Non-Unif"

    def __init__(self):
        super().__init__((N, 3), "max")
        self.ndarray = self.ndarray[:, 0:2]
        self.tensor = self.tensor[:, 0:2]


if __name__ == "__main__":
    results = profile_all([
        TensorReduce0,
        TensorReduce10,
        TensorReduce20,

        TensorReduce1,
        TensorReduce11,
        TensorReduce21,

        TensorReduce2,
        TensorReduce12,
        TensorReduce22,

        TensorReduce3,
        TensorReduce13,
        TensorReduce23,
    ])
    plot_barplot(results, "Tensor Reduction Benchmark")
