from __future__ import annotations

from typing import Callable, Type
from collections import defaultdict
import time

from .result import Result
from .util import compile_rust, get_class_from_method

profile_methods = defaultdict(dict)
rust_methods = defaultdict(dict)


# noinspection PyDecorator
@classmethod
def cls_profile(cls, *args, n: int = 10, verbose: bool = True, **kwargs) -> dict[str, Result]:
    total_time = defaultdict(list)
    if verbose:
        print("\033[31m", cls.__name__, f"– \"{cls.name}\"" if cls.name else "", "\033[0m")

    for _ in range(n):
        suite_obj = cls(*args, **kwargs)

        for label, function in cls.rust_methods.items():
            elapsed = function(suite_obj)
            total_time[label].append(elapsed)

        for label, function in cls.perf_methods.items():
            start = time.process_time_ns()
            function(suite_obj)
            end = time.process_time_ns()

            total_time[label].append(end - start)

    results = {}
    for label, times in total_time.items():
        results[label] = Result(label, times)
        if verbose:
            print("\t", results[label])

    if verbose:
        print()
    return results


# noinspection PyUnresolvedReferences
def profile(suite: Type["TimingSuite"], *args, n: int = 10, **kwargs) -> dict[str, Result]:
    return suite.profile(*args, n=n, **kwargs)


# noinspection PyUnresolvedReferences
def profile_all(suites: list[Type["TimingSuite"]], *args, n: int = 10, **kwargs) -> dict[str, dict[str, Result]]:
    return {suite.name: profile(suite, *args, n=n, **kwargs) for suite in suites}


"""
Decorators for use by TimingSuite classes

Example:

class TensorEinsum(TimingSuite):
    ...
    
    @measure_performance("NumPy")
    def run(self):
        np.einsum(self.einsum_string, *self.ndarrays)
        
    @measure_rust_performance("Chela CPU", target="einsum")
    def run(self, executable):
        return self.run_rust(executable, self.ID)
"""


def measure_performance(label: str) -> Callable[[Callable], Callable]:
    def decorator(function):
        clsname = get_class_from_method(function)
        profile_methods[clsname][label] = function
        return function

    return decorator


def measure_rust_performance(label: str, target: str) -> Callable[[Callable], Callable]:
    def decorator(function):
        executable = compile_rust(target)

        def wrapper(self, *args, **kwargs):
            return function(self, executable, *args, **kwargs)

        clsname = get_class_from_method(function)
        rust_methods[clsname][label] = wrapper
        return wrapper

    return decorator


__all__ = ["measure_performance", "measure_rust_performance", "profile", "profile_all"]
