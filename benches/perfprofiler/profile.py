from typing import Callable
from collections import defaultdict
import time

from .result import Result
from .util import compile_rust, get_method_class

profile_methods = defaultdict(dict)
rust_methods = defaultdict(dict)


# noinspection PyDecorator
@classmethod
def profile(cls, *args, n: int = 100, verbose: bool = True, **kwargs) -> dict[str, Result]:
    total_time = defaultdict(list)

    for _ in range(n):
        suite_obj = cls(*args, **kwargs)

        for label, function in cls.perf_methods.items():
            start = time.process_time_ns()
            function(suite_obj)
            end = time.process_time_ns()

            total_time[label].append(end - start)

        for label, function in cls.rust_methods.items():
            elapsed = function(suite_obj)
            total_time[label].append(elapsed)

    results = {}
    for label, times in total_time.items():
        results[label] = Result(times)
        if verbose:
            print(f"{label} took {results[label]} seconds to execute.")

    return results


def measure_performance(label: str) -> Callable:
    def decorator(function):
        clsname = get_method_class(function)
        profile_methods[clsname][label] = function
        return function

    return decorator


def measure_rust_performance(label: str, target: str) -> Callable:
    def decorator(function):
        executable = compile_rust(target)

        def wrapper(self, *args, **kwargs):
            return function(self, executable, *args, **kwargs)

        clsname = get_method_class(function)
        rust_methods[clsname][label] = wrapper
        return wrapper

    return decorator

__all__ = ["measure_performance", "measure_rust_performance"]
