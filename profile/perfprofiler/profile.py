from typing import Callable
from collections import defaultdict
import time

import numpy as np

profile_methods = defaultdict(dict)


# noinspection PyDecorator
@classmethod
def profile(cls, *args, n: int = 10, **kwargs):
    total_time = {label: [] for label in cls.perf_methods.keys()}

    for _ in range(n):
        suite_obj = cls(*args, **kwargs)

        for label, function in cls.perf_methods.items():
            start = time.perf_counter_ns()
            function(suite_obj)
            end = time.perf_counter_ns()

            total_time[label].append(end - start)

    for label, times in total_time.items():
        total_time[label] = np.mean(times) / 1e9
        print(f"{label} took {total_time[label]} seconds to execute.")

    return total_time


def measure_performance(label: str) -> Callable:
    def decorator(function):
        clsname = function.__qualname__.split(".")[0]
        profile_methods[clsname][label] = function
        return function

    return decorator


__all__ = ["measure_performance"]
