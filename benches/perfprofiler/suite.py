from typing import Iterable
import subprocess

from tqdm import tqdm

# noinspection PyProtectedMember
from .profile import profile, profile_methods, rust_methods
from .util import merge_dicts
from .result import Result


# from tqdm import tqdm


class TimingSuiteMeta(type):
    perf_methods = {}

    def __new__(cls, name, bases, attrs):
        attrs["perf_methods"] = profile_methods.pop(name, {})
        attrs["rust_methods"] = rust_methods.pop(name, {})
        attrs["profile"] = profile
        return type.__new__(cls, name, bases, attrs)


class TimingSuite(metaclass=TimingSuiteMeta):
    def run_rust(self, executable, *argv) -> float:
        value = subprocess.check_output(f"{executable} {''.join(map(str, argv))}", shell=True)
        return int(value)

    def run_cpp(self, executable, *argv) -> float:
        value = subprocess.check_output(f"{executable} {''.join(map(str, argv))}", shell=True)
        return int(value)

    def profile(*args, **kwargs) -> dict[str, Result]:
        raise NotImplementedError()  # implemented by TimingSuiteMeta

    @classmethod
    def profile_each(cls, args_array, n: int = 100) -> dict[str, list[Result]]:
        results = []

        for i, args in enumerate(tqdm(args_array)):
            if isinstance(args, dict):
                result = cls.profile(**args, n=n, verbose=False)
            elif isinstance(args, Iterable):
                result = cls.profile(*args, n=n, verbose=False)
            else:
                result = cls.profile(args, n=n, verbose=False)

            results.append(result)

        return merge_dicts(results)


__all__ = ["TimingSuite"]
