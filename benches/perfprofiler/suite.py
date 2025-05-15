from __future__ import annotations

from typing import Iterable
from tqdm import tqdm

from .profile import cls_profile, profile_methods, rust_methods


class TimingSuiteMeta(type):
    perf_methods = {}

    def __new__(cls, name, bases, attrs):
        attrs["perf_methods"] = profile_methods.get(name, {})
        attrs["rust_methods"] = rust_methods.get(name, {})
        attrs["profile"] = cls_profile

        for base in bases:
            attrs["perf_methods"].update(profile_methods.get(base.__name__, {}))
            attrs["rust_methods"].update(rust_methods.get(base.__name__, {}))

        return type.__new__(cls, name, bases, attrs)


class TimingSuite(metaclass=TimingSuiteMeta):
    name: str = ""
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

# noinspection PyProtectedMember
from .util import merge_dicts
from .result import Result
