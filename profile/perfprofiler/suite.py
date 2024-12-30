from typing import Iterable

# noinspection PyProtectedMember
from .profile import profile, profile_methods
from .util import merge_dicts


class TimingSuiteMeta(type):
    perf_methods = {}

    def __new__(cls, name, bases, attrs):
        attrs["perf_methods"] = profile_methods.pop(name, {})
        attrs["profile"] = profile
        return type.__new__(cls, name, bases, attrs)


class TimingSuite(metaclass=TimingSuiteMeta):
    def profile(*args, **kwargs):
        raise NotImplementedError()  # implemented by TimingSuiteMeta

    @classmethod
    def profile_each(cls, args_array, n: int = 10):
        results = []

        for i, args in enumerate(args_array):
            print(f"{i}. {args=}")

            if isinstance(args, dict):
                result = cls.profile(**args, n=n)
            elif isinstance(args, Iterable):
                result = cls.profile(*args, n=n)
            else:
                result = cls.profile(args, n=n)

            results.append(result)
            print()

        return merge_dicts(results)


__all__ = ["TimingSuite"]
