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
            if isinstance(args, dict):
                result = cls.profile(**args, n=n, verbose=False)
            elif isinstance(args, Iterable):
                result = cls.profile(*args, n=n, verbose=False)
            else:
                result = cls.profile(args, n=n, verbose=False)

            results.append(result)

        return merge_dicts(results)


__all__ = ["TimingSuite"]
