import subprocess
import json

import numpy as np
import torch


def merge_dicts(list_of_dicts):
    result = {}
    for k in list_of_dicts[0].keys():
        result[k] = [d[k] for d in list_of_dicts]
    return result


target_to_executable = {}


def compile_rust(target: str) -> str:
    if target not in target_to_executable:
        out = subprocess.check_output(f"$HOME/.cargo/bin/cargo build --release --bin {target} "
                                      "-v --message-format=json", shell=True)
        out = out.split(b"\n")[-3]
        target_to_executable[target] = json.loads(out)["executable"]

    return target_to_executable[target]


def run_rust(executable, *argv) -> float:
    value = subprocess.check_output(f"{executable} {''.join(map(str, argv))}", shell=True)
    return int(value)


def get_class_from_method(method) -> str:
    return method.__qualname__.split(".")[0]


def rand_ndarrays_with_shape(shapes: list[tuple[int, ...]], dtype: str = "int32") -> list[np.ndarray]:
    return [np.random.rand(*shape).astype(dtype) for shape in shapes]


def rand_tensors_with_shape(shapes: list[tuple[int, ...]], dtype=torch.float32, **kwargs) -> list[torch.Tensor]:
    return [torch.rand(shape, dtype=dtype, **kwargs) for shape in shapes]


__all__ = ["merge_dicts", "compile_rust", "run_rust", "get_class_from_method",
           "rand_ndarrays_with_shape", "rand_tensors_with_shape"]
