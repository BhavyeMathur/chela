import subprocess
import json


def merge_dicts(list_of_dicts):
    result = {}
    for k in list_of_dicts[0].keys():
        result[k] = [d[k] for d in list_of_dicts]
    return result


def compile_rust(target: str) -> str:
    out = subprocess.check_output(f"$HOME/.cargo/bin/cargo build --release --bin {target} "
                                  "-v --message-format=json", shell=True)
    out = out.split(b"\n")[-3]
    return json.loads(out)["executable"]

def get_method_class(method) -> str:
    return method.__qualname__.split(".")[0]


__all__ = ["merge_dicts", "compile_rust", "get_method_class"]
