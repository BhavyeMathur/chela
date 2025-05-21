from collections import defaultdict
import colorsys

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
import numpy as np

from .result import Result


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return r, g, b


def scale_lightness(hex_color: str, scale_l: float) -> tuple[float, float, float]:
    rgb = hex_to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=min(1, s * scale_l))


colors = {
    "NumPy":        "#4dabcf",
    "PyTorch CPU":  "#f2765d",
    "PyTorch MPS":  "#812ce5",
    "Chela CPU":    "#ce422b",
    "Chela Einsum": "#ba2323"
}


def plot_results(x, results: dict[str, list[Result]],
                 title: str,
                 xlog=True,
                 ylog=False) -> None:
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    plt.title(title)

    for label, result in results.items():
        upper_bound = [res.mean + res.se() for res in result]
        lower_bound = [res.mean - res.se() for res in result]

        color = colors.get(label)

        ax.fill_between(x, upper_bound, lower_bound, color=color, alpha=0.2, ec=None)
        ax.plot(x, result, label=label, color=color)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.3f} ms"))

    if xlog:
        plt.xscale("symlog")
    if ylog:
        plt.yscale("symlog")

    plt.legend()

    plt.show()


def plot_barplot(results: dict[str, dict[str, Result]],
                 title: str,
                 normalize: str = "NumPy") -> None:

    suites = []
    data: dict[str, np.ndarray] | defaultdict = defaultdict(list)
    errors = defaultdict(list)
    max_ = 0

    for suite, results in results.items():
        suites.append(suite.replace("->", "→"))

        for series, result in results.items():
            data[series].append(1000 * float(result))
            errors[series].append(1000 * result.se())

    normalizer = np.array(data[normalize])
    for (series, result), (_, error) in zip(data.items(), errors.items()):
        data[series] = np.array(result) / normalizer
        errors[series] = np.array(error) / normalizer
        max_ = max(max_, data[series].max())

    x = np.arange(len(suites))
    width = 0.25
    multiplier = 0

    fig, ax = plt.subplots(figsize=(max(7, int(1.5 * len(suites))), 7))
    ax.set_ylabel(f"Time (relative to {normalize})")
    ax.set_title(title)

    for (series, measurement), (_, error) in zip(data.items(), errors.items()):
        color = colors.get(series)
        ecolor = scale_lightness(color, 0.9)

        offset = width * multiplier

        rects = ax.bar(x + offset, height=measurement, width=width, color=color, label=series,
                       yerr=error, error_kw={"elinewidth": 1, "mew": 1, "capsize": 4, "ecolor": ecolor,
                                             "mfc":        ecolor, "mec": ecolor, "ms": 4, "fmt": "o"})
        ax.bar_label(rects, fmt="{:.2f}x", padding=3, size=6)
        multiplier += 1

    ax.set_xticks(x + width, suites, size=7)
    ax.set_ylim(0, min(1.1 * max_, 10))
    ax.legend(loc="best", ncols=3)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}x"))

    plt.show()


__all__ = ["plot_results", "plot_barplot"]
