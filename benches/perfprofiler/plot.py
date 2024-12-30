import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .result import Result

colors = {
    "NumPy":       "#4dabcf",
    "PyTorch CPU": "#f2765d",
    "PyTorch MPS": "#812ce5",
}


def plot_results(x, results: dict[str, list[Result]]) -> None:
    plt.figure()
    ax = plt.gca()

    for label, result in results.items():
        upper_bound = [res.mean + res.se() for res in result]
        lower_bound = [res.mean - res.se() for res in result]

        print(upper_bound, lower_bound)

        color = colors.get(label)

        ax.fill_between(x, upper_bound, lower_bound, color=color, alpha=0.2, ec=None)
        ax.plot(x, result, label=label, color=color)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.3f} ms"))
    plt.xscale("symlog")
    # plt.yscale("symlog")
    plt.legend()

    plt.show()


__all__ = ["plot_results"]
