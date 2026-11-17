"""Графики для отчёта (аналог рис. 4.1–4.3 книги)"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12})

COLORS = {2: "#1f77b4", 4: "#ff7f0e", 6: "#2ca02c", 8: "#d62728", 10: "#9467bd"}
MARKERS = {2: "o", 4: "s", 6: "^", 8: "D", 10: "v"}

STR_COLS = {"func", "param"}
INT_COLS = {"dim"}


def load_data(filename: str = "experiment.csv") -> Dict[str, List]:
    """Читает CSV из results/ (или по абсолютному пути)."""
    path = Path(filename)
    if not path.is_absolute():
        path = PROJECT_ROOT / "results" / filename
    if not path.exists():
        raise FileNotFoundError(f"Не найден {path}. Запустите run_experiment.py")

    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} пуст")

    data: Dict[str, List] = {k: [] for k in rows[0]}
    for row in rows:
        for k, v in row.items():
            if k in STR_COLS:
                data[k].append(v)
            elif k in INT_COLS:
                data[k].append(int(v))
            else:
                data[k].append(float(v))
    return data


def filter_data(data: Dict[str, List], **criteria) -> Dict[str, List]:
    mask = np.ones(len(data["func"]), dtype=bool)
    for key, val in criteria.items():
        mask &= np.array(data[key]) == val
    return {k: [v for v, m in zip(data[k], mask) if m] for k in data}


def fmt_value(param: str, v: float) -> str:
    """Подпись значения параметра: n_c — целое, alpha — 2 знака."""
    return f"{int(v)}" if param == "n_c" else f"{v:.2f}"


def param_label(param: str) -> str:
    return "Число клонов $n_c$" if param == "n_c" else "Коэффициент мутации α"


def save_figure(fig: plt.Figure, name: str) -> None:
    fig_dir = PROJECT_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"   {name}.png/pdf")


def plot_dependence(data: Dict[str, List], func_name: str, param: str) -> None:
    sub = filter_data(data, func=func_name, param=param)
    if not sub["func"]:
        print(f"   нет данных {func_name}/{param}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    short = "$n_c$" if param == "n_c" else "α"

    for dim in sorted(set(sub["dim"])):
        d = filter_data(sub, dim=dim)
        order = np.argsort(d["value"])
        x = np.array(d["value"], dtype=float)[order]
        kw = dict(marker=MARKERS.get(dim, "o"),
                  color=COLORS.get(dim, None),
                  label=f"|X|={dim}", linewidth=2, markersize=6)

        axes[0].plot(x, np.array(d["n_iter_mean"])[order], **kw)
        axes[1].plot(x, np.array(d["n_eval_mean"])[order] / 1000, **kw)
        axes[2].plot(x, np.array(d["f_best_min"])[order], **kw)

    specs = [
        ("Среднее число итераций $t_{end}$", f"(а) $t_{{end}}$ от {short}", False),
        ("Среднее число испытаний $n_f$ (тыс.)", f"(б) $n_f$ от {short}", False),
        ("Лучшее достигнутое $\\tilde{f}^*$", f"(в) $\\tilde{{f}}^*$ от {short}", True),
    ]
    for ax, (ylabel, title, log) in zip(axes, specs):
        ax.set_xlabel(param_label(param))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        if log:
            ax.set_yscale("log")
        if param == "n_c":
            ax.set_xticks(sorted(set(sub["value"])))
            ax.set_xticklabels([fmt_value(param, v) for v in sorted(set(sub["value"]))])

    fig.suptitle(f"{func_name.capitalize()}: зависимость критериев от {short}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, f"{func_name}_dependence_on_{param}")


def plot_reliability_heatmap(data: Dict[str, List], func_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, param in zip(axes, ["n_c", "alpha"]):
        sub = filter_data(data, func=func_name, param=param)
        if not sub["func"]:
            ax.set_title(f"нет данных {param}")
            continue

        dims = sorted(set(sub["dim"]))
        vals = sorted(set(sub["value"]))
        lookup = {(d, v): r for d, v, r in zip(sub["dim"], sub["value"], sub["reliability"])}
        matrix = np.array([[lookup.get((d, v), np.nan) for v in vals] for d in dims])

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(vals)), [fmt_value(param, v) for v in vals])
        ax.set_yticks(range(len(dims)), [str(d) for d in dims])
        ax.set_xlabel(param_label(param))
        ax.set_ylabel("Размерность |X|")
        ax.set_title(f"Надёжность vs {'$n_c$' if param == 'n_c' else 'α'}")
        ax.grid(False)

        for i in range(len(dims)):
            for j in range(len(vals)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                            color="black" if matrix[i, j] < 0.6 else "white", fontsize=9)
        fig.colorbar(im, ax=ax, label="Reliability")

    fig.suptitle(f"{func_name.capitalize()}: тепловая карта надёжности",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, f"{func_name}_reliability_heatmap")


def main() -> None:
    filename = sys.argv[1] if len(sys.argv) > 1 else "experiment.csv"
    print(f"Загрузка {filename} ...")
    data = load_data(filename)
    print(f" {len(data['func'])} строк\n")

    for func in sorted(set(data["func"])):
        print(f"{func.capitalize()}:")
        plot_dependence(data, func, "n_c")
        plot_dependence(data, func, "alpha")
        plot_reliability_heatmap(data, func)

    print("\nresults/figures/")


if __name__ == "__main__":
    main()
