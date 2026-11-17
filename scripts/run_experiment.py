"""Полный эксперимент по методике книги + сравнение с RS."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from functions import get_function, TestFunction
from ais import AISConfig
from experiment import run_multistart, aggregate


# --- RS для baseline (из раздела 1.2.1 книги) ---
def random_search(test_func: TestFunction, n_evals: int, seed: int) -> float:
    """Простейший случайный поиск - baseline."""
    rng = np.random.default_rng(seed)
    best_f = np.inf
    for _ in range(n_evals):
        x = test_func.random_point(rng)
        f = test_func(x)
        if f < best_f:
            best_f = f
    return best_f


def main() -> None:
    # Параметры эксперимента
    DIMS = [2, 4, 6, 8, 10]
    FUNCTIONS = ["rosenbrock", "rastrigin"]
    N_RUNS = 10
    N_C_VALUES = [50, 100, 150, 200]
    ALPHA_VALUES = [0.02, 0.04, 0.06, 0.08]

    # Базовая конфигурация
    base = AISConfig(
        population_size=50,
        n_best=10,
        n_clones=100,
        n_keep_after_mutation=5,
        alpha=0.04,
        stagnation_limit=20,
    )

    rows: list[dict] = []

    # --- 1. Зависимость от n_c ---
    print("=== Dependence on n_c ===")
    for fname in FUNCTIONS:
        for dim in DIMS:
            tf = get_function(fname, dim)
            for n_c in N_C_VALUES:
                cfg = AISConfig(**{**vars(base), "n_clones": n_c})
                res = run_multistart(tf, cfg, n_runs=N_RUNS)
                agg = aggregate(res, tf.info.f_star)
                rows.append({
                    "func": fname, "dim": dim,
                    "param": "n_c", "value": n_c, **agg,
                })
                print(
                    f"{fname:12s} dim={dim:2d} n_c={n_c:3d} "
                    f"f*={agg['f_best_min']:.3e} evals={agg['n_eval_mean']:.0f} "
                    f"rel={agg['reliability']:.2f}"
                )

    # --- 2. Зависимость от alpha ---
    print("\n=== Dependence on alpha ===")
    for fname in FUNCTIONS:
        for dim in DIMS:
            tf = get_function(fname, dim)
            for a in ALPHA_VALUES:
                cfg = AISConfig(**{**vars(base), "alpha": a})
                res = run_multistart(tf, cfg, n_runs=N_RUNS)
                agg = aggregate(res, tf.info.f_star)
                rows.append({
                    "func": fname, "dim": dim,
                    "param": "alpha", "value": a, **agg,
                })
                print(
                    f"{fname:12s} dim={dim:2d} alpha={a:.2f} "
                    f"f*={agg['f_best_min']:.3e} evals={agg['n_eval_mean']:.0f}"
                )

    # --- 3. Сравнение с RS при том же числе испытаний ---
    print("\n=== Comparison with Random Search ===")
    for fname in FUNCTIONS:
        for dim in DIMS:
            tf = get_function(fname, dim)
            cfg = AISConfig(**vars(base))
            res = run_multistart(tf, cfg, n_runs=N_RUNS)
            ais_evals = int(np.mean([r.n_evaluations for r in res]))
            rs_f = [random_search(tf, ais_evals, s) for s in range(N_RUNS)]
            ais_mean = np.mean([r.f_best for r in res])
            print(
                f"{fname:12s} dim={dim:2d} | AIS: {ais_mean:.3e} "
                f"| RS: {np.mean(rs_f):.3e} (evals={ais_evals})"
            )

    # --- Сохраняем CSV ---
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "experiment.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
