"""Полный эксперимент по методике книги (стр. 202-204) + сравнение с RS."""
import sys, csv, time
sys.path.insert(0, "src")

import numpy as np
import matplotlib.pyplot as plt
from functions import get_function
from ais import AISConfig
from experiment import run_multistart, aggregate

# --- RS для baseline (из раздела 1.2.1 книги) ---
def random_search(func, bounds, dim, n_evals, seed):
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    best_f = np.inf
    for _ in range(n_evals):
        x = rng.uniform(lo, hi, size=dim)
        f = func(x)
        if f < best_f:
            best_f = f
    return best_f


if __name__ == "__main__":
    # Параметры эксперимента
    DIMS = [2, 4, 6, 8, 10]
    FUNCTIONS = ["rosenbrock", "rastrigin"]  # + himmelblau отдельно (dim=2)
    N_RUNS = 10                                # мультистарт
    N_C_VALUES = [50, 100, 150, 200]
    ALPHA_VALUES = [0.02, 0.04, 0.06, 0.08]

    # Базовая конфигурация (фиксированные параметры)
    base = AISConfig(population_size=50, n_best=10, n_clones=100,
                     n_keep_after_mutation=5, alpha=0.04,
                     stagnation_limit=20)

    # --- 1. Зависимость от n_c ---
    rows = []
    for fname in FUNCTIONS:
        for dim in DIMS:
            tf = get_function(fname, dim)
            for n_c in N_C_VALUES:
                cfg = AISConfig(**{**base.__dict__, "n_clones": n_c})
                res = run_multistart(tf, cfg, n_runs=N_RUNS)
                agg = aggregate(res, tf.f_star)
                rows.append({"func": fname, "dim": dim, "param": "n_c",
                             "value": n_c, **agg})
                print(f"{fname:12s} dim={dim:2d} n_c={n_c:3d} "
                      f"f*={agg['f_best_min']:.3e} evals={agg['n_eval_mean']:.0f} "
                      f"rel={agg['reliability']:.2f}")

    # --- 2. Зависимость от alpha ---
    for fname in FUNCTIONS:
        for dim in DIMS:
            tf = get_function(fname, dim)
            for a in ALPHA_VALUES:
                cfg = AISConfig(**{**base.__dict__, "alpha": a})
                res = run_multistart(tf, cfg, n_runs=N_RUNS)
                agg = aggregate(res, tf.f_star)
                rows.append({"func": fname, "dim": dim, "param": "alpha",
                             "value": a, **agg})
                print(f"{fname:12s} dim={dim:2d} alpha={a:.2f} "
                      f"f*={agg['f_best_min']:.3e} evals={agg['n_eval_mean']:.0f}")

    # --- 3. Сравнение с RS при том же числе испытаний ---
    print("\n--- Comparison with Random Search ---")
    for fname in FUNCTIONS:
        for dim in DIMS:
            tf = get_function(fname, dim)
            cfg = AISConfig(**base.__dict__)
            res = run_multistart(tf, cfg, n_runs=N_RUNS)
            ais_evals = int(np.mean([r.n_evaluations for r in res]))
            rs_f = [random_search(tf.func, tf.bounds, dim, ais_evals, s)
                    for s in range(N_RUNS)]
            print(f"{fname:12s} dim={dim:2d} | AIS: {np.mean([r.f_best for r in res]):.3e} "
                  f"| RS: {np.mean(rs_f):.3e} (evals={ais_evals})")

    # --- Сохраняем CSV ---
    with open("results/experiment.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved results/experiment.csv")
