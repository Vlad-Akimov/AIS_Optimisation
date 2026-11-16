"""Мультистарт, метрики, прогон эксперимента."""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

import numpy as np

from ais import ImmuneNetworkOptimizer, AISConfig, AISResult

if TYPE_CHECKING:
    from functions import TestFunction


@dataclass
class RunMetrics:
    """Метрики одного запуска."""
    f_best: float
    n_iterations: int
    n_evaluations: int
    wall_time: float


def run_multistart(
    test_func: "TestFunction",
    config: AISConfig,
    n_runs: int = 30,
    seed_base: int = 0,
) -> list[RunMetrics]:
    """n_runs запусков с разными seed — методика мультистарта (стр. 12 книги)."""
    results: list[RunMetrics] = []
    for r in range(n_runs):
        cfg_dict = asdict(config)
        cfg_dict["seed"] = seed_base + r
        cfg = AISConfig(**cfg_dict)

        optimizer = ImmuneNetworkOptimizer(test_func, cfg)
        t0 = time.perf_counter()
        res: AISResult = optimizer.optimize()
        wall = time.perf_counter() - t0

        results.append(
            RunMetrics(
                f_best=res.f_best,
                n_iterations=res.n_iterations,
                n_evaluations=res.n_evaluations,
                wall_time=wall,
            )
        )
    return results


def aggregate(
    results: list[RunMetrics], f_star: float, tol: float = 1e-4
) -> dict:
    """Средние метрики + надёжность (доля запусков, где f_best < f_star + tol)."""
    f_bests = np.array([r.f_best for r in results])
    n_iters = np.array([r.n_iterations for r in results])
    n_evals = np.array([r.n_evaluations for r in results])
    times = np.array([r.wall_time for r in results])

    # Для f_star == 0 используем абсолютный допуск, иначе относительный
    if f_star == 0:
        success = f_bests < tol
    else:
        success = f_bests < (f_star + tol * abs(f_star))

    return {
        "f_best_mean": float(f_bests.mean()),
        "f_best_min": float(f_bests.min()),
        "f_best_max": float(f_bests.max()),
        "f_best_std": float(f_bests.std()),
        "n_iter_mean": float(n_iters.mean()),
        "n_iter_std": float(n_iters.std()),
        "n_eval_mean": float(n_evals.mean()),
        "n_eval_std": float(n_evals.std()),
        "time_mean": float(times.mean()),
        "reliability": float(success.mean()),
    }
