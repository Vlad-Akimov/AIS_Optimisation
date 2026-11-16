"""Мультистарт, метрики, прогон эксперимента."""
import time
import numpy as np
from dataclasses import dataclass
from .functions import TestFunction
from .ais import ImmuneNetworkOptimizer, AISConfig


@dataclass
class RunMetrics:
    f_best: float
    n_iterations: int
    n_evaluations: int
    wall_time: float


def run_multistart(test_func: TestFunction, config: AISConfig,
                   n_runs: int = 30, seed_base: int = 0) -> list[RunMetrics]:
    """n_runs запусков с разными seed — методика мультистарта (стр. 12)."""
    results = []
    for r in range(n_runs):
        cfg = AISConfig(**{**config.__dict__, "seed": seed_base + r})
        optimizer = ImmuneNetworkOptimizer(test_func.func, test_func.bounds, test_func.dim, cfg)
        t0 = time.perf_counter()
        res = optimizer.optimize()
        wall = time.perf_counter() - t0
        results.append(RunMetrics(
            f_best=res.f_best,
            n_iterations=res.n_iterations,
            n_evaluations=res.n_evaluations,
            wall_time=wall,
        ))
    return results


def aggregate(results: list[RunMetrics], f_star: float, tol: float = 1e-4):
    """Средние метрики + надёжность (доля запусков, где f_best < f_star + tol)."""
    f_bests = np.array([r.f_best for r in results])
    n_iters = np.array([r.n_iterations for r in results])
    n_evals = np.array([r.n_evaluations for r in results])
    times = np.array([r.wall_time for r in results])

    success = f_bests < (f_star + tol) if f_star == 0 else \
              f_bests < (f_star + tol * abs(f_star))
    return {
        "f_best_mean": float(f_bests.mean()),
        "f_best_min": float(f_bests.min()),
        "f_best_std": float(f_bests.std()),
        "n_iter_mean": float(n_iters.mean()),
        "n_eval_mean": float(n_evals.mean()),
        "time_mean": float(times.mean()),
        "reliability": float(success.mean()),   # доля успешных запусков
    }
