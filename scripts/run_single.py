"""Один запуск — проверить, что алгоритм вообще работает."""
from __future__ import annotations

import sys
from pathlib import Path

# Корректный путь к src через pathlib
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from functions import get_function
from ais import ImmuneNetworkOptimizer, AISConfig


def main() -> None:
    tf = get_function("rosenbrock", dim=2)
    cfg = AISConfig(
        population_size=50,
        n_best=10,
        n_clones=100,
        n_keep_after_mutation=5,
        alpha=0.04,
        stagnation_limit=20,
        seed=42,
    )
    opt = ImmuneNetworkOptimizer(tf, cfg)
    res = opt.optimize()

    print(f"Function : {tf.info.name}, dim={tf.dim}")
    print(f"f*       = {tf.info.f_star}")
    print(f"f_best   = {res.f_best:.6e}")
    print(f"x_best   = {res.x_best}")
    print(f"x*       = {tf.info.x_star[0]}")
    print(f"iters    = {res.n_iterations}")
    print(f"evals    = {res.n_evaluations}")
    print(f"dist_opt = {tf.distance_to_nearest_optimum(res.x_best):.3e}")


if __name__ == "__main__":
    main()
