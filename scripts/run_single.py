"""Один запуск - проверить, что алгоритм вообще работает."""
import sys
sys.path.insert(0, "src")

from functions import get_function
from ais import ImmuneNetworkOptimizer, AISConfig

if __name__ == "__main__":
    tf = get_function("rosenbrock", dim=2)
    cfg = AISConfig(
        population_size=50, n_best=10, n_clones=100,
        n_keep_after_mutation=5, alpha=0.04,
        stagnation_limit=20, seed=42,
    )
    opt = ImmuneNetworkOptimizer(tf.func, tf.bounds, tf.dim, cfg)
    res = opt.optimize()
    print(f"Function: {tf.name}, dim={tf.dim}")
    print(f"f*     = {tf.f_star}")
    print(f"f_best = {res.f_best:.6e}")
    print(f"x_best = {res.x_best}")
    print(f"x*     = {tf.x_star}")
    print(f"iters  = {res.n_iterations}")
    print(f"evals  = {res.n_evaluations}")
