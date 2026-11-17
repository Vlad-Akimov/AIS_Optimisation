"""
Упрощённая искусственная иммунная сеть (п. 4.1.4 книги Карпенко).
Реализация для задачи МИНИМИЗАЦИИ.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from functions import TestFunction


@dataclass
class AISConfig:
    """Параметры алгоритма (стр. 202 книги)."""
    population_size: int = 50       # |S^b|
    n_best: int = 10                # n_b - лучших для клонирования
    n_clones: int = 100             # n_c - клонов на одно антитело
    n_keep_after_mutation: int = 5  # n_d - лучших клонов после мутации
    alpha: float = 0.04             # коэффициент мутации
    stagnation_limit: int = 20      # delta_t
    max_iterations: int = 10_000    # страховка от бесконечного цикла
    seed: int | None = None

    def __post_init__(self) -> None:
        """Валидация параметров - чтобы ошибки ловились сразу."""
        if self.population_size <= 0:
            raise ValueError("population_size must be > 0")
        if not (0 < self.n_best <= self.population_size):
            raise ValueError("n_best must be in (0, population_size]")
        if self.n_clones <= 0:
            raise ValueError("n_clones must be > 0")
        if not (0 < self.n_keep_after_mutation <= self.n_best * self.n_clones):
            raise ValueError(
                "n_keep_after_mutation must be in (0, n_best * n_clones]"
            )
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")
        if self.stagnation_limit <= 0:
            raise ValueError("stagnation_limit must be > 0")


@dataclass
class AISResult:
    """Результат одного запуска."""
    x_best: np.ndarray
    f_best: float
    n_iterations: int       # t_end - номер итерации стагнации
    n_evaluations: int      # n_f - число вычислений функции
    history: list[float] = field(default_factory=list)


class ImmuneNetworkOptimizer:
    """
    Упрощённая иммунная сеть (4.1.4).

    Отличия от полной модели 4.1.2:
      - только BG-аффинность (= фитнес);
      - нет BB-аффинности и подавления;
      - сжатие только по качеству решения.
    """

    def __init__(self, test_func: "TestFunction", config: AISConfig) -> None:
        self.test_func = test_func
        self.dim = test_func.dim

        # Извлекаем границы (для наших функций - единые для всех координат)
        bounds = test_func.info.bounds
        if isinstance(bounds, tuple):
            self.lo, self.hi = bounds
        else:
            self.lo = min(b[0] for b in bounds)
            self.hi = max(b[1] for b in bounds)

        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.n_evaluations = 0
        self.history: list[float] = []

    # ---------- служебные ----------

    def _evaluate(self, X: np.ndarray) -> float:
        """Однократное вычисление целевой функции (счётчик испытаний)."""
        self.n_evaluations += 1
        return float(self.test_func(X))

    def _evaluate_population(self, S: np.ndarray) -> np.ndarray:
        """Вычисление фитнеса для популяции."""
        return np.array([self._evaluate(x) for x in S])

    # ---------- шаги алгоритма ----------

    def _initialize(self) -> np.ndarray:
        """Шаг 1: случайная инициализация |S^b| антител."""
        return self.test_func.random_population(
            size=self.cfg.population_size, rng=self.rng
        )

    def _clone(self, S_best: np.ndarray) -> np.ndarray:
        """Шаг 2: клонирование n_b лучших по n_c копий."""
        return np.repeat(S_best, self.cfg.n_clones, axis=0)

    def _mutate(self, S_clones: np.ndarray) -> np.ndarray:
        """Шаг 3: мутация X += alpha * U(-0.5, 0.5).
        Выход за границы обрезаем через test_func.clip()."""
        noise = self.rng.uniform(-0.5, 0.5, size=S_clones.shape)
        S_mut = S_clones + self.cfg.alpha * noise
        return self.test_func.clip(S_mut)

    def _compress(
        self, S: np.ndarray, fitness: np.ndarray, n_keep: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Сжатие: оставить n_keep лучших (наименьший fitness)."""
        idx = np.argsort(fitness)[:n_keep]
        return S[idx], fitness[idx]

    # ---------- главный цикл ----------

    def optimize(self) -> AISResult:
        # Шаг 1: инициализация
        S = self._initialize()
        fitness = self._evaluate_population(S)
        f_best = float(np.min(fitness))
        self.history.append(f_best)
        stagnation = 0
        t_end = 0

        for t in range(1, self.cfg.max_iterations + 1):
            t_end = t
            # Шаг 2.1: отобрать n_b лучших антител
            S_best, _ = self._compress(S, fitness, self.cfg.n_best)
            # Шаг 2.2: клонировать
            S_clones = self._clone(S_best)
            # Шаг 3: мутировать
            S_mut = self._mutate(S_clones)
            # Шаг 4: сжать клоны до n_d лучших
            fit_mut = self._evaluate_population(S_mut)
            S_m, fit_m = self._compress(
                S_mut, fit_mut, self.cfg.n_keep_after_mutation
            )
            # Шаг 5: объединить S^b ∪ S^m и сжать до |S^b|
            S_all = np.vstack([S, S_m])
            fit_all = np.concatenate([fitness, fit_m])
            S, fitness = self._compress(S_all, fit_all, self.cfg.population_size)

            # Учёт стагнации
            new_best = float(fitness[0])
            if new_best < f_best - 1e-12:
                f_best = new_best
                stagnation = 0
            else:
                stagnation += 1

            self.history.append(f_best)

            # Шаг 6: условие остановки
            if stagnation >= self.cfg.stagnation_limit:
                break

        return AISResult(
            x_best=S[0].copy(),
            f_best=float(fitness[0]),
            n_iterations=t_end,
            n_evaluations=self.n_evaluations,
            history=self.history,
        )
