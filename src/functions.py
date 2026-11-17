"""
Тестовые функции из Приложения Б книги Карпенко.
Все функции - для МИНИМИЗАЦИИ.
"""
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union


@dataclass
class FunctionInfo:
    """Метаданные тестовой функции."""
    name: str
    dim: int                                    # размерность (None = любая)
    bounds: Union[Tuple[float, float],          # единые границы для всех координат
                  List[Tuple[float, float]]]    # или по координатам
    f_star: float                               # известное значение f*
    x_star: List[np.ndarray]                    # ВСЕ известные точки минимума
    description: str = ""                       # краткое описание


class TestFunction(ABC):
    """Абстрактный базовый класс для тестовых функций."""

    def __init__(self, dim: int):
        self.dim = dim
        self.info = self._build_info()
        self._validate_dim()

    @abstractmethod
    def __call__(self, X: np.ndarray) -> float:
        """Вычисление значения функции в точке X."""
        ...

    @abstractmethod
    def _build_info(self) -> FunctionInfo:
        """Построение метаданных функции."""
        ...


    def _validate_dim(self) -> None:
        if self.info.dim is not None and self.dim != self.info.dim:
            raise ValueError(
                f"{self.info.name} определена только для dim={self.info.dim}, "
                f"получено dim={self.dim}"
            )

    def _get_bounds_array(self) -> np.ndarray:
        """Возвращает границы в виде массива shape (dim, 2)."""
        b = self.info.bounds
        if isinstance(b, tuple):
            return np.tile(np.array(b), (self.dim, 1))
        return np.array(b)

    def is_feasible(self, X: np.ndarray) -> bool:
        """Проверка попадания точки в допустимую область."""
        bounds = self._get_bounds_array()
        return bool(np.all((X >= bounds[:, 0]) & (X <= bounds[:, 1])))

    def clip(self, X: np.ndarray) -> np.ndarray:
        """Обрезка координат до границ области поиска."""
        bounds = self._get_bounds_array()
        return np.clip(X, bounds[:, 0], bounds[:, 1])

    def random_point(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Равномерно распределённая случайная точка в области поиска."""
        if rng is None:
            rng = np.random.default_rng()
        bounds = self._get_bounds_array()
        return rng.uniform(bounds[:, 0], bounds[:, 1], size=self.dim)

    def random_population(self, size: int,
                          rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Случайная популяция shape (size, dim)."""
        if rng is None:
            rng = np.random.default_rng()
        bounds = self._get_bounds_array()
        low = np.tile(bounds[:, 0], (size, 1))
        high = np.tile(bounds[:, 1], (size, 1))
        return rng.uniform(low, high)

    def distance_to_nearest_optimum(self, X: np.ndarray) -> float:
        """Евклидово расстояние до ближайшего известного оптимума."""
        X = np.asarray(X)
        return float(min(np.linalg.norm(X - x_opt) for x_opt in self.info.x_star))


class RosenbrockFunction(TestFunction):
    """
    Функция Розенброка ("овраг").
    f(X) = Σ [100·(x_{i+1} - x_i²)² + (1 - x_i)²] → min

    Область поиска: x_i ∈ [-2.048; 2.048]  (по п. 4.1.4 книги)
    Глобальный минимум: f* = 0 при x_i = 1 для всех i.
    """

    def __call__(self, X: np.ndarray) -> float:
        X = np.asarray(X).ravel()
        if X.size < 2:
            raise ValueError(f"Rosenbrock требует dim >= 2, получено {X.size}")
        return float(np.sum(100.0 * (X[1:] - X[:-1]**2)**2 + (1.0 - X[:-1])**2))

    def _build_info(self) -> FunctionInfo:
        return FunctionInfo(
            name="Rosenbrock",
            dim=self.dim,
            bounds=(-2.048, 2.048),
            f_star=0.0,
            x_star=[np.ones(self.dim)],
            description="Овражная функция, классический тест для популяционных алгоритмов",
        )


class HimmelblauFunction(TestFunction):
    """
    Функция Химмельблау.
    f(X) = (x₁² + x₂ - 11)² + (x₁ + x₂² - 7)² → min

    Область поиска: x₁, x₂ ∈ [-6; 6]
    Глобальный минимум: f* = 0 (4 точки!)
    """

    OPTIMA = [
        np.array([ 3.0,        2.0]),
        np.array([-2.805118,   3.131312]),
        np.array([ 3.584428,  -1.848126]),
        np.array([-3.779310,  -3.283186]),
    ]

    def __call__(self, X: np.ndarray) -> float:
        X = np.asarray(X).ravel()
        if X.size != 2:
            raise ValueError(f"Himmelblau требует dim = 2, получено {X.size}")
        x1, x2 = X[0], X[1]
        return float((x1**2 + x2 - 11.0)**2 + (x1 + x2**2 - 7.0)**2)

    def _build_info(self) -> FunctionInfo:
        return FunctionInfo(
            name="Himmelblau",
            dim=2,
            bounds=(-6.0, 6.0),
            f_star=0.0,
            x_star=self.OPTIMA.copy(),
            description="4 одинаковых глобальных минимума - проверяет способность ИИС находить несколько решений",
        )


class RastriginFunction(TestFunction):
    """
    Функция Растригина.
    f(X) = Σ [x_i² - 10·cos(2π·x_i) + 10] → min

    Область поиска: x_i ∈ [-5.12; 5.12]
    Глобальный минимум: f* = 0 при x_i = 0 для всех i.
    Сильная мультимодальность - много локальных минимумов.
    """

    def __call__(self, X: np.ndarray) -> float:
        X = np.asarray(X).ravel()
        if X.size < 1:
            raise ValueError("Rastrigin требует dim >= 1")
        return float(np.sum(X**2 - 10.0 * np.cos(2.0 * np.pi * X) + 10.0))

    def _build_info(self) -> FunctionInfo:
        return FunctionInfo(
            name="Rastrigin",
            dim=self.dim,
            bounds=(-5.12, 5.12),
            f_star=0.0,
            x_star=[np.zeros(self.dim)],
            description="Сильно мультимодальная функция - проверяет способность не застревать в локальных минимумах",
        )


FUNCTION_REGISTRY = {
    "rosenbrock": RosenbrockFunction,
    "himmelblau": HimmelblauFunction,
    "rastrigin":  RastriginFunction,
}


def get_function(name: str, dim: int = 2) -> TestFunction:
    """Фабрика: возвращает экземпляр тестовой функции по имени."""
    name_lower = name.lower()
    if name_lower not in FUNCTION_REGISTRY:
        available = ", ".join(FUNCTION_REGISTRY.keys())
        raise KeyError(f"Неизвестная функция: '{name}'. Доступные: {available}")
    return FUNCTION_REGISTRY[name_lower](dim=dim)


def list_functions() -> List[str]:
    """Список всех зарегистрированных функций."""
    return list(FUNCTION_REGISTRY.keys())
