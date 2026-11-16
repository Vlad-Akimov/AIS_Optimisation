"""
Тестовые функции из Приложения Б книги Карпенко.
Все функции — для МИНИМИЗАЦИИ.
"""
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class TestFunction:
    name: str
    func: Callable[[np.ndarray], float]
    bounds: tuple[float, float]   # [x_min, x_max] для всех координат
    dim: int | None               # None = любая размерность
    x_star: np.ndarray | None     # эталонное решение (если известно)
    f_star: float                 # эталонное значение


def rosenbrock(X: np.ndarray) -> float:
    """Функция Розенброка (классическая форма).
    В книге (стр. 428) есть опечатка в индексах — используем классическую."""
    return float(np.sum(100.0 * (X[1:] - X[:-1]**2)**2 + (1.0 - X[:-1])**2))


def himmelblau(X: np.ndarray) -> float:
    """Функция Химмельблау. 4 глобальных минимума, f* = 0."""
    x1, x2 = X[0], X[1]
    return float((x1**2 + x2 - 11.0)**2 + (x1 + x2**2 - 7.0)**2)


def rastrigin(X: np.ndarray) -> float:
    """Функция Растригина. Мультимодальная, f* = 0 при x_i = 0."""
    n = len(X)
    return float(np.sum(X**2 - 10.0 * np.cos(2.0 * np.pi * X) + 10.0))


def get_function(name: str, dim: int) -> TestFunction:
    """Фабрика: возвращает функцию с нужной размерностью и эталоном."""
    if name == "rosenbrock":
        return TestFunction(
            name="Rosenbrock",
            func=rosenbrock,
            bounds=(-100.0, 100.0),
            dim=dim,
            x_star=np.ones(dim),
            f_star=0.0,
        )
    if name == "himmelblau":
        if dim != 2:
            raise ValueError("Himmelblau определена только для dim=2")
        return TestFunction(
            name="Himmelblau",
            func=himmelblau,
            bounds=(-6.0, 6.0),
            dim=2,
            x_star=np.array([3.0, 2.0]),  # один из 4 минимумов
            f_star=0.0,
        )
    if name == "rastrigin":
        return TestFunction(
            name="Rastrigin",
            func=rastrigin,
            bounds=(-5.0, 5.0),
            dim=dim,
            x_star=np.zeros(dim),
            f_star=0.0,
        )
    raise ValueError(f"Unknown function: {name}")
