"""Вспомогательные утилиты проекта."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Создать директорию, если её нет. Возвращает путь."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_time(seconds: float) -> str:
    """Форматировать время в читаемый вид."""
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    """Среднее и стандартное отклонение."""
    arr = np.asarray(list(values))
    return float(arr.mean()), float(arr.std())
