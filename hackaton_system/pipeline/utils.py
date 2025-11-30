"""Простые утилиты для вычислений внутри пайплайна."""

from __future__ import annotations

import math
from typing import Any


def to_list(obj: Any) -> list[Any]:
    """Преобразует тензоры/списки numpy в обычный Python-список."""
    array = obj.cpu() if hasattr(obj, 'cpu') else obj
    if hasattr(array, 'tolist'):
        array = array.tolist()
    return list(array)


def center_of_bbox(
    bbox: tuple[int, int, int, int],
) -> tuple[float, float]:
    """Возвращает координаты центра прямоугольника."""
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def speed(
    prev: tuple[float, float], curr: tuple[float, float], duration: float
) -> float:
    """Считает скорость перемещения между двумя точками за промежуток."""
    if duration <= 0.0:
        return 0.0
    return math.hypot(curr[0] - prev[0], curr[1] - prev[1]) / duration
