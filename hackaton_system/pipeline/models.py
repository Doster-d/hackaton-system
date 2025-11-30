"""Общие структуры данных, которыми обмениваются компоненты пайплайна."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol


class DetectorProtocol(Protocol):
    """Минимальный интерфейс, который должен предоставлять YOLO-модель."""

    def track(self, *args: Any, **kwargs: Any) -> Iterable[Any]: ...


@dataclass(slots=True)
class DetectionResult:
    """Нормализованная запись о детекции и треке."""
    frame_id: int
    time_sec: float
    bbox: tuple[int, int, int, int]
    confidence: float
    track_id: int
    embedding: list[float] | None = None
    class_id: int = 0  # ID класса COCO (0 — человек, 6 — поезд и т. д.)


@dataclass(slots=True)
class PoseResult:
    """Результаты позовой модели (скелет) для трека."""
    frame_id: int
    time_sec: float
    track_id: int
    keypoints: list[dict[str, float]]
    confidence: float


@dataclass(slots=True)
class ActivityResult:
    """Итоговая активность для конкретного трека."""
    track_id: int
    activity_class: str
    start_sec: float
    end_sec: float
    confidence: float


@dataclass(slots=True)
class RoleAssignment:
    """Назначение роли/типа сотрудника по данным трека."""
    track_id: int
    person_type: str
    confidence: float


@dataclass(slots=True)
class FrameState:
    """Промежуточное состояние трека между двумя кадрами."""
    track_id: int
    label: str
    start_sec: float
    end_sec: float
