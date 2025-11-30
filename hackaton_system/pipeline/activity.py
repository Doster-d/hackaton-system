"""Эвристики для определения активностей людей на кадре."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from hackaton_system.config import Settings, ZoneDefinition
from hackaton_system.pipeline.models import (
    ActivityResult,
    DetectionResult,
    FrameState,
)
from hackaton_system.pipeline.utils import center_of_bbox, speed


class ActivityInferrer:
    """Определяет активности людей на основе последовательностей детекций."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def infer_activities(
        self,
        tracks: Mapping[int, list[DetectionResult]],
        fps: float | None = None,
    ) -> list[ActivityResult]:
        """Эвристическое определение активностей под заводской сценарий."""
        if not tracks:
            return []

        effective_fps = fps if fps and fps > 0 else 25.0

        frame_states: list[FrameState] = []
        for track_id, dets in tracks.items():
            if not dets:
                continue
            frame_states.extend(
                self._classify_track_states(track_id, dets, effective_fps)
            )

        if not frame_states:
            return []

        activities: list[ActivityResult] = []
        min_duration = self.settings.activity_min_interval_sec

        frame_states.sort(key=lambda s: (s.track_id, s.start_sec))
        merged: list[FrameState] = []
        for state in frame_states:
            if not merged:
                merged.append(state)
                continue
            last = merged[-1]
            if (
                state.track_id == last.track_id
                and state.label == last.label
                and state.start_sec <= last.end_sec + 1e-3
            ):
                last.end_sec = max(last.end_sec, state.end_sec)
            else:
                merged.append(state)

        for state in merged:
            duration = state.end_sec - state.start_sec
            if duration < min_duration:
                continue
            activities.append(
                ActivityResult(
                    track_id=state.track_id,
                    activity_class=state.label,
                    start_sec=state.start_sec,
                    end_sec=state.end_sec,
                    confidence=1.0,
                )
            )

        return activities

    def _classify_track_states(
        self, track_id: int, detections: Sequence[DetectionResult], fps: float
    ) -> list[FrameState]:
        """Для каждой пары кадров вычисляет состояние (label) трека."""
        if len(detections) == 0:
            return []

        frame_step = 1.0 / fps if fps else 1.0 / 25.0
        states: list[FrameState] = []
        prev = detections[0]
        prev_center = center_of_bbox(prev.bbox)

        for curr in detections[1:]:
            curr_center = center_of_bbox(curr.bbox)
            duration = max(curr.time_sec - prev.time_sec, frame_step)
            speed_val = speed(prev_center, curr_center, duration)
            zone = self._zone_for_point(prev_center)
            label = self._activity_label(zone, speed_val)
            states.append(
                FrameState(
                    track_id=track_id,
                    label=label,
                    start_sec=prev.time_sec,
                    end_sec=prev.time_sec + duration,
                )
            )
            prev = curr
            prev_center = curr_center

        # Добавляем хвостовую секцию, чтобы покрыть последний кадр.
        tail_zone = self._zone_for_point(prev_center)
        states.append(
            FrameState(
                track_id=track_id,
                label=self._activity_label(tail_zone, 0.0),
                start_sec=prev.time_sec,
                end_sec=prev.time_sec + frame_step,
            )
        )
        return states

    def _zone_for_point(
        self, center: tuple[float, float]
    ) -> ZoneDefinition | None:
        """Ищет зону, внутри которой находится указанная точка."""
        x, y = center
        for zone in self.settings.zones:
            if zone.x_min <= x <= zone.x_max and zone.y_min <= y <= zone.y_max:
                return zone
        return None

    def _activity_label(
        self, zone: ZoneDefinition | None, speed: float
    ) -> str:
        """Подбирает текстовую метку активности в зависимости от зоны и скорости."""
        move = self.settings.activity_velocity_move_thresh
        idle = self.settings.activity_velocity_idle_thresh
        category = zone.zone_category if zone else None

        if category == 'restricted':
            return 'in_restricted_zone'
        if category == 'station':
            return self._station_activity_label(speed, move, idle)
        # Для коридора, визитёров или при отсутствии зоны.
        return self._movement_activity_label(speed, move, idle)

    def _station_activity_label(
        self, speed: float, move: float, idle: float
    ) -> str:
        """Возвращает метку активности для зоны станции."""
        if speed >= move:
            return 'working'
        if speed <= idle:
            return 'idle_at_station'
        # Промежуточные значения скорости относим к рабочему состоянию.
        return 'working'

    def _movement_activity_label(
        self, speed: float, move: float, idle: float
    ) -> str:
        """Возвращает метку активности для коридоров и «гостевых» зон."""
        if speed >= move:
            return 'walking'
        if speed <= idle:
            return 'standing'
        # При средней скорости считаем, что человек всё равно движется.
        return 'walking'
