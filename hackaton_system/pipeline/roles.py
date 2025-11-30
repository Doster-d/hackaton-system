"""Назначение ролей людям по истории их перемещений."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from hackaton_system.config import Settings, ZoneDefinition
from hackaton_system.pipeline.models import DetectionResult, RoleAssignment
from hackaton_system.pipeline.utils import center_of_bbox


class RoleInferrer:
    """Определяет тип сотрудника на основе зон, где он проводит время."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def infer_person_roles(
        self, tracks: Mapping[int, list[DetectionResult]], fps: float
    ) -> dict[int, RoleAssignment]:
        """Анализирует треки и возвращает наиболее вероятную роль."""
        assignments: dict[int, RoleAssignment] = {}
        for track_id, dets in tracks.items():
            if not dets:
                continue
            role_totals = self._accumulate_role_time(dets, fps)
            if not role_totals:
                assignments[track_id] = RoleAssignment(
                    track_id, 'unknown', 0.0
                )
                continue
            best_role, best_duration = max(
                role_totals.items(), key=lambda item: item[1]
            )
            total_time = sum(role_totals.values())
            confidence = best_duration / total_time if total_time else 0.0
            person_type = (
                best_role
                if confidence >= self.settings.role_assignment_threshold
                else 'unknown'
            )
            assignments[track_id] = RoleAssignment(
                track_id, person_type, confidence
            )
        return assignments

    def _accumulate_role_time(
        self, detections: Sequence[DetectionResult], fps: float
    ) -> dict[str, float]:
        """Считает, сколько времени человек провёл в каждой зоне."""
        frame_step = 1.0 / fps if fps else 1.0 / 25.0
        totals: defaultdict[str, float] = defaultdict(float)
        prev = detections[0]
        prev_center = center_of_bbox(prev.bbox)

        for curr in detections[1:]:
            duration = max(curr.time_sec - prev.time_sec, frame_step)
            zone = self._zone_for_point(prev_center)
            role_name = self._role_name_for_zone(zone)
            totals[role_name] += duration
            prev = curr
            prev_center = center_of_bbox(curr.bbox)

        role_name = self._role_name_for_zone(self._zone_for_point(prev_center))
        totals[role_name] += frame_step
        return totals

    def _zone_for_point(
        self, center: tuple[float, float]
    ) -> ZoneDefinition | None:
        """Возвращает зону, в которой находится точка, или None."""
        x, y = center
        for zone in self.settings.zones:
            if zone.x_min <= x <= zone.x_max and zone.y_min <= y <= zone.y_max:
                return zone
        return None

    def _role_name_for_zone(self, zone: ZoneDefinition | None) -> str:
        """Преобразует зону в целевой label роли."""
        if zone is None:
            return 'visitor'
        if zone.person_type == 'restricted':
            return 'visitor'
        if not zone.person_type:
            return 'visitor'
        return zone.person_type
