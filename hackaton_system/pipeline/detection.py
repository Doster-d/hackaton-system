"""Групповые функции, которые запускают YOLO-детектор и разбирают результаты."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from hackaton_system.config import DETECTION_CLASSES_DEFAULT, Settings
from hackaton_system.pipeline.models import (
    DetectionResult,
    DetectorProtocol,
    PoseResult,
)
from hackaton_system.pipeline.utils import to_list

LOGGER = logging.getLogger(__name__)


class DetectionHandler:
    """Оборачивает вызов YOLO и преобразование результатов в наши структуры."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run_detection(
        self, detector: DetectorProtocol, video_path: Path, fps: float
    ) -> tuple[list[DetectionResult], list[PoseResult]]:
        """Выполняем детекцию + трекинг через Ultralytics YOLO."""
        detections: list[DetectionResult] = []
        poses: list[PoseResult] = []
        frame_idx = 0

        try:
            results_stream = self._get_detection_stream(detector, video_path)
            for result in results_stream:
                parsed = self._parse_detection_result(result, frame_idx, fps)
                if parsed is None:
                    frame_idx += 1
                    continue
                frame_detections, frame_poses = parsed
                detections.extend(frame_detections)
                poses.extend(frame_poses)
                frame_idx += 1

        except Exception as exc:
            LOGGER.error('Detection failed: %s', exc, exc_info=True)
            return [], []

        LOGGER.info(
            'Detection complete: %s frames processed, '
            '%s tracks, %s pose entries',
            frame_idx,
            len(detections),
            len(poses),
        )
        return detections, poses

    def _get_detection_stream(
        self, model: DetectorProtocol, video_path: Path
    ) -> Iterable[Any]:
        """Возвращает генератор результатов YOLO для заданного видео."""
        # Берём список классов из настроек, иначе используем значения по умолчанию.
        classes = getattr(
            self.settings, 'detection_classes', DETECTION_CLASSES_DEFAULT.copy()
        )
        if not classes:
            classes = DETECTION_CLASSES_DEFAULT.copy()

        return model.track(
            source=str(video_path),
            tracker=self.settings.tracker_config_path,
            stream=True,
            save=False,
            show=False,
            verbose=False,
            classes=classes,
            conf=self.settings.detection_conf,
            iou=self.settings.detection_iou,
            imgsz=self.settings.detection_imgsz,
        )

    def _parse_detection_result(
        self, result: Any, frame_idx: int, fps: float
    ) -> tuple[list[DetectionResult], list[PoseResult]] | None:
        """Разбирает результат трекинга для одного кадра."""
        boxes: Any = getattr(result, 'boxes', None)
        if boxes is None:
            return None
        xyxy = self._extract_xyxy(boxes)
        if xyxy is None:
            return None

        confs = self._extract_confidences(boxes, len(xyxy))
        classes = self._extract_classes(boxes, len(xyxy))
        track_ids = self._extract_track_ids(boxes, len(xyxy))
        keypoints_xy, keypoints_conf = self._extract_keypoints(
            result, len(xyxy)
        )

        frame_time = frame_idx / fps if fps else frame_idx

        detections: list[DetectionResult] = []
        poses: list[PoseResult] = []

        for bbox, conf, cls_id, track_id, kp_xy, kp_conf in zip(
            xyxy,
            confs,
            classes,
            track_ids,
            keypoints_xy,
            keypoints_conf,
            strict=True,
        ):
            if not self._is_valid_detection(cls_id, track_id, conf):
                continue
            bbox_tuple = cast(
                tuple[int, int, int, int],
                tuple(int(coord) for coord in bbox),
            )
            detections.append(
                DetectionResult(
                    frame_id=frame_idx,
                    time_sec=frame_time,
                    bbox=bbox_tuple,
                    confidence=float(conf),
                    track_id=int(track_id),
                    embedding=None,  # Позже заполнится блоком ReID.
                    class_id=int(cls_id) if cls_id is not None else 0,
                )
            )
            if kp_xy is not None:
                pose = self._create_pose_result(
                    frame_idx, frame_time, track_id, kp_xy, kp_conf, conf
                )
                poses.append(pose)

        return detections, poses

    def _extract_xyxy(self, boxes: Any) -> list[list[float]] | None:
        """Извлекает координаты прямоугольников xyxy."""
        xyxy_attr = getattr(boxes, 'xyxy', None)
        if xyxy_attr is None:
            return None
        xyxy_source = (
            xyxy_attr.cpu() if hasattr(xyxy_attr, 'cpu') else xyxy_attr
        )
        xyxy_data = (
            xyxy_source.tolist()
            if hasattr(xyxy_source, 'tolist')
            else xyxy_source
        )
        return cast(list[list[float]], xyxy_data)

    def _extract_confidences(self, boxes: Any, count: int) -> list[float]:
        """Возвращает confidence для каждой детекции (или заполняет нулями)."""
        conf_attr = getattr(boxes, 'conf', None)
        if conf_attr is not None:
            return cast(list[float], to_list(conf_attr))
        return [0.0] * count

    def _extract_classes(self, boxes: Any, count: int) -> list[int | None]:
        """Возвращает классы объектов из результата YOLO."""
        cls_attr = getattr(boxes, 'cls', None)
        if cls_attr is not None:
            return cast(list[int | None], to_list(cls_attr))
        return [0 for _ in range(count)]

    def _extract_track_ids(self, boxes: Any, count: int) -> list[int | None]:
        """Извлекает track_id, присвоенные трекером."""
        id_attr = getattr(boxes, 'id', None)
        if id_attr is not None:
            return cast(list[int | None], to_list(id_attr))
        return [None for _ in range(count)]

    def _extract_keypoints(
        self, result: Any, count: int
    ) -> tuple[list[list[list[float]] | None], list[list[float] | None]]:
        """Возвращает массивы точек позы и confidence, если они есть."""
        keypoints_attr = getattr(result, 'keypoints', None)
        if keypoints_attr is None:
            return [None] * count, [None] * count

        kp_xy_attr = getattr(keypoints_attr, 'xy', None)
        kp_conf_attr = getattr(keypoints_attr, 'conf', None)

        keypoints_xy = self._parse_keypoint_xy(kp_xy_attr, count)
        keypoints_conf = self._parse_keypoint_conf(kp_conf_attr, count)

        return keypoints_xy, keypoints_conf

    def _parse_keypoint_xy(
        self, kp_xy_attr: Any, count: int
    ) -> list[list[list[float]] | None]:
        """Преобразует координаты keypoints в обычные списки."""
        if kp_xy_attr is None:
            return [None] * count
        kp_xy_source = (
            kp_xy_attr.cpu() if hasattr(kp_xy_attr, 'cpu') else kp_xy_attr
        )
        kp_xy_data = (
            kp_xy_source.tolist()
            if hasattr(kp_xy_source, 'tolist')
            else kp_xy_source
        )
        keypoints_xy = cast(list[list[list[float]]], kp_xy_data)
        if len(keypoints_xy) != count:
            return [None] * count
        return keypoints_xy

    def _parse_keypoint_conf(
        self, kp_conf_attr: Any, count: int
    ) -> list[list[float] | None]:
        """Преобразует confidence точек позы к спискам."""
        if kp_conf_attr is None:
            return [None] * count
        kp_conf_source = (
            kp_conf_attr.cpu()
            if hasattr(kp_conf_attr, 'cpu')
            else kp_conf_attr
        )
        kp_conf_data = (
            kp_conf_source.tolist()
            if hasattr(kp_conf_source, 'tolist')
            else kp_conf_source
        )
        keypoints_conf = cast(list[list[float]], kp_conf_data)
        if len(keypoints_conf) != count:
            return [None] * count
        return keypoints_conf

    def _is_valid_detection(
        self, cls_id: int | None, track_id: int | None, conf: float
    ) -> bool:
        """Проверяет, разрешён ли класс и хватает ли уверенности."""
        # Получаем список разрешённых классов.
        allowed_classes = getattr(
            self.settings, 'detection_classes', DETECTION_CLASSES_DEFAULT.copy()
        )
        if not allowed_classes:
            allowed_classes = DETECTION_CLASSES_DEFAULT.copy()

        # Убеждаемся, что класс разрешён и есть track_id.
        if cls_id is not None and int(cls_id) not in allowed_classes:
            return False
        if track_id is None:
            return False
        return conf >= self.settings.detection_conf

    def _create_pose_result(
        self,
        frame_idx: int,
        frame_time: float,
        track_id: int,
        kp_xy: list[list[float]],
        kp_conf: list[float] | None,
        default_conf: float,
    ) -> PoseResult:
        """Создаёт PoseResult, нормализуя массивы точек и confident scores."""
        confidences = kp_conf or []
        valid_scores = [score for score in confidences if score is not None]
        avg_conf = (
            float(sum(valid_scores) / len(valid_scores))
            if valid_scores
            else float(default_conf)
        )
        padded_conf = confidences if confidences else [None] * len(kp_xy)
        keypoints_payload: list[dict[str, float]] = []
        for idx_point, point in enumerate(kp_xy):
            entry: dict[str, float] = {
                'x': float(point[0]),
                'y': float(point[1]),
            }
            score = (
                padded_conf[idx_point]
                if idx_point < len(padded_conf)
                else None
            )
            if score is not None:
                entry['confidence'] = float(score)
            keypoints_payload.append(entry)
        return PoseResult(
            frame_id=frame_idx,
            time_sec=frame_time,
            track_id=int(track_id),
            keypoints=keypoints_payload,
            confidence=avg_conf,
        )
