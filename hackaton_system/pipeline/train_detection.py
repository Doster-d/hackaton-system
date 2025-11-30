"""Обнаружение поездов и определение событий прибытия/отправления."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from hackaton_system.config import Settings
from hackaton_system.pipeline.models import DetectorProtocol
from hackaton_system.pipeline.videollama3 import VideoLLaMA3ActionInferrer

LOGGER = logging.getLogger(__name__)

# В COCO класс 6 соответствует поезду (train); 0 — человек.
TRAIN_CLASS_IDS = [6]  # учитываем только поезд.


@dataclass(slots=True)
class TrainDetection:
    """Одна детекция поезда от YOLO."""

    frame_id: int
    time_sec: float
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int


@dataclass(slots=True)
class TrainEventData:
    """Событие прибытия, отправления или присутствия поезда."""

    event_type: str  # Значения: arrival, departure или presence.
    time_sec: float
    frame_id: int
    confidence: float
    bbox: tuple[int, int, int, int] | None = None


class TrainDetector:
    """Обнаруживает поезда и фиксирует события с помощью VideoLLaMA3/эвристик."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._min_confidence = getattr(settings, 'train_detection_conf', 0.3)
        self._min_frames_present = getattr(
            settings, 'train_min_frames_present', 10
        )
        self._max_frames_absent = getattr(
            settings, 'train_max_frames_absent', 30
        )
        self._use_videollama3 = getattr(
            settings, 'train_videollama3_enabled', True
        )
        # Поддержка старых настроек, где использовался X-CLIP.
        if not self._use_videollama3:
            self._use_videollama3 = getattr(
                settings, 'train_xclip_enabled', True
            )
        self._action_inferrer: VideoLLaMA3ActionInferrer | None = None

    def detect_trains(
        self, detector: DetectorProtocol, video_path: Path, fps: float
    ) -> list[TrainDetection]:
        """Запускает YOLO для поиска поездов на видео."""
        detections: list[TrainDetection] = []
        frame_idx = 0

        try:
            results_stream = self._get_train_detection_stream(
                detector, video_path
            )
            for result in results_stream:
                frame_detections = self._parse_train_result(
                    result, frame_idx, fps
                )
                detections.extend(frame_detections)
                frame_idx += 1

        except Exception as exc:
            LOGGER.error('Train detection failed: %s', exc, exc_info=True)
            return []

        LOGGER.info(
            'Train detection complete: %s frames processed, %s detections',
            frame_idx,
            len(detections),
        )
        return detections

    def _get_train_detection_stream(
        self, model: DetectorProtocol, video_path: Path
    ) -> Iterable[Any]:
        """Возвращает поток детекций YOLO, ограниченный классом поезда."""
        return model.track(
            source=str(video_path),
            tracker=self.settings.tracker_config_path,
            stream=True,
            save=False,
            show=False,
            verbose=False,
            classes=TRAIN_CLASS_IDS,
            conf=self._min_confidence,
            iou=self.settings.detection_iou,
            imgsz=self.settings.detection_imgsz,
        )

    def _parse_train_result(
        self, result: Any, frame_idx: int, fps: float
    ) -> list[TrainDetection]:
        """Разбирает результат YOLO для конкретного кадра."""
        boxes: Any = getattr(result, 'boxes', None)
        if boxes is None:
            return []

        xyxy = self._extract_xyxy(boxes)
        if xyxy is None:
            return []

        confs = self._extract_confidences(boxes, len(xyxy))
        classes = self._extract_classes(boxes, len(xyxy))
        frame_time = frame_idx / fps if fps else frame_idx

        detections: list[TrainDetection] = []
        for bbox, conf, cls_id in zip(xyxy, confs, classes, strict=True):
            if float(conf) < self._min_confidence:
                continue
            bbox_tuple = cast(
                tuple[int, int, int, int],
                tuple(int(coord) for coord in bbox),
            )
            detections.append(
                TrainDetection(
                    frame_id=frame_idx,
                    time_sec=frame_time,
                    bbox=bbox_tuple,
                    confidence=float(conf),
                    class_id=int(cls_id),
                )
            )

        return detections

    def _extract_xyxy(self, boxes: Any) -> list[list[float]] | None:
        """Извлекает координаты рамок."""
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            return xyxy.tolist()
        except Exception:
            return None

    def _extract_confidences(self, boxes: Any, count: int) -> list[float]:
        """Возвращает список confidences или заполняет нулями."""
        try:
            confs = boxes.conf.cpu().numpy()
            return confs.tolist()
        except Exception:
            return [0.0] * count

    def _extract_classes(self, boxes: Any, count: int) -> list[int]:
        """Возвращает классы детекций."""
        try:
            classes = boxes.cls.cpu().numpy()
            return classes.astype(int).tolist()
        except Exception:
            return [0] * count

    def detect_arrival_departure(
        self,
        detections: list[TrainDetection],
        video_path: Path | None = None,
        fps: float | None = None,
    ) -> list[TrainEventData]:
        """Определяет прибытия, отправления и присутствие поездов."""
        if not detections:
            return []

        if self._use_videollama3 and video_path and fps:
            return self._detect_with_videollama3(detections, video_path, fps)
        return self._detect_with_heuristics(detections)

    def _detect_with_videollama3(
        self,
        detections: list[TrainDetection],
        video_path: Path,
        fps: float,
    ) -> list[TrainEventData]:
        """Использует VideoLLaMA3 для классификации событий поезда."""
        if self._action_inferrer is None:
            from hackaton_system.pipeline.videollama3 import (
                get_shared_videollama3_instance,
            )

            self._action_inferrer = get_shared_videollama3_instance(
                self.settings
            )

        prompts = getattr(self.settings, 'train_prompts', [])
        if not prompts:
            LOGGER.warning('No train prompts configured, using heuristics')
            return self._detect_with_heuristics(detections)

        # Переводим детекции поездов в формат DetectionResult.
        # Группируем по кадрам и формируем треки.
        frames_with_train: dict[int, TrainDetection] = {}
        for det in detections:
            existing = frames_with_train.get(det.frame_id)
            if existing is None or det.confidence > existing.confidence:
                frames_with_train[det.frame_id] = det

        if not frames_with_train:
            return []

        # Создаём единый трек из всех детекций поезда.
        from hackaton_system.pipeline.models import DetectionResult

        train_tracks: dict[int, list[DetectionResult]] = {}
        track_id = 0  # Один трек, описывающий поезд целиком.
        for frame_id, det in sorted(frames_with_train.items()):
            if track_id not in train_tracks:
                train_tracks[track_id] = []
            train_tracks[track_id].append(
                DetectionResult(
                    frame_id=det.frame_id,
                    time_sec=det.time_sec,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    track_id=track_id,
                    class_id=det.class_id,
                )
            )

        # Используем общий инферер VideoLLaMA3 для действий.
        activities = self._action_inferrer.infer_actions(
            video_path, train_tracks, target_class='train'
        )

        # Преобразуем активности в события поездов.
        events: list[TrainEventData] = []
        previous_state: str | None = None

        for activity in activities:
            # Ищем ближайшую детекцию для времени активности.
            frame_id = int(activity.start_sec * fps) if fps else 0
            det = frames_with_train.get(frame_id)
            if det is None:
                # Если точного кадра нет — берём ближайший.
                closest_frame = min(
                    frames_with_train.keys(),
                    key=lambda f: abs(f - frame_id),
                )
                det = frames_with_train[closest_frame]

                # Определяем тип события по смене состояния.
                state = SimpleNamespace(
                    label=activity.activity_class,
                    confidence=activity.confidence,
                )
                event_type = self._determine_event_type(previous_state, state)
                if event_type:
                    events.append(
                        TrainEventData(
                            event_type=event_type,
                            time_sec=activity.start_sec,
                            frame_id=det.frame_id,
                            confidence=activity.confidence,
                            bbox=det.bbox,
                        )
                    )

                # Добавляем события presence для стоящих поездов.
                presence_label = (
                    'train is present and stationary at the station'
                )
                if activity.activity_class == presence_label:
                    events.append(
                        TrainEventData(
                            event_type='presence',
                            time_sec=activity.start_sec,
                            frame_id=det.frame_id,
                            confidence=activity.confidence,
                            bbox=det.bbox,
                        )
                    )

            previous_state = activity.activity_class

        LOGGER.info(
            'Detected %s train events via VideoLLaMA3: %s arrivals, '
            '%s departures, %s presence',
            len(events),
            sum(1 for e in events if e.event_type == 'arrival'),
            sum(1 for e in events if e.event_type == 'departure'),
            sum(1 for e in events if e.event_type == 'presence'),
        )

        return events

    def _detect_with_heuristics(
        self, detections: list[TrainDetection], fps: float | None = None
    ) -> list[TrainEventData]:
        """Фиксирует события поезда с помощью простых эвристик."""
        # Группируем детекции по кадрам, чтобы отслеживать присутствие поезда.
        frames_with_train: dict[int, list[TrainDetection]] = {}
        for det in detections:
            if det.frame_id not in frames_with_train:
                frames_with_train[det.frame_id] = []
            frames_with_train[det.frame_id].append(det)

        if not frames_with_train:
            return []

        sorted_frames = sorted(frames_with_train.keys())
        events: list[TrainEventData] = []

        # Следим за состоянием «поезд присутствует/отсутствует».
        train_present = False
        consecutive_absent_frames = 0
        last_present_frame = None

        # Не считаем прибытие, если поезд в кадре с самого начала (до ~1 секунды).
        first_second_frames = int(fps) if fps else 30

        for frame_idx in sorted_frames:
            if frame_idx in frames_with_train:
                # Поезд виден в текущем кадре.
                if not train_present:
                    # Поезд только появился, учитываем прибытие.
                    # Но игнорируем, если это самые первые кадры.
                    if frame_idx > first_second_frames:
                        dets = frames_with_train[frame_idx]
                        best_det = max(dets, key=lambda d: d.confidence)
                        events.append(
                            TrainEventData(
                                event_type='arrival',
                                time_sec=best_det.time_sec,
                                frame_id=frame_idx,
                                confidence=best_det.confidence,
                                bbox=best_det.bbox,
                            )
                        )
                    train_present = True
                else:
                    # Поезд стоит — периодически фиксируем presence.
                    dets = frames_with_train[frame_idx]
                    best_det = max(dets, key=lambda d: d.confidence)
                    # Добавляем presence каждые N кадров.
                    if frame_idx % 30 == 0:
                        events.append(
                            TrainEventData(
                                event_type='presence',
                                time_sec=best_det.time_sec,
                                frame_id=frame_idx,
                                confidence=best_det.confidence,
                                bbox=best_det.bbox,
                            )
                        )
                consecutive_absent_frames = 0
                last_present_frame = frame_idx
            # Поезда нет в кадре.
            elif train_present:
                consecutive_absent_frames += 1
                if consecutive_absent_frames >= self._max_frames_absent:
                    # Считаем, что поезд уехал.
                    if last_present_frame is not None:
                        last_dets = frames_with_train[last_present_frame]
                        best_det = max(last_dets, key=lambda d: d.confidence)
                        events.append(
                            TrainEventData(
                                event_type='departure',
                                time_sec=best_det.time_sec,
                                frame_id=last_present_frame,
                                confidence=best_det.confidence,
                                bbox=best_det.bbox,
                            )
                        )
                    train_present = False
                    consecutive_absent_frames = 0

        LOGGER.info(
            'Detected %s train events (heuristic): %s arrivals, '
            '%s departures, %s presence',
            len(events),
            sum(1 for e in events if e.event_type == 'arrival'),
            sum(1 for e in events if e.event_type == 'departure'),
            sum(1 for e in events if e.event_type == 'presence'),
        )

        return events

    def _determine_event_type(
        self, previous_state: str | None, current_state: SimpleNamespace
    ) -> str | None:
        """Определяет тип события по смене предикта VideoLLaMA3."""
        current_label = current_state.label

        if previous_state is None:
            if 'arriving' in current_label.lower():
                return 'arrival'
            return None

        current_lower = current_label.lower()
        prev_lower = previous_state.lower() if previous_state else ''
        if 'arriving' in current_lower and 'arriving' not in prev_lower:
            return 'arrival'
        if 'departing' in current_lower and 'departing' not in prev_lower:
            return 'departure'

        return None
