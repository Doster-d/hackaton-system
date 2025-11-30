"""Компоненты переидентификации для сшивания треков."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from hackaton_system.config import Settings
from hackaton_system.pipeline.models import DetectionResult

LOGGER = logging.getLogger(__name__)


class ReIDHandler:
    """Отвечает за переидентификацию и сшивание треков по эмбеддингам."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._reid_model: torch.nn.Module | None = None
        self._reid_device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self._reid_transform = transforms.Compose(
            [
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def stitch_tracks_with_reid(
        self, detections: list[DetectionResult]
    ) -> list[DetectionResult]:
        """Ставит одинаковые track_id для похожих треков с учётом ReID."""
        if not detections or not any(det.embedding for det in detections):
            return detections

        tracks = self._group_detections_by_track(detections)
        summaries = self._create_track_summaries(tracks)
        if not summaries:
            return detections

        mapping = self._compute_track_mapping(summaries, tracks)
        self._apply_track_mapping(detections, mapping)
        return detections

    def _group_detections_by_track(
        self, detections: list[DetectionResult]
    ) -> dict[int, list[DetectionResult]]:
        """Группирует детекции по track_id и сортирует их по времени."""
        grouped: dict[int, list[DetectionResult]] = defaultdict(list)
        for det in detections:
            grouped[det.track_id].append(det)
        for dets in grouped.values():
            dets.sort(key=lambda d: d.time_sec)
        return grouped

    def _create_track_summaries(
        self, tracks: dict[int, list[DetectionResult]]
    ) -> list[dict[str, Any]]:
        """Строит усреднённые эмбеддинги по каждому треку."""
        summaries: list[dict[str, Any]] = []
        for track_id, dets in tracks.items():
            summary = self._create_track_summary(track_id, dets)
            if summary is not None:
                summaries.append(summary)
        summaries.sort(key=lambda item: item['start'])
        return summaries

    def _create_track_summary(
        self, track_id: int, dets: list[DetectionResult]
    ) -> dict[str, Any] | None:
        """Возвращает нормализованный эмбеддинг одного трека."""
        embeddings = [
            np.array(det.embedding, dtype=np.float32)
            for det in dets
            if det.embedding
        ]
        if not embeddings:
            return None
        mean_vec = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_vec)
        if not norm:
            return None
        return {
            'track_id': track_id,
            'start': dets[0].time_sec,
            'end': dets[-1].time_sec,
            'embedding': mean_vec / norm,
        }

    def _compute_track_mapping(
        self,
        summaries: list[dict[str, Any]],
        tracks: dict[int, list[DetectionResult]],
    ) -> dict[int, int]:
        """Строит отображение track_id → новый track_id по близости эмбеддингов."""
        gap_limit = getattr(self.settings, 'reid_time_gap_sec', 2.5)
        similarity_thresh = getattr(
            self.settings, 'reid_similarity_threshold', 0.6
        )

        mapping: dict[int, int] = {track_id: track_id for track_id in tracks}
        history: list[dict[str, Any]] = []
        for summary in summaries:
            assigned = mapping[summary['track_id']]
            best_match_id = self._find_best_match(
                summary, history, gap_limit, similarity_thresh
            )
            if best_match_id is not None:
                mapping[summary['track_id']] = best_match_id
                summary['assigned_id'] = best_match_id
            else:
                summary['assigned_id'] = assigned
            history.append(summary)
        return mapping

    def _find_best_match(
        self,
        summary: dict[str, Any],
        history: list[dict[str, Any]],
        gap_limit: float,
        similarity_thresh: float,
    ) -> int | None:
        """Находит лучшего кандидата из истории с учётом порога."""
        best_match_id = None
        best_similarity = similarity_thresh
        for candidate in history:
            if not self._is_valid_candidate(summary, candidate, gap_limit):
                continue
            sim = float(np.dot(summary['embedding'], candidate['embedding']))
            if sim >= best_similarity:
                best_similarity = sim
                best_match_id = candidate['assigned_id']
        return best_match_id

    def _is_valid_candidate(
        self,
        summary: dict[str, Any],
        candidate: dict[str, Any],
        gap_limit: float,
    ) -> bool:
        """Проверяет, можно ли сравнивать текущий трек с кандидатом."""
        if summary['start'] < candidate['end']:
            return False
        gap = summary['start'] - candidate['end']
        return 0 <= gap <= gap_limit

    def _apply_track_mapping(
        self, detections: list[DetectionResult], mapping: dict[int, int]
    ) -> None:
        """Применяет новое значение track_id к каждой детекции."""
        for det in detections:
            det.track_id = mapping.get(det.track_id, det.track_id)

    def compute_track_embeddings(
        self, tracks: Mapping[int, list[DetectionResult]]
    ) -> dict[int, str]:
        """Сохраняет усреднённые эмбеддинги треков в текстовом виде."""
        result: dict[int, str] = {}
        for track_id, dets in tracks.items():
            embeddings = [
                np.array(det.embedding, dtype=np.float32)
                for det in dets
                if det.embedding
            ]
            if not embeddings:
                continue
            mean_vec = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(mean_vec)
            if not norm:
                continue
            normalized = (mean_vec / norm).tolist()
            result[track_id] = json.dumps(normalized)
        return result

    def extract_embedding(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> list[float] | None:
        """Вырезает область кадра и получает для неё эмбеддинг ReID."""
        model = self._ensure_reid_model()
        if model is None:
            return None

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None

        crop = frame[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = (
            self._reid_transform(pil_img).unsqueeze(0).to(self._reid_device)
        )
        with torch.no_grad():
            emb = model(tensor)
            emb = torch_functional.normalize(emb, dim=1)
        return emb.squeeze(0).cpu().tolist()

    def _ensure_reid_model(self) -> torch.nn.Module | None:
        """Лениво загружает модель ReID и возвращает её экземпляр."""
        if not getattr(self.settings, 'reid_enabled', True):
            return None
        if self._reid_model is None:
            try:
                model = resnet18(weights=ResNet18_Weights.DEFAULT)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning('Failed to load ReID backbone: %s', exc)
                return None
            model.fc = torch.nn.Identity()
            model.eval()
            model.to(self._reid_device)
            self._reid_model = model
        return self._reid_model
