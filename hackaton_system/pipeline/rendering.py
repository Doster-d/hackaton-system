"""Отрисовка превьюшек и визуализация результатов детекции."""

from __future__ import annotations

import contextlib
import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hackaton_system.config import Settings
from hackaton_system.pipeline.models import DetectionResult
from hackaton_system.utils.ffmpeg_subprocess import start_ffmpeg_writer

LOGGER = logging.getLogger(__name__)


class VideoRenderer:
    """Отвечает за отрисовку превью с рамками и ID треков."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def render_preview(
        self,
        source: Path,
        video_id: int,
        detections: Sequence[DetectionResult],
        fps: float,
    ) -> Path | None:
        """Создаёт отдельный файл-визуализацию с боксами и треками."""
        if not source.exists():
            return None

        render_config = self._prepare_render_config(source, fps)
        if render_config is None:
            return None

        output_path = self._prepare_output_path(video_id, source)
        process = self._create_ffmpeg_process(render_config, output_path)
        if process is None:
            return None

        frame_map = self._build_frame_map(detections)
        self._render_frames(render_config, frame_map, process)

        return output_path if output_path.exists() else None

    def _prepare_render_config(
        self, source: Path, fps: float
    ) -> dict[str, Any] | None:
        """Формирует конфигурацию рендера: размеры, шаг кадров и т. п."""
        output_dir = getattr(
            self.settings, 'video_output_dir', Path('runs/visualizations')
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_step = max(
            1, int(getattr(self.settings, 'preview_frame_step', 2))
        )
        max_side = max(1, int(getattr(self.settings, 'preview_max_side', 640)))

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps_src = fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
        if width <= 0 or height <= 0:
            cap.release()
            return None

        target_width, target_height, scale = self._calculate_target_size(
            width, height, max_side
        )
        fps_out = max(fps_src / frame_step, 1.0)

        return {
            'cap': cap,
            'width': width,
            'height': height,
            'target_width': target_width,
            'target_height': target_height,
            'scale': scale,
            'fps_out': fps_out,
            'frame_step': frame_step,
        }

    def _calculate_target_size(
        self, width: int, height: int, max_side: int
    ) -> tuple[int, int, float]:
        """Подбирает итоговые размеры видео с учётом ограничения по стороне."""
        scale = 1.0
        if max(width, height) > max_side:
            scale = max_side / max(width, height)
        target_width = max(1, round(width * scale))
        target_height = max(1, round(height * scale))
        if target_width % 2:
            target_width += 1
        if target_height % 2:
            target_height += 1
        return target_width, target_height, scale

    def _prepare_output_path(self, video_id: int, source: Path) -> Path:
        """Определяет конечный путь для превью и удаляет предыдущее."""
        output_dir = getattr(
            self.settings, 'video_output_dir', Path('runs/visualizations')
        )
        output_path = output_dir / f'{video_id}_{source.stem}.mp4'
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                LOGGER.warning(
                    'Could not remove existing preview: %s', output_path
                )
        return output_path

    def _create_ffmpeg_process(
        self, config: dict[str, Any], output_path: Path
    ) -> Any | None:
        """Запускает ffmpeg-процесс для записи кадров."""
        try:
            return start_ffmpeg_writer(
                width=config['target_width'],
                height=config['target_height'],
                fps=config['fps_out'],
                output_path=output_path,
                ffmpeg_path=getattr(self.settings, 'preview_ffmpeg_path', None),
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning('ffmpeg pipeline creation failed: %s', exc)
            return None

    def _build_frame_map(
        self, detections: Sequence[DetectionResult]
    ) -> dict[int, list[DetectionResult]]:
        """Группирует детекции по номеру кадра."""
        frame_map: dict[int, list[DetectionResult]] = defaultdict(list)
        for det in detections:
            frame_map[det.frame_id].append(det)
        return frame_map

    def _render_frames(
        self,
        config: dict[str, Any],
        frame_map: dict[int, list[DetectionResult]],
        process: Any,
    ) -> None:
        """Читает кадры исходного видео и отправляет их в ffmpeg."""
        cap = config['cap']
        width = config['width']
        height = config['height']
        target_width = config['target_width']
        target_height = config['target_height']
        scale = config['scale']
        frame_step = config['frame_step']

        frame_idx = 0
        process_alive = True
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_step != 0:
                    frame_idx += 1
                    continue

                # Перед отрисовкой кадра убеждаемся, что процесс ffmpeg ещё жив.
                if not process_alive:
                    LOGGER.debug(
                        'FFmpeg process terminated early, stopping frame rendering'
                    )
                    break

                annotated = self._annotate_frame(
                    frame,
                    frame_map.get(frame_idx, []),
                    width,
                    height,
                    target_width,
                    target_height,
                    scale,
                )

                # Пишем кадр в stdin ffmpeg с обработкой ошибок.
                if process.stdin and process_alive:
                    try:
                        process.stdin.write(annotated.tobytes())
                    except (BrokenPipeError, OSError) as e:
                        LOGGER.debug(
                            'FFmpeg process pipe broken or closed: %s', e
                        )
                        process_alive = False
                        break
                    except Exception as e:
                        LOGGER.warning(
                            'Unexpected error writing to ffmpeg: %s', e
                        )
                        process_alive = False
                        break

                    try:
                        process.stdin.flush()
                    except (BrokenPipeError, OSError) as e:
                        LOGGER.debug(
                            'FFmpeg process pipe broken during flush: %s', e
                        )
                        process_alive = False
                        break
                    except Exception as e:
                        LOGGER.warning(
                            'Unexpected error flushing ffmpeg: %s', e
                        )
                        process_alive = False
                        break
                frame_idx += 1
        finally:
            cap.release()
            # Safely close stdin and wait for process
            # Аккуратно закрываем пайп и ждём завершения процесса.
            if process.stdin:
                try:
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    # Процесс уже мог закрыть stdin.
                    pass
            try:
                # Дожидаемся завершения без таймаута ради совместимости.
                process.wait()
            except Exception as e:
                LOGGER.debug('Error waiting for ffmpeg process: %s', e)
                # Пробуем завершить процесс, если он завис.
                try:
                    if hasattr(process, 'terminate'):
                        process.terminate()
                except Exception:
                    pass

                with contextlib.suppress(Exception):
                    process.wait()

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: list[DetectionResult],
        width: int,
        height: int,
        target_width: int,
        target_height: int,
        scale: float,
    ) -> np.ndarray:
        """Наносит на кадр рамки и подписи треков."""
        if scale != 1.0:
            annotated = cv2.resize(
                frame,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            annotated = frame
        ratio_x = annotated.shape[1] / width
        ratio_y = annotated.shape[0] / height
        for det in detections:
            color = self._color_for_track(det.track_id)
            x1, y1, x2, y2 = det.bbox
            x1 = int(x1 * ratio_x)
            x2 = int(x2 * ratio_x)
            y1 = int(y1 * ratio_y)
            y2 = int(y2 * ratio_y)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f'ID {det.track_id}'
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                lineType=cv2.LINE_AA,
            )
        return annotated

    @staticmethod
    def _color_for_track(track_id: int) -> tuple[int, int, int]:
        """Детерминированный цвет (BGR) для конкретного трека."""
        value = (track_id * 37) % 255
        return (value, (value * 2) % 255, (value * 3) % 255)
