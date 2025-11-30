"""Модуль инференса действий через VideoLLaMA3 для людей и поездов."""

from __future__ import annotations

import contextlib
import logging
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from pathlib import Path as PathLib
from types import SimpleNamespace
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from hackaton_system.config import Settings
from hackaton_system.pipeline.models import ActivityResult, DetectionResult
from hackaton_system.pipeline.yolo_classes import normalize_class_key

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER = logging.getLogger(__name__)

# Глобальный кеш-инстанс, чтобы не загружать модель несколько раз.
_VIDEOLLAMA3_INSTANCE: VideoLLaMA3ActionInferrer | None = None
_VIDEOLLAMA3_LOCK = None

try:
    import threading

    _VIDEOLLAMA3_LOCK = threading.Lock()
except ImportError:
    # Если threading недоступен, просто пропускаем блокировку.
    pass


def get_shared_videollama3_instance(
    settings: Settings,
) -> VideoLLaMA3ActionInferrer:
    """Возвращает общий экземпляр VideoLLaMA3, чтобы избежать повторной загрузки модели."""
    global _VIDEOLLAMA3_INSTANCE

    if _VIDEOLLAMA3_INSTANCE is None:
        if _VIDEOLLAMA3_LOCK is not None:
            with _VIDEOLLAMA3_LOCK:
                if _VIDEOLLAMA3_INSTANCE is None:
                    LOGGER.info('Creating shared VideoLLaMA3 instance')
                    _VIDEOLLAMA3_INSTANCE = VideoLLaMA3ActionInferrer(settings)
        else:
            LOGGER.info('Creating shared VideoLLaMA3 instance (no lock)')
            _VIDEOLLAMA3_INSTANCE = VideoLLaMA3ActionInferrer(settings)
    else:
        LOGGER.debug('Reusing existing VideoLLaMA3 instance')

    return _VIDEOLLAMA3_INSTANCE


@dataclass(slots=True)
class ClipSegment:
    """Временной отрезок кадров для конкретного трека."""

    track_id: int
    start_sec: float
    end_sec: float
    frame_indices: list[int]
    target_class: str = 'person'


class VideoLLaMA3ActionInferrer:
    """Единый инферер действий через VideoLLaMA3-2B для людей и поездов."""

    def __init__(self, settings: Settings) -> None:
        """Сохраняет настройки и подготавливает отложенные объекты модели."""
        self.settings = settings
        self._model: AutoModelForCausalLM | None = None
        self._processor: AutoProcessor | None = None
        self._device: str | None = None

    def infer_actions(
        self,
        video_path: Path | str,
        tracks: Mapping[int, list[DetectionResult]],
        target_class: str = 'person',
    ) -> list[ActivityResult]:
        """Определяет действия для всех треков с помощью VideoLLaMA3."""
        if not tracks:
            return []

        video_path = Path(video_path)
        if not video_path.exists():
            LOGGER.warning('Video file not found: %s', video_path)
            return []

        # Get prompts based on target class
        prompts = self._get_prompts_for_class(target_class)
        if not prompts:
            LOGGER.warning('No prompts configured for class: %s', target_class)
            return []

        segments = self._build_clip_segments(video_path, tracks, target_class)
        if not segments:
            return []

        # Log segment information before processing
        # Group segments by track_id for better readability
        segments_by_track: dict[int, list[ClipSegment]] = {}
        for segment in segments:
            if segment.track_id not in segments_by_track:
                segments_by_track[segment.track_id] = []
            segments_by_track[segment.track_id].append(segment)
        
        LOGGER.info(
            'Built %s clip segments for %s class across %s tracks. Segments to process:',
            len(segments),
            target_class,
            len(segments_by_track),
        )
        
        # Show summary per track, then details for first few tracks
        for track_id, track_segments in sorted(segments_by_track.items()):
            total_duration = sum(
                seg.end_sec - seg.start_sec for seg in track_segments
            )
            LOGGER.info(
                '  Track ID %s: %s segments, total duration %.2fs, time range [%.2fs - %.2fs]',
                track_id,
                len(track_segments),
                total_duration,
                track_segments[0].start_sec,
                track_segments[-1].end_sec,
            )
            # Show details for first 3 segments of each track (or all if <= 3)
            show_count = min(3, len(track_segments))
            for i, segment in enumerate(track_segments[:show_count], 1):
                duration = segment.end_sec - segment.start_sec
                frame_range = (
                    f'{segment.frame_indices[0]}..{segment.frame_indices[-1]}'
                    if segment.frame_indices
                    else 'none'
                )
                LOGGER.info(
                    '    [%s/%s] time=[%.3fs, %.3fs] (%.3fs), frames=%s (%s)',
                    i,
                    len(track_segments),
                    segment.start_sec,
                    segment.end_sec,
                    duration,
                    len(segment.frame_indices),
                    frame_range,
                )
            if len(track_segments) > show_count:
                LOGGER.info(
                    '    ... and %s more segments',
                    len(track_segments) - show_count,
                )

        scored = self._score_segments(video_path, segments, prompts)
        min_conf = self.settings.action_min_confidence

        # Log results for each segment
        LOGGER.info(
            'Processing results for %s segments (%s scored):',
            len(segments),
            len(scored),
        )
        # Group results by track_id for better readability
        results_by_track: dict[int, list[SimpleNamespace]] = {}
        for result in scored:
            track_id = result.segment.track_id
            if track_id not in results_by_track:
                results_by_track[track_id] = []
            results_by_track[track_id].append(result)
        
        for track_id, track_results in sorted(results_by_track.items()):
            LOGGER.info(
                '  Track ID %s: %s results',
                track_id,
                len(track_results),
            )
            for i, result in enumerate(track_results, 1):
                segment = result.segment
                status = '✓' if result.confidence >= min_conf else '✗'
                LOGGER.info(
                    '    [%s/%s] %s time=[%.3fs, %.3fs] → %s (confidence: %.3f%s)',
                    i,
                    len(track_results),
                    status,
                    segment.start_sec,
                    segment.end_sec,
                    result.label,
                    result.confidence,
                    f' >= {min_conf:.3f}' if result.confidence >= min_conf else f' < {min_conf:.3f} (filtered)',
                )

        activities: list[ActivityResult] = []
        for result in scored:
            if result.confidence < min_conf:
                continue
            activities.append(
                ActivityResult(
                    track_id=result.segment.track_id,
                    activity_class=result.label,
                    start_sec=result.segment.start_sec,
                    end_sec=result.segment.end_sec,
                    confidence=result.confidence,
                )
            )

        LOGGER.info(
            'Final activities after filtering (confidence >= %.3f): %s',
            min_conf,
            len(activities),
        )

        return activities

    def ask_question(
        self,
        video_path: Path | str,
        question: str,
        start_sec: float | None = None,
        end_sec: float | None = None,
        max_frames: int | None = None,
        max_new_tokens: int = 256,
    ) -> str | None:
        """Формулирует ответ на вопрос о видео с помощью VideoLLaMA3-2B."""
        if self._model is None or self._processor is None:
            self._load_model()

        video_path = Path(video_path)
        if not video_path.exists():
            LOGGER.warning('Video file not found: %s', video_path)
            return None

        # If segment specified, extract frames and create temp video
        if start_sec is not None and end_sec is not None:
            temp_video_path = self._extract_video_segment(
                video_path, start_sec, end_sec
            )
            if temp_video_path is None:
                return None
            video_to_use = temp_video_path
        else:
            video_to_use = video_path

        # Get FPS from video
        try:
            cap = cv2.VideoCapture(str(video_to_use))
        except Exception as e:
            LOGGER.error('Failed to create VideoCapture: %s', e)
            return None

        if not cap.isOpened():
            LOGGER.error('Failed to open video: %s', video_to_use)
            return None

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        except Exception as e:
            LOGGER.error('Failed to get FPS: %s', e)
            cap.release()
            return None

        try:
            cap.release()
        except Exception as e:
            LOGGER.warning('Failed to release VideoCapture: %s', e)

        # Use max_frames from settings if not specified
        if max_frames is None:
            max_frames = getattr(self.settings, 'videollama3_max_frames', 128)

        # Build conversation
        conversation = [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'video',
                        'video': {
                            'video_path': str(video_to_use),
                            'fps': fps,
                            'max_frames': max_frames,
                        },
                    },
                    {'type': 'text', 'text': question},
                ],
            }
        ]

        # Process conversation
        try:
            inputs = self._processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors='pt',
            )
        except Exception as e:
            LOGGER.error('Failed to process conversation: %s', e)
            return None

        try:
            inputs = {
                k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            # Convert pixel_values dtype
            if 'pixel_values' in inputs and isinstance(
                inputs['pixel_values'], torch.Tensor
            ):
                inputs['pixel_values'] = inputs['pixel_values'].to(
                    torch.bfloat16
                )
        except Exception as e:
            LOGGER.error('Failed to process inputs: %s', e)
            return None

        # Generate response
        try:
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )
        except Exception as e:
            LOGGER.error('Failed to generate response: %s', e)
            return None

        # Decode response
        try:
            response = self._processor.batch_decode(
                output_ids, skip_special_tokens=True
            )[0].strip()
        except Exception as e:
            LOGGER.error('Failed to decode response: %s', e)
            return None

        # Extract only the generated part (remove input)
        # The response includes the full conversation, we need only the answer
        if question in response:
            # Find where the answer starts (after the question)
            answer_start = response.find(question) + len(question)
            response = response[answer_start:].strip()

        # Clean up temp video if created
        if (
            start_sec is not None
            and end_sec is not None
            and video_to_use != video_path
            and video_to_use.exists()
        ):
            try:
                video_to_use.unlink()
            except Exception as e:
                LOGGER.warning('Failed to delete temp video file: %s', e)

        return response

    def _extract_video_segment(
        self, video_path: Path, start_sec: float, end_sec: float
    ) -> Path | None:
        """Вырезает фрагмент видео во временный файл и возвращает его путь."""
        try:
            cap = cv2.VideoCapture(str(video_path))
        except Exception as e:
            LOGGER.error('Failed to create VideoCapture: %s', e)
            return None

        if not cap.isOpened():
            LOGGER.error('Failed to open video: %s', video_path)
            return None

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        except Exception as e:
            LOGGER.error('Failed to get FPS: %s', e)
            cap.release()
            return None

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        except Exception as e:
            LOGGER.error('Failed to get frame width: %s', e)
            cap.release()
            return None

        try:
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except Exception as e:
            LOGGER.error('Failed to get frame height: %s', e)
            cap.release()
            return None

        # Calculate frame indices
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        # Create temporary video file
        try:
            with tempfile.NamedTemporaryFile(
                suffix='.mp4', delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
        except Exception as e:
            LOGGER.error('Failed to create temporary file: %s', e)
            cap.release()
            return None

        # Write frames to video
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        except Exception as e:
            LOGGER.error('Failed to create VideoWriter codec: %s', e)
            cap.release()
            return None

        try:
            out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
        except Exception as e:
            LOGGER.error('Failed to create VideoWriter: %s', e)
            cap.release()
            return None

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        except Exception as e:
            LOGGER.error('Failed to set frame position: %s', e)
            out.release()
            cap.release()
            return None

        try:
            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
        except Exception as e:
            LOGGER.error('Failed to write frames: %s', e)
            out.release()
            cap.release()
            return None

        try:
            out.release()
        except Exception as e:
            LOGGER.warning('Failed to release VideoWriter: %s', e)

        try:
            cap.release()
        except Exception as e:
            LOGGER.warning('Failed to release VideoCapture: %s', e)

        return temp_path

    def _get_prompts_for_class(self, target_class: str) -> list[str]:
        """Возвращает список промптов для класса (class_prompts → action/train_prompts)."""
        # Normalize class key (handle both names and IDs)
        normalized_key = normalize_class_key(target_class)

        # Get class_prompts dict
        class_prompts = getattr(self.settings, 'class_prompts', {})

        # Try normalized key first
        if class_prompts and normalized_key in class_prompts:
            prompts = class_prompts[normalized_key]
            if isinstance(prompts, list) and prompts:
                return prompts

        # Try original key
        if class_prompts and target_class in class_prompts:
            prompts = class_prompts[target_class]
            if isinstance(prompts, list) and prompts:
                return prompts

        # Try as string ID
        if class_prompts and str(target_class) in class_prompts:
            prompts = class_prompts[str(target_class)]
            if isinstance(prompts, list) and prompts:
                return prompts

        # Fallback to action_prompts for person or train_prompts for train
        if normalized_key == 'person' or target_class == 'person':
            action_prompts = getattr(self.settings, 'action_prompts', [])
            if action_prompts:
                return action_prompts
        elif normalized_key == 'train' or target_class == 'train':
            train_prompts = getattr(self.settings, 'train_prompts', [])
            if train_prompts:
                return train_prompts

        # No prompts found for this class
        return []

    def _build_clip_segments(
        self,
        video_path: Path,  # noqa: ARG002
        tracks: Mapping[int, list[DetectionResult]],
        target_class: str = 'person',
    ) -> list[ClipSegment]:
        """Разбивает треки на клипы по шаблону take/skip или шаговому stride."""
        segments: list[ClipSegment] = []
        
        # Check if take-skip pattern is enabled
        take_duration = getattr(self.settings, 'action_clip_take_duration_sec', None)
        skip_duration = getattr(self.settings, 'action_clip_skip_duration_sec', None)
        
        use_take_skip = (
            take_duration is not None 
            and take_duration > 0 
            and skip_duration is not None 
            and skip_duration >= 0
        )
        
        if use_take_skip:
            # Use take-skip pattern
            for track_id, detections in tracks.items():
                if not detections:
                    continue
                
                # Sort detections by time to ensure correct order
                sorted_dets = sorted(detections, key=lambda d: d.time_sec)
                current_time = sorted_dets[0].time_sec
                end_time = sorted_dets[-1].time_sec
                
                while current_time <= end_time:
                    # Take segment: all detections within take_duration from current_time
                    take_end_time = current_time + take_duration
                    segment_dets = [
                        d for d in sorted_dets
                        if current_time <= d.time_sec < take_end_time
                    ]
                    
                    if segment_dets:
                        segments.append(
                            ClipSegment(
                                track_id=track_id,
                                start_sec=segment_dets[0].time_sec,
                                end_sec=segment_dets[-1].time_sec,
                                frame_indices=[d.frame_id for d in segment_dets],
                                target_class=target_class,
                            )
                        )
                    
                    # Move to next segment: skip duration after the end of taken segment
                    current_time = take_end_time + skip_duration
        else:
            # Legacy stride-based approach
            clip_frames = self.settings.action_clip_frames
            stride = self.settings.action_clip_stride

            for track_id, detections in tracks.items():
                if len(detections) < clip_frames:
                    # If track is too short, use all available frames
                    if detections:
                        segments.append(
                            ClipSegment(
                                track_id=track_id,
                                start_sec=detections[0].time_sec,
                                end_sec=detections[-1].time_sec,
                                frame_indices=[d.frame_id for d in detections],
                                target_class=target_class,
                            )
                        )
                    continue

                # Extract overlapping segments with stride
                for i in range(0, len(detections) - clip_frames + 1, stride):
                    window = detections[i : i + clip_frames]
                    segments.append(
                        ClipSegment(
                            track_id=track_id,
                            start_sec=window[0].time_sec,
                            end_sec=window[-1].time_sec,
                            frame_indices=[d.frame_id for d in window],
                            target_class=target_class,
                        )
                    )

        return segments

    def _score_segments(
        self,
        video_path: Path,
        segments: Sequence[ClipSegment],
        prompts: list[str],
    ) -> list[SimpleNamespace]:
        """Оценивает сегменты клипов по набору промптов VideoLLaMA3."""
        if not segments:
            return []

        # Lazy load model
        if self._model is None:
            self._load_model()

        if not prompts:
            LOGGER.warning('No action prompts configured')
            return []

        results: list[SimpleNamespace] = []

        # Process segments in batches for efficiency
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            LOGGER.error('Failed to open video: %s', video_path)
            return []

        try:
            for segment in segments:
                frames = self._extract_frames(cap, segment.frame_indices)
                if not frames:
                    LOGGER.debug(
                        'No frames extracted for segment track_id=%s, '
                        'frames=%s',
                        segment.track_id,
                        segment.frame_indices[:5],
                    )
                    continue

                # Score video segment against prompts using VideoLLaMA3-2B
                # Save frames to temporary video file
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

                # Limit frames to max_frames to prevent memory issues
                max_frames = getattr(
                    self.settings, 'videollama3_max_frames', 32
                )
                if len(frames) > max_frames:
                    # Take evenly spaced frames
                    indices = np.linspace(
                        0, len(frames) - 1, max_frames, dtype=int
                    )
                    frames = [frames[i] for i in indices]
                    LOGGER.debug(
                        'Limited segment frames from %s to %s for track_id=%s',
                        len(segment.frame_indices),
                        len(frames),
                        segment.track_id,
                    )

                temp_video_path = self._save_frames_as_video(frames, fps)
                if temp_video_path is None:
                    LOGGER.debug(
                        'Failed to create temp video for segment track_id=%s',
                        segment.track_id,
                    )
                    continue

                try:
                    # Score segment against all prompts using frames directly
                    # This prevents processor from loading entire video file
                    similarities = self._score_frames_against_prompts(
                        frames, prompts, fps
                    )
                    if similarities is None or len(similarities) == 0:
                        continue
                finally:
                    # Clean up temp video
                    if temp_video_path and temp_video_path.exists():
                        temp_video_path.unlink()

                # Find best match
                best_idx = int(np.argmax(similarities))
                confidence = float(similarities[best_idx])
                label = prompts[best_idx]

                # Build raw scores dict
                raw_scores = {
                    prompt: float(sim)
                    for prompt, sim in zip(prompts, similarities, strict=True)
                }

                results.append(
                    SimpleNamespace(
                        segment=segment,
                        label=label,
                        confidence=confidence,
                        raw_scores=raw_scores,
                    )
                )
        finally:
            cap.release()

        return results

    def _load_model(self) -> None:
        """Лениво загружает модель и процессор VideoLLaMA3 (2B/7B)."""
        model_name = getattr(
            self.settings,
            'videollama3_model_name',
            'DAMO-NLP-SG/VideoLLaMA3-7B',
        )
        device_str = getattr(self.settings, 'videollama3_device', 'auto')

        if device_str == 'auto':
            if torch.cuda.is_available():
                self._device = 'cuda:0'
            else:
                self._device = 'cpu'
        else:
            self._device = device_str

        LOGGER.info(
            'Loading VideoLLaMA3 model: %s on %s', model_name, self._device
        )

        # Copy our fixed custom code to HuggingFace cache before loading
        # This ensures transformers uses our fixes (VideoInput import, Path serialization)
        # Determine which model variant (7B or 2B) and use appropriate local code
        try:
            if '7B' in model_name:
                model_variant = 'videollama3'
                cache_model_name = 'VideoLLaMA3_hyphen_7B'
            elif '2B' in model_name:
                model_variant = 'videollama3_2b'
                cache_model_name = 'VideoLLaMA3_hyphen_2B'
            else:
                # Default to 7B if unclear
                model_variant = 'videollama3'
                cache_model_name = 'VideoLLaMA3_hyphen_7B'
        except Exception as e:
            LOGGER.error('Failed to determine model variant: %s', e)
            raise

        try:
            cache_dir = (
                PathLib.home()
                / '.cache'
                / 'huggingface'
                / 'modules'
                / 'transformers_modules'
                / 'DAMO_hyphen_NLP_hyphen_SG'
                / cache_model_name
            )
        except Exception as e:
            LOGGER.error('Failed to construct cache directory path: %s', e)
            raise

        # Get local model directory with fixes
        try:
            local_model_dir = (
                PathLib(__file__).parent.parent / 'models' / model_variant
            )
        except Exception as e:
            LOGGER.error(
                'Failed to construct local model directory path: %s', e
            )
            raise

        # Try to load processor first (this will create cache if needed)
        # If it fails due to import error, we'll fix the cache and retry
        processor_loaded = False
        processor_error = None
        try:
            self._processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            processor_loaded = True
        except Exception as e:
            error_msg = str(e)
            processor_error = e
            if 'VideoInput' in error_msg and 'image_utils' in error_msg:
                LOGGER.info(
                    'Processor load failed due to VideoInput import issue. '
                    'Will fix cache and retry.'
                )
            else:
                # For other errors, we still want to apply fixes, but log the error
                LOGGER.warning(
                    'Processor load failed: %s. Will apply fixes and retry.',
                    processor_error,
                )

        # Copy fixed files to cache (whether cache existed before or was just created)
        # This ensures our fixes are always applied
        try:
            commit_dirs = (
                list(cache_dir.glob('*/')) if cache_dir.exists() else []
            )
            # Filter out __pycache__ directories and non-directories
            commit_dirs = [
                d
                for d in commit_dirs
                if d.is_dir() and not d.name.startswith('_')
            ]
        except Exception as e:
            LOGGER.error('Failed to list commit directories: %s', e)
            commit_dirs = []

        latest_dir = None
        if commit_dirs:
            try:
                latest_dir = max(commit_dirs, key=lambda p: p.stat().st_mtime)
            except Exception as e:
                LOGGER.error('Failed to find latest commit directory: %s', e)

        if latest_dir:
            try:
                # Copy fixed files
                for py_file in local_model_dir.glob('*.py'):
                    if py_file.name != '__init__.py':
                        shutil.copy2(py_file, latest_dir / py_file.name)
                        LOGGER.debug(
                            'Copied fixed file to cache: %s -> %s',
                            py_file.name,
                            latest_dir / py_file.name,
                        )

                # Clear __pycache__ to force Python to recompile with fixed imports
                pycache_dir = latest_dir / '__pycache__'
                if pycache_dir.exists():
                    try:
                        shutil.rmtree(pycache_dir)
                        LOGGER.debug('Cleared __pycache__ directory')
                    except Exception as e:
                        LOGGER.warning('Failed to clear __pycache__: %s', e)

                LOGGER.info(
                    'Copied fixed custom code to cache (%s): %s',
                    model_variant,
                    latest_dir,
                )
            except Exception as e:
                LOGGER.error(
                    'Failed to copy fixed custom code to cache: %s', e
                )
                if not processor_loaded:
                    raise

        # If processor wasn't loaded (due to import error), retry now
        if not processor_loaded:
            try:
                self._processor = AutoProcessor.from_pretrained(
                    model_name, trust_remote_code=True
                )
                LOGGER.info(
                    'Successfully loaded processor after applying fixes'
                )
            except Exception as e:
                LOGGER.error(
                    'Failed to load VideoLLaMA3 processor after fixes: %s', e
                )
                raise

        # Configure device_map - use single device
        device_map = {'': self._device}

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=device_map,
            dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
        )
        # Set model to eval mode
        try:
            self._model.eval()
        except Exception as e:
            LOGGER.error('Failed to set model to eval mode: %s', e)
            raise

    def _extract_frames(
        self, cap: cv2.VideoCapture, frame_indices: list[int]
    ) -> list[np.ndarray]:
        """Считывает кадры по указанным индексам."""
        frames: list[np.ndarray] = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_idx in frame_indices:
            # Validate frame index
            if frame_idx < 0 or (
                total_frames > 0 and frame_idx >= total_frames
            ):
                LOGGER.warning(
                    'Frame index %s out of range (total: %s)',
                    frame_idx,
                    total_frames,
                )
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                LOGGER.debug(
                    'cap.read() returned False/None for frame %s', frame_idx
                )
                continue
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                LOGGER.debug('Frame %s is invalid', frame_idx)
                continue
            try:
                if len(frame.shape) < 2:
                    continue
                frames.append(frame)
            except (AttributeError, TypeError, IndexError) as e:
                LOGGER.debug('Frame %s validation error: %s', frame_idx, e)
                continue
        return frames

    def _save_frames_as_video(
        self, frames: list[np.ndarray], fps: float
    ) -> Path | None:
        """Сохраняет кадры во временный видеофайл для инференса."""
        if not frames:
            return None

        # Create temporary video file
        try:
            with tempfile.NamedTemporaryFile(
                suffix='.mp4', delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
        except Exception as e:
            LOGGER.error('Failed to create temporary file: %s', e)
            return None

        # Get frame dimensions
        try:
            height, width = frames[0].shape[:2]
        except Exception as e:
            LOGGER.error('Failed to get frame dimensions: %s', e)
            if temp_path.exists():
                temp_path.unlink()
            return None

        # Write frames to video using OpenCV
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        except Exception as e:
            LOGGER.error('Failed to create VideoWriter codec: %s', e)
            if temp_path.exists():
                temp_path.unlink()
            return None

        try:
            out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
        except Exception as e:
            LOGGER.error('Failed to create VideoWriter: %s', e)
            if temp_path.exists():
                temp_path.unlink()
            return None

        try:
            for frame in frames:
                out.write(frame)
        except Exception as e:
            LOGGER.error('Failed to write frames: %s', e)
            out.release()
            if temp_path.exists():
                temp_path.unlink()
            return None

        try:
            out.release()
        except Exception as e:
            LOGGER.warning('Failed to release VideoWriter: %s', e)
            if temp_path.exists():
                temp_path.unlink()
            return None

        return temp_path

    def _extract_limited_frames_video(
        self, video_path: Path, max_frames: int, fps: float
    ) -> Path | None:
        """Вырезает ограниченное число кадров, чтобы не упереться в память."""
        try:
            cap = cv2.VideoCapture(str(video_path))
        except Exception as e:
            LOGGER.error('Failed to create VideoCapture: %s', e)
            return None

        if not cap.isOpened():
            LOGGER.error('Failed to open video: %s', video_path)
            return None

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except Exception as e:
            LOGGER.error('Failed to get video properties: %s', e)
            cap.release()
            return None

        if total_frames <= max_frames:
            # Video is already small enough
            cap.release()
            return video_path

        # Calculate evenly spaced frame indices
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

        # Create temporary video file
        try:
            with tempfile.NamedTemporaryFile(
                suffix='.mp4', delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
        except Exception as e:
            LOGGER.error('Failed to create temporary file: %s', e)
            cap.release()
            return None

        # Write limited frames to video
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
        except Exception as e:
            LOGGER.error('Failed to create VideoWriter: %s', e)
            cap.release()
            if temp_path.exists():
                temp_path.unlink()
            return None

        try:
            for frame_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                out.write(frame)
        except Exception as e:
            LOGGER.error('Failed to write frames: %s', e)
            out.release()
            cap.release()
            if temp_path.exists():
                temp_path.unlink()
            return None

        try:
            out.release()
            cap.release()
        except Exception as e:
            LOGGER.warning('Failed to release resources: %s', e)

        return temp_path

    def _score_video_against_prompts(
        self, video_path: Path, prompts: list[str], fps: float
    ) -> np.ndarray | None:
        """Оценивает видео целиком по промптам VideoLLaMA3."""
        if self._model is None or self._processor is None:
            return None

        if not prompts:
            return None

        max_frames = getattr(self.settings, 'videollama3_max_frames', 32)

        # CRITICAL: Verify video is small enough before processing
        # Processor may try to load entire video, causing OOM
        limited_video_path: Path | None = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                # If video has more frames than max_frames, extract limited frames
                if total_frames > max_frames:
                    LOGGER.warning(
                        'Video has %s frames, limiting to %s to prevent OOM',
                        total_frames,
                        max_frames,
                    )
                    limited_video_path = self._extract_limited_frames_video(
                        video_path, max_frames, fps
                    )
                    if limited_video_path is None:
                        LOGGER.error('Failed to create limited frames video')
                        return None
                    video_path = limited_video_path
        except Exception as e:
            LOGGER.warning(
                'Failed to check video size, proceeding with original: %s', e
            )

        # Build conversation with video and question
        # For each prompt, ask if it's happening in the video
        scores = []
        for prompt in prompts:
            conversation = [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'video',
                            'video': {
                                'video_path': str(video_path),
                                'fps': fps,
                                'max_frames': max_frames,
                            },
                        },
                        {
                            'type': 'text',
                            'text': f'Is this action happening in the video: "{prompt}"? Answer only "Yes" or "No".',
                        },
                    ],
                }
            ]

            # Process conversation
            inputs = None
            try:
                inputs = self._processor(
                    conversation=conversation,
                    add_system_prompt=True,
                    add_generation_prompt=True,
                    return_tensors='pt',
                )
            except Exception as e:
                LOGGER.error(
                    'Failed to process conversation for prompt "%s": %s',
                    prompt,
                    e,
                )
                scores.append(0.5)  # Default score on error
                continue

            try:
                inputs = {
                    k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }
            except Exception as e:
                LOGGER.error(
                    'Failed to process inputs for prompt "%s": %s',
                    prompt,
                    e,
                )
                scores.append(0.5)
                continue

            if 'pixel_values' in inputs:
                try:
                    inputs['pixel_values'] = inputs['pixel_values'].to(
                        torch.bfloat16
                        if self._device != 'cpu'
                        else torch.float32
                    )
                except Exception as e:
                    LOGGER.error(
                        'Failed to convert pixel_values dtype for prompt "%s": %s',
                        prompt,
                        e,
                    )
                    scores.append(0.5)
                    continue

            # Generate response
            output_ids: torch.Tensor | None = None
            inputs_for_cleanup = inputs  # Keep reference for cleanup
            try:
                if inputs is None:
                    scores.append(0.5)
                    continue

                with torch.no_grad():
                    output_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        temperature=0.0,
                    )
                # Clear inputs from memory immediately after generation
                del inputs
                inputs_for_cleanup = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                LOGGER.error(
                    'Failed to generate response for prompt "%s": %s',
                    prompt,
                    e,
                )
                # Clear inputs on error too
                if inputs_for_cleanup is not None:
                    del inputs_for_cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                scores.append(0.5)
                continue

            # Decode response
            output_ids_for_cleanup = output_ids  # Keep reference for cleanup
            try:
                if output_ids is None:
                    scores.append(0.5)
                    continue

                response = self._processor.batch_decode(
                    output_ids, skip_special_tokens=True
                )[0].strip()
                # Clear output_ids from memory immediately
                del output_ids
                output_ids_for_cleanup = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                LOGGER.error(
                    'Failed to decode response for prompt "%s": %s', prompt, e
                )
                if output_ids_for_cleanup is not None:
                    del output_ids_for_cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                scores.append(0.5)
                continue

            # Extract score from response
            # If response contains "Yes", score is high; if "No", score is low
            if 'yes' in response.lower():
                score = 0.9
            elif 'no' in response.lower():
                score = 0.1
            else:
                # If unclear, use a medium score
                score = 0.5

            scores.append(score)
            # Clear response from memory
            del response

            # Force garbage collection every few prompts to free memory
            if len(scores) % 3 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Clean up limited video if it was created
        if limited_video_path is not None and limited_video_path.exists():
            with contextlib.suppress(Exception):
                limited_video_path.unlink()

        if not scores:
            return None

        return np.array(scores)

    def _score_frames_against_prompts(
        self, frames: list[np.ndarray], prompts: list[str], fps: float
    ) -> np.ndarray | None:
        """Оценивает отдельные кадры напрямую, чтобы не перегружать память."""
        if self._model is None or self._processor is None:
            return None

        if not prompts:
            return None

        if not frames:
            return None

        # Convert BGR frames to RGB
        rgb_frames = self._convert_frames_to_rgb(frames)
        if not rgb_frames:
            LOGGER.warning('No valid RGB frames after conversion')
            return None

        # Limit frames to max_frames (should already be limited, but double-check)
        max_frames = getattr(self.settings, 'videollama3_max_frames', 8)
        if len(rgb_frames) > max_frames:
            indices = np.linspace(
                0, len(rgb_frames) - 1, max_frames, dtype=int
            )
            rgb_frames = [rgb_frames[i] for i in indices]
            LOGGER.debug(
                'Limited frames from %s to %s for scoring',
                len(frames),
                len(rgb_frames),
            )

        # Clear original frames from memory
        del frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Score against each prompt
        scores = []
        for prompt in prompts:
            # Build conversation with frames as images (not video_path)
            # This prevents processor from loading entire video file
            try:
                # Use only first few frames to reduce memory
                # VideoLLaMA3 can work with multiple images in conversation
                frame_sample = rgb_frames[: min(4, len(rgb_frames))]

                conversation = [
                    {
                        'role': 'user',
                        'content': [
                            *[
                                {'type': 'image', 'image': frame}
                                for frame in frame_sample
                            ],
                            {
                                'type': 'text',
                                'text': f'Is this action happening in the video: "{prompt}"? Answer only "Yes" or "No".',
                            },
                        ],
                    }
                ]

                inputs = self._processor(
                    conversation=conversation,
                    add_system_prompt=True,
                    add_generation_prompt=True,
                    return_tensors='pt',
                )
            except Exception as e:
                LOGGER.error(
                    'Failed to process conversation for prompt "%s": %s',
                    prompt,
                    e,
                )
                scores.append(0.5)
                continue

            # Validate inputs is a dict-like object before processing
            if inputs is None:
                LOGGER.error(
                    'Processor returned None for prompt "%s"',
                    prompt,
                )
                scores.append(0.5)
                continue

            # Check if inputs is dict-like (has items method)
            if not hasattr(inputs, 'items'):
                LOGGER.error(
                    'Processor returned non-dict-like for prompt "%s": %s',
                    prompt,
                    type(inputs),
                )
                scores.append(0.5)
                continue

            try:
                # Convert to dict and move all tensors to device
                # Handle both dict and dict-like objects (e.g., BatchEncoding)
                inputs_dict = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        # Move to device and convert dtype if needed
                        if k == 'pixel_values':
                            # Convert pixel_values to bfloat16 and move to device
                            inputs_dict[k] = v.to(self._device).to(
                                torch.bfloat16
                                if self._device != 'cpu'
                                else torch.float32
                            )
                        else:
                            # Move other tensors to device
                            inputs_dict[k] = v.to(self._device)
                    elif isinstance(v, (list, tuple)):
                        # Handle nested structures
                        inputs_dict[k] = v
                    else:
                        inputs_dict[k] = v
                inputs = inputs_dict
            except Exception as e:
                LOGGER.error(
                    'Failed to process inputs for prompt "%s": %s',
                    prompt,
                    e,
                )
                scores.append(0.5)
                continue

            # Generate response
            output_ids: torch.Tensor | None = None
            inputs_for_cleanup = inputs  # Keep reference for cleanup
            try:
                if inputs is None:
                    scores.append(0.5)
                    continue

                with torch.no_grad():
                    output_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        temperature=0.0,
                    )
                # Clear inputs from memory immediately after generation
                del inputs
                inputs_for_cleanup = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                LOGGER.error(
                    'Failed to generate response for prompt "%s": %s',
                    prompt,
                    e,
                )
                # Clear inputs on error too
                if inputs_for_cleanup is not None:
                    del inputs_for_cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                scores.append(0.5)
                continue

            # Decode response
            output_ids_for_cleanup = output_ids  # Keep reference for cleanup
            try:
                if output_ids is None:
                    scores.append(0.5)
                    continue

                response = self._processor.batch_decode(
                    output_ids, skip_special_tokens=True
                )[0].strip()
                # Clear output_ids from memory immediately
                del output_ids
                output_ids_for_cleanup = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                LOGGER.error(
                    'Failed to decode response for prompt "%s": %s', prompt, e
                )
                if output_ids_for_cleanup is not None:
                    del output_ids_for_cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                scores.append(0.5)
                continue

            # Extract score from response
            if 'yes' in response.lower():
                score = 0.9
            elif 'no' in response.lower():
                score = 0.1
            else:
                score = 0.5

            scores.append(score)
            del response

            # Force garbage collection every few prompts
            if len(scores) % 3 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not scores:
            return None

        return np.array(scores)

    def _encode_video(self, frames: list[np.ndarray]) -> torch.Tensor | None:
        """Формирует видеозакодированное представление через VideoLLaMA3."""
        if self._model is None or self._processor is None:
            return None

        if not frames:
            return None

        valid_frames = self._validate_and_filter_frames(frames)
        if not valid_frames:
            return None

        rgb_frames = self._convert_frames_to_rgb(valid_frames)
        if not rgb_frames:
            return None

        inputs = self._process_frames_with_processor(rgb_frames)
        if inputs is None:
            return None

        return self._get_video_embeddings(inputs)

    def _validate_and_filter_frames(
        self, frames: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Проверяет кадры и фильтрует некорректные."""
        valid_frames = []
        for f in frames:
            if f is None or not isinstance(f, np.ndarray) or f.size == 0:
                continue

            try:
                if len(f.shape) < 2:
                    continue
                if len(f.shape) == 3:
                    if f.shape[2] == 3:
                        valid_frames.append(f)
                    else:
                        continue
                elif len(f.shape) == 2:
                    try:
                        bgr_frame = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                        if bgr_frame is not None and bgr_frame.size > 0:
                            valid_frames.append(bgr_frame)
                    except Exception:
                        continue
                else:
                    continue
            except (AttributeError, IndexError):
                continue

        if not valid_frames:
            LOGGER.warning('No valid frames to encode')
        return valid_frames

    def _convert_frames_to_rgb(
        self, frames: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Преобразует кадры из BGR в RGB."""
        rgb_frames = []
        for f in frames:
            if f is None or not isinstance(f, np.ndarray) or f.size == 0:
                continue

            try:
                if len(f.shape) < 2:
                    continue
            except (AttributeError, TypeError):
                continue

            try:
                rgb_frame = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            except Exception as e:
                LOGGER.debug('Failed to convert frame to RGB: %s', e)
                continue

            if (
                rgb_frame is None
                or not isinstance(rgb_frame, np.ndarray)
                or rgb_frame.size == 0
            ):
                continue

            try:
                if len(rgb_frame.shape) < 2:
                    continue
            except (AttributeError, TypeError):
                continue

            rgb_frames.append(rgb_frame)

        if not rgb_frames:
            LOGGER.warning('No valid RGB frames after conversion')
        return rgb_frames

    def _process_frames_with_processor(
        self, rgb_frames: list[np.ndarray]
    ) -> dict[str, torch.Tensor] | None:
        """Прогоняет RGB-кадры через процессор VideoLLaMA3."""
        if not rgb_frames:
            LOGGER.warning('No RGB frames to process')
            return None

        # Final validation of frames before processing
        valid_rgb_frames = []
        for f in rgb_frames:
            if (
                f is None
                or not isinstance(f, np.ndarray)
                or f.size == 0
                or len(f.shape) < 3
                or f.shape[2] != 3
                or f.shape[0] == 0
                or f.shape[1] == 0
            ):
                continue
            valid_rgb_frames.append(f)

        if not valid_rgb_frames:
            LOGGER.warning('No valid RGB frames after final validation')
            return None

        LOGGER.debug(
            'Processing %s frames, first frame shape: %s, dtype: %s',
            len(valid_rgb_frames),
            valid_rgb_frames[0].shape if valid_rgb_frames else None,
            valid_rgb_frames[0].dtype if valid_rgb_frames else None,
        )

        try:
            # Ensure uint8 dtype and proper range [0, 255]
            processed_frames = []
            for f in valid_rgb_frames:
                if f.dtype != np.uint8:
                    if f.max() <= 1.0:
                        f = (f * 255).astype(np.uint8)
                    else:
                        f = f.astype(np.uint8)
                processed_frames.append(f)

            # VideoLLaMA3 processor expects list of numpy arrays (H, W, C) in RGB
            # Use processor's video/image processing capability
            inputs = None
            if hasattr(self._processor, 'image_processor'):
                # For video, we process each frame as an image and stack
                # SigLIP-NaViT can handle variable number of frames
                max_frames = getattr(
                    self.settings, 'videollama3_max_frames', 32
                )
                if len(processed_frames) > max_frames:
                    # Take evenly spaced frames
                    indices = np.linspace(
                        0,
                        len(processed_frames) - 1,
                        max_frames,
                        dtype=int,
                    )
                    processed_frames = [processed_frames[i] for i in indices]

                # Process frames through image processor
                # SigLIP-NaViT expects batched images
                if hasattr(self._processor, 'image_processor') and self._processor.image_processor is not None:
                    video_inputs = self._processor.image_processor(
                        processed_frames, return_tensors='pt'
                    )
                    inputs = video_inputs
                elif hasattr(self._processor, 'video_processor') and self._processor.video_processor is not None:
                    # Fallback to video processor if available
                    video_inputs = self._processor.video_processor(
                        processed_frames, return_tensors='pt'
                    )
                    inputs = video_inputs
            
            if inputs is None:
                # Direct processor call as last resort
                inputs = self._processor(
                    images=processed_frames, return_tensors='pt', padding=True
                )
        except Exception as e:
            LOGGER.error(
                'Failed to process frames with VideoLLaMA3 processor: %s',
                e,
                exc_info=True,
            )
            return None

        if inputs is None:
            LOGGER.error('Processor returned None')
            return None

        if not hasattr(inputs, 'items'):
            LOGGER.error('Processor returned non-dict-like: %s', type(inputs))
            return None

        # Check what keys processor returned
        input_keys = list(inputs.keys())
        LOGGER.debug('Processor returned keys: %s', input_keys)

        if not input_keys:
            LOGGER.error('Processor returned empty dict')
            return None

        try:
            processed_inputs = {}
            for k, v in inputs.items():
                if v is None:
                    LOGGER.error('Input key %s has None value', k)
                    return None
                if not isinstance(v, torch.Tensor):
                    LOGGER.error(
                        'Input key %s is not a tensor: %s', k, type(v)
                    )
                    return None
                try:
                    if k == 'pixel_values':
                        # Convert pixel_values to bfloat16 and move to device
                        processed_inputs[k] = v.to(self._device).to(
                            torch.bfloat16
                            if self._device != 'cpu'
                            else torch.float32
                        )
                    else:
                        # Move other tensors to device
                        processed_inputs[k] = v.to(self._device)
                except Exception as e:
                    LOGGER.error('Failed to process input %s: %s', k, e)
                    return None
            inputs = processed_inputs
        except Exception as e:
            LOGGER.error('Failed to move inputs to device: %s', e)
            return None

        # Check for pixel_values or pixel_value (SigLIP uses pixel_value)
        if 'pixel_values' not in inputs and 'pixel_value' not in inputs:
            LOGGER.error(
                'pixel_values/pixel_value not found in inputs. Available keys: %s',
                list(inputs.keys()),
            )
            return None

        return inputs

    def _get_video_embeddings(
        self, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor | None:
        """Получает видеоэмбеддинги из модели."""
        if inputs is None:
            LOGGER.error('Inputs is None')
            return None

        if self._model is None:
            LOGGER.error('Model is None')
            return None

        # Validate inputs before passing to model
        for k, v in inputs.items():
            if v is None:
                LOGGER.error('Input key %s is None', k)
                return None
            if not isinstance(v, torch.Tensor):
                LOGGER.error('Input key %s is not a tensor: %s', k, type(v))
                return None

        try:
            with torch.no_grad():
                # VideoLLaMA3 (VL3-SigLIP-NaViT) processing
                # SigLIP-NaViT processes images/videos through vision encoder
                pixel_key = (
                    'pixel_values'
                    if 'pixel_values' in inputs
                    else 'pixel_value'
                )
                pixel_values = inputs[pixel_key]

                # SigLIP-NaViT expects [batch, num_frames, C, H, W] or [batch, C, H, W]
                # Handle both single image and video batch cases
                if pixel_values.dim() == 4:
                    # Single image case: [batch, C, H, W]
                    # Expand to video format: [batch, 1, C, H, W]
                    pixel_values = pixel_values.unsqueeze(1)
                elif pixel_values.dim() == 5:
                    # Video case: [batch, num_frames, C, H, W]
                    pass
                else:
                    LOGGER.error(
                        'Unexpected pixel_values shape: %s', pixel_values.shape
                    )
                    return None

                batch_size, num_frames, channels, height, width = (
                    pixel_values.shape
                )

                # Reshape for vision_model: [batch * num_frames, C, H, W]
                pixel_values_reshaped = pixel_values.view(
                    batch_size * num_frames, channels, height, width
                )

                # Process through vision encoder (SigLIP-NaViT)
                if hasattr(self._model, 'vision_model'):
                    vision_outputs = self._model.vision_model(
                        pixel_values=pixel_values_reshaped
                    )
                elif hasattr(self._model, 'model'):
                    # Some model structures use 'model' as the vision encoder
                    vision_outputs = self._model.model(
                        pixel_values=pixel_values_reshaped
                    )
                else:
                    # Direct call
                    vision_outputs = self._model(
                        pixel_values=pixel_values_reshaped
                    )

                # Extract hidden states
                if isinstance(vision_outputs, tuple):
                    hidden_states = vision_outputs[0]
                elif hasattr(vision_outputs, 'last_hidden_state'):
                    hidden_states = vision_outputs.last_hidden_state
                else:
                    hidden_states = vision_outputs

                # Reshape back: [batch, num_frames, seq_len, hidden_dim]
                seq_len = hidden_states.shape[1]
                hidden_dim = hidden_states.shape[2]
                hidden_states = hidden_states.view(
                    batch_size, num_frames, seq_len, hidden_dim
                )

                # Take CLS token (first token) for each frame
                cls_tokens = hidden_states[
                    :, :, 0, :
                ]  # [batch, frames, hidden_dim]

                # Average over frames to get video-level embedding
                video_emb = cls_tokens.mean(dim=1)  # [batch, hidden_dim]

        except Exception as e:
            LOGGER.error('Failed to get model outputs: %s', e, exc_info=True)
            return None

        if video_emb is None:
            LOGGER.error('Failed to extract video embeddings')
            return None

        try:
            video_emb = video_emb / video_emb.norm(dim=-1, keepdim=True)
        except Exception as e:
            LOGGER.error('Failed to normalize embeddings: %s', e)
            return None

        try:
            return video_emb.squeeze(0)
        except Exception as e:
            LOGGER.error('Failed to squeeze embeddings: %s', e)
            return None

    def _encode_texts(self, prompts: list[str]) -> torch.Tensor | None:
        """Преобразует текстовые промпты в эмбеддинги."""
        if self._model is None or self._processor is None:
            return None

        try:
            # Process text through tokenizer
            text_inputs = self._processor(
                text=prompts, return_tensors='pt', padding=True
            )
            text_inputs = {
                k: v.to(self._device) for k, v in text_inputs.items()
            }

            with torch.no_grad():
                # Use text_model if available (SigLIP has text encoder)
                if hasattr(self._model, 'text_model'):
                    text_outputs = self._model.text_model(**text_inputs)
                    if isinstance(text_outputs, tuple):
                        text_emb = text_outputs[0]
                    else:
                        text_emb = text_outputs.last_hidden_state
                    # Apply text_projection if available
                    if hasattr(self._model, 'text_projection'):
                        text_emb = self._model.text_projection(text_emb)
                    # Take CLS token or mean
                    if text_emb.dim() == 3:
                        text_emb = text_emb[:, 0, :]
                    else:
                        text_emb = text_emb.mean(dim=1)
                elif hasattr(self._model, 'get_text_features'):
                    # SigLIP-style text encoding
                    text_emb = self._model.get_text_features(**text_inputs)
                else:
                    # Fallback: use full model
                    outputs = self._model(**text_inputs)
                    if hasattr(outputs, 'text_embeds'):
                        text_emb = outputs.text_embeds
                    else:
                        text_emb = outputs.last_hidden_state.mean(dim=1)

            return text_emb / text_emb.norm(dim=-1, keepdim=True)
        except Exception as e:
            LOGGER.error('Failed to encode texts: %s', e, exc_info=True)
            return None

    def _compute_similarities(
        self, video_emb: torch.Tensor, text_embs: torch.Tensor
    ) -> np.ndarray:
        """Вычисляет косинусные сходства между видео- и текстовыми эмбеддингами."""
        # Both are already normalized, so dot product = cosine similarity
        similarities = torch.matmul(text_embs, video_emb.unsqueeze(1)).squeeze(
            1
        )
        return similarities.cpu().numpy()
