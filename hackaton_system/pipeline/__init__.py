"""Утилиты и публичные объекты для пайплайна обработки видео."""

from .models import (
    ActivityResult,
    DetectionResult,
    DetectorProtocol,
    FrameState,
    PoseResult,
    RoleAssignment,
)
from .video_processor import VideoProcessor

__all__ = [
    'ActivityResult',
    'DetectionResult',
    'DetectorProtocol',
    'FrameState',
    'PoseResult',
    'RoleAssignment',
    'VideoProcessor',
]
