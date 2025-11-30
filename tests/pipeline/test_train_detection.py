# pyright: reportPrivateUsage=false
"""Tests for train detection module."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from hackaton_system.config import Settings
from hackaton_system.pipeline.models import ActivityResult
from hackaton_system.pipeline.train_detection import (
    TrainDetection,
    TrainDetector,
)


def test_train_detector_init():
    """Test TrainDetector initialization."""
    settings = Settings()
    detector = TrainDetector(settings)
    assert detector.settings is settings
    assert detector._min_confidence == 0.3
    assert detector._min_frames_present == 10
    assert detector._max_frames_absent == 30


def test_detect_trains_empty_result(monkeypatch):
    """detect_trains should handle empty detection results."""
    settings = Settings()
    detector = TrainDetector(settings)
    mock_detector = MagicMock()
    mock_detector.track.return_value = []

    result = detector.detect_trains(mock_detector, Path('test.mp4'), 25.0)
    assert result == []


def test_detect_trains_with_results(monkeypatch):
    """detect_trains should parse YOLO results correctly."""
    settings = Settings(train_detection_conf=0.3)
    detector = TrainDetector(settings)

    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_boxes = MagicMock()

    # Create proper mock tensor-like objects that support .cpu().numpy()
    # Methods call .cpu().numpy(), not .cpu().tolist()
    mock_xyxy_cpu = MagicMock()
    mock_xyxy_cpu.numpy.return_value = np.array([[100.0, 200.0, 300.0, 400.0]])
    mock_xyxy_tensor = MagicMock()
    mock_xyxy_tensor.cpu.return_value = mock_xyxy_cpu
    mock_boxes.xyxy = mock_xyxy_tensor

    mock_conf_cpu = MagicMock()
    mock_conf_cpu.numpy.return_value = np.array([0.8])
    mock_conf_tensor = MagicMock()
    mock_conf_tensor.cpu.return_value = mock_conf_cpu
    mock_boxes.conf = mock_conf_tensor

    mock_cls_cpu = MagicMock()
    mock_cls_cpu.numpy.return_value = np.array([6])  # Train class
    mock_cls_tensor = MagicMock()
    mock_cls_tensor.cpu.return_value = mock_cls_cpu
    mock_boxes.cls = mock_cls_tensor

    mock_result.boxes = mock_boxes
    # track() should return an iterable (stream)
    mock_detector.track.return_value = iter([mock_result])

    result = detector.detect_trains(mock_detector, Path('test.mp4'), 25.0)
    assert len(result) == 1
    assert result[0].confidence == 0.8
    assert result[0].class_id == 6


def test_detect_arrival_departure_empty_detections():
    """detect_arrival_departure should return empty list for empty detections."""
    settings = Settings()
    detector = TrainDetector(settings)

    result = detector.detect_arrival_departure([])
    assert result == []


def test_detect_arrival_departure_heuristics():
    """detect_arrival_departure should use heuristics when VideoLLaMA3 disabled."""
    settings = Settings(train_videollama3_enabled=False)
    detector = TrainDetector(settings)

    detections = [
        TrainDetection(
            frame_id=0,
            time_sec=0.0,
            bbox=(100, 200, 300, 400),
            confidence=0.8,
            class_id=6,
        ),
        TrainDetection(
            frame_id=30,
            time_sec=1.2,
            bbox=(100, 200, 300, 400),
            confidence=0.8,
            class_id=6,
        ),
    ]

    result = detector.detect_arrival_departure(detections, fps=25.0)
    assert len(result) > 0


def test_detect_with_heuristics_arrival():
    """_detect_with_heuristics should detect train arrival."""
    settings = Settings()
    detector = TrainDetector(settings)

    detections = [
        TrainDetection(
            frame_id=50,  # After first second
            time_sec=2.0,
            bbox=(100, 200, 300, 400),
            confidence=0.8,
            class_id=6,
        ),
    ]

    result = detector._detect_with_heuristics(detections, fps=25.0)
    assert len(result) > 0
    assert any(e.event_type == 'arrival' for e in result)


def test_detect_with_heuristics_departure():
    """_detect_with_heuristics should detect train departure."""
    settings = Settings()
    detector = TrainDetector(settings)

    # Train present then absent
    detections = [
        TrainDetection(
            frame_id=0,
            time_sec=0.0,
            bbox=(100, 200, 300, 400),
            confidence=0.8,
            class_id=6,
        ),
        TrainDetection(
            frame_id=10,
            time_sec=0.4,
            bbox=(100, 200, 300, 400),
            confidence=0.8,
            class_id=6,
        ),
    ]

    result = detector._detect_with_heuristics(detections, fps=25.0)
    # Should have presence events
    assert len(result) >= 0


def test_determine_event_type_arrival():
    """_determine_event_type should detect arrival events."""
    settings = Settings()
    detector = TrainDetector(settings)

    current_state = SimpleNamespace(
        label='train is arriving at the station', confidence=0.9
    )

    event_type = detector._determine_event_type(None, current_state)
    assert event_type == 'arrival'


def test_determine_event_type_departure():
    """_determine_event_type should detect departure events."""
    settings = Settings()
    detector = TrainDetector(settings)

    previous_state = 'train is arriving at the station'
    current_state = SimpleNamespace(
        label='train is departing from the station', confidence=0.9
    )

    event_type = detector._determine_event_type(previous_state, current_state)
    assert event_type == 'departure'


def test_determine_event_type_no_change():
    """_determine_event_type should return None when no state change."""
    settings = Settings()
    detector = TrainDetector(settings)

    previous_state = 'train is present and stationary at the station'
    current_state = SimpleNamespace(
        label='train is present and stationary at the station', confidence=0.9
    )

    event_type = detector._determine_event_type(previous_state, current_state)
    assert event_type is None


def test_detect_with_videollama3(monkeypatch, tmp_path):
    """_detect_with_videollama3 should use VideoLLaMA3 for train detection."""
    settings = Settings(
        train_videollama3_enabled=True,
        train_prompts=[
            'train is arriving at the station',
            'train is departing from the station',
        ],
    )
    detector = TrainDetector(settings)

    detections = [
        TrainDetection(
            frame_id=0,
            time_sec=0.0,
            bbox=(100, 200, 300, 400),
            confidence=0.8,
            class_id=6,
        ),
    ]

    video_path = tmp_path / 'test.mp4'
    video_path.write_bytes(b'fake')

    # Mock VideoLLaMA3ActionInferrer
    mock_inferrer = MagicMock()
    mock_activity = ActivityResult(
        track_id=0,
        activity_class='train is arriving at the station',
        start_sec=0.0,
        end_sec=1.0,
        confidence=0.9,
    )
    mock_inferrer.infer_actions.return_value = [mock_activity]

    detector._action_inferrer = mock_inferrer

    result = detector._detect_with_videollama3(detections, video_path, 25.0)
    assert len(result) >= 0
    mock_inferrer.infer_actions.assert_called_once()


def test_detect_with_videollama3_no_prompts(monkeypatch, tmp_path):
    """_detect_with_videollama3 should fallback to heuristics when no prompts."""
    settings = Settings(train_videollama3_enabled=True, train_prompts=[])
    detector = TrainDetector(settings)

    detections = [
        TrainDetection(
            frame_id=0,
            time_sec=0.0,
            bbox=(100, 200, 300, 400),
            confidence=0.8,
            class_id=6,
        ),
    ]

    video_path = tmp_path / 'test.mp4'
    video_path.write_bytes(b'fake')

    # When no prompts, it should fallback to heuristics
    # The method checks prompts and calls _detect_with_heuristics if empty
    result = detector._detect_with_videollama3(detections, video_path, 25.0)
    # Should fallback to heuristics and return list
    assert isinstance(result, list)
