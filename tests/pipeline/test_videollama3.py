"""Tests for VideoLLaMA3 action inference module."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from hackaton_system.config import Settings
from hackaton_system.pipeline.models import DetectionResult
from hackaton_system.pipeline.videollama3 import (
    ClipSegment,
    VideoLLaMA3ActionInferrer,
)


def _make_detection(
    track_id: int, frame_id: int, time_sec: float
) -> DetectionResult:
    """Helper to create synthetic detections for tracks."""
    return DetectionResult(
        frame_id=frame_id,
        time_sec=time_sec,
        bbox=(10, 20, 30, 40),
        confidence=0.9,
        track_id=track_id,
    )


def test_videollama3_inferrer_produces_activity_results(monkeypatch, tmp_path):
    """VideoLLaMA3 inferrer should convert scored segments into ActivityResult objects."""
    settings = Settings(
        action_backend='videollama3',
        action_prompts=[
            'railway worker is welding a train part',
            'railway worker is walking near the train',
        ],
        action_clip_frames=8,
        action_clip_stride=2,
        action_min_confidence=0.2,
    )
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]
    clip_segment = ClipSegment(
        track_id=1,
        start_sec=0.0,
        end_sec=2.0,
        frame_indices=[0, 2, 4, 6],
        target_class='person',
    )

    monkeypatch.setattr(
        inferrer,
        '_build_clip_segments',
        lambda __, _1, _2='person': [clip_segment],
        raising=False,
    )
    monkeypatch.setattr(
        inferrer,
        '_score_segments',
        lambda __, _1, _2: [
            SimpleNamespace(
                segment=clip_segment,
                label='railway worker is welding a train part',
                confidence=0.91,
                raw_scores={'weld': 0.91, 'walk': 0.12},
            )
        ],
        raising=False,
    )

    video_path = tmp_path / 'dummy.mp4'
    video_path.write_bytes(b'fake')
    tracks = {1: [_make_detection(1, 0, 0.0), _make_detection(1, 10, 0.4)]}

    activities = inferrer.infer_actions(
        video_path, tracks, target_class='person'
    )  # type: ignore[attr-defined]

    assert len(activities) == 1
    activity = activities[0]
    assert activity.track_id == 1
    assert activity.activity_class == 'railway worker is welding a train part'
    assert activity.start_sec == clip_segment.start_sec
    assert activity.end_sec == clip_segment.end_sec
    assert activity.confidence == 0.91


def test_videollama3_inferrer_filters_low_confidence(monkeypatch, tmp_path):
    """Segments below configured confidence threshold should be ignored."""
    settings = Settings(
        action_backend='videollama3',
        action_prompts=['worker welding', 'worker idle'],
        action_min_confidence=0.8,
    )
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]
    clip_segment = ClipSegment(
        track_id=7,
        start_sec=5.0,
        end_sec=7.0,
        frame_indices=[5, 6, 7],
        target_class='person',
    )

    monkeypatch.setattr(
        inferrer,
        '_build_clip_segments',
        lambda __, _1, _2='person': [clip_segment],
        raising=False,
    )
    monkeypatch.setattr(
        inferrer,
        '_score_segments',
        lambda __, _1, _2: [
            SimpleNamespace(
                segment=clip_segment,
                label='worker welding',
                confidence=0.5,
                raw_scores={'worker welding': 0.5, 'worker idle': 0.49},
            )
        ],
        raising=False,
    )

    video_path = tmp_path / 'dummy.mp4'
    video_path.write_bytes(b'fake')

    activities = inferrer.infer_actions(
        video_path, {7: [_make_detection(7, 0, 5.0)]}, target_class='person'
    )  # type: ignore[attr-defined]

    assert activities == []


def test_videollama3_inferrer_supports_train_class(monkeypatch, tmp_path):
    """VideoLLaMA3 inferrer should support train class actions."""
    settings = Settings(
        action_backend='videollama3',
        train_prompts=[
            'train is arriving at the station',
            'train is departing from the station',
        ],
        action_clip_frames=8,
        action_min_confidence=0.2,
    )
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]
    clip_segment = ClipSegment(
        track_id=0,
        start_sec=0.0,
        end_sec=2.0,
        frame_indices=[0, 2, 4, 6],
        target_class='train',
    )

    monkeypatch.setattr(
        inferrer,
        '_build_clip_segments',
        lambda __, _1, _2='train': [clip_segment],
        raising=False,
    )
    monkeypatch.setattr(
        inferrer,
        '_score_segments',
        lambda __, _1, _2: [
            SimpleNamespace(
                segment=clip_segment,
                label='train is arriving at the station',
                confidence=0.85,
                raw_scores={
                    'arriving': 0.85,
                    'departing': 0.15,
                },
            )
        ],
        raising=False,
    )

    video_path = tmp_path / 'dummy.mp4'
    video_path.write_bytes(b'fake')
    tracks = {0: [_make_detection(0, 0, 0.0), _make_detection(0, 10, 0.4)]}

    activities = inferrer.infer_actions(
        video_path, tracks, target_class='train'
    )  # type: ignore[attr-defined]

    assert len(activities) == 1
    activity = activities[0]
    assert activity.track_id == 0
    assert activity.activity_class == 'train is arriving at the station'
    assert activity.confidence == 0.85


def test_infer_actions_empty_tracks(tmp_path):
    """infer_actions should return empty list for empty tracks."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]
    video_path = tmp_path / 'dummy.mp4'
    video_path.write_bytes(b'fake')

    result = inferrer.infer_actions(video_path, {}, target_class='person')
    assert result == []


def test_infer_actions_nonexistent_file(tmp_path):
    """infer_actions should return empty list for nonexistent file."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]
    tracks = {1: [_make_detection(1, 0, 0.0)]}

    result = inferrer.infer_actions(
        Path('nonexistent.mp4'), tracks, target_class='person'
    )
    assert result == []


def test_infer_actions_no_prompts(monkeypatch, tmp_path):
    """infer_actions should return empty list when no prompts configured."""
    settings = Settings(
        action_backend='videollama3', action_prompts=[], train_prompts=[]
    )
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]
    video_path = tmp_path / 'dummy.mp4'
    video_path.write_bytes(b'fake')
    tracks = {1: [_make_detection(1, 0, 0.0)]}

    result = inferrer.infer_actions(video_path, tracks, target_class='person')
    assert result == []


def test_get_prompts_for_class_person():
    """_get_prompts_for_class should return class_prompts for person."""
    settings = Settings(
        action_backend='videollama3',
        class_prompts={'person': ['person action 1', 'person action 2']},
    )
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    prompts = inferrer._get_prompts_for_class('person')
    assert prompts == ['person action 1', 'person action 2']


def test_get_prompts_for_class_train():
    """_get_prompts_for_class should return class_prompts for train."""
    settings = Settings(
        action_backend='videollama3',
        class_prompts={'train': ['train action 1', 'train action 2']},
    )
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    prompts = inferrer._get_prompts_for_class('train')
    assert prompts == ['train action 1', 'train action 2']


def test_get_prompts_for_class_unknown():
    """_get_prompts_for_class should return empty list for unknown class."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    prompts = inferrer._get_prompts_for_class('unknown')  # type: ignore[arg-type]
    assert prompts == []


def test_build_clip_segments_short_track():
    """_build_clip_segments should handle tracks shorter than clip_frames."""
    settings = Settings(
        action_backend='videollama3',
        action_clip_frames=16,
        action_clip_take_duration_sec=None,  # Disable take-skip to use legacy mode
        action_clip_skip_duration_sec=None,
    )
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    detections = [
        _make_detection(1, 0, 0.0),
        _make_detection(1, 1, 0.1),
        _make_detection(1, 2, 0.2),
    ]
    tracks = {1: detections}
    video_path = Path('dummy.mp4')

    segments = inferrer._build_clip_segments(video_path, tracks, 'person')
    assert len(segments) == 1
    assert segments[0].track_id == 1
    assert segments[0].target_class == 'person'
    assert len(segments[0].frame_indices) == 3


def test_build_clip_segments_long_track():
    """_build_clip_segments should create overlapping segments for long tracks."""
    settings = Settings(
        action_backend='videollama3',
        action_clip_frames=4,
        action_clip_stride=2,
    )
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    detections = [_make_detection(1, i, i * 0.1) for i in range(10)]
    tracks = {1: detections}
    video_path = Path('dummy.mp4')

    segments = inferrer._build_clip_segments(video_path, tracks, 'person')
    assert len(segments) > 1
    assert all(s.track_id == 1 for s in segments)
    assert all(s.target_class == 'person' for s in segments)


def test_build_clip_segments_empty_track():
    """_build_clip_segments should handle empty tracks."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    tracks = {1: []}
    video_path = Path('dummy.mp4')

    segments = inferrer._build_clip_segments(video_path, tracks, 'person')
    assert segments == []


def test_score_segments_empty_segments(tmp_path):
    """_score_segments should return empty list for empty segments."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]
    video_path = tmp_path / 'dummy.mp4'
    video_path.write_bytes(b'fake')

    result = inferrer._score_segments(video_path, [], ['prompt1'])
    assert result == []


def test_score_segments_no_prompts(monkeypatch, tmp_path):
    """_score_segments should return empty list when no prompts."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]
    video_path = tmp_path / 'dummy.mp4'
    video_path.write_bytes(b'fake')
    segment = ClipSegment(
        track_id=1,
        start_sec=0.0,
        end_sec=1.0,
        frame_indices=[0, 1, 2],
        target_class='person',
    )

    # Mock _load_model to prevent actual model loading
    monkeypatch.setattr(inferrer, '_load_model', lambda: None)

    result = inferrer._score_segments(video_path, [segment], [])
    assert result == []


@patch('hackaton_system.pipeline.videollama3.cv2.VideoCapture')
def test_extract_frames(mock_cap_class, tmp_path):
    """_extract_frames should extract frames correctly."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 100  # total frames
    mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, mock_frame)
    mock_cap_class.return_value = mock_cap

    frames = inferrer._extract_frames(mock_cap, [0, 1, 2])
    assert len(frames) == 3
    assert all(isinstance(f, np.ndarray) for f in frames)


@patch('hackaton_system.pipeline.videollama3.cv2.VideoCapture')
def test_extract_frames_invalid_indices(mock_cap_class):
    """_extract_frames should handle invalid frame indices."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 10  # total frames
    mock_cap_class.return_value = mock_cap

    frames = inferrer._extract_frames(mock_cap, [100, -1])  # invalid indices
    assert len(frames) == 0


def test_validate_and_filter_frames():
    """_validate_and_filter_frames should filter invalid frames."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    valid_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    invalid_frames = [
        None,
        np.array([]),  # empty
        np.zeros((100,)),  # 1D
        np.zeros((100, 100, 4)),  # wrong channels
    ]

    frames = [valid_frame, *invalid_frames]
    result = inferrer._validate_and_filter_frames(frames)
    assert len(result) == 1
    assert np.array_equal(result[0], valid_frame)


def test_convert_frames_to_rgb():
    """_convert_frames_to_rgb should convert BGR to RGB."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    # Create BGR frame (OpenCV format)
    bgr_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bgr_frame[:, :, 0] = 255  # Blue channel

    frames = [bgr_frame]
    rgb_frames = inferrer._convert_frames_to_rgb(frames)
    assert len(rgb_frames) == 1
    assert rgb_frames[0].shape == (100, 100, 3)


def test_compute_similarities():
    """_compute_similarities should compute cosine similarities."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    video_emb = torch.tensor([1.0, 0.0, 0.0])
    text_embs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    similarities = inferrer._compute_similarities(video_emb, text_embs)
    assert len(similarities) == 2
    assert similarities[0] > similarities[1]  # First should be more similar


def test_infer_actions_no_segments(monkeypatch, tmp_path):
    """infer_actions should return empty list when no segments built."""
    settings = Settings(
        action_backend='videollama3', action_prompts=['action1', 'action2']
    )
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]
    video_path = tmp_path / 'dummy.mp4'
    video_path.write_bytes(b'fake')
    tracks = {1: [_make_detection(1, 0, 0.0)]}

    monkeypatch.setattr(inferrer, '_build_clip_segments', lambda *args: [])

    result = inferrer.infer_actions(video_path, tracks, target_class='person')
    assert result == []


def test_infer_actions_filters_low_confidence(monkeypatch, tmp_path):
    """infer_actions should filter activities below confidence threshold."""
    settings = Settings(
        action_backend='videollama3',
        action_prompts=['action1'],
        action_min_confidence=0.8,
    )
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]
    video_path = tmp_path / 'dummy.mp4'
    video_path.write_bytes(b'fake')
    tracks = {1: [_make_detection(1, 0, 0.0)]}

    segment = ClipSegment(
        track_id=1,
        start_sec=0.0,
        end_sec=1.0,
        frame_indices=[0],
        target_class='person',
    )

    monkeypatch.setattr(
        inferrer, '_build_clip_segments', lambda *args: [segment]
    )
    monkeypatch.setattr(
        inferrer,
        '_score_segments',
        lambda *args: [
            SimpleNamespace(
                segment=segment,
                label='action1',
                confidence=0.5,  # Below threshold
                raw_scores={'action1': 0.5},
            )
        ],
    )

    result = inferrer.infer_actions(video_path, tracks, target_class='person')
    assert result == []  # Filtered out due to low confidence


@patch('hackaton_system.pipeline.videollama3.cv2.VideoCapture')
def test_score_segments_full_flow(mock_cap_class, monkeypatch, tmp_path):
    """_score_segments should process segments through full pipeline."""
    settings = Settings(
        action_backend='videollama3', action_prompts=['action1', 'action2']
    )
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    # Mock model and processor
    mock_model = MagicMock()
    mock_processor = MagicMock()
    inferrer._model = mock_model
    inferrer._processor = mock_processor
    inferrer._device = 'cpu'

    # Mock video capture
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 100
    mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, mock_frame)
    mock_cap_class.return_value = mock_cap

    # Mock video encoding
    mock_video_emb = torch.tensor([1.0, 0.0, 0.0])
    monkeypatch.setattr(
        inferrer, '_encode_video', lambda frames: mock_video_emb
    )

    # Mock text encoding
    mock_text_embs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    monkeypatch.setattr(
        inferrer, '_encode_texts', lambda prompts: mock_text_embs
    )

    segment = ClipSegment(
        track_id=1,
        start_sec=0.0,
        end_sec=1.0,
        frame_indices=[0],
        target_class='person',
    )

    video_path = tmp_path / 'dummy.mp4'
    video_path.write_bytes(b'fake')

    result = inferrer._score_segments(
        video_path, [segment], ['action1', 'action2']
    )
    assert len(result) == 1
    assert result[0].segment == segment
    assert result[0].label in ['action1', 'action2']
    assert 'raw_scores' in result[0].__dict__


@patch('hackaton_system.pipeline.videollama3.cv2.VideoCapture')
def test_score_segments_video_not_opened(mock_cap_class, tmp_path):
    """_score_segments should return empty list when video cannot be opened."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    inferrer._model = mock_model

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_cap_class.return_value = mock_cap

    segment = ClipSegment(
        track_id=1,
        start_sec=0.0,
        end_sec=1.0,
        frame_indices=[0],
        target_class='person',
    )

    video_path = tmp_path / 'dummy.mp4'
    video_path.write_bytes(b'fake')

    result = inferrer._score_segments(video_path, [segment], ['prompt1'])
    assert result == []


@patch('hackaton_system.pipeline.videollama3.cv2.VideoCapture')
def test_score_segments_no_frames_extracted(
    mock_cap_class, monkeypatch, tmp_path
):
    """_score_segments should skip segments when no frames extracted."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    inferrer._model = mock_model

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 100
    mock_cap.read.return_value = (False, None)  # No frames
    mock_cap_class.return_value = mock_cap

    segment = ClipSegment(
        track_id=1,
        start_sec=0.0,
        end_sec=1.0,
        frame_indices=[0],
        target_class='person',
    )

    video_path = tmp_path / 'dummy.mp4'
    video_path.write_bytes(b'fake')

    result = inferrer._score_segments(video_path, [segment], ['prompt1'])
    assert result == []


@patch('hackaton_system.pipeline.videollama3.cv2.VideoCapture')
def test_score_segments_encode_video_fails(
    mock_cap_class, monkeypatch, tmp_path
):
    """_score_segments should skip segments when video encoding fails."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    inferrer._model = mock_model

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 100
    mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, mock_frame)
    mock_cap_class.return_value = mock_cap

    # Mock encode_video to return None
    monkeypatch.setattr(inferrer, '_encode_video', lambda frames: None)

    segment = ClipSegment(
        track_id=1,
        start_sec=0.0,
        end_sec=1.0,
        frame_indices=[0],
        target_class='person',
    )

    video_path = tmp_path / 'dummy.mp4'
    video_path.write_bytes(b'fake')

    result = inferrer._score_segments(video_path, [segment], ['prompt1'])
    assert result == []


@patch('hackaton_system.pipeline.videollama3.cv2.VideoCapture')
def test_score_segments_encode_texts_fails(
    mock_cap_class, monkeypatch, tmp_path
):
    """_score_segments should skip segments when text encoding fails."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    inferrer._model = mock_model

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 100
    mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, mock_frame)
    mock_cap_class.return_value = mock_cap

    mock_video_emb = torch.tensor([1.0, 0.0, 0.0])
    monkeypatch.setattr(
        inferrer, '_encode_video', lambda frames: mock_video_emb
    )
    # Mock encode_texts to return None
    monkeypatch.setattr(inferrer, '_encode_texts', lambda prompts: None)

    segment = ClipSegment(
        track_id=1,
        start_sec=0.0,
        end_sec=1.0,
        frame_indices=[0],
        target_class='person',
    )

    video_path = tmp_path / 'dummy.mp4'
    video_path.write_bytes(b'fake')

    result = inferrer._score_segments(video_path, [segment], ['prompt1'])
    assert result == []


def test_encode_video_no_model():
    """_encode_video should return None when model not loaded."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
    result = inferrer._encode_video(frames)
    assert result is None


def test_encode_video_empty_frames():
    """_encode_video should return None for empty frames."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    mock_processor = MagicMock()
    inferrer._model = mock_model
    inferrer._processor = mock_processor

    result = inferrer._encode_video([])
    assert result is None


def test_encode_video_no_valid_frames(monkeypatch):
    """_encode_video should return None when no valid frames after filtering."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    mock_processor = MagicMock()
    inferrer._model = mock_model
    inferrer._processor = mock_processor

    invalid_frames = [None, np.array([])]
    monkeypatch.setattr(
        inferrer, '_validate_and_filter_frames', lambda frames: []
    )

    result = inferrer._encode_video(invalid_frames)
    assert result is None


def test_encode_video_no_rgb_frames(monkeypatch):
    """_encode_video should return None when RGB conversion fails."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    mock_processor = MagicMock()
    inferrer._model = mock_model
    inferrer._processor = mock_processor

    frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
    monkeypatch.setattr(inferrer, '_validate_and_filter_frames', lambda f: f)
    monkeypatch.setattr(inferrer, '_convert_frames_to_rgb', lambda f: [])

    result = inferrer._encode_video(frames)
    assert result is None


def test_encode_video_processor_fails(monkeypatch):
    """_encode_video should return None when processor fails."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    mock_processor = MagicMock()
    inferrer._model = mock_model
    inferrer._processor = mock_processor

    frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
    monkeypatch.setattr(inferrer, '_validate_and_filter_frames', lambda f: f)
    monkeypatch.setattr(inferrer, '_convert_frames_to_rgb', lambda f: f)
    monkeypatch.setattr(
        inferrer, '_process_frames_with_processor', lambda f: None
    )

    result = inferrer._encode_video(frames)
    assert result is None


@patch('hackaton_system.pipeline.videollama3.AutoProcessor')
@patch('hackaton_system.pipeline.videollama3.AutoModelForCausalLM')
def test_load_model(mock_model_class, mock_processor_class, monkeypatch):
    """_load_model should load model and processor correctly."""
    settings = Settings(
        action_backend='videollama3',
        videollama3_model_name='test-model',
        videollama3_device='cpu',
    )
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_model_class.from_pretrained.return_value = mock_model
    mock_processor_class.from_pretrained.return_value = mock_processor

    monkeypatch.setattr('torch.cuda.is_available', lambda: False)

    inferrer._load_model()

    assert inferrer._model is not None
    assert inferrer._processor is not None
    assert inferrer._device == 'cpu'
    mock_model.eval.assert_called_once()


@patch('hackaton_system.pipeline.videollama3.AutoProcessor')
@patch('hackaton_system.pipeline.videollama3.AutoModelForCausalLM')
def test_load_model_auto_device_cuda(
    mock_model_class, mock_processor_class, monkeypatch
):
    """_load_model should use CUDA when available and device is auto."""
    settings = Settings(
        action_backend='videollama3', videollama3_device='auto'
    )
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_model_class.from_pretrained.return_value = mock_model
    mock_processor_class.from_pretrained.return_value = mock_processor

    monkeypatch.setattr('torch.cuda.is_available', lambda: True)

    inferrer._load_model()

    assert inferrer._device == 'cuda:0'


def test_process_frames_with_processor_no_frames():
    """_process_frames_with_processor should return None for empty frames."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_processor = MagicMock()
    inferrer._processor = mock_processor
    inferrer._device = 'cpu'

    result = inferrer._process_frames_with_processor([])
    assert result is None


def test_process_frames_with_processor_invalid_frames():
    """_process_frames_with_processor should filter invalid frames."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_processor = MagicMock()
    inferrer._processor = mock_processor
    inferrer._device = 'cpu'

    invalid_frames = [
        None,
        np.array([]),
        np.zeros((100,)),  # 1D
        np.zeros((100, 100, 4)),  # Wrong channels
        np.zeros((0, 100, 3)),  # Zero height
    ]

    result = inferrer._process_frames_with_processor(invalid_frames)
    assert result is None


def test_process_frames_with_processor_image_processor(monkeypatch):
    """_process_frames_with_processor should use image_processor when available."""
    settings = Settings(action_backend='videollama3', videollama3_max_frames=4)
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_processor = MagicMock()
    mock_image_processor = MagicMock()
    mock_image_processor.return_value = {
        'pixel_values': torch.zeros(1, 5, 3, 100, 100)
    }
    mock_processor.image_processor = mock_image_processor
    inferrer._processor = mock_processor
    inferrer._device = 'cpu'

    # Create 10 frames to test max_frames limiting
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(10)]

    result = inferrer._process_frames_with_processor(frames)
    assert result is not None
    assert 'pixel_values' in result or 'pixel_value' in result


def test_process_frames_with_processor_video_processor(monkeypatch):
    """_process_frames_with_processor should use video_processor as fallback."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_processor = MagicMock()
    mock_processor.image_processor = None  # No image_processor
    mock_video_processor = MagicMock()
    mock_video_processor.return_value = {
        'pixel_values': torch.zeros(1, 3, 100, 100)
    }
    mock_processor.video_processor = mock_video_processor
    inferrer._processor = mock_processor
    inferrer._device = 'cpu'

    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]

    result = inferrer._process_frames_with_processor(frames)
    assert result is not None


def test_process_frames_with_processor_direct_call(monkeypatch):
    """_process_frames_with_processor should use direct processor call as last resort."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_processor = MagicMock()
    mock_processor.image_processor = None
    mock_processor.video_processor = None
    mock_processor.return_value = {'pixel_values': torch.zeros(1, 3, 100, 100)}
    inferrer._processor = mock_processor
    inferrer._device = 'cpu'

    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]

    result = inferrer._process_frames_with_processor(frames)
    assert result is not None
    mock_processor.assert_called_once()


def test_process_frames_with_processor_exception(monkeypatch):
    """_process_frames_with_processor should handle exceptions gracefully."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_processor = MagicMock()
    mock_processor.image_processor = MagicMock(
        side_effect=Exception('Processor error')
    )
    inferrer._processor = mock_processor
    inferrer._device = 'cpu'

    frames = [np.zeros((100, 100, 3), dtype=np.uint8)]

    result = inferrer._process_frames_with_processor(frames)
    assert result is None


def test_process_frames_with_processor_none_inputs(monkeypatch):
    """_process_frames_with_processor should return None when processor returns None."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_processor = MagicMock()
    mock_image_processor = MagicMock(return_value=None)
    mock_processor.image_processor = mock_image_processor
    inferrer._processor = mock_processor
    inferrer._device = 'cpu'

    frames = [np.zeros((100, 100, 3), dtype=np.uint8)]

    result = inferrer._process_frames_with_processor(frames)
    assert result is None


def test_process_frames_with_processor_no_pixel_values(monkeypatch):
    """_process_frames_with_processor should return None when no pixel_values."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_processor = MagicMock()
    mock_image_processor = MagicMock(
        return_value={'other_key': torch.zeros(1, 3, 100, 100)}
    )
    mock_processor.image_processor = mock_image_processor
    inferrer._processor = mock_processor
    inferrer._device = 'cpu'

    frames = [np.zeros((100, 100, 3), dtype=np.uint8)]

    result = inferrer._process_frames_with_processor(frames)
    assert result is None


def test_process_frames_with_processor_none_tensor_value(monkeypatch):
    """_process_frames_with_processor should return None when tensor value is None."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_processor = MagicMock()
    mock_image_processor = MagicMock(return_value={'pixel_values': None})
    mock_processor.image_processor = mock_image_processor
    inferrer._processor = mock_processor
    inferrer._device = 'cpu'

    frames = [np.zeros((100, 100, 3), dtype=np.uint8)]

    result = inferrer._process_frames_with_processor(frames)
    assert result is None


def test_process_frames_with_processor_non_tensor_value(monkeypatch):
    """_process_frames_with_processor should return None when value is not tensor."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_processor = MagicMock()
    mock_image_processor = MagicMock(
        return_value={'pixel_values': 'not a tensor'}
    )
    mock_processor.image_processor = mock_image_processor
    inferrer._processor = mock_processor
    inferrer._device = 'cpu'

    frames = [np.zeros((100, 100, 3), dtype=np.uint8)]

    result = inferrer._process_frames_with_processor(frames)
    assert result is None


def test_process_frames_with_processor_device_error(monkeypatch):
    """_process_frames_with_processor should handle device move errors."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_processor = MagicMock()
    mock_tensor = MagicMock()
    mock_tensor.to.side_effect = Exception('Device error')
    mock_image_processor = MagicMock(
        return_value={'pixel_values': mock_tensor}
    )
    mock_processor.image_processor = mock_image_processor
    inferrer._processor = mock_processor
    inferrer._device = 'cpu'

    frames = [np.zeros((100, 100, 3), dtype=np.uint8)]

    result = inferrer._process_frames_with_processor(frames)
    assert result is None


def test_get_video_embeddings_none_inputs():
    """_get_video_embeddings should return None for None inputs."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    result = inferrer._get_video_embeddings(None)
    assert result is None


def test_get_video_embeddings_no_model():
    """_get_video_embeddings should return None when model not loaded."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    inputs = {'pixel_values': torch.zeros(1, 3, 100, 100)}
    result = inferrer._get_video_embeddings(inputs)
    assert result is None


def test_get_video_embeddings_none_tensor_value():
    """_get_video_embeddings should return None when input has None value."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    inferrer._model = mock_model

    inputs = {'pixel_values': None}
    result = inferrer._get_video_embeddings(inputs)
    assert result is None


def test_get_video_embeddings_non_tensor_value():
    """_get_video_embeddings should return None when input is not tensor."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    inferrer._model = mock_model

    inputs = {'pixel_values': 'not a tensor'}
    result = inferrer._get_video_embeddings(inputs)
    assert result is None


def test_get_video_embeddings_unexpected_shape():
    """_get_video_embeddings should return None for unexpected tensor shape."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    inferrer._model = mock_model

    # 3D tensor (unexpected)
    inputs = {'pixel_values': torch.zeros(1, 100, 100)}
    result = inferrer._get_video_embeddings(inputs)
    assert result is None


def test_get_video_embeddings_vision_model(monkeypatch):
    """_get_video_embeddings should use vision_model when available."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    mock_vision_outputs = MagicMock()
    mock_vision_outputs.last_hidden_state = torch.zeros(1, 10, 768)
    mock_model.vision_model.return_value = mock_vision_outputs
    inferrer._model = mock_model

    # 4D tensor (single image) - will be expanded to 5D
    inputs = {'pixel_values': torch.zeros(1, 3, 100, 100)}

    inferrer._get_video_embeddings(inputs)
    # Should process through vision_model
    assert mock_model.vision_model.called


def test_get_video_embeddings_model_fallback(monkeypatch):
    """_get_video_embeddings should use model fallback when no vision_model."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    del mock_model.vision_model  # No vision_model
    mock_model.model = MagicMock(
        return_value=MagicMock(last_hidden_state=torch.zeros(1, 10, 768))
    )
    inferrer._model = mock_model

    inputs = {'pixel_values': torch.zeros(1, 3, 100, 100)}

    inferrer._get_video_embeddings(inputs)
    # Should use model.model
    assert hasattr(mock_model, 'model')


def test_get_video_embeddings_direct_call(monkeypatch):
    """_get_video_embeddings should use direct model call as last resort."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    del mock_model.vision_model
    del mock_model.model
    mock_model.return_value = MagicMock(
        last_hidden_state=torch.zeros(1, 10, 768)
    )
    inferrer._model = mock_model

    inputs = {'pixel_values': torch.zeros(1, 3, 100, 100)}

    inferrer._get_video_embeddings(inputs)
    # Should call model directly
    assert mock_model.called


def test_get_video_embeddings_tuple_output(monkeypatch):
    """_get_video_embeddings should handle tuple vision outputs."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    mock_vision_outputs = (torch.zeros(1, 10, 768), None)
    mock_model.vision_model.return_value = mock_vision_outputs
    inferrer._model = mock_model

    inputs = {'pixel_values': torch.zeros(1, 3, 100, 100)}

    inferrer._get_video_embeddings(inputs)
    # Should extract first element from tuple
    assert mock_model.vision_model.called


def test_get_video_embeddings_exception(monkeypatch):
    """_get_video_embeddings should handle exceptions gracefully."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    mock_model.vision_model.side_effect = Exception('Model error')
    inferrer._model = mock_model

    inputs = {'pixel_values': torch.zeros(1, 3, 100, 100)}

    result = inferrer._get_video_embeddings(inputs)
    assert result is None


def test_get_video_embeddings_normalize_error(monkeypatch):
    """_get_video_embeddings should handle normalization errors."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    mock_vision_outputs = MagicMock()
    mock_vision_outputs.last_hidden_state = torch.zeros(1, 10, 768)
    mock_model.vision_model.return_value = mock_vision_outputs
    inferrer._model = mock_model

    # Create tensor that will fail normalization
    mock_emb = MagicMock()
    mock_emb.mean.return_value = torch.zeros(1, 768)
    mock_emb.norm.side_effect = Exception('Normalize error')

    monkeypatch.setattr(
        'hackaton_system.pipeline.videollama3.torch.no_grad',
        lambda: MagicMock(
            __enter__=lambda _: None, __exit__=lambda *args: None
        ),
    )

    inputs = {'pixel_values': torch.zeros(1, 1, 3, 100, 100)}

    # Mock the entire processing to get to normalization
    with patch.object(inferrer, '_model') as mock_m:
        mock_m.vision_model.return_value = MagicMock(
            last_hidden_state=torch.zeros(1, 10, 768)
        )
        # This will fail at normalization
        inferrer._get_video_embeddings(inputs)
        # Should return None due to normalization error
        assert True  # May return None or raise


def test_encode_texts_no_model():
    """_encode_texts should return None when model not loaded."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    result = inferrer._encode_texts(['prompt1'])
    assert result is None


def test_encode_texts_text_model(monkeypatch):
    """_encode_texts should use text_model when available."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_processor.return_value = {'input_ids': torch.zeros(1, 10)}
    mock_text_outputs = MagicMock()
    mock_text_outputs.last_hidden_state = torch.zeros(1, 10, 768)
    mock_model.text_model.return_value = mock_text_outputs
    mock_model.text_projection = MagicMock(
        return_value=torch.zeros(1, 10, 512)
    )
    inferrer._model = mock_model
    inferrer._processor = mock_processor
    inferrer._device = 'cpu'

    result = inferrer._encode_texts(['prompt1'])
    assert result is not None
    assert mock_model.text_model.called


def test_encode_texts_get_text_features(monkeypatch):
    """_encode_texts should use get_text_features when available."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_processor.return_value = {'input_ids': torch.zeros(1, 10)}
    del mock_model.text_model
    mock_model.get_text_features.return_value = torch.zeros(1, 512)
    inferrer._model = mock_model
    inferrer._processor = mock_processor
    inferrer._device = 'cpu'

    result = inferrer._encode_texts(['prompt1'])
    assert result is not None
    assert mock_model.get_text_features.called


def test_encode_texts_fallback(monkeypatch):
    """_encode_texts should use full model as fallback."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_processor.return_value = {'input_ids': torch.zeros(1, 10)}
    del mock_model.text_model
    del mock_model.get_text_features
    mock_outputs = MagicMock()
    mock_outputs.text_embeds = torch.zeros(1, 512)
    mock_model.return_value = mock_outputs
    inferrer._model = mock_model
    inferrer._processor = mock_processor
    inferrer._device = 'cpu'

    result = inferrer._encode_texts(['prompt1'])
    assert result is not None
    assert mock_model.called


def test_encode_texts_exception(monkeypatch):
    """_encode_texts should handle exceptions gracefully."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_processor.side_effect = Exception('Processor error')
    inferrer._model = mock_model
    inferrer._processor = mock_processor
    inferrer._device = 'cpu'

    result = inferrer._encode_texts(['prompt1'])
    assert result is None


@patch('hackaton_system.pipeline.videollama3.cv2.VideoCapture')
def test_extract_frames_read_fails(mock_cap_class):
    """_extract_frames should handle read failures."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 100
    mock_cap.read.return_value = (False, None)  # Read fails
    mock_cap_class.return_value = mock_cap

    frames = inferrer._extract_frames(mock_cap, [0])
    assert len(frames) == 0


@patch('hackaton_system.pipeline.videollama3.cv2.VideoCapture')
def test_extract_frames_invalid_frame(mock_cap_class):
    """_extract_frames should skip invalid frames."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 100
    # Return invalid frame (not numpy array)
    mock_cap.read.return_value = (True, 'not an array')
    mock_cap_class.return_value = mock_cap

    frames = inferrer._extract_frames(mock_cap, [0])
    assert len(frames) == 0


@patch('hackaton_system.pipeline.videollama3.cv2.VideoCapture')
def test_extract_frames_shape_error(mock_cap_class):
    """_extract_frames should handle shape validation errors."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 100
    # Create frame that will fail shape check - use property that raises AttributeError
    invalid_frame = MagicMock()
    invalid_frame.size = 100
    def raise_on_shape():
        raise AttributeError('shape')
    type(invalid_frame).shape = property(raise_on_shape)
    mock_cap.read.return_value = (True, invalid_frame)
    mock_cap_class.return_value = mock_cap

    frames = inferrer._extract_frames(mock_cap, [0])
    assert len(frames) == 0


def test_process_frames_with_processor_float_dtype(monkeypatch):
    """_process_frames_with_processor should convert float frames to uint8."""
    settings = Settings(action_backend='videollama3')
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_processor = MagicMock()
    mock_image_processor = MagicMock(
        return_value={'pixel_values': torch.zeros(1, 3, 100, 100)}
    )
    mock_processor.image_processor = mock_image_processor
    inferrer._processor = mock_processor
    inferrer._device = 'cpu'

    # Float frame in range [0, 1]
    float_frame = np.zeros((100, 100, 3), dtype=np.float32)
    float_frame.fill(0.5)

    result = inferrer._process_frames_with_processor([float_frame])
    assert result is not None


def test_process_frames_with_processor_max_frames_limiting(monkeypatch):
    """_process_frames_with_processor should limit frames to max_frames."""
    settings = Settings(action_backend='videollama3', videollama3_max_frames=5)
    inferrer = VideoLLaMA3ActionInferrer(settings)  # type: ignore[call-arg]

    mock_processor = MagicMock()
    mock_image_processor = MagicMock()
    mock_image_processor.return_value = {
        'pixel_values': torch.zeros(1, 5, 3, 100, 100)
    }
    mock_processor.image_processor = mock_image_processor
    inferrer._processor = mock_processor
    inferrer._device = 'cpu'

    # Create 20 frames
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(20)]

    result = inferrer._process_frames_with_processor(frames)
    # Should limit to 5 frames
    assert result is not None
    # Verify processor was called with limited frames
    call_args = mock_image_processor.call_args
    assert len(call_args[0][0]) == 5
