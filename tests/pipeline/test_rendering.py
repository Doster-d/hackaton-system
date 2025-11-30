# pyright: reportPrivateUsage=false
"""Tests for rendering module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from hackaton_system.config import Settings
from hackaton_system.pipeline.models import DetectionResult
from hackaton_system.pipeline.rendering import VideoRenderer


def test_render_preview_returns_none_when_prepare_fails(monkeypatch, tmp_path):
    """render_preview returns None when configuration cannot be prepared."""

    settings = Settings()
    renderer = VideoRenderer(settings)
    video_path = tmp_path / 'test.mp4'
    video_path.write_bytes(b'fake data')

    mock_prepare = MagicMock(return_value=None)
    monkeypatch.setattr(renderer, '_prepare_render_config', mock_prepare)
    mock_create = MagicMock()
    monkeypatch.setattr(renderer, '_create_ffmpeg_process', mock_create)

    result = renderer.render_preview(
        video_path, video_id=1, detections=[], fps=25.0
    )

    assert result is None
    mock_prepare.assert_called_once()
    mock_create.assert_not_called()


def test_render_preview_runs_pipeline(monkeypatch, tmp_path):
    """render_preview orchestrates helpers and returns output path."""

    settings = Settings()
    renderer = VideoRenderer(settings)
    video_path = tmp_path / 'video.mp4'
    video_path.write_bytes(b'data')

    config = {'cap': MagicMock()}
    mock_prepare = MagicMock(return_value=config)
    mock_output_path = tmp_path / 'preview.mp4'
    mock_process = MagicMock(stdin=None)

    monkeypatch.setattr(renderer, '_prepare_render_config', mock_prepare)
    monkeypatch.setattr(
        renderer,
        '_prepare_output_path',
        MagicMock(return_value=mock_output_path),
    )
    monkeypatch.setattr(
        renderer,
        '_create_ffmpeg_process',
        MagicMock(return_value=mock_process),
    )
    monkeypatch.setattr(
        renderer, '_build_frame_map', MagicMock(return_value={})
    )

    def fake_render_frames(config_arg, frame_map_arg, process_arg):
        assert config_arg is config
        assert frame_map_arg == {}
        assert process_arg is mock_process
        mock_output_path.touch()

    monkeypatch.setattr(renderer, '_render_frames', fake_render_frames)

    result = renderer.render_preview(
        video_path, video_id=1, detections=[], fps=25.0
    )

    assert result == mock_output_path


def test_render_preview_returns_none_when_ffmpeg_fails(monkeypatch, tmp_path):
    """render_preview returns None when ffmpeg process cannot be created."""

    settings = Settings()
    renderer = VideoRenderer(settings)
    video_path = tmp_path / 'video.mp4'
    video_path.write_bytes(b'data')

    config = {'cap': MagicMock()}
    monkeypatch.setattr(
        renderer, '_prepare_render_config', MagicMock(return_value=config)
    )
    monkeypatch.setattr(
        renderer, '_create_ffmpeg_process', MagicMock(return_value=None)
    )

    result = renderer.render_preview(
        video_path, video_id=1, detections=[], fps=25.0
    )

    assert result is None


def test_prepare_render_config(monkeypatch, tmp_path):
    """Test _prepare_render_config prepares render configuration."""

    settings = Settings()
    renderer = VideoRenderer(settings)

    test_video = tmp_path / 'test.mp4'
    test_video.write_bytes(b'fake video')

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        3: 1920,  # CAP_PROP_FRAME_WIDTH
        4: 1080,  # CAP_PROP_FRAME_HEIGHT
    }.get(prop, 0)

    monkeypatch.setattr(
        'hackaton_system.pipeline.rendering.cv2.VideoCapture',
        lambda _: mock_cap,
    )

    config = renderer._prepare_render_config(test_video, fps=25.0)

    assert config is not None
    assert 'cap' in config
    assert 'width' in config
    assert 'height' in config


def test_calculate_target_size():
    """Test _calculate_target_size calculates target size."""

    settings = Settings()
    renderer = VideoRenderer(settings)

    width, height, _ = renderer._calculate_target_size(
        1920, 1080, max_side=1280
    )

    assert width <= 1280
    assert height <= 1280
    assert abs((width / height) - (1920 / 1080)) < 0.01


def test_build_frame_map():
    """Test _build_frame_map builds frame to detections mapping."""

    settings = Settings()
    renderer = VideoRenderer(settings)

    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            track_id=1,
        ),
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(10, 10, 20, 20),
            confidence=0.8,
            track_id=2,
        ),
        DetectionResult(
            frame_id=1,
            time_sec=1.0,
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            track_id=1,
        ),
    ]

    frame_map = renderer._build_frame_map(detections)

    assert 0 in frame_map
    assert 1 in frame_map
    assert len(frame_map[0]) == 2
    assert len(frame_map[1]) == 1


def test_prepare_render_config_with_video(monkeypatch, tmp_path):
    """Test _prepare_render_config prepares config for video."""

    settings = Settings()
    renderer = VideoRenderer(settings)

    test_video = tmp_path / 'test.mp4'
    test_video.write_bytes(b'fake video')

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        3: 1920,  # CAP_PROP_FRAME_WIDTH
        4: 1080,  # CAP_PROP_FRAME_HEIGHT
    }.get(prop, 0)

    monkeypatch.setattr(
        'hackaton_system.pipeline.rendering.cv2.VideoCapture',
        lambda _: mock_cap,
    )

    config = renderer._prepare_render_config(test_video, fps=25.0)

    assert config is not None
    assert 'cap' in config
    assert 'width' in config
    assert 'height' in config


def test_prepare_render_config_video_not_opened(monkeypatch, tmp_path):
    """Test _prepare_render_config returns None when video not opened."""

    settings = Settings()
    renderer = VideoRenderer(settings)

    test_video = tmp_path / 'test.mp4'
    test_video.write_bytes(b'fake video')

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    monkeypatch.setattr(
        'hackaton_system.pipeline.rendering.cv2.VideoCapture',
        lambda _: mock_cap,
    )

    config = renderer._prepare_render_config(test_video, fps=25.0)

    assert config is None


def test_calculate_target_size_landscape():
    """Test _calculate_target_size for landscape video."""

    settings = Settings()
    renderer = VideoRenderer(settings)

    width, height, _ = renderer._calculate_target_size(
        1920, 1080, max_side=1280
    )

    assert width == 1280
    assert height == 720
    assert abs((width / height) - (1920 / 1080)) < 0.01


def test_calculate_target_size_portrait():
    """Test _calculate_target_size for portrait video."""

    settings = Settings()
    renderer = VideoRenderer(settings)

    width, height, _ = renderer._calculate_target_size(
        1080, 1920, max_side=1280
    )

    assert width == 720
    assert height == 1280
    assert abs((width / height) - (1080 / 1920)) < 0.01


def test_calculate_target_size_small_video():
    """Test _calculate_target_size for small video (no scaling)."""

    settings = Settings()
    renderer = VideoRenderer(settings)

    width, height, scale = renderer._calculate_target_size(
        640, 480, max_side=1280
    )

    assert width == 640
    assert height == 480
    assert scale == 1.0


def test_calculate_target_size_odd_dimensions():
    """Test _calculate_target_size handles odd dimensions."""

    settings = Settings()
    renderer = VideoRenderer(settings)

    width, height, _ = renderer._calculate_target_size(
        1921, 1081, max_side=1280
    )

    assert width % 2 == 0
    assert height % 2 == 0


def test_prepare_render_config_invalid_dimensions(monkeypatch, tmp_path):
    """Test _prepare_render_config returns None for invalid dimensions."""

    settings = Settings()
    renderer = VideoRenderer(settings)

    test_video = tmp_path / 'test.mp4'
    test_video.write_bytes(b'fake video')

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        3: 0,  # CAP_PROP_FRAME_WIDTH - invalid
        4: 0,  # CAP_PROP_FRAME_HEIGHT - invalid
    }.get(prop, 0)
    mock_cap.release = MagicMock()

    monkeypatch.setattr(
        'hackaton_system.pipeline.rendering.cv2.VideoCapture',
        lambda _: mock_cap,
    )

    config = renderer._prepare_render_config(test_video, fps=25.0)

    assert config is None
    mock_cap.release.assert_called_once()


def test_annotate_frame():
    """Test _annotate_frame annotates frame with bounding boxes."""

    settings = Settings()
    renderer = VideoRenderer(settings)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(10, 20, 30, 40),
            confidence=0.9,
            track_id=1,
        ),
    ]

    annotated = renderer._annotate_frame(
        frame,
        detections,
        width=100,
        height=100,
        target_width=100,
        target_height=100,
        scale=1.0,
    )

    assert annotated.shape == (100, 100, 3)
    assert annotated.dtype == np.uint8


def test_render_preview_source_not_exists(monkeypatch, tmp_path):
    """Test _render_preview returns None when source doesn't exist."""

    settings = Settings()
    renderer = VideoRenderer(settings)

    result = renderer.render_preview(
        source=Path('nonexistent.mp4'),
        video_id=1,
        detections=[],
        fps=25.0,
    )

    assert result is None


def test_render_frames_with_broken_pipe(monkeypatch, tmp_path):
    """_render_frames should handle broken pipe errors gracefully."""
    settings = Settings()
    renderer = VideoRenderer(settings)

    mock_cap = MagicMock()
    mock_cap.read.return_value = (
        True,
        np.zeros((100, 100, 3), dtype=np.uint8),
    )
    mock_cap.release = MagicMock()

    mock_process = MagicMock()
    mock_stdin = MagicMock()
    mock_stdin.write.side_effect = BrokenPipeError('Broken pipe')
    mock_process.stdin = mock_stdin
    mock_process.wait = MagicMock()

    config = {
        'cap': mock_cap,
        'width': 100,
        'height': 100,
        'target_width': 100,
        'target_height': 100,
        'scale': 1.0,
        'frame_step': 1,
    }
    frame_map = {0: []}

    renderer._render_frames(config, frame_map, mock_process)
    mock_cap.release.assert_called_once()


def test_render_frames_process_terminates_early(monkeypatch):
    """_render_frames should stop when process terminates early."""
    settings = Settings()
    renderer = VideoRenderer(settings)

    mock_cap = MagicMock()
    mock_cap.read.return_value = (
        True,
        np.zeros((100, 100, 3), dtype=np.uint8),
    )
    mock_cap.release = MagicMock()

    mock_process = MagicMock()
    mock_stdin = MagicMock()
    mock_stdin.write.side_effect = OSError('Process terminated')
    mock_process.stdin = mock_stdin
    mock_process.wait = MagicMock()

    config = {
        'cap': mock_cap,
        'width': 100,
        'height': 100,
        'target_width': 100,
        'target_height': 100,
        'scale': 1.0,
        'frame_step': 1,
    }
    frame_map = {0: []}

    renderer._render_frames(config, frame_map, mock_process)
    mock_cap.release.assert_called_once()


def test_render_frames_no_stdin(monkeypatch):
    """_render_frames should handle process with no stdin."""
    settings = Settings()
    renderer = VideoRenderer(settings)

    mock_cap = MagicMock()
    mock_cap.read.return_value = (False, None)
    mock_cap.release = MagicMock()

    mock_process = MagicMock()
    mock_process.stdin = None
    mock_process.wait = MagicMock()

    config = {
        'cap': mock_cap,
        'width': 100,
        'height': 100,
        'target_width': 100,
        'target_height': 100,
        'scale': 1.0,
        'frame_step': 1,
    }
    frame_map = {}

    renderer._render_frames(config, frame_map, mock_process)
    mock_cap.release.assert_called_once()


def test_annotate_frame_with_scaling():
    """_annotate_frame should scale frames correctly."""
    settings = Settings()
    renderer = VideoRenderer(settings)

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(50, 50, 150, 150),
            confidence=0.9,
            track_id=1,
        ),
    ]

    annotated = renderer._annotate_frame(
        frame,
        detections,
        width=200,
        height=200,
        target_width=100,
        target_height=100,
        scale=0.5,
    )

    assert annotated.shape == (100, 100, 3)


def test_annotate_frame_multiple_detections():
    """_annotate_frame should annotate multiple detections."""
    settings = Settings()
    renderer = VideoRenderer(settings)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(10, 10, 30, 30),
            confidence=0.9,
            track_id=1,
        ),
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(50, 50, 70, 70),
            confidence=0.8,
            track_id=2,
        ),
    ]

    annotated = renderer._annotate_frame(
        frame,
        detections,
        width=100,
        height=100,
        target_width=100,
        target_height=100,
        scale=1.0,
    )

    assert annotated.shape == (100, 100, 3)


def test_prepare_output_path():
    """_prepare_output_path should create correct output path."""
    settings = Settings()
    renderer = VideoRenderer(settings)

    output_path = renderer._prepare_output_path(1, Path('test_video.mp4'))
    assert '1_test_video.mp4' in str(output_path)


def test_prepare_output_path_removes_existing(tmp_path):
    """_prepare_output_path should remove existing file."""
    settings = Settings(video_output_dir=tmp_path)
    renderer = VideoRenderer(settings)

    existing_path = tmp_path / '1_test.mp4'
    existing_path.write_bytes(b'existing')

    output_path = renderer._prepare_output_path(1, Path('test.mp4'))
    assert not existing_path.exists() or output_path != existing_path


def test_create_ffmpeg_process(monkeypatch, tmp_path):
    """_create_ffmpeg_process should create ffmpeg process."""
    settings = Settings()
    renderer = VideoRenderer(settings)

    config = {
        'target_width': 640,
        'target_height': 480,
        'fps_out': 25.0,
    }
    output_path = tmp_path / 'output.mp4'

    with patch(
        'hackaton_system.pipeline.rendering.start_ffmpeg_writer'
    ) as mock_start:
        mock_process = MagicMock()
        mock_start.return_value = mock_process

        result = renderer._create_ffmpeg_process(config, output_path)
        assert result is mock_process
        mock_start.assert_called_once()


def test_create_ffmpeg_process_failure(monkeypatch, tmp_path):
    """_create_ffmpeg_process should return None on failure."""
    settings = Settings()
    renderer = VideoRenderer(settings)

    config = {
        'target_width': 640,
        'target_height': 480,
        'fps_out': 25.0,
    }
    output_path = tmp_path / 'output.mp4'

    with patch(
        'hackaton_system.pipeline.rendering.start_ffmpeg_writer'
    ) as mock_start:
        mock_start.side_effect = RuntimeError('FFmpeg error')

        result = renderer._create_ffmpeg_process(config, output_path)
        assert result is None
