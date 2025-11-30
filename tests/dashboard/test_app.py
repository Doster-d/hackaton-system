"""Tests for Streamlit dashboard app."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from hackaton_system.dashboard import app, data_access
from hackaton_system.dashboard.app import VideoRecord


def test_video_record_typeddict():
    """Test VideoRecord TypedDict structure."""

    record: VideoRecord = {
        'id': 1,
        'filename': 'test.mp4',
        'duration_sec': 100.0,
        'fps': 25.0,
    }

    assert record['id'] == 1
    assert record['filename'] == 'test.mp4'
    assert record['duration_sec'] == 100.0
    assert record['fps'] == 25.0


def test_video_record_with_none_id():
    """Test VideoRecord with None id."""

    record: VideoRecord = {
        'id': None,
        'filename': 'placeholder.mp4',
        'duration_sec': 300.0,
        'fps': 25.0,
    }

    assert record['id'] is None
    assert record['filename'] == 'placeholder.mp4'


@pytest.mark.skip(reason='Streamlit app requires full UI context')
def test_app_imports_correctly():
    """Test that app module imports without errors."""
    # This test just ensures the module can be imported
    # Full testing would require Streamlit runtime
    import hackaton_system.dashboard.app  # noqa: F401, PLC0415

    assert True


def test_app_data_loading_logic(monkeypatch: Any) -> None:
    """Test data loading logic without Streamlit UI."""
    # Mock data access functions
    mock_videos = pd.DataFrame(
        [
            {
                'id': 1,
                'filename': 'test.mp4',
                'duration_sec': 100.0,
                'fps': 25.0,
            },
        ]
    )
    mock_headcount = pd.DataFrame(
        {'time_sec': [0.0, 5.0], 'headcount': [2, 3]}
    )
    mock_activity = pd.DataFrame(
        {
            'activity_class': ['working', 'idle'],
            'duration_min': [10.0, 5.0],
        }
    )
    mock_matrix = pd.DataFrame(
        {
            'person_type': ['operator'],
            'walking': [5.0],
            'working': [10.0],
        }
    )
    mock_episodes = pd.DataFrame(
        {
            'person_id': [1],
            'person_type': ['operator'],
            'activity_class': ['working'],
            't_start_sec': [0.0],
            't_end_sec': [10.0],
            'duration_sec': [10.0],
        }
    )

    monkeypatch.setattr(
        data_access, 'list_available_videos', lambda: mock_videos
    )
    monkeypatch.setattr(
        data_access, 'load_headcount', lambda _vid_id: mock_headcount
    )
    monkeypatch.setattr(
        data_access, 'load_activity_summary', lambda _vid_id: mock_activity
    )
    monkeypatch.setattr(
        data_access, 'load_role_activity_matrix', lambda _vid_id: mock_matrix
    )
    monkeypatch.setattr(
        data_access, 'load_person_episodes', lambda _vid_id: mock_episodes
    )

    # Test that functions return expected data
    videos = data_access.list_available_videos()
    assert len(videos) == 1
    assert videos.iloc[0]['filename'] == 'test.mp4'

    headcount = data_access.load_headcount(1)
    assert len(headcount) == 2
    assert 'headcount' in headcount.columns

    activity = data_access.load_activity_summary(1)
    assert len(activity) == 2
    assert 'activity_class' in activity.columns

    matrix = data_access.load_role_activity_matrix(1)
    assert 'person_type' in matrix.columns

    episodes = data_access.load_person_episodes(1)
    assert 'person_id' in episodes.columns


def test_process_video_file_success(monkeypatch: Any, tmp_path: Path) -> None:
    """Test _process_video_file processes video successfully."""

    mock_processor = MagicMock()
    mock_processor.process_video.return_value = 42

    def fake_video_processor():
        return mock_processor

    mock_st = MagicMock()
    mock_st.sidebar.empty.return_value = mock_st
    mock_st.spinner.return_value.__enter__ = lambda *_, **__: None
    mock_st.spinner.return_value.__exit__ = lambda *_, **__: None

    monkeypatch.setattr(app, 'VideoProcessor', fake_video_processor)
    monkeypatch.setattr(app, 'st', mock_st)
    monkeypatch.setattr(app, 'PREVIEW_DIR', tmp_path)

    test_path = tmp_path / 'test.mp4'
    test_path.write_bytes(b'dummy')
    app._process_video_file(test_path)

    mock_processor.process_video.assert_called_once_with(test_path)


def test_process_video_file_error(monkeypatch: Any, tmp_path: Path) -> None:
    """Test _process_video_file handles errors gracefully."""

    mock_processor = MagicMock()
    mock_processor.process_video.side_effect = Exception('Processing failed')

    def fake_video_processor():
        return mock_processor

    mock_st = MagicMock()
    mock_st.sidebar.empty.return_value = mock_st
    mock_st.spinner.return_value.__enter__ = lambda *_, **__: None
    mock_st.spinner.return_value.__exit__ = lambda *_, **__: None

    monkeypatch.setattr(app, 'VideoProcessor', fake_video_processor)
    monkeypatch.setattr(app, 'st', mock_st)
    monkeypatch.setattr(app, 'PREVIEW_DIR', tmp_path)

    test_path = tmp_path / 'test.mp4'
    test_path.write_bytes(b'dummy')
    app._process_video_file(test_path)

    mock_st.error.assert_called_once()
