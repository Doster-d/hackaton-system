# pyright: reportPrivateUsage=false
"""Tests for video_processor module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hackaton_system.config import Settings
from hackaton_system.db import (
    Activity,
    Person,
    PoseKeypoints,
    Video,
    get_session,
    init_db,
)
from hackaton_system.pipeline.models import (
    ActivityResult,
    DetectionResult,
    PoseResult,
    RoleAssignment,
)
from hackaton_system.pipeline.video_processor import VideoProcessor


def test_group_detections_by_track():
    """Test _group_detections_by_track groups detections correctly."""
    processor = VideoProcessor()
    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            track_id=1,
        ),
        DetectionResult(
            frame_id=1,
            time_sec=1.0,
            bbox=(1, 1, 11, 11),
            confidence=0.8,
            track_id=1,
        ),
        DetectionResult(
            frame_id=2,
            time_sec=2.0,
            bbox=(100, 100, 110, 110),
            confidence=0.7,
            track_id=2,
        ),
    ]

    tracks = VideoProcessor._group_detections_by_track(processor, detections)

    assert len(tracks) == 2
    assert len(tracks[1]) == 2
    assert len(tracks[2]) == 1
    assert tracks[1][0].time_sec == 0.0
    assert tracks[1][1].time_sec == 1.0
    assert tracks[2][0].time_sec == 2.0


def test_group_detections_by_track_sorts_by_time():
    """Test _group_detections_by_track sorts detections by time."""
    processor = VideoProcessor()
    detections = [
        DetectionResult(
            frame_id=2,
            time_sec=2.0,
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            track_id=1,
        ),
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            track_id=1,
        ),
        DetectionResult(
            frame_id=1,
            time_sec=1.0,
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            track_id=1,
        ),
    ]

    tracks = VideoProcessor._group_detections_by_track(processor, detections)

    assert tracks[1][0].time_sec == 0.0
    assert tracks[1][1].time_sec == 1.0
    assert tracks[1][2].time_sec == 2.0


def test_ensure_person_records():
    """Test _ensure_person_records creates Person records."""

    settings = Settings()
    processor = VideoProcessor(settings=settings)
    init_db()

    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            track_id=1,
        ),
        DetectionResult(
            frame_id=1,
            time_sec=1.0,
            bbox=(0, 0, 10, 10),
            confidence=0.8,
            track_id=2,
        ),
        DetectionResult(
            frame_id=2,
            time_sec=2.0,
            bbox=(0, 0, 10, 10),
            confidence=0.7,
            track_id=1,
        ),
    ]

    with get_session() as session:
        video = Video(filename='test.mp4', fps=25.0, duration_sec=10.0)
        session.add(video)
        session.flush()

        person_map = processor._ensure_person_records(
            session, video, detections
        )

        assert len(person_map) == 2  # Two unique track_ids
        assert 1 in person_map
        assert 2 in person_map
        assert person_map[1].track_id == 1
        assert person_map[2].track_id == 2
        assert person_map[1].person_type == 'unknown'
        assert person_map[1].person_type_conf == 0.0


def test_ensure_person_records_empty_detections():
    """Test _ensure_person_records with empty detections."""

    settings = Settings()
    processor = VideoProcessor(settings=settings)
    init_db()

    with get_session() as session:
        video = Video(filename='test.mp4', fps=25.0, duration_sec=10.0)
        session.add(video)
        session.flush()

        person_map = processor._ensure_person_records(session, video, [])

        assert person_map == {}


def test_load_detector_lazy_initialization(monkeypatch):
    """Test _load_detector lazy initialization."""

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    # Mock ultralytics module
    mock_ultralytics = MagicMock()
    mock_yolo_class = MagicMock()
    mock_yolo_instance = MagicMock()
    mock_yolo_class.return_value = mock_yolo_instance
    mock_ultralytics.YOLO = mock_yolo_class

    with monkeypatch.context() as m:
        m.setattr(
            'hackaton_system.pipeline.video_processor.import_module',
            lambda _: mock_ultralytics,
        )

        detector1 = processor._load_detector()
        detector2 = processor._load_detector()

        # Should return same instance (lazy initialization)
        assert detector1 is detector2
        # Should only be called once
        assert mock_yolo_class.call_count == 1


def test_load_detector_with_custom_weights(monkeypatch):
    """Test _load_detector uses custom weights if provided."""

    settings = Settings(detection_model_path='custom_weights.pt')
    processor = VideoProcessor(settings=settings)

    mock_ultralytics = MagicMock()
    mock_yolo_class = MagicMock()
    mock_yolo_instance = MagicMock()
    mock_yolo_class.return_value = mock_yolo_instance
    mock_ultralytics.YOLO = mock_yolo_class

    with monkeypatch.context() as m:
        m.setattr(
            'hackaton_system.pipeline.video_processor.import_module',
            lambda _: mock_ultralytics,
        )

        processor._load_detector()

        # Should use custom weights
        mock_yolo_class.assert_called_once_with('custom_weights.pt')


def test_process_video_file_not_found():
    """Test process_video raises FileNotFoundError for non-existent file."""

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    with pytest.raises(FileNotFoundError):
        processor.process_video(Path('nonexistent_video.mp4'))


def test_extract_video_metadata(monkeypatch):
    """Test _extract_video_metadata extracts video metadata correctly."""

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    # Mock cv2.VideoCapture
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        5: 25.0,  # CAP_PROP_FPS
        7: 750,  # CAP_PROP_FRAME_COUNT
    }.get(prop, 0)
    mock_cap.release = MagicMock()

    with monkeypatch.context() as m:
        m.setattr(
            'hackaton_system.pipeline.video_processor.cv2.VideoCapture',
            lambda _: mock_cap,
        )

        fps, frames, duration = processor._extract_video_metadata(
            Path('test.mp4')
        )

        assert fps == 25.0
        assert frames == 750
        assert duration == 30.0  # 750 / 25.0
        mock_cap.release.assert_called_once()


def test_extract_video_metadata_default_fps(monkeypatch):
    """Test _extract_video_metadata uses default FPS when not available."""

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 0  # No FPS available
    mock_cap.release = MagicMock()

    with monkeypatch.context() as m:
        m.setattr(
            'hackaton_system.pipeline.video_processor.cv2.VideoCapture',
            lambda _: mock_cap,
        )

        fps, _, _ = processor._extract_video_metadata(Path('test.mp4'))

        assert fps == 25.0  # Default FPS
        mock_cap.release.assert_called_once()


def test_extract_video_metadata_cannot_open(monkeypatch):
    """Test _extract_video_metadata raises RuntimeError when video cannot be opened."""

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with monkeypatch.context() as m:
        m.setattr(
            'hackaton_system.pipeline.video_processor.cv2.VideoCapture',
            lambda _: mock_cap,
        )

        with pytest.raises(RuntimeError, match='Cannot open video'):
            processor._extract_video_metadata(Path('test.mp4'))


def test_process_video_full_flow(monkeypatch, tmp_path):
    """Test process_video full flow with all mocks."""

    settings = Settings(activity_min_interval_sec=0.1)
    processor = VideoProcessor(settings=settings)
    init_db()

    # Create a real file for testing
    test_video_path = tmp_path / 'test_video.mp4'
    test_video_path.touch()  # Create empty file

    # Mock cv2.VideoCapture
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        5: 25.0,  # CAP_PROP_FPS
        7: 250,  # CAP_PROP_FRAME_COUNT
    }.get(prop, 0)
    mock_cap.release = MagicMock()
    mock_cap.read.return_value = (False, None)

    # Mock YOLO detector
    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_boxes = MagicMock()

    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value = mock_xyxy
    mock_xyxy.tolist.return_value = [[10, 20, 30, 40]]
    mock_boxes.xyxy = mock_xyxy
    mock_boxes.conf = MagicMock()
    mock_boxes.conf.cpu.return_value = mock_boxes.conf
    mock_boxes.conf.tolist.return_value = [0.9]
    mock_boxes.cls = MagicMock()
    mock_boxes.cls.cpu.return_value = mock_boxes.cls
    mock_boxes.cls.tolist.return_value = [0]
    mock_boxes.id = MagicMock()
    mock_boxes.id.cpu.return_value = mock_boxes.id
    mock_boxes.id.tolist.return_value = [1]

    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    with monkeypatch.context() as m:
        # Mock cv2
        m.setattr(
            'hackaton_system.pipeline.video_processor.cv2.VideoCapture',
            lambda _: mock_cap,
        )
        # Mock detector
        processor._detector = mock_detector

        video_id = processor.process_video(test_video_path)

        # Verify video was created
        with get_session() as session:
            video = session.query(Video).filter_by(id=video_id).first()
            assert video is not None
            assert video.filename == 'test_video.mp4'
            assert video.fps == 25.0


def test_process_video_skips_missing_track_in_person_map(
    monkeypatch, tmp_path
):
    """Test process_video skips tracks not in person_map for role assignment."""

    settings = Settings(activity_min_interval_sec=0.1)
    processor = VideoProcessor(settings=settings)
    init_db()

    test_video_path = tmp_path / 'test_video.mp4'
    test_video_path.touch()

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        5: 25.0,
        7: 250,
    }.get(prop, 0)
    mock_cap.release = MagicMock()
    mock_cap.read.return_value = (False, None)

    # Create detections with track_id 1
    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_boxes = MagicMock()

    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value = mock_xyxy
    mock_xyxy.tolist.return_value = [[10, 20, 30, 40]]
    mock_boxes.xyxy = mock_xyxy
    mock_boxes.conf = MagicMock()
    mock_boxes.conf.cpu.return_value = mock_boxes.conf
    mock_boxes.conf.tolist.return_value = [0.9]
    mock_boxes.cls = MagicMock()
    mock_boxes.cls.cpu.return_value = mock_boxes.cls
    mock_boxes.cls.tolist.return_value = [0]
    mock_boxes.id = MagicMock()
    mock_boxes.id.cpu.return_value = mock_boxes.id
    mock_boxes.id.tolist.return_value = [1]

    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    # Mock _infer_person_roles to return assignment for track_id 999 (not in person_map)
    def mock_infer_roles(tracks, fps):
        return {
            999: RoleAssignment(999, 'operator', 0.9)
        }  # Track not in person_map

    with monkeypatch.context() as m:
        m.setattr(
            'hackaton_system.pipeline.video_processor.cv2.VideoCapture',
            lambda _: mock_cap,
        )
        processor._detector = mock_detector
        processor._infer_person_roles = mock_infer_roles

        video_id = processor.process_video(test_video_path)

        # Should complete without error, skipping track 999
        assert video_id is not None


def test_process_video_skips_activity_with_missing_track(
    monkeypatch, tmp_path
):
    """Test process_video skips activities for tracks not in person_map."""

    settings = Settings(activity_min_interval_sec=0.1)
    processor = VideoProcessor(settings=settings)
    init_db()

    test_video_path = tmp_path / 'test_video.mp4'
    test_video_path.touch()

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        5: 25.0,
        7: 250,
    }.get(prop, 0)
    mock_cap.release = MagicMock()
    mock_cap.read.return_value = (False, None)

    # Create detections with track_id 1
    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_boxes = MagicMock()

    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value = mock_xyxy
    mock_xyxy.tolist.return_value = [[10, 20, 30, 40]]
    mock_boxes.xyxy = mock_xyxy
    mock_boxes.conf = MagicMock()
    mock_boxes.conf.cpu.return_value = mock_boxes.conf
    mock_boxes.conf.tolist.return_value = [0.9]
    mock_boxes.cls = MagicMock()
    mock_boxes.cls.cpu.return_value = mock_boxes.cls
    mock_boxes.cls.tolist.return_value = [0]
    mock_boxes.id = MagicMock()
    mock_boxes.id.cpu.return_value = mock_boxes.id
    mock_boxes.id.tolist.return_value = [1]

    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    # Mock _infer_activities to return activity for track_id 999 (not in person_map)
    def mock_infer_activities(tracks, fps):
        return [
            ActivityResult(999, 'working', 0.0, 10.0, 1.0)
        ]  # Track not in person_map

    with monkeypatch.context() as m:
        m.setattr(
            'hackaton_system.pipeline.video_processor.cv2.VideoCapture',
            lambda _: mock_cap,
        )
        processor._detector = mock_detector
        processor._infer_activities = mock_infer_activities

        video_id = processor.process_video(test_video_path)

        # Should complete without error, skipping activity for track 999
        assert video_id is not None


def test_load_detector_import_error(monkeypatch):
    """Test _load_detector raises RuntimeError on ImportError."""

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    def mock_import_module(name):
        raise ImportError("No module named 'ultralytics'")

    with monkeypatch.context() as m:
        m.setattr(
            'hackaton_system.pipeline.video_processor.import_module',
            mock_import_module,
        )

        with pytest.raises(
            RuntimeError, match='Ultralytics package is not installed'
        ):
            processor._load_detector()


def test_load_detector_no_yolo_class(monkeypatch):
    """Test _load_detector raises RuntimeError when YOLO class not found."""

    settings = Settings()
    processor = VideoProcessor(settings=settings)

    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO = None  # YOLO class not available

    with monkeypatch.context() as m:
        m.setattr(
            'hackaton_system.pipeline.video_processor.import_module',
            lambda _: mock_ultralytics,
        )

        with pytest.raises(
            RuntimeError, match='does not expose the YOLO class'
        ):
            processor._load_detector()


def test_save_activities(monkeypatch, tmp_path):
    """Test _save_activities saves activities to database."""

    settings = Settings()
    processor = VideoProcessor(settings=settings)
    init_db()

    with get_session() as session:
        video = Video(filename='test.mp4', fps=25.0, duration_sec=10.0)
        session.add(video)
        session.flush()

        person = Person(video_id=video.id, track_id=1, person_type='operator')
        session.add(person)
        session.flush()

        person_map = {1: person}

        activities = [
            ActivityResult(
                track_id=1,
                activity_class='working',
                start_sec=0.0,
                end_sec=5.0,
                confidence=0.9,
            ),
            ActivityResult(
                track_id=1,
                activity_class='walking',
                start_sec=5.0,
                end_sec=10.0,
                confidence=0.8,
            ),
        ]

        processor._save_activities(session, video, person_map, activities)
        session.flush()

        saved = (
            session.query(Activity)
            .filter_by(video_id=video.id)
            .order_by(Activity.t_start_sec)
            .all()
        )
        assert len(saved) == 2
        assert saved[0].activity_class == 'working'
        assert saved[1].activity_class == 'walking'


def test_save_activities_skips_missing_track(monkeypatch, tmp_path):
    """Test _save_activities skips activities for missing tracks."""

    settings = Settings()
    processor = VideoProcessor(settings=settings)
    init_db()

    with get_session() as session:
        video = Video(filename='test.mp4', fps=25.0, duration_sec=10.0)
        session.add(video)
        session.flush()

        person_map = {}

        activities = [
            ActivityResult(
                track_id=999,
                activity_class='working',
                start_sec=0.0,
                end_sec=5.0,
                confidence=0.9,
            ),
        ]

        processor._save_activities(session, video, person_map, activities)
        session.flush()

        saved = session.query(Activity).filter_by(video_id=video.id).all()
        assert len(saved) == 0


def test_save_poses_enabled(monkeypatch, tmp_path):
    """Test _save_poses saves poses when enabled."""

    settings = Settings(enable_pose_capture=True)
    processor = VideoProcessor(settings=settings)
    init_db()

    with get_session() as session:
        video = Video(filename='test.mp4', fps=25.0, duration_sec=10.0)
        session.add(video)
        session.flush()

        person = Person(video_id=video.id, track_id=1, person_type='operator')
        session.add(person)
        session.flush()

        person_map = {1: person}

        poses = [
            PoseResult(
                frame_id=0,
                time_sec=0.0,
                track_id=1,
                keypoints=[{'x': 10.0, 'y': 20.0, 'confidence': 0.9}],
                confidence=0.9,
            ),
        ]

        processor._save_poses(session, video, person_map, poses)
        session.flush()

        saved = session.query(PoseKeypoints).filter_by(video_id=video.id).all()
        assert len(saved) == 1
        assert saved[0].keypoints == poses[0].keypoints


def test_save_poses_disabled(monkeypatch, tmp_path):
    """Test _save_poses does not save when disabled."""

    settings = Settings(enable_pose_capture=False)
    processor = VideoProcessor(settings=settings)
    init_db()

    with get_session() as session:
        video = Video(filename='test.mp4', fps=25.0, duration_sec=10.0)
        session.add(video)
        session.flush()

        person_map = {}

        poses = [
            PoseResult(
                frame_id=0,
                time_sec=0.0,
                track_id=1,
                keypoints=[{'x': 10.0, 'y': 20.0}],
                confidence=0.9,
            ),
        ]

        processor._save_poses(session, video, person_map, poses)
        session.flush()

        saved = session.query(PoseKeypoints).filter_by(video_id=video.id).all()
        assert len(saved) == 0


def test_save_poses_empty_list(monkeypatch, tmp_path):
    """Test _save_poses handles empty pose list."""

    settings = Settings(enable_pose_capture=True)
    processor = VideoProcessor(settings=settings)
    init_db()

    with get_session() as session:
        video = Video(filename='test.mp4', fps=25.0, duration_sec=10.0)
        session.add(video)
        session.flush()

        processor._save_poses(session, video, {}, [])
        session.flush()

        saved = session.query(PoseKeypoints).filter_by(video_id=video.id).all()
        assert len(saved) == 0


def test_save_poses_skips_missing_track(monkeypatch, tmp_path):
    """Test _save_poses skips poses for missing tracks."""

    settings = Settings(enable_pose_capture=True)
    processor = VideoProcessor(settings=settings)
    init_db()

    with get_session() as session:
        video = Video(filename='test.mp4', fps=25.0, duration_sec=10.0)
        session.add(video)
        session.flush()

        person_map = {}  # Empty map

        poses = [
            PoseResult(
                frame_id=0,
                time_sec=0.0,
                track_id=999,  # Not in person_map
                keypoints=[{'x': 10.0, 'y': 20.0}],
                confidence=0.9,
            ),
        ]

        processor._save_poses(session, video, person_map, poses)
        session.flush()

        saved = session.query(PoseKeypoints).filter_by(video_id=video.id).all()
        assert len(saved) == 0
