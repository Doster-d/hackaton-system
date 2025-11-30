# pyright: reportPrivateUsage=false
"""Tests for detection module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from hackaton_system.config import Settings
from hackaton_system.pipeline.detection import DetectionHandler


def test_run_detection_with_mocked_yolo(monkeypatch):
    """Test run_detection processes YOLO results correctly."""

    settings = Settings(detection_conf=0.5)
    handler = DetectionHandler(settings)

    # Mock YOLO detector
    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_boxes = MagicMock()

    # Mock boxes attributes
    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value = mock_xyxy
    mock_xyxy.tolist.return_value = [[10, 20, 30, 40], [50, 60, 70, 80]]
    mock_boxes.xyxy = mock_xyxy
    mock_boxes.conf = MagicMock()
    mock_boxes.conf.cpu.return_value = mock_boxes.conf
    mock_boxes.conf.tolist.return_value = [0.9, 0.8]
    mock_boxes.cls = MagicMock()
    mock_boxes.cls.cpu.return_value = mock_boxes.cls
    mock_boxes.cls.tolist.return_value = [0, 0]  # Class 0 = person
    mock_boxes.id = MagicMock()
    mock_boxes.id.cpu.return_value = mock_boxes.id
    mock_boxes.id.tolist.return_value = [1, 2]  # Track IDs

    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    detections, _ = handler.run_detection(
        mock_detector, Path('test.mp4'), fps=25.0
    )

    assert len(detections) == 2
    assert detections[0].track_id == 1
    assert detections[0].confidence == 0.9
    assert detections[1].track_id == 2
    assert detections[1].confidence == 0.8


def test_run_detection_filters_low_confidence(monkeypatch):
    """Test run_detection filters detections below confidence threshold."""

    settings = Settings(detection_conf=0.7)  # High threshold
    handler = DetectionHandler(settings)

    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_boxes = MagicMock()

    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value = mock_xyxy
    mock_xyxy.tolist.return_value = [[10, 20, 30, 40]]
    mock_boxes.xyxy = mock_xyxy
    mock_boxes.conf = MagicMock()
    mock_boxes.conf.cpu.return_value = mock_boxes.conf
    mock_boxes.conf.tolist.return_value = [0.5]  # Below threshold
    mock_boxes.cls = MagicMock()
    mock_boxes.cls.cpu.return_value = mock_boxes.cls
    mock_boxes.cls.tolist.return_value = [0]
    mock_boxes.id = MagicMock()
    mock_boxes.id.cpu.return_value = mock_boxes.id
    mock_boxes.id.tolist.return_value = [1]

    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    detections, _ = handler.run_detection(
        mock_detector, Path('test.mp4'), fps=25.0
    )

    # Should be filtered out due to low confidence
    assert len(detections) == 0


def test_run_detection_filters_non_person_classes(monkeypatch):
    """Test _run_detection filters non-person classes."""

    settings = Settings()
    handler = DetectionHandler(settings)

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
    mock_boxes.cls.tolist.return_value = [1]  # Not class 0 (person)
    mock_boxes.id = MagicMock()
    mock_boxes.id.cpu.return_value = mock_boxes.id
    mock_boxes.id.tolist.return_value = [1]

    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    detections, _ = handler.run_detection(
        mock_detector, Path('test.mp4'), fps=25.0
    )

    # Should be filtered out (not a person)
    assert len(detections) == 0


def test_run_detection_handles_no_boxes(monkeypatch):
    """Test _run_detection handles results without boxes."""

    settings = Settings()
    handler = DetectionHandler(settings)

    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes = None  # No boxes
    mock_detector.track.return_value = [mock_result]

    detections, _ = handler.run_detection(
        mock_detector, Path('test.mp4'), fps=25.0
    )

    # Should handle gracefully and return empty list
    assert len(detections) == 0


def test_run_detection_handles_exception(monkeypatch):
    """Test _run_detection handles exceptions gracefully."""

    settings = Settings()
    handler = DetectionHandler(settings)

    mock_detector = MagicMock()
    mock_detector.track.side_effect = Exception('YOLO error')

    detections, poses = handler.run_detection(
        mock_detector, Path('test.mp4'), fps=25.0
    )

    # Should return empty list on error
    assert detections == []
    assert poses == []


def test_run_detection_no_conf_attr(monkeypatch):
    """Test _run_detection handles missing conf attribute."""

    settings = Settings()
    handler = DetectionHandler(settings)

    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_boxes = MagicMock()

    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value = mock_xyxy
    mock_xyxy.tolist.return_value = [[10, 20, 30, 40]]
    mock_boxes.xyxy = mock_xyxy
    mock_boxes.conf = None  # No conf attribute
    mock_boxes.cls = MagicMock()
    mock_boxes.cls.cpu.return_value = mock_boxes.cls
    mock_boxes.cls.tolist.return_value = [0]
    mock_boxes.id = MagicMock()
    mock_boxes.id.cpu.return_value = mock_boxes.id
    mock_boxes.id.tolist.return_value = [1]

    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    detections, _ = handler.run_detection(
        mock_detector, Path('test.mp4'), fps=25.0
    )

    # Should use default confidence 0.0, but will be filtered by threshold
    assert len(detections) == 0


def test_run_detection_no_cls_attr(monkeypatch):
    """Test _run_detection handles missing cls attribute."""

    settings = Settings()
    handler = DetectionHandler(settings)

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
    mock_boxes.cls = None  # No cls attribute
    mock_boxes.id = MagicMock()
    mock_boxes.id.cpu.return_value = mock_boxes.id
    mock_boxes.id.tolist.return_value = [1]

    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    detections, _ = handler.run_detection(
        mock_detector, Path('test.mp4'), fps=25.0
    )

    # Should default to class 0 (person)
    assert len(detections) == 1
    assert detections[0].track_id == 1


def test_run_detection_no_id_attr(monkeypatch):
    """Test _run_detection handles missing id attribute."""

    settings = Settings()
    handler = DetectionHandler(settings)

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
    mock_boxes.id = None  # No id attribute

    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    detections, _ = handler.run_detection(
        mock_detector, Path('test.mp4'), fps=25.0
    )

    # Should filter out (no track_id)
    assert len(detections) == 0


def test_run_detection_no_xyxy_attr(monkeypatch):
    """Test _run_detection handles missing xyxy attribute."""

    settings = Settings()
    handler = DetectionHandler(settings)

    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_boxes = MagicMock()
    mock_boxes.xyxy = None  # No xyxy attribute

    mock_result.boxes = mock_boxes
    mock_detector.track.return_value = [mock_result]

    detections, _ = handler.run_detection(
        mock_detector, Path('test.mp4'), fps=25.0
    )

    # Should skip frame
    assert len(detections) == 0


def test_create_pose_result():
    """Test _create_pose_result creates PoseResult correctly."""

    settings = Settings()
    handler = DetectionHandler(settings)

    kp_xy = [[10.0, 20.0], [30.0, 40.0]]
    kp_conf = [0.9, 0.8]

    result = handler._create_pose_result(
        frame_idx=0,
        frame_time=0.0,
        track_id=1,
        kp_xy=kp_xy,
        kp_conf=kp_conf,
        default_conf=0.5,
    )

    assert result.frame_id == 0
    assert result.time_sec == 0.0
    assert result.track_id == 1
    assert len(result.keypoints) == 2
    assert result.keypoints[0]['x'] == 10.0
    assert result.keypoints[0]['y'] == 20.0
    assert result.keypoints[0]['confidence'] == 0.9
    assert abs(result.confidence - 0.85) < 0.001


def test_create_pose_result_no_conf():
    """Test _create_pose_result uses default confidence when no conf."""

    settings = Settings()
    handler = DetectionHandler(settings)

    kp_xy = [[10.0, 20.0]]

    result = handler._create_pose_result(
        frame_idx=0,
        frame_time=0.0,
        track_id=1,
        kp_xy=kp_xy,
        kp_conf=None,
        default_conf=0.7,
    )

    assert result.confidence == 0.7
    assert result.keypoints[0].get('confidence') is None


def test_extract_keypoints_with_keypoints(monkeypatch):
    """Test _extract_keypoints extracts keypoints when available."""

    settings = Settings()
    handler = DetectionHandler(settings)

    mock_result = MagicMock()
    mock_keypoints = MagicMock()
    mock_xy = MagicMock()
    mock_xy.cpu.return_value = mock_xy
    mock_xy.tolist.return_value = [[[10.0, 20.0], [30.0, 40.0]]]
    mock_conf = MagicMock()
    mock_conf.cpu.return_value = mock_conf
    mock_conf.tolist.return_value = [[0.9, 0.8]]
    mock_keypoints.xy = mock_xy
    mock_keypoints.conf = mock_conf
    mock_result.keypoints = mock_keypoints

    kp_xy, _ = handler._extract_keypoints(mock_result, count=1)

    assert len(kp_xy) == 1
    assert kp_xy[0] is not None
    assert len(kp_xy[0]) == 2


def test_extract_keypoints_no_keypoints(monkeypatch):
    """Test _extract_keypoints returns None when no keypoints."""

    settings = Settings()
    handler = DetectionHandler(settings)

    mock_result = MagicMock()
    mock_result.keypoints = None

    kp_xy, kp_conf = handler._extract_keypoints(mock_result, count=2)

    assert kp_xy == [None, None]
    assert kp_conf == [None, None]


def test_parse_keypoint_xy_with_data():
    """Test _parse_keypoint_xy parses keypoint coordinates."""

    settings = Settings()
    handler = DetectionHandler(settings)

    mock_attr = MagicMock()
    mock_attr.cpu.return_value = mock_attr
    mock_attr.tolist.return_value = [[[10.0, 20.0]], [[30.0, 40.0]]]

    result = handler._parse_keypoint_xy(mock_attr, count=2)

    assert len(result) == 2
    assert result[0] == [[10.0, 20.0]]


def test_parse_keypoint_xy_none():
    """Test _parse_keypoint_xy returns None list when attr is None."""

    settings = Settings()
    handler = DetectionHandler(settings)

    result = handler._parse_keypoint_xy(None, count=3)

    assert result == [None, None, None]


def test_parse_keypoint_xy_wrong_count():
    """Test _parse_keypoint_xy returns None when count mismatch."""

    settings = Settings()
    handler = DetectionHandler(settings)

    mock_attr = MagicMock()
    mock_attr.cpu.return_value = mock_attr
    mock_attr.tolist.return_value = [[[10.0, 20.0]]]

    result = handler._parse_keypoint_xy(mock_attr, count=2)

    assert result == [None, None]


def test_parse_keypoint_conf_with_data():
    """Test _parse_keypoint_conf parses confidence scores."""

    settings = Settings()
    handler = DetectionHandler(settings)

    mock_attr = MagicMock()
    mock_attr.cpu.return_value = mock_attr
    mock_attr.tolist.return_value = [[0.9, 0.8], [0.7, 0.6]]

    result = handler._parse_keypoint_conf(mock_attr, count=2)

    assert len(result) == 2
    assert result[0] == [0.9, 0.8]


def test_parse_keypoint_conf_none():
    """Test _parse_keypoint_conf returns None list when attr is None."""

    settings = Settings()
    handler = DetectionHandler(settings)

    result = handler._parse_keypoint_conf(None, count=2)

    assert result == [None, None]


def test_parse_keypoint_conf_wrong_count():
    """Test _parse_keypoint_conf returns None when count mismatch."""

    settings = Settings()
    handler = DetectionHandler(settings)

    mock_attr = MagicMock()
    mock_attr.cpu.return_value = mock_attr
    mock_attr.tolist.return_value = [[0.9]]

    result = handler._parse_keypoint_conf(mock_attr, count=2)

    assert result == [None, None]


def test_is_valid_detection_valid():
    """Test _is_valid_detection returns True for valid detection."""

    settings = Settings()
    handler = DetectionHandler(settings)

    assert handler._is_valid_detection(cls_id=0, track_id=1, conf=0.9) is True
    assert handler._is_valid_detection(cls_id=0, track_id=2, conf=0.8) is True


def test_is_valid_detection_invalid_class():
    """Test _is_valid_detection returns False for non-person class."""

    settings = Settings()
    handler = DetectionHandler(settings)

    assert handler._is_valid_detection(cls_id=1, track_id=1, conf=0.9) is False


def test_is_valid_detection_no_track_id():
    """Test _is_valid_detection returns False when no track_id."""

    settings = Settings()
    handler = DetectionHandler(settings)

    assert (
        handler._is_valid_detection(cls_id=0, track_id=None, conf=0.9) is False
    )


def test_is_valid_detection_low_confidence():
    """Test _is_valid_detection returns False for low confidence."""

    settings = Settings(detection_conf=0.7)
    handler = DetectionHandler(settings)

    assert handler._is_valid_detection(cls_id=0, track_id=1, conf=0.5) is False
    assert handler._is_valid_detection(cls_id=0, track_id=1, conf=0.8) is True


def test_parse_detection_result_with_pose(monkeypatch):
    """Test _parse_detection_result creates pose when keypoints available."""

    settings = Settings()
    handler = DetectionHandler(settings)

    mock_result = MagicMock()
    mock_boxes = MagicMock()

    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value = mock_xyxy
    mock_xyxy.tolist.return_value = [[10, 20, 30, 40]]
    mock_boxes.xyxy = mock_xyxy

    mock_conf = MagicMock()
    mock_conf.cpu.return_value = mock_conf
    mock_conf.tolist.return_value = [0.9]
    mock_boxes.conf = mock_conf

    mock_cls = MagicMock()
    mock_cls.cpu.return_value = mock_cls
    mock_cls.tolist.return_value = [0]
    mock_boxes.cls = mock_cls

    mock_id = MagicMock()
    mock_id.cpu.return_value = mock_id
    mock_id.tolist.return_value = [1]
    mock_boxes.id = mock_id

    mock_result.boxes = mock_boxes

    mock_keypoints = MagicMock()
    mock_xy = MagicMock()
    mock_xy.cpu.return_value = mock_xy
    mock_xy.tolist.return_value = [[[10.0, 20.0]]]
    mock_conf_kp = MagicMock()
    mock_conf_kp.cpu.return_value = mock_conf_kp
    mock_conf_kp.tolist.return_value = [[0.9]]
    mock_keypoints.xy = mock_xy
    mock_keypoints.conf = mock_conf_kp
    mock_result.keypoints = mock_keypoints

    mock_extract_keypoints = MagicMock(
        return_value=([[[10.0, 20.0]]], [[0.9]])
    )
    monkeypatch.setattr(handler, '_extract_keypoints', mock_extract_keypoints)

    result = handler._parse_detection_result(
        mock_result, frame_idx=0, fps=25.0
    )

    assert result is not None
    detections, poses = result
    assert len(detections) == 1
    assert len(poses) == 1
    assert poses[0].track_id == 1
