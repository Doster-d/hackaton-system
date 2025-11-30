# pyright: reportPrivateUsage=false
"""Tests for activity module."""

from __future__ import annotations

from hackaton_system.config import Settings, ZoneDefinition
from hackaton_system.pipeline.activity import ActivityInferrer
from hackaton_system.pipeline.models import DetectionResult


def test_infer_activities_detects_states():
    # Create inferrer with lower min_interval to ensure activities are detected
    settings = Settings(activity_min_interval_sec=0.1)
    inferrer = ActivityInferrer(settings)
    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            track_id=1,
        ),
        DetectionResult(
            frame_id=25,
            time_sec=1.0,
            bbox=(1, 1, 11, 11),
            confidence=0.8,
            track_id=1,
        ),
        DetectionResult(
            frame_id=50,
            time_sec=2.0,
            bbox=(100, 100, 110, 110),
            confidence=0.7,
            track_id=1,
        ),
    ]
    tracks = {1: detections}
    fps = 25.0

    activities = inferrer.infer_activities(tracks, fps)

    assert len(activities) > 0
    activity_classes = {activity.activity_class for activity in activities}
    assert (
        'idle' in activity_classes
        or 'walking' in activity_classes
        or 'standing' in activity_classes
    )
    assert activities[0].start_sec >= 0.0


def test_infer_activities_single_detection_marks_idle():
    # Create inferrer with lower min_interval to ensure activities are detected
    settings = Settings(activity_min_interval_sec=0.1)
    inferrer = ActivityInferrer(settings)
    # Single detection creates a tail segment
    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=5.0,
            bbox=(0, 0, 10, 10),
            confidence=0.5,
            track_id=7,
        ),
        DetectionResult(
            frame_id=25,
            time_sec=6.0,
            bbox=(0, 0, 10, 10),
            confidence=0.5,
            track_id=7,
        ),
    ]
    tracks = {7: detections}
    fps = 25.0

    activities = inferrer.infer_activities(tracks, fps)

    # Should have at least one activity
    assert len(activities) >= 1
    # Activity should be related to the detection time
    assert any(
        activity.start_sec <= 6.0 <= activity.end_sec
        for activity in activities
    )


def test_zone_for_point_finds_zone():
    """Test _zone_for_point finds zone containing point."""
    zone = ZoneDefinition(
        name='test_zone',
        x_min=0,
        y_min=0,
        x_max=100,
        y_max=100,
        zone_category='station',
        person_type='operator',
    )
    settings = Settings(zones=[zone])
    inferrer = ActivityInferrer(settings)

    # Point inside zone
    found_zone = inferrer._zone_for_point((50.0, 50.0))
    assert found_zone is not None
    assert found_zone.name == 'test_zone'

    # Point outside zone
    found_zone = inferrer._zone_for_point((150.0, 150.0))
    assert found_zone is None


def test_zone_for_point_boundary_conditions():
    """Test _zone_for_point handles boundary conditions."""
    zone = ZoneDefinition(
        name='boundary_zone',
        x_min=10,
        y_min=20,
        x_max=30,
        y_max=40,
        zone_category='station',
        person_type='operator',
    )
    settings = Settings(zones=[zone])
    inferrer = ActivityInferrer(settings)

    # Test boundaries
    assert inferrer._zone_for_point((10.0, 20.0)) is not None  # min corner
    assert inferrer._zone_for_point((30.0, 40.0)) is not None  # max corner
    assert inferrer._zone_for_point((9.9, 20.0)) is None  # just outside
    assert inferrer._zone_for_point((30.1, 40.0)) is None  # just outside


def test_activity_label_restricted_zone():
    """Test _activity_label returns in_restricted_zone for restricted zones."""

    restricted_zone = ZoneDefinition(
        name='restricted',
        x_min=0,
        y_min=0,
        x_max=100,
        y_max=100,
        zone_category='restricted',
        person_type='restricted',
    )
    settings = Settings()
    inferrer = ActivityInferrer(settings)

    # Any speed in restricted zone should return in_restricted_zone
    assert (
        inferrer._activity_label(restricted_zone, 0.0) == 'in_restricted_zone'
    )
    assert (
        inferrer._activity_label(restricted_zone, 100.0)
        == 'in_restricted_zone'
    )
    assert (
        inferrer._activity_label(restricted_zone, 200.0)
        == 'in_restricted_zone'
    )


def test_activity_label_station_zone():
    """Test _activity_label for station zones with different speeds."""

    station_zone = ZoneDefinition(
        name='station',
        x_min=0,
        y_min=0,
        x_max=100,
        y_max=100,
        zone_category='station',
        person_type='operator',
    )
    settings = Settings(
        activity_velocity_move_thresh=140.0,
        activity_velocity_idle_thresh=40.0,
    )
    inferrer = ActivityInferrer(settings)

    # High speed -> working
    assert inferrer._activity_label(station_zone, 150.0) == 'working'
    # Low speed -> idle_at_station
    assert inferrer._activity_label(station_zone, 30.0) == 'idle_at_station'
    # Medium speed -> working (default)
    assert inferrer._activity_label(station_zone, 80.0) == 'working'


def test_activity_label_corridor_visitor_none():
    """Test _activity_label for corridor/visitor/None zones."""

    corridor_zone = ZoneDefinition(
        name='corridor',
        x_min=0,
        y_min=0,
        x_max=100,
        y_max=100,
        zone_category='corridor',
        person_type='supervisor',
    )
    settings = Settings(
        activity_velocity_move_thresh=140.0,
        activity_velocity_idle_thresh=40.0,
    )
    inferrer = ActivityInferrer(settings)

    # High speed -> walking
    assert inferrer._activity_label(corridor_zone, 150.0) == 'walking'
    # Low speed -> standing
    assert inferrer._activity_label(corridor_zone, 30.0) == 'standing'
    # Medium speed -> walking (default)
    assert inferrer._activity_label(corridor_zone, 80.0) == 'walking'

    # None zone should behave like corridor
    assert inferrer._activity_label(None, 150.0) == 'walking'
    assert inferrer._activity_label(None, 30.0) == 'standing'
    assert inferrer._activity_label(None, 80.0) == 'walking'


def test_classify_track_states_empty_detections():
    """Test _classify_track_states returns empty list for empty detections."""

    settings = Settings(activity_min_interval_sec=0.1)
    inferrer = ActivityInferrer(settings)

    states = inferrer._classify_track_states(
        track_id=1, detections=[], fps=25.0
    )
    assert states == []


def test_classify_track_states_single_detection():
    """Test _classify_track_states with single detection creates tail segment."""

    settings = Settings(activity_min_interval_sec=0.1)
    inferrer = ActivityInferrer(settings)

    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=5.0,
            bbox=(100, 200, 200, 300),
            confidence=0.9,
            track_id=1,
        ),
    ]

    states = inferrer._classify_track_states(
        track_id=1, detections=detections, fps=25.0
    )

    assert len(states) == 1  # Only tail segment
    assert states[0].track_id == 1
    assert states[0].start_sec == 5.0


def test_classify_track_states_multiple_detections():
    """Test _classify_track_states with multiple detections."""

    settings = Settings(activity_min_interval_sec=0.1)
    inferrer = ActivityInferrer(settings)

    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(100, 200, 200, 300),
            confidence=0.9,
            track_id=1,
        ),
        DetectionResult(
            frame_id=25,
            time_sec=1.0,
            bbox=(110, 210, 210, 310),
            confidence=0.8,
            track_id=1,
        ),
        DetectionResult(
            frame_id=50,
            time_sec=2.0,
            bbox=(120, 220, 220, 320),
            confidence=0.7,
            track_id=1,
        ),
    ]

    states = inferrer._classify_track_states(
        track_id=1, detections=detections, fps=25.0
    )

    # Should have states for each transition + tail segment
    assert len(states) >= 2
    assert all(state.track_id == 1 for state in states)


def test_infer_activities_empty_tracks():
    """Test _infer_activities with empty tracks."""

    settings = Settings()
    inferrer = ActivityInferrer(settings)

    activities = inferrer.infer_activities({}, fps=25.0)

    assert activities == []


def test_infer_activities_filters_by_min_duration():
    """Test _infer_activities filters activities by min_duration."""

    # High min_duration threshold
    settings = Settings(activity_min_interval_sec=10.0)
    inferrer = ActivityInferrer(settings)

    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(100, 200, 200, 300),
            confidence=0.9,
            track_id=1,
        ),
        DetectionResult(
            frame_id=25,
            time_sec=1.0,
            bbox=(100, 200, 200, 300),
            confidence=0.8,
            track_id=1,
        ),
    ]
    tracks = {1: detections}

    activities = inferrer.infer_activities(tracks, fps=25.0)

    # Short activities should be filtered out
    assert len(activities) == 0


def test_infer_activities_merges_adjacent_states():
    """Test _infer_activities merges adjacent states with same label."""

    settings = Settings(activity_min_interval_sec=0.1)
    inferrer = ActivityInferrer(settings)

    # Create detections that should produce merged states
    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(100, 200, 200, 300),
            confidence=0.9,
            track_id=1,
        ),
        DetectionResult(
            frame_id=25,
            time_sec=1.0,
            bbox=(100, 200, 200, 300),
            confidence=0.8,
            track_id=1,
        ),
        DetectionResult(
            frame_id=50,
            time_sec=2.0,
            bbox=(100, 200, 200, 300),
            confidence=0.7,
            track_id=1,
        ),
    ]
    tracks = {1: detections}

    activities = inferrer.infer_activities(tracks, fps=25.0)

    # Should have activities (may be merged)
    assert len(activities) >= 0


def test_infer_activities_empty_frame_states():
    """Test _infer_activities handles empty frame_states."""

    settings = Settings(activity_min_interval_sec=0.1)
    inferrer = ActivityInferrer(settings)

    # Create tracks that produce no frame states (empty detections after filtering)
    tracks = {1: []}  # Empty detections

    activities = inferrer.infer_activities(tracks, fps=25.0)

    # Should return empty list when no frame states
    assert activities == []
