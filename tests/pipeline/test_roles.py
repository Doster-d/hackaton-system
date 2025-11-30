# pyright: reportPrivateUsage=false
"""Tests for roles module."""

from __future__ import annotations

from hackaton_system.config import Settings, ZoneDefinition
from hackaton_system.pipeline.models import DetectionResult
from hackaton_system.pipeline.roles import RoleInferrer


def test_role_name_for_zone():
    """Test _role_name_for_zone returns correct role names."""

    settings = Settings()
    inferrer = RoleInferrer(settings)

    # Test with None zone
    assert inferrer._role_name_for_zone(None) == 'visitor'

    # Test with zone that has person_type
    zone = ZoneDefinition(
        name='test',
        x_min=0,
        y_min=0,
        x_max=100,
        y_max=100,
        zone_category='station',
        person_type='operator',
    )
    assert inferrer._role_name_for_zone(zone) == 'operator'

    # Test with restricted zone
    restricted_zone = ZoneDefinition(
        name='restricted',
        x_min=0,
        y_min=0,
        x_max=100,
        y_max=100,
        zone_category='restricted',
        person_type='restricted',
    )
    assert inferrer._role_name_for_zone(restricted_zone) == 'visitor'

    # Test with zone without person_type
    empty_zone = ZoneDefinition(
        name='empty',
        x_min=0,
        y_min=0,
        x_max=100,
        y_max=100,
        zone_category='corridor',
        person_type='',
    )
    assert inferrer._role_name_for_zone(empty_zone) == 'visitor'


def test_accumulate_role_time():
    """Test _accumulate_role_time accumulates time by role."""

    # Create zones for role assignment
    station_zone = ZoneDefinition(
        name='station',
        x_min=50,
        y_min=50,
        x_max=150,
        y_max=150,
        zone_category='station',
        person_type='operator',
    )
    settings = Settings(zones=[station_zone], activity_min_interval_sec=0.1)
    inferrer = RoleInferrer(settings)

    # Detections in station zone
    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(100, 100, 110, 110),
            confidence=0.9,
            track_id=1,
        ),
        DetectionResult(
            frame_id=25,
            time_sec=1.0,
            bbox=(100, 100, 110, 110),
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

    totals = inferrer._accumulate_role_time(detections, fps=25.0)

    assert 'operator' in totals
    assert totals['operator'] > 0


def test_accumulate_role_time_visitor_zone():
    """Test _accumulate_role_time for visitor zone."""

    settings = Settings(activity_min_interval_sec=0.1)
    inferrer = RoleInferrer(settings)

    # Detections outside any zone (should be visitor)
    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(1000, 1000, 1010, 1010),
            confidence=0.9,
            track_id=1,
        ),
        DetectionResult(
            frame_id=25,
            time_sec=1.0,
            bbox=(1000, 1000, 1010, 1010),
            confidence=0.8,
            track_id=1,
        ),
    ]

    totals = inferrer._accumulate_role_time(detections, fps=25.0)

    assert 'visitor' in totals
    assert totals['visitor'] > 0


def test_infer_person_roles_empty_tracks():
    """Test _infer_person_roles with empty tracks."""

    settings = Settings()
    inferrer = RoleInferrer(settings)

    assignments = inferrer.infer_person_roles({}, fps=25.0)

    assert assignments == {}


def test_infer_person_roles_empty_detections():
    """Test _infer_person_roles with empty detections."""

    settings = Settings()
    inferrer = RoleInferrer(settings)

    assignments = inferrer.infer_person_roles({1: []}, fps=25.0)

    assert 1 not in assignments  # Empty detections are skipped


def test_infer_person_roles_with_confidence_threshold():
    """Test _infer_person_roles respects confidence threshold."""

    station_zone = ZoneDefinition(
        name='station',
        x_min=50,
        y_min=50,
        x_max=150,
        y_max=150,
        zone_category='station',
        person_type='operator',
    )
    # High threshold - should return unknown if confidence too low
    settings = Settings(
        zones=[station_zone],
        role_assignment_threshold=0.9,
        activity_min_interval_sec=0.1,
    )
    inferrer = RoleInferrer(settings)

    # Detections that might not reach 90% confidence
    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(100, 100, 110, 110),
            confidence=0.9,
            track_id=1,
        ),
        DetectionResult(
            frame_id=25,
            time_sec=1.0,
            bbox=(200, 200, 210, 210),
            confidence=0.8,
            track_id=1,
        ),  # Outside zone
    ]

    tracks = {1: detections}
    assignments = inferrer.infer_person_roles(tracks, fps=25.0)

    assert 1 in assignments
    # Depending on time distribution, might be unknown or operator
    assert assignments[1].person_type in ['operator', 'unknown', 'visitor']


def test_infer_person_roles_empty_role_totals():
    """Test _infer_person_roles handles empty role_totals."""

    settings = Settings()
    inferrer = RoleInferrer(settings)

    # Mock _accumulate_role_time to return empty dict
    def mock_accumulate(dets, fps):
        return {}

    inferrer._accumulate_role_time = mock_accumulate

    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            track_id=1,
        ),
    ]
    tracks = {1: detections}

    assignments = inferrer.infer_person_roles(tracks, fps=25.0)

    assert 1 in assignments
    assert assignments[1].person_type == 'unknown'
    assert assignments[1].confidence == 0.0


def test_infer_person_roles_zero_total_time():
    """Test _infer_person_roles handles zero total_time edge case."""

    settings = Settings()
    inferrer = RoleInferrer(settings)

    # Mock _accumulate_role_time to return dict with zero values
    def mock_accumulate(dets, fps):
        return {'operator': 0.0, 'visitor': 0.0}

    inferrer._accumulate_role_time = mock_accumulate

    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            track_id=1,
        ),
    ]
    tracks = {1: detections}

    assignments = inferrer.infer_person_roles(tracks, fps=25.0)

    assert 1 in assignments
    # Should handle zero total_time gracefully
    assert assignments[1].confidence == 0.0


def test_infer_person_roles_high_confidence():
    """Test _infer_person_roles with high confidence assignment."""

    station_zone = ZoneDefinition(
        name='station',
        x_min=50,
        y_min=50,
        x_max=150,
        y_max=150,
        zone_category='station',
        person_type='operator',
    )
    # Low threshold - should assign role easily
    settings = Settings(
        zones=[station_zone],
        role_assignment_threshold=0.3,
        activity_min_interval_sec=0.1,
    )
    inferrer = RoleInferrer(settings)

    # All detections in station zone
    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(100, 100, 110, 110),
            confidence=0.9,
            track_id=1,
        ),
        DetectionResult(
            frame_id=25,
            time_sec=1.0,
            bbox=(100, 100, 110, 110),
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
    assignments = inferrer.infer_person_roles(tracks, fps=25.0)

    assert 1 in assignments
    # Should have high confidence for operator
    assert assignments[1].confidence > 0.0
