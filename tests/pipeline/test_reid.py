# pyright: reportPrivateUsage=false
"""Tests for reid module."""

from __future__ import annotations

import numpy as np

from hackaton_system.config import Settings
from hackaton_system.pipeline.models import DetectionResult
from hackaton_system.pipeline.reid import ReIDHandler


def test_create_track_summaries():
    """Test _create_track_summaries creates summaries from tracks."""

    settings = Settings()
    handler = ReIDHandler(settings)

    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            track_id=1,
            embedding=[1.0, 2.0, 3.0],
        ),
        DetectionResult(
            frame_id=1,
            time_sec=1.0,
            bbox=(1, 1, 11, 11),
            confidence=0.8,
            track_id=1,
            embedding=[2.0, 3.0, 4.0],
        ),
    ]

    tracks = {1: detections}
    summaries = handler._create_track_summaries(tracks)

    assert len(summaries) == 1
    assert summaries[0]['track_id'] == 1


def test_create_track_summary_with_embeddings():
    """Test _create_track_summary creates summary with embeddings."""

    settings = Settings()
    handler = ReIDHandler(settings)

    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            track_id=1,
            embedding=[1.0, 2.0],
        ),
    ]

    summary = handler._create_track_summary(1, detections)

    assert summary is not None
    assert summary['track_id'] == 1


def test_create_track_summary_no_embeddings():
    """Test _create_track_summary returns None when no embeddings."""

    settings = Settings()
    handler = ReIDHandler(settings)

    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            track_id=1,
            embedding=None,
        ),
    ]

    summary = handler._create_track_summary(1, detections)

    assert summary is None


def test_find_best_match_finds_match():
    """Test _find_best_match finds matching track."""

    settings = Settings()
    handler = ReIDHandler(settings)

    summary = {
        'track_id': 2,
        'start': 5.0,
        'embedding': np.array([1.0, 0.0], dtype=np.float32),
    }

    history = [
        {
            'track_id': 1,
            'end': 3.0,
            'assigned_id': 1,
            'embedding': np.array([1.0, 0.0], dtype=np.float32),
        },
    ]

    match_id = handler._find_best_match(
        summary, history, gap_limit=2.5, similarity_thresh=0.6
    )

    assert match_id == 1


def test_find_best_match_no_match():
    """Test _find_best_match returns None when no match."""

    settings = Settings()
    handler = ReIDHandler(settings)

    summary = {
        'track_id': 2,
        'start': 10.0,
        'embedding': np.array([1.0, 0.0], dtype=np.float32),
    }

    history = [
        {
            'track_id': 1,
            'end': 3.0,
            'embedding': np.array([0.0, 1.0], dtype=np.float32),
        },
    ]

    match_id = handler._find_best_match(
        summary, history, gap_limit=2.5, similarity_thresh=0.6
    )

    assert match_id is None


def test_is_valid_candidate():
    """Test _is_valid_candidate checks candidate validity."""

    settings = Settings()
    handler = ReIDHandler(settings)

    summary = {
        'start': 5.0,
    }

    candidate = {
        'track_id': 1,
        'end': 3.0,
    }

    gap_limit = 2.5

    assert handler._is_valid_candidate(summary, candidate, gap_limit) is True

    summary['start'] = 10.0
    assert handler._is_valid_candidate(summary, candidate, gap_limit) is False


def test_apply_track_mapping():
    """Test _apply_track_mapping applies track ID mapping."""

    settings = Settings()
    handler = ReIDHandler(settings)

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
    ]

    mapping = {1: 1, 2: 1}

    handler._apply_track_mapping(detections, mapping)

    assert detections[0].track_id == 1
    assert detections[1].track_id == 1


def test_compute_track_embeddings():
    """Test _compute_track_embeddings computes normalized embeddings."""

    settings = Settings()
    handler = ReIDHandler(settings)

    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            track_id=1,
            embedding=[1.0, 0.0],
        ),
        DetectionResult(
            frame_id=1,
            time_sec=1.0,
            bbox=(0, 0, 10, 10),
            confidence=0.8,
            track_id=1,
            embedding=[2.0, 0.0],
        ),
    ]

    tracks = {1: detections}
    embeddings = handler.compute_track_embeddings(tracks)

    assert 1 in embeddings
    assert isinstance(embeddings[1], str)


def test_compute_track_embeddings_no_embeddings():
    """Test _compute_track_embeddings skips tracks without embeddings."""

    settings = Settings()
    handler = ReIDHandler(settings)

    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            track_id=1,
            embedding=None,
        ),
    ]

    tracks = {1: detections}
    embeddings = handler.compute_track_embeddings(tracks)

    assert 1 not in embeddings


def test_stitch_tracks_with_reid():
    """Test _stitch_tracks_with_reid stitches tracks using ReID."""

    settings = Settings()
    handler = ReIDHandler(settings)

    detections = [
        DetectionResult(
            frame_id=0,
            time_sec=0.0,
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            track_id=1,
            embedding=[1.0, 0.0],
        ),
        DetectionResult(
            frame_id=10,
            time_sec=5.0,
            bbox=(0, 0, 10, 10),
            confidence=0.8,
            track_id=2,
            embedding=[1.0, 0.0],
        ),
    ]

    result = handler.stitch_tracks_with_reid(detections)

    assert len(result) == 2
