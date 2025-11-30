# pyright: reportPrivateUsage=false
"""Tests for dashboard data access utilities."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd

from hackaton_system.dashboard import data_access as da

RowValue = float | int | str
Row = dict[str, RowValue]


def test_load_headcount_df_groups_unique_people(monkeypatch: Any) -> None:
    rows: list[Row] = [
        {'time_sec': 0.0, 'person_id': 1},
        {'time_sec': 0.0, 'person_id': 2},
        {'time_sec': 5.0, 'person_id': 2},
        {'time_sec': 5.0, 'person_id': 2},
        {'time_sec': 10.0, 'person_id': 3},
    ]

    def fake_fetch_rows(stmt: Any) -> list[Row]:
        return rows.copy()

    monkeypatch.setattr(da, '_fetch_rows', fake_fetch_rows, raising=False)

    df: pd.DataFrame = da._load_headcount_df(video_id=1)

    expected = pd.DataFrame(
        [
            {'time_sec': 0.0, 'headcount': 2},
            {'time_sec': 5.0, 'headcount': 1},
            {'time_sec': 10.0, 'headcount': 1},
        ]
    )
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)


def test_load_person_activity_matrix_builds_manual_pivot(
    monkeypatch: Any,
) -> None:
    rows: list[Row] = [
        {
            'person_type': 'mechanic',
            'activity_class': 'walking',
            'duration_sec': 10.0,
        },
        {
            'person_type': 'mechanic',
            'activity_class': 'welding',
            'duration_sec': 5.0,
        },
        {
            'person_type': 'inspector',
            'activity_class': 'walking',
            'duration_sec': 7.0,
        },
    ]

    def fake_fetch_rows(stmt: Any) -> list[Row]:
        return rows.copy()

    monkeypatch.setattr(da, '_fetch_rows', fake_fetch_rows, raising=False)

    df: pd.DataFrame = da._load_person_activity_matrix(video_id=42)

    assert set(df.columns) == {'person_type', 'walking', 'welding'}
    mechanic = df[df['person_type'] == 'mechanic'].iloc[0]
    inspector = df[df['person_type'] == 'inspector'].iloc[0]
    assert mechanic['walking'] == 10.0
    assert mechanic['welding'] == 5.0
    assert inspector['walking'] == 7.0
    assert inspector['welding'] == 0.0


def test_load_activity_df_falls_back_to_placeholder(monkeypatch: Any) -> None:
    empty_df = pd.DataFrame()

    def fake_loader(statement: Any) -> pd.DataFrame:
        return empty_df

    monkeypatch.setattr(da, '_load_dataframe', fake_loader, raising=False)

    result: pd.DataFrame = da._load_activity_df(video_id=7)

    assert result.equals(da._placeholder_activity_df())


def test_load_activity_summary_with_data(monkeypatch: Any) -> None:
    """Test load_activity_summary calculates duration_min correctly."""
    activity_df = pd.DataFrame(
        {
            'activity_class': ['walking', 'walking', 'idle'],
            't_start_sec': [0.0, 60.0, 120.0],
            't_end_sec': [30.0, 90.0, 180.0],
        }
    )

    def fake_load_activity_df(video_id: Any) -> pd.DataFrame:
        return activity_df.copy()

    monkeypatch.setattr(
        da, '_load_activity_df', fake_load_activity_df, raising=False
    )

    result = da.load_activity_summary(video_id=1)

    assert 'activity_class' in result.columns
    assert 'duration_min' in result.columns
    assert len(result) == 2  # walking merged, idle separate
    walking_total = result[result['activity_class'] == 'walking'][
        'duration_min'
    ].sum()
    assert walking_total == (30.0 + 30.0) / 60.0


def test_load_activity_summary_empty_returns_placeholder(
    monkeypatch: Any,
) -> None:
    """Test load_activity_summary returns placeholder for empty data."""
    empty_df = pd.DataFrame()

    def fake_load_activity_df(video_id: Any) -> pd.DataFrame:
        return empty_df.copy()

    monkeypatch.setattr(
        da, '_load_activity_df', fake_load_activity_df, raising=False
    )

    result = da.load_activity_summary(video_id=1)

    assert not result.empty
    assert 'activity_class' in result.columns
    assert 'duration_min' in result.columns


def test_load_headcount_wraps_internal_function(monkeypatch: Any) -> None:
    """Test load_headcount calls _load_headcount_df."""
    expected_df = pd.DataFrame({'time_sec': [0.0, 5.0], 'headcount': [2, 3]})

    def fake_load_headcount_df(video_id: Any) -> pd.DataFrame:
        return expected_df.copy()

    monkeypatch.setattr(
        da, '_load_headcount_df', fake_load_headcount_df, raising=False
    )

    result = da.load_headcount(video_id=1)

    pd.testing.assert_frame_equal(result, expected_df)


def test_load_role_activity_matrix_converts_to_minutes(
    monkeypatch: Any,
) -> None:
    """Test load_role_activity_matrix converts seconds to minutes."""
    matrix_df = pd.DataFrame(
        {
            'person_type': ['mechanic', 'inspector'],
            'walking': [120.0, 60.0],  # seconds
            'welding': [180.0, 0.0],
        }
    )

    def fake_load_person_activity_matrix(video_id: Any) -> pd.DataFrame:
        return matrix_df.copy()

    monkeypatch.setattr(
        da,
        '_load_person_activity_matrix',
        fake_load_person_activity_matrix,
        raising=False,
    )

    result = da.load_role_activity_matrix(video_id=1)

    assert result['walking'].iloc[0] == 2.0  # 120 seconds = 2 minutes
    assert result['walking'].iloc[1] == 1.0  # 60 seconds = 1 minute
    assert result['welding'].iloc[0] == 3.0  # 180 seconds = 3 minutes


def test_load_role_activity_matrix_empty_returns_placeholder(
    monkeypatch: Any,
) -> None:
    """Test load_role_activity_matrix returns placeholder for empty data."""
    empty_df = pd.DataFrame()

    def fake_load_person_activity_matrix(video_id: Any) -> pd.DataFrame:
        return empty_df.copy()

    monkeypatch.setattr(
        da,
        '_load_person_activity_matrix',
        fake_load_person_activity_matrix,
        raising=False,
    )

    result = da.load_role_activity_matrix(video_id=1)

    assert not result.empty
    assert 'person_type' in result.columns


def test_load_person_episodes_with_none_returns_placeholder() -> None:
    """Test load_person_episodes returns placeholder for None video_id."""
    result = da.load_person_episodes(video_id=None)

    assert not result.empty
    assert 'person_id' in result.columns
    assert 'activity_class' in result.columns


def test_load_person_episodes_with_data(monkeypatch: Any) -> None:
    """Test load_person_episodes loads episodes from database."""
    episodes_df = pd.DataFrame(
        {
            'person_id': [1, 1, 2],
            'person_type': ['operator', 'operator', 'supervisor'],
            'activity_class': ['working', 'idle_at_station', 'walking'],
            't_start_sec': [0.0, 60.0, 120.0],
            't_end_sec': [30.0, 90.0, 180.0],
            'duration_sec': [30.0, 30.0, 60.0],
        }
    )

    def fake_load_dataframe(stmt: Any) -> pd.DataFrame:
        return episodes_df.copy()

    monkeypatch.setattr(
        da, '_load_dataframe', fake_load_dataframe, raising=False
    )

    result = da.load_person_episodes(video_id=1)

    assert len(result) == 3
    assert 'person_id' in result.columns
    assert 'activity_class' in result.columns
    assert 'duration_sec' in result.columns


def test_load_person_episodes_empty_returns_placeholder(
    monkeypatch: Any,
) -> None:
    """Test load_person_episodes returns placeholder for empty data."""
    empty_df = pd.DataFrame()

    def fake_load_dataframe(stmt: Any) -> pd.DataFrame:
        return empty_df.copy()

    monkeypatch.setattr(
        da, '_load_dataframe', fake_load_dataframe, raising=False
    )

    result = da.load_person_episodes(video_id=1)

    assert not result.empty
    assert 'person_id' in result.columns


def test_list_available_videos_returns_placeholder_when_empty(
    monkeypatch: Any,
) -> None:
    """Test list_available_videos returns placeholder when no videos."""
    empty_df = pd.DataFrame()

    def fake_load_dataframe(stmt: Any) -> pd.DataFrame:
        return empty_df.copy()

    monkeypatch.setattr(
        da, '_load_dataframe', fake_load_dataframe, raising=False
    )

    result = da.list_available_videos()

    assert len(result) == 1
    assert result.iloc[0]['id'] is None
    assert result.iloc[0]['filename'] == 'Demo placeholder'


def test_list_available_videos_returns_videos_when_present(
    monkeypatch: Any,
) -> None:
    """Test list_available_videos returns videos when present."""
    videos_df = pd.DataFrame(
        [
            {
                'id': 1,
                'filename': 'test1.mp4',
                'duration_sec': 100.0,
                'fps': 25.0,
            },
        ]
    )

    def fake_load_dataframe(stmt: Any) -> pd.DataFrame:
        return videos_df.copy()

    monkeypatch.setattr(
        da, '_load_dataframe', fake_load_dataframe, raising=False
    )

    result = da.list_available_videos()

    assert len(result) == 1
    assert result.iloc[0]['filename'] == 'test1.mp4'


def test_load_dashboard_data(monkeypatch: Any) -> None:
    """Test load_dashboard_data returns DashboardData."""
    activities_df = pd.DataFrame(
        {'activity_class': ['walking'], 'duration_min': [10.0]}
    )
    headcount_df = pd.DataFrame({'time_sec': [0.0], 'headcount': [2]})
    matrix_df = pd.DataFrame({'person_type': ['operator'], 'walking': [5.0]})

    def fake_load_activity_df(video_id: Any) -> pd.DataFrame:
        return activities_df.copy()

    def fake_load_headcount_df(video_id: Any) -> pd.DataFrame:
        return headcount_df.copy()

    def fake_load_person_activity_matrix(video_id: Any) -> pd.DataFrame:
        return matrix_df.copy()

    monkeypatch.setattr(
        da, '_load_activity_df', fake_load_activity_df, raising=False
    )
    monkeypatch.setattr(
        da, '_load_headcount_df', fake_load_headcount_df, raising=False
    )
    monkeypatch.setattr(
        da,
        '_load_person_activity_matrix',
        fake_load_person_activity_matrix,
        raising=False,
    )

    result = da.load_dashboard_data(video_id=1)

    assert result.activities.equals(activities_df)
    assert result.headcount.equals(headcount_df)
    assert result.person_matrix.equals(matrix_df)


def test_placeholder_activity_df() -> None:
    """Test _placeholder_activity_df returns correct structure."""
    result = da._placeholder_activity_df()

    assert 'activity_class' in result.columns
    assert 'duration_min' in result.columns
    assert len(result) == 4


def test_placeholder_headcount_df() -> None:
    """Test _placeholder_headcount_df returns correct structure."""
    result = da._placeholder_headcount_df()

    assert 'time_sec' in result.columns
    assert 'headcount' in result.columns
    assert len(result) > 0


def test_placeholder_matrix_df() -> None:
    """Test _placeholder_matrix_df returns correct structure."""
    result = da._placeholder_matrix_df()

    assert 'person_type' in result.columns
    assert 'walking' in result.columns
    assert len(result) == 3


def test_placeholder_episodes_df() -> None:
    """Test _placeholder_episodes_df returns correct structure."""
    result = da._placeholder_episodes_df()

    assert 'person_id' in result.columns
    assert 'activity_class' in result.columns
    assert len(result) == 4


def test_delete_video_and_related_not_found(monkeypatch: Any) -> None:
    """Test delete_video_and_related returns 0 when video not found."""
    mock_session = MagicMock()
    mock_session.query.return_value.filter.return_value.first.return_value = (
        None
    )

    def fake_get_session():
        @contextmanager
        def session_ctx():
            yield mock_session

        return session_ctx()

    monkeypatch.setattr(da, 'get_session', fake_get_session, raising=False)

    result = da.delete_video_and_related(video_id=999)

    assert result == 0


def _reload_dashboard_modules(monkeypatch: Any, tmp_path: Path):
    """Reload config/session/data_access against temporary SQLite DB."""
    db_path = tmp_path / 'manual_actions.db'
    monkeypatch.setenv('HACKATON_DATABASE_URL', f'sqlite:///{db_path}')
    monkeypatch.setenv('HACKATON_DATA_DIR', str(tmp_path))

    import hackaton_system.config as config_module  # noqa: PLC0415
    import hackaton_system.dashboard.data_access as data_access_module  # noqa: PLC0415
    import hackaton_system.db.models as models_module  # noqa: PLC0415
    import hackaton_system.db.session as session_module  # noqa: PLC0415

    config_module = importlib.reload(config_module)
    session_module = importlib.reload(session_module)
    models_module = importlib.reload(models_module)
    data_access_module = importlib.reload(data_access_module)

    session_module.init_db()

    return data_access_module, session_module, models_module


def test_save_manual_activity_creates_manual_entry(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Manual action saves should insert reviewed activity rows."""
    da, session_module, models_module = _reload_dashboard_modules(
        monkeypatch, tmp_path
    )
    with session_module.get_session() as session:
        video = models_module.Video(
            filename='train.mp4', fps=25.0, duration_sec=120.0
        )
        session.add(video)
        session.flush()
        person = models_module.Person(
            video_id=video.id,
            track_id=10,
            person_type='operator',
            person_type_conf=0.5,
        )
        session.add(person)
        session.flush()
        person_id = person.id
        video_id = video.id

    saved = da.save_manual_activity(
        video_id=video_id,
        person_id=person_id,
        track_id=10,
        activity_class='railway worker is welding a train part',
        start_sec=5.0,
        end_sec=15.0,
        reviewer='safety_officer',
        notes='verified on-site',
    )

    assert saved.activity_class == 'railway worker is welding a train part'
    assert saved.person_id == person_id
    assert saved.source == 'manual'
    assert saved.review_status == 'approved'
    assert saved.reviewer == 'safety_officer'
    assert saved.review_notes == 'verified on-site'

    with session_module.get_session() as session:
        stored = session.query(models_module.Activity).all()
        assert len(stored) == 1
        assert stored[0].activity_class == saved.activity_class


def test_save_manual_activity_updates_existing_entry(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Manual saves against existing activity rows should update labels/source."""
    da, session_module, models_module = _reload_dashboard_modules(
        monkeypatch, tmp_path
    )
    with session_module.get_session() as session:
        video = models_module.Video(
            filename='train.mp4', fps=25.0, duration_sec=60.0
        )
        session.add(video)
        session.flush()
        person = models_module.Person(
            video_id=video.id, track_id=4, person_type='operator'
        )
        session.add(person)
        session.flush()
        activity = models_module.Activity(
            video_id=video.id,
            person_id=person.id,
            activity_class='auto_prediction',
            t_start_sec=0.0,
            t_end_sec=5.0,
            activity_conf=0.55,
            source='videollama3',
            review_status='pending',
        )
        session.add(activity)
        session.flush()
        video_id = video.id
        person_id = person.id
        existing_id = activity.id

    updated = da.save_manual_activity(
        video_id=video_id,
        person_id=person_id,
        track_id=4,
        activity_class='manual_override_activity',
        start_sec=0.0,
        end_sec=5.0,
        reviewer='chief_inspector',
        activity_id=existing_id,
    )

    assert updated.id == existing_id
    assert updated.activity_class == 'manual_override_activity'
    assert updated.source == 'manual'
    assert updated.review_status == 'approved'
    assert updated.reviewer == 'chief_inspector'

    with session_module.get_session() as session:
        refreshed = session.get(models_module.Activity, existing_id)
        assert refreshed is not None
        assert refreshed.activity_class == 'manual_override_activity'
        assert refreshed.source == 'manual'
        assert refreshed.review_status == 'approved'
