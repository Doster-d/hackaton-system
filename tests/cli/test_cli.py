"""Tests for CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from typer.testing import CliRunner

from hackaton_system.cli import app

runner = CliRunner()


def test_setup_database_command(monkeypatch: Any, tmp_path: Path) -> None:
    """Test setup-database command."""
    mock_init_db = MagicMock()
    monkeypatch.setattr('hackaton_system.cli.init_db', mock_init_db)

    result = runner.invoke(app, ['setup-database'])

    assert result.exit_code == 0
    assert 'Database initialized' in result.output
    mock_init_db.assert_called_once()


def test_seed_demo_command_no_force(monkeypatch: Any) -> None:
    """Test seed-demo command without force flag."""
    mock_seed = MagicMock(return_value=0)
    monkeypatch.setattr('hackaton_system.cli.seed_demo_data', mock_seed)

    result = runner.invoke(app, ['seed-demo'])

    assert result.exit_code == 0
    assert 'Demo data already present' in result.output
    mock_seed.assert_called_once_with(force=False)


def test_seed_demo_command_with_force(monkeypatch: Any) -> None:
    """Test seed-demo command with force flag."""
    mock_seed = MagicMock(return_value=100)
    monkeypatch.setattr('hackaton_system.cli.seed_demo_data', mock_seed)

    result = runner.invoke(app, ['seed-demo', '--force'])

    assert result.exit_code == 0
    assert 'Inserted 100 demo rows' in result.output
    mock_seed.assert_called_once_with(force=True)


def test_process_video_command(monkeypatch: Any, tmp_path: Path) -> None:
    """Test process-video command."""
    test_video = tmp_path / 'test_video.mp4'
    test_video.write_bytes(b'fake video data')

    mock_processor = MagicMock()
    mock_processor.process_video.return_value = 42
    mock_video_processor = MagicMock(return_value=mock_processor)
    monkeypatch.setattr(
        'hackaton_system.cli.VideoProcessor', mock_video_processor
    )

    result = runner.invoke(app, ['process-video', str(test_video)])

    assert result.exit_code == 0
    assert 'Video stored with id=42' in result.output
    mock_video_processor.assert_called_once()
    mock_processor.process_video.assert_called_once_with(test_video)


def test_process_video_command_file_not_found() -> None:
    """Test process-video command with non-existent file."""
    result = runner.invoke(app, ['process-video', 'nonexistent.mp4'])

    assert result.exit_code != 0
    output = result.output
    assert (
        'does not exist' in output
        or 'Error' in output
        or 'No such file' in output
    )


def test_cli_help() -> None:
    """Test CLI help output."""
    result = runner.invoke(app, ['--help'])

    assert result.exit_code == 0
    assert 'Hackathon video analytics toolkit' in result.output
    assert 'setup-database' in result.output
    assert 'process-video' in result.output
    assert 'seed-demo' in result.output
