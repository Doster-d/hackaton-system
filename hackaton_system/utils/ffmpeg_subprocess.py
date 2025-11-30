"""Utilities for delegating video work to the system FFmpeg binaries."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_FFMPEG_ENV = 'HACKATON_FFMPEG_BIN'
DEFAULT_FFPROBE_ENV = 'HACKATON_FFPROBE_BIN'


def _resolve_binary(env_var: str, preferred: str | None, fallback: str) -> str:
    if preferred:
        return preferred
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value
    return fallback


def get_ffmpeg_binary(preferred: str | None = None) -> str:
    """Return FFmpeg binary path honoring overrides."""

    return _resolve_binary(DEFAULT_FFMPEG_ENV, preferred, 'ffmpeg')


def get_ffprobe_binary(preferred: str | None = None) -> str:
    """Return FFprobe binary path honoring overrides."""

    return _resolve_binary(DEFAULT_FFPROBE_ENV, preferred, 'ffprobe')


def start_ffmpeg_writer(
    *,
    width: int,
    height: int,
    fps: float,
    output_path: Path | str,
    ffmpeg_path: str | None = None,
    input_pix_fmt: str = 'bgr24',
    vcodec: str = 'libx264',
    preset: str = 'veryfast',
    crf: int | str = 30,
    loglevel: str = 'error',
) -> subprocess.Popen[bytes]:
    """Spawn FFmpeg process that accepts raw frames on stdin."""

    ffmpeg_bin = get_ffmpeg_binary(ffmpeg_path)
    cmd = [
        ffmpeg_bin,
        '-y',
        '-loglevel',
        loglevel,
        '-f',
        'rawvideo',
        '-pix_fmt',
        input_pix_fmt,
        '-s',
        f'{width}x{height}',
        '-r',
        f'{fps}',
        '-i',
        'pipe:0',
        '-an',
        '-c:v',
        vcodec,
        '-pix_fmt',
        'yuv420p',
        '-preset',
        preset,
        '-crf',
        str(crf),
        str(output_path),
    ]
    return subprocess.Popen(  # noqa: S603
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def run_ffprobe(
    video_path: Path | str,
    *,
    ffprobe_path: str | None = None,
) -> dict[str, Any]:
    """Return ffprobe metadata as a Python dict."""

    ffprobe_bin = get_ffprobe_binary(ffprobe_path)
    cmd = [
        ffprobe_bin,
        '-v',
        'error',
        '-show_streams',
        '-show_format',
        '-print_format',
        'json',
        str(video_path),
    ]
    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - passthrough
        raise RuntimeError(
            f'ffprobe failed for {video_path}: {exc.stderr}'
        ) from exc
    return json.loads(result.stdout or '{}')


def run_ffmpeg_rawvideo(
    video_path: Path | str,
    *,
    pix_fmt: str,
    filters: Sequence[str] | None = None,
    input_options: Mapping[str, Any] | None = None,
    output_options: Mapping[str, Any] | None = None,
    ffmpeg_path: str | None = None,
    verbose: bool = False,
) -> bytes:
    """Run FFmpeg and return raw video bytes from stdout."""

    ffmpeg_bin = get_ffmpeg_binary(ffmpeg_path)
    cmd: list[str] = [
        ffmpeg_bin,
        '-hide_banner',
        '-loglevel',
        'info' if verbose else 'error',
    ]
    for key, value in (input_options or {}).items():
        cmd.extend([f'-{key}', str(value)])
    cmd.extend(['-i', str(video_path)])
    if filters:
        cmd.extend(['-vf', ','.join(filters)])
    for key, value in (output_options or {}).items():
        cmd.extend([f'-{key}', str(value)])
    cmd.extend(['-f', 'rawvideo', '-pix_fmt', pix_fmt, 'pipe:1'])
    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - passthrough
        stderr = exc.stderr.decode(errors='ignore') if isinstance(exc.stderr, bytes) else exc.stderr
        raise RuntimeError(f'ffmpeg failed for {video_path}: {stderr}') from exc
    return result.stdout
