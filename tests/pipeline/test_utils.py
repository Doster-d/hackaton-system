# pyright: reportPrivateUsage=false
"""Tests for utility functions."""

from __future__ import annotations

from hackaton_system.pipeline.utils import center_of_bbox, speed, to_list


def test_to_list_handles_torch_like_objects():
    class Dummy:
        def __init__(self) -> None:
            self.cpu_called = False

        def cpu(self):
            self.cpu_called = True
            return self

        def tolist(self):
            return [1, 2, 3]

    dummy = Dummy()

    assert to_list(dummy) == [1, 2, 3]
    assert dummy.cpu_called


def test_to_list_handles_regular_list():
    """Test to_list with regular Python list."""
    regular_list = [1, 2, 3]
    assert to_list(regular_list) == [1, 2, 3]


def test_to_list_handles_object_with_tolist():
    """Test to_list with object that has tolist but no cpu."""

    class ListLike:
        def tolist(self):
            return [4, 5, 6]

    obj = ListLike()
    assert to_list(obj) == [4, 5, 6]


def test_center_of_bbox():
    """Test center_of_bbox calculates center correctly."""
    bbox = (10, 20, 30, 40)
    result = center_of_bbox(bbox)
    assert result == (20.0, 30.0)


def test_speed_calculates_distance():
    """Test speed calculates speed correctly."""
    prev = (0.0, 0.0)
    curr = (3.0, 4.0)  # distance = 5.0
    duration = 1.0
    result = speed(prev, curr, duration)
    assert abs(result - 5.0) < 0.001  # sqrt(3^2 + 4^2) / 1.0 = 5.0


def test_speed_handles_zero_duration():
    """Test speed returns 0 for zero duration."""
    prev = (0.0, 0.0)
    curr = (10.0, 10.0)
    duration = 0.0
    result = speed(prev, curr, duration)
    assert result == 0.0
