"""Tests for yolo_classes module."""

from __future__ import annotations

import pytest

from hackaton_system.pipeline.yolo_classes import (
    get_class_id,
    get_class_name,
    get_class_name_ru,
    normalize_class_key,
)


def test_get_class_name_valid_id():
    """get_class_name should return class name for valid ID."""
    assert get_class_name(0) == 'person'
    assert get_class_name(6) == 'train'
    assert get_class_name(1) == 'bicycle'


def test_get_class_name_invalid_id():
    """get_class_name should return class_N for invalid ID."""
    assert get_class_name(999) == 'class_999'
    assert get_class_name(-1) == 'class_-1'


def test_get_class_id_valid_name():
    """get_class_id should return class ID for valid name."""
    assert get_class_id('person') == 0
    assert get_class_id('train') == 6
    assert get_class_id('PERSON') == 0  # Case insensitive
    assert get_class_id('Train') == 6


def test_get_class_id_invalid_name():
    """get_class_id should return None for invalid name."""
    assert get_class_id('invalid_class') is None
    assert get_class_id('') is None


def test_get_class_name_ru_valid():
    """get_class_name_ru should return Russian translation for known class."""
    assert get_class_name_ru('person') == 'Человек'
    assert get_class_name_ru('train') == 'Поезд'
    assert get_class_name_ru('PERSON') == 'Человек'  # Case insensitive


def test_get_class_name_ru_unknown():
    """get_class_name_ru should return original name for unknown class."""
    assert get_class_name_ru('unknown_class') == 'unknown_class'
    assert get_class_name_ru('') == ''


def test_normalize_class_key_int():
    """normalize_class_key should convert int to string."""
    assert normalize_class_key(0) == '0'
    assert normalize_class_key(6) == '6'
    assert normalize_class_key(999) == '999'


def test_normalize_class_key_string_int():
    """normalize_class_key should convert string int to class name."""
    assert normalize_class_key('0') == 'person'
    assert normalize_class_key('6') == 'train'
    assert normalize_class_key('1') == 'bicycle'


def test_normalize_class_key_string_name():
    """normalize_class_key should lowercase class name."""
    assert normalize_class_key('person') == 'person'
    assert normalize_class_key('PERSON') == 'person'
    assert normalize_class_key('Train') == 'train'


def test_normalize_class_key_invalid_string_int():
    """normalize_class_key should handle invalid string int."""
    # Very large number that doesn't exist as class
    result = normalize_class_key('999')
    assert result == 'class_999'  # get_class_name returns class_999 for invalid ID


def test_normalize_class_key_unknown_string():
    """normalize_class_key should lowercase unknown string."""
    assert normalize_class_key('unknown') == 'unknown'
    assert normalize_class_key('UNKNOWN') == 'unknown'

