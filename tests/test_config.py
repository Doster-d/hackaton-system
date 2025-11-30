"""Tests for hackaton_system.config module."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from hackaton_system.config import (
    DETECTION_CLASSES_DEFAULT,
    Settings,
    ZoneDefinition,
)


def test_get_settings_reads_env(monkeypatch: Any, tmp_path: Path) -> None:
    """Settings should reflect environment overrides and use caching."""

    db_path = tmp_path / 'custom.db'
    data_dir = tmp_path / 'artifacts'

    monkeypatch.setenv('HACKATON_DATABASE_URL', f'sqlite:///{db_path}')
    monkeypatch.setenv('HACKATON_DATA_DIR', str(data_dir))

    from hackaton_system import config  # noqa: PLC0415

    config = importlib.reload(config)
    settings = config.get_settings()

    assert settings.database_url == f'sqlite:///{db_path}'
    assert settings.data_dir == Path(data_dir)
    # lru_cache ensures repeated calls return the same object
    assert config.get_settings() is settings


def test_pose_capture_flag_follow_env(monkeypatch: Any) -> None:
    """Pose persistence flag should follow environment overrides."""

    monkeypatch.setenv('HACKATON_ENABLE_POSE_CAPTURE', 'false')

    from hackaton_system import config  # noqa: PLC0415

    config = importlib.reload(config)
    settings = config.get_settings()
    assert settings.enable_pose_capture is False


def test_parse_action_prompts_list():
    """parse_action_prompts should handle list input."""
    prompts = ['action1', 'action2', 'action3']
    result = Settings.parse_action_prompts(prompts)
    assert result == prompts


def test_parse_action_prompts_json_string():
    """parse_action_prompts should parse JSON string."""
    prompts_json = '["action1", "action2"]'
    result = Settings.parse_action_prompts(prompts_json)
    assert result == ['action1', 'action2']


def test_parse_action_prompts_comma_separated():
    """parse_action_prompts should parse comma-separated string."""
    prompts_str = 'action1, action2, action3'
    result = Settings.parse_action_prompts(prompts_str)
    assert result == ['action1', 'action2', 'action3']


def test_parse_action_prompts_single_value():
    """parse_action_prompts should handle single value."""
    result = Settings.parse_action_prompts('single_action')
    assert result == ['single_action']


def test_parse_action_prompts_malformed_json():
    """parse_action_prompts should handle malformed JSON."""
    malformed = '[\n    "action1",\n    "action2"\n]'
    result = Settings.parse_action_prompts(malformed)
    assert len(result) > 0


def test_parse_action_prompts_empty_string():
    """parse_action_prompts should return empty list for empty string."""
    result = Settings.parse_action_prompts('')
    assert result == []


def test_parse_action_prompts_invalid_type():
    """parse_action_prompts should return empty list for invalid type."""
    result = Settings.parse_action_prompts(None)
    assert result == []


def test_parse_train_prompts_list():
    """parse_train_prompts should handle list input."""
    prompts = ['train action1', 'train action2']
    result = Settings.parse_train_prompts(prompts)
    assert result == prompts


def test_parse_train_prompts_json_string():
    """parse_train_prompts should parse JSON string."""
    prompts_json = '["train arriving", "train departing"]'
    result = Settings.parse_train_prompts(prompts_json)
    assert result == ['train arriving', 'train departing']


def test_parse_train_prompts_comma_separated():
    """parse_train_prompts should parse comma-separated string."""
    prompts_str = 'train arriving, train departing'
    result = Settings.parse_train_prompts(prompts_str)
    assert result == ['train arriving', 'train departing']


def test_parse_train_prompts_single_value():
    """parse_train_prompts should handle single value."""
    result = Settings.parse_train_prompts('train present')
    assert result == ['train present']


def test_parse_train_prompts_empty_string():
    """parse_train_prompts should return empty list for empty string."""
    result = Settings.parse_train_prompts('')
    assert result == []


def test_settings_default_values():
    """Settings should have correct default values."""
    settings = Settings()
    assert settings.action_backend == 'heuristic'
    assert settings.action_min_confidence == 0.2
    assert settings.action_clip_frames == 16
    assert settings.videollama3_model_name == 'DAMO-NLP-SG/VideoLLaMA3-2B'
    assert settings.train_videollama3_enabled is True


def test_settings_videollama3_backend():
    """Settings should support videollama3 backend."""
    settings = Settings(action_backend='videollama3')
    assert settings.action_backend == 'videollama3'


def test_settings_zones_default():
    """Settings should have default zones."""
    settings = Settings()
    assert len(settings.zones) > 0
    assert all(isinstance(zone, ZoneDefinition) for zone in settings.zones)


def test_zone_definition():
    """ZoneDefinition should create zone correctly."""
    zone = ZoneDefinition(
        name='test_zone',
        x_min=0,
        y_min=0,
        x_max=100,
        y_max=100,
        zone_category='station',
        person_type='operator',
    )
    assert zone.name == 'test_zone'
    assert zone.zone_category == 'station'
    assert zone.person_type == 'operator'


def test_settings_custom_action_prompts():
    """Settings should accept custom action prompts."""
    custom_prompts = ['custom action 1', 'custom action 2']
    settings = Settings(action_prompts=custom_prompts)
    assert settings.action_prompts == custom_prompts


def test_settings_custom_train_prompts():
    """Settings should accept custom train prompts."""
    custom_prompts = ['custom train action 1', 'custom train action 2']
    settings = Settings(train_prompts=custom_prompts)
    assert settings.train_prompts == custom_prompts


def test_settings_deprecated_xclip_fields():
    """Settings should have deprecated xclip fields for backward compatibility."""
    settings = Settings()
    assert hasattr(settings, 'xclip_model_name')
    assert hasattr(settings, 'xclip_device')
    assert hasattr(settings, 'train_xclip_enabled')


def test_parse_zones_from_dict_list():
    """parse_zones should handle list of dicts."""
    zones_data = [
        {
            'name': 'test_zone',
            'x_min': 0,
            'y_min': 0,
            'x_max': 100,
            'y_max': 100,
            'zone_category': 'station',
            'person_type': 'operator',
        }
    ]
    result = Settings.parse_zones(zones_data)
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert result[0]['name'] == 'test_zone'


def test_parse_zones_single_dict():
    """parse_zones should handle single dict."""
    zone_data = {
        'name': 'test_zone',
        'x_min': 0,
        'y_min': 0,
        'x_max': 100,
        'y_max': 100,
        'zone_category': 'station',
        'person_type': 'operator',
    }
    result = Settings.parse_zones(zone_data)
    assert len(result) == 1
    assert isinstance(result[0], dict)


def test_parse_zones_dict_with_zones_key():
    """parse_zones should handle dict with zones key."""
    zones_data = {
        'zones': [
            {
                'name': 'test_zone',
                'x_min': 0,
                'y_min': 0,
                'x_max': 100,
                'y_max': 100,
                'zone_category': 'station',
                'person_type': 'operator',
            }
        ]
    }
    result = Settings.parse_zones(zones_data)
    assert len(result) == 1


def test_parse_detection_classes_list():
    """parse_detection_classes should handle list input."""
    result = Settings.parse_detection_classes([0, 6, 1])
    assert result == [0, 6, 1]


def test_parse_detection_classes_json_string():
    """parse_detection_classes should parse JSON string."""
    result = Settings.parse_detection_classes('[0, 6, 1]')
    assert result == [0, 6, 1]


def test_parse_detection_classes_invalid_string():
    """parse_detection_classes should return default for invalid string."""
    result = Settings.parse_detection_classes('invalid')
    assert result == DETECTION_CLASSES_DEFAULT


def test_parse_detection_classes_invalid_type():
    """parse_detection_classes should return default for invalid type."""
    result = Settings.parse_detection_classes(None)
    assert result == DETECTION_CLASSES_DEFAULT


def test_parse_class_prompts_dict_with_list():
    """parse_class_prompts should handle dict with list values."""
    prompts = {
        'person': ['action1', 'action2'],
        'train': ['state1', 'state2'],
    }
    result = Settings.parse_class_prompts(prompts)
    assert result == {'person': ['action1', 'action2'], 'train': ['state1', 'state2']}


def test_parse_class_prompts_dict_with_single_value():
    """parse_class_prompts should convert single value to list."""
    prompts = {
        'person': 'single_action',
        'train': 'single_state',
    }
    result = Settings.parse_class_prompts(prompts)
    assert result == {'person': ['single_action'], 'train': ['single_state']}


def test_parse_class_prompts_json_string():
    """parse_class_prompts should parse JSON string."""
    prompts_json = '{"person": ["action1", "action2"], "train": ["state1"]}'
    result = Settings.parse_class_prompts(prompts_json)
    assert result == {'person': ['action1', 'action2'], 'train': ['state1']}


def test_parse_class_prompts_json_string_single_value():
    """parse_class_prompts should handle JSON string with single values."""
    prompts_json = '{"person": "action1", "train": "state1"}'
    result = Settings.parse_class_prompts(prompts_json)
    assert result == {'person': ['action1'], 'train': ['state1']}


def test_parse_class_prompts_invalid_string():
    """parse_class_prompts should return empty dict for invalid string."""
    result = Settings.parse_class_prompts('invalid')
    assert result == {}


def test_parse_class_prompts_invalid_type():
    """parse_class_prompts should return empty dict for invalid type."""
    result = Settings.parse_class_prompts(None)
    assert result == {}


def test_parse_class_prompts_empty_dict():
    """parse_class_prompts should handle empty dict."""
    result = Settings.parse_class_prompts({})
    assert result == {}


def test_parse_action_prompts_malformed_json_with_brackets():
    """parse_action_prompts should extract from malformed JSON with brackets."""
    malformed = '[\n    "action1",\n    "action2"\n]'
    result = Settings.parse_action_prompts(malformed)
    assert len(result) >= 2
    assert 'action1' in result or 'action2' in result


def test_parse_train_prompts_malformed_json_with_brackets():
    """parse_train_prompts should extract from malformed JSON with brackets."""
    malformed = '[\n    "train1",\n    "train2"\n]'
    result = Settings.parse_train_prompts(malformed)
    assert len(result) >= 2
    assert 'train1' in result or 'train2' in result


def test_parse_zones_mixed_list():
    """parse_zones should handle mixed list with dicts and other types."""
    from hackaton_system.config import ZoneDefinition
    
    zones_data = [
        ZoneDefinition(
            name='zone1',
            x_min=0, y_min=0, x_max=100, y_max=100,
            zone_category='station',
            person_type='operator',
        ),
        {
            'name': 'zone2',
            'x_min': 10, 'y_min': 10, 'x_max': 110, 'y_max': 110,
            'zone_category': 'corridor',
            'person_type': 'supervisor',
        }
    ]
    result = Settings.parse_zones(zones_data)
    assert len(result) == 2
    assert all(isinstance(z, dict) for z in result)


def test_parse_zones_single_zonedefinition():
    """parse_zones should handle single ZoneDefinition instance."""
    from hackaton_system.config import ZoneDefinition
    
    zone = ZoneDefinition(
        name='test_zone',
        x_min=0, y_min=0, x_max=100, y_max=100,
        zone_category='station',
        person_type='operator',
    )
    result = Settings.parse_zones(zone)
    assert len(result) == 1
    assert isinstance(result[0], dict)


def test_parse_zones_return_unchanged():
    """parse_zones should return unchanged value for non-handled types."""
    result = Settings.parse_zones(123)
    assert result == 123


def test_parse_action_prompts_malformed_json_exception():
    """parse_action_prompts should handle exceptions in malformed JSON parsing."""
    # This should trigger the exception handler in the malformed JSON parsing
    malformed = '[\n    "action1"\n    "action2"\n]'  # Missing comma, will cause exception
    result = Settings.parse_action_prompts(malformed)
    # Should fall back to comma-separated or single value
    assert isinstance(result, list)


def test_parse_train_prompts_malformed_json_exception():
    """parse_train_prompts should handle exceptions in malformed JSON parsing."""
    # This should trigger the exception handler in the malformed JSON parsing
    malformed = '[\n    "train1"\n    "train2"\n]'  # Missing comma, will cause exception
    result = Settings.parse_train_prompts(malformed)
    # Should fall back to comma-separated or single value
    assert isinstance(result, list)
