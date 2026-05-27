"""Tests for data loading."""

import numpy as np
import pytest

from data_loader import FireEvent, load_all_events, load_fire_event


class TestLoadFireEvent:
    def test_load_bear(self, tiny_config):
        event_info = tiny_config['data']['train_events']['Bear_2020']
        event = load_fire_event('Bear_2020', tiny_config, event_info)
        assert isinstance(event, FireEvent)
        assert event.event_name == 'Bear_2020'
        assert event.static_cnn_input.shape[0] == 7  # 7 static channels
        assert event.fire_seq.ndim == 3
        assert event.fuel_mask.ndim == 2

    def test_spatial_crop(self, tiny_config):
        event_info = tiny_config['data']['train_events']['Bear_2020']
        event = load_fire_event('Bear_2020', tiny_config, event_info)
        y1, y2 = event_info['y1'], event_info['y2']
        x1, x2 = event_info['x1'], event_info['x2']
        expected_h = y2 - y1
        expected_w = x2 - x1
        assert event.grid_shape == (expected_h, expected_w)

    def test_day_limit(self, tiny_config):
        event_info = tiny_config['data']['train_events']['Bear_2020']
        event = load_fire_event('Bear_2020', tiny_config, event_info)
        day_start, day_end = event_info['days']
        assert event.n_days <= day_end - day_start
        assert event.fire_seq.shape[0] == event.n_days + 1
        assert event.wind_u.shape[0] == event.n_days

    def test_day_range_offset(self, tiny_config):
        """Test that days=[1, 3] uses day 0 as initial condition."""
        event_info = dict(tiny_config['data']['train_events']['Bear_2020'])
        event_info['days'] = [1, 3]
        event = load_fire_event('Bear_2020', tiny_config, event_info)
        assert event.n_days == 2
        assert event.fire_seq.shape[0] == 3  # initial (day 0) + 2 targets

    def test_ca_inputs_shapes(self, tiny_config):
        event_info = tiny_config['data']['train_events']['Bear_2020']
        event = load_fire_event('Bear_2020', tiny_config, event_info)
        H, W = event.grid_shape
        assert event.slope_ca.shape == (H, W)
        assert event.aspect_upslope.shape == (H, W)
        assert event.fuel_type_map.shape == (H, W)
        assert event.fuel_type_map.dtype == np.int32


class TestLoadAllEvents:
    def test_loads_train_and_test(self, tiny_config):
        train_events, test_events = load_all_events(tiny_config)
        assert len(train_events) == len(tiny_config['data']['train_events'])
        assert len(test_events) == len(tiny_config['data']['test_events'])
