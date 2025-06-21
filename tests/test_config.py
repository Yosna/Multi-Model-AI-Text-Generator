import json
from unittest.mock import patch

import pytest

from run.config import (
    _build_button_window,
    _build_config_window,
    _clean_config_value,
    _collapse_config,
    _expand_config,
    _save_config,
    add_config_button,
    build_window,
    initialize_dpg,
    run_config,
)


def build_file(tmp_path, file_name, content):
    file = tmp_path / file_name
    file.write_text(content)
    return file


def test_run_config(tmp_path):
    with patch("run.config.build_window") as build_window, patch(
        "run.config.initialize_dpg"
    ) as initialize_dpg:
        path = build_file(tmp_path, "config.json", "{}")
        run_config(path=path)
        build_window.assert_called_once()
        initialize_dpg.assert_called_once()


@patch("run.config.dpg")
def test_build_window(dpg):
    path = "run.config._build"
    with patch(f"{path}_config_window") as config_window, patch(
        f"{path}_button_window"
    ) as button_window:
        config_window.return_value = ([], {})
        build_window("test", 100, 100, {"test": "test"})
        config_window.assert_called_once()
        button_window.assert_called_once()


@patch("run.config.dpg")
def test_build_config_window(dpg):
    config = {"test": {"value": 1}}
    with patch("run.config.add_config_item") as add_config_item:
        headers, widgets = _build_config_window(config, 100, 100)
        dpg.collapsing_header.assert_called_once()
        dpg.group.assert_called_once()
        add_config_item.assert_called_once()
        assert len(headers) == 1
        assert len(widgets) == 1


@patch("run.config.dpg")
def test_build_button_window(dpg):
    widgets = {}
    _build_button_window("test", [], widgets, 100, 100)
    dpg.child_window.assert_called_once()
    dpg.table.assert_called_once()
    assert dpg.add_table_column.call_count == 4
    dpg.table_row.assert_called_once()
    assert dpg.table_cell.call_count == 4


def test_add_config_button():
    with patch("run.config.dpg") as dpg:
        add_config_button("test", 100, lambda: None)
        dpg.add_button.assert_called_once()


def test_expand_config():
    with patch("run.config.dpg") as dpg:
        _expand_config(None, None, {"headers": ["test"]})
        dpg.set_value.assert_called_once()


def test_collapse_config():
    with patch("run.config.dpg") as dpg:
        _collapse_config(None, None, {"headers": ["test"]})
        dpg.set_value.assert_called_once()


def test_save_config(tmp_path):
    path = build_file(tmp_path, "config.json", '{"test": {"value": 1}}')
    user_data = {
        "widgets": {"test.value": "test.value"},
        "path": path,
        "config": {},
    }
    with patch("run.config.dpg"):
        _save_config(None, None, user_data)
        config = json.load(path.open())
        assert config == {"test": {"value": 1}}


@pytest.mark.parametrize("value", [True, 1, 1.0, "test"])
def test_clean_config_value(value):
    assert _clean_config_value([], value) == value


def test_clean_config_value_none():
    with patch("run.config.dpg") as dpg:
        dpg.get_value.side_effect = [1, 10, 1]
        value = _clean_config_value(["test"], None)
        assert dpg.get_value.call_count == 3
        assert value == [i for i in range(1, 11, 1)]


def test_initialize_dpg():
    with patch("run.config.dpg") as dpg:
        initialize_dpg("test", 100, 100)
        dpg.create_viewport.assert_called_once()
        dpg.setup_dearpygui.assert_called_once()
        dpg.show_viewport.assert_called_once()
        dpg.start_dearpygui.assert_called_once()
        dpg.destroy_context.assert_called_once()
