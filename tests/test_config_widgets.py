from unittest.mock import ANY, patch

import pytest

from run.helpers.widgets import (
    _set_bool_input,
    _set_float_input,
    _set_int_input,
    _set_list_input,
    _set_str_input,
    add_config_item,
)


@pytest.mark.parametrize(
    "input, value",
    [("bool", True), ("int", 1), ("float", 1.0), ("list", [1, 2, 3]), ("str", "test")],
)
@patch("run.helpers.widgets.dpg")
def test_add_config_item(dpg, input, value):
    with patch(f"run.helpers.widgets._set_{input}_input") as func:
        tag = add_config_item("test", "test", value, "test", 0)
        func.assert_called_once()
        assert tag == func.return_value


@patch("run.helpers.widgets.dpg")
def test_add_config_item_error(dpg):
    with pytest.raises(ValueError):
        add_config_item("test", "test", {}, "test", 0)


@patch("run.helpers.widgets.dpg")
def test_set_bool_input(dpg):
    tag = _set_bool_input(True, "test")
    assert tag == "test"


@patch("run.helpers.widgets.dpg")
def test_set_bool_input_update(dpg):
    dpg.add_text.return_value = "text"
    dpg.add_checkbox.return_value = "widget"
    _set_bool_input(True, "test")

    args = dpg.set_item_callback.call_args[0]
    assert args[0] == "widget"
    args[1](None, True)
    dpg.set_value.assert_called_with("text", "enabled ")
    args[1](None, False)
    dpg.set_value.assert_called_with("text", "disabled")


@pytest.mark.parametrize(
    "key, value, parents",
    [
        ("max_new_tokens", 256, []),
        ("max_checkpoints", 10, []),
        ("steps", 1000, ["runtime"]),
        ("interval", 500, ["other", "runtime"]),
        ("batch_size", 16, []),
        ("block_size", 128, []),
        ("n_trials", 50, []),
        ("step_divisor", 5, []),
        ("steps", 1000, ["test"]),
        ("test", 99, []),
    ],
)
@patch("run.helpers.widgets.dpg")
def test_set_int_input(dpg, key, value, parents):
    tag = _set_int_input(key, value, "test", parents, 0)
    assert tag == "test"


@pytest.mark.parametrize(
    "key, value, parents",
    [
        ("temperature", 1.0, []),
        ("weight", 0.5, ["visualization"]),
        ("test", 99, []),
    ],
)
@patch("run.helpers.widgets.dpg")
def test_set_float_input(dpg, key, value, parents):
    tag = _set_float_input(key, value, "test", parents, 0)
    assert tag == "test"


@patch("run.helpers.widgets.dpg")
def test_set_list_input(dpg):
    tag = _set_list_input([1, 2, 3], "test", 0)
    assert tag == "test"


@pytest.mark.parametrize(
    "key, value, parents",
    [
        ("type", "test", ["tuning_ranges", "hparam"]),
        ("source", "local", []),
        ("data_name", "test", ["library"]),
        ("token_level", "char", []),
        ("pruner", "median", []),
        ("test", "test", []),
    ],
)
@patch("run.helpers.widgets.dpg")
def test_set_str_input(dpg, key, value, parents):
    tag = _set_str_input(key, value, "test", parents, 0)
    assert tag == "test"
