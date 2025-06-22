"""A module for creating and managing configuration widgets.

Includes:
- add_config_item: Add a configuration item to the window.
- build_int_list: Build a list of integers for an integer input widget.
- _set_bool_input: Set a boolean input widget.
- _set_int_input: Set an integer input widget.
- _set_float_input: Set a float input widget.
- _set_list_input: Set a list input widget.
"""

from typing import Any

import dearpygui.dearpygui as dpg


def add_config_item(
    label: str, key: str, value: Any, tag: str, width: int, parents: list[str] = []
) -> str:
    """Add a configuration item to the window.

    Args:
        label (str): The label for the config item.
        key (str): The key for the config item.
        value (Any): The value of the config item.
        tag (str): The tag for the config item.
        width (int): The width of the config item.
        parents (list[str], optional): The parents of the config item. Defaults to [].

    Raises:
        ValueError: If the value type is not supported.

    Returns:
        str: The tag of the config item.
    """
    with dpg.table(header_row=False, policy=dpg.mvTable_SizingFixedFit):
        label_width = int(width * 0.4125)
        input_width = int(width * 0.5025)
        dpg.add_table_column(init_width_or_weight=label_width)
        dpg.add_table_column(init_width_or_weight=input_width)

        with dpg.table_row():
            with dpg.table_cell():
                dpg.add_text(label)

            with dpg.table_cell():
                if isinstance(value, bool):
                    tag = _set_bool_input(value, tag)
                elif isinstance(value, int):
                    tag = _set_int_input(key, value, tag, parents, input_width)
                elif isinstance(value, float):
                    tag = _set_float_input(key, value, tag, parents, input_width)
                elif isinstance(value, list):
                    tag = _set_list_input(value, tag, input_width)
                elif isinstance(value, str):
                    tag = _set_str_input(key, value, tag, parents, input_width)
                else:
                    raise ValueError(f"Invalid value type: {type(value)}")
            return tag


def _set_bool_input(value: bool, tag: str) -> str:
    """Set a boolean input widget.

    Args:
        value (bool): The default value.
        tag (str): The tag for the widget.

    Returns:
        str: The tag of the widget.
    """
    with dpg.group(horizontal=True):
        text_id = dpg.add_text("enabled " if value else "disabled")
        widget_id = dpg.add_checkbox(default_value=value, tag=tag)

        def update_text(_, app_data):
            dpg.set_value(text_id, "enabled " if app_data else "disabled")

        dpg.set_item_callback(widget_id, update_text)
        return tag


def _set_int_input(
    key: str, value: int, tag: str, parents: list[str], width: int
) -> str:
    """Set an integer input widget.

    Args:
        key (str): The key for the widget.
        value (int): The default value.
        tag (str): The tag for the widget.
        parents (list[str]): The parents of the widget.
        width (int): The width of the widget.

    Returns:
        str: The tag of the widget.
    """

    def when(key: str, parent: int | None = None) -> bool:
        return False if not parents else parents[parent or -1] == key

    options = {
        "max_new_tokens": {"data": ["pow", [11]], "when": True},
        "max_checkpoints": {"data": ["mult", [1, 10, 100]], "when": True},
        "steps": {"data": ["mult", [1000, 10000]], "when": when("runtime")},
        "interval": {"data": ["mult", [100, 1000]], "when": when("runtime")},
        "patience": {"data": ["mult", [1, 10]], "when": when("runtime")},
        "batch_size": {"data": ["pow", [1, 9]], "when": True},
        "block_size": {"data": ["pow", [2, 10]], "when": True},
        "n_trials": {"data": ["mult", [1, 10, 100]], "when": True},
        "step_divisor": {"data": ["mult", [1, 10]], "when": True},
    }

    if key in options and options[key]["when"]:
        kind, values = options[key]["data"]
        kind = "powers_of_two" if kind == "pow" else "multiples_of_ten"
        items = build_int_list(kind, values)
        dpg.add_combo(items=items, default_value=str(value), width=width, tag=tag)
    else:
        dpg.add_input_int(default_value=value, width=width, tag=tag)
    return tag


def build_int_list(kind: str, values: list[int]) -> list[str]:
    """Build a list of integers for an integer input widget.

    Args:
        kind (str): The kind of the widget. "powers_of_two" or "multiples_of_ten".
        values (list[int]): The range or multiples for the items.

    Returns:
        list[str]: The items for the widget.

    Raises:
        ValueError: If the kind is not valid.
    """
    if kind == "powers_of_two":
        items = [str(2**i) for i in range(*values)]
    elif kind == "multiples_of_ten":
        items = []
        for multiple in values:
            items += [str(i * multiple) for i in range(1, 11)]
    else:
        raise ValueError(
            f"""Invalid kind: {kind}
            Valid kinds: powers_of_two, multiples_of_ten"""
        )
    return list(dict.fromkeys(items))


def _set_float_input(
    key: str, value: float, tag: str, parents: list[str], width: int
) -> str:
    """Set a float input widget.

    Args:
        key (str): The key for the widget.
        value (float): The default value.
        tag (str): The tag for the widget.
        parents (list[str]): The parents of the widget.
        width (int): The width of the widget.

    Returns:
        str: The tag of the widget.
    """
    add_slider_widget = key == "temperature" or (
        key == "weight" and parents[-1] == "visualization"
    )
    if add_slider_widget:
        dpg.add_slider_float(
            default_value=value,
            min_value=0,
            max_value=2 if key == "temperature" else 1,
            width=width,
            format="%.2f",
            tag=tag,
        )
    else:
        dpg.add_input_float(
            default_value=value, width=width, format=f"%.{len(str(value))-2}f", tag=tag
        )
    return tag


def _set_list_input(value: list, tag: str, width: int) -> str:
    """Set a list input widget.

    Args:
        value (list): The default value.
        tag (str): The tag for the widget.
        width (int): The width of the widget.

    Returns:
        str: The tag of the widget.
    """
    with dpg.group(horizontal=True):
        with dpg.table(header_row=False):
            for _ in range(3):
                dpg.add_table_column()

            with dpg.table_row():
                [dpg.add_text(text) for text in [" min:", " max:", " step:"]]

            with dpg.table_row():
                width = int(width * 0.3125)
                keys = {"min": value[0], "max": value[-1], "step": value[1] - value[0]}
                for key in keys:
                    dpg.add_input_int(
                        default_value=keys[key], width=width, tag=f"{tag}.{key}"
                    )
        return tag


def _set_str_input(
    key: str, value: str, tag: str, parents: list[str], width: int
) -> str:
    """Set a string input widget.

    Args:
        key (str): The key for the widget.
        value (str): The default value.
        tag (str): The tag for the widget.
        parents (list[str]): The parents of the widget.
        width (int): The width of the widget.

    Returns:
        str: The tag of the widget.
    """

    def when(key: str, parent: int | None = None) -> bool:
        return False if not parents else parents[parent or -1] == key

    sources = ["local", "library", "huggingface"]
    libraries = ["news", "squad", "science", "movies", "yelp"]
    libraries.extend(["tiny_stories", "stackoverflow", "wikipedia"])
    options = {
        "source": {"data": sources, "when": when("datasets")},
        "data_name": {"data": libraries, "when": when("library")},
        "token_level": {"data": ["char", "word"], "when": True},
        "pruner": {"data": ["median", "halving", "hyperband"], "when": True},
    }

    if key == "type" and when("tuning_ranges", -2):
        dpg.add_text(value, tag=tag)
    elif key in options and options[key]["when"]:
        items = options[key]["data"]
        dpg.add_combo(items=items, default_value=value, width=width, tag=tag)
    else:
        dpg.add_input_text(default_value=value, width=width, tag=tag)
    return tag
