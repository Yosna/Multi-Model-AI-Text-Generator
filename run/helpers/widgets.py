"""A module for creating and managing configuration widgets."""

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
    if key == "max_new_tokens":
        items = [str(2**i) for i in range(11)]
        dpg.add_combo(items=items, default_value=str(value), width=width, tag=tag)
    elif key == "max_checkpoints":
        items = [str(i) for i in range(1, 11)]
        items += [str(i * 10) for i in range(2, 11)]
        items += [str(i * 100) for i in range(2, 11)]
        dpg.add_combo(items=items, default_value=str(value), width=width, tag=tag)
    elif key == "steps" and parents[-1] == "runtime":
        items = [str(i * 1000) for i in range(1, 11)]
        items += [str(i * 10000) for i in range(2, 11)]
        dpg.add_combo(items=items, default_value=str(value), width=width, tag=tag)
    elif key == "interval" and parents[-1] == "runtime":
        items = [str(i * 100) for i in range(1, 11)]
        items += [str(i * 1000) for i in range(2, 11)]
        dpg.add_combo(items=items, default_value=str(value), width=width, tag=tag)
    elif key == "batch_size":
        items = [str(2**i) for i in range(1, 9)]
        dpg.add_combo(items=items, default_value=str(value), width=width, tag=tag)
    elif key == "block_size":
        items = [str(2**i) for i in range(2, 10)]
        dpg.add_combo(items=items, default_value=str(value), width=width, tag=tag)
    elif key == "n_trials":
        items = [str(i) for i in range(1, 11)]
        items += [str(i * 10) for i in range(2, 11)]
        items += [str(i * 100) for i in range(2, 11)]
        dpg.add_combo(items=items, default_value=str(value), width=width, tag=tag)
    elif key == "step_divisor":
        items = [str(i) for i in range(1, 11)]
        items += [str(i * 10) for i in range(2, 11)]
        dpg.add_combo(items=items, default_value=str(value), width=width, tag=tag)
    else:
        dpg.add_input_int(default_value=value, width=width, tag=tag)
    return tag


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
    if key == "temperature":
        dpg.add_slider_float(
            default_value=value,
            min_value=0,
            max_value=2,
            width=width,
            format="%.2f",
            tag=tag,
        )
    elif key == "weight" and parents[-1] == "visualization":
        dpg.add_slider_float(
            default_value=value,
            min_value=0,
            max_value=1,
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
                with dpg.group(horizontal=True):
                    dpg.add_text(" min:")
                with dpg.group(horizontal=True):
                    dpg.add_text(" max:")
                with dpg.group(horizontal=True):
                    dpg.add_text(" step:")

            with dpg.table_row():
                width = int(width * 0.3125)
                keys = {"min": value[0], "max": value[-1], "step": value[1] - value[0]}
                for key in keys:
                    with dpg.group(horizontal=True):
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
    if key == "type" and parents[-2] == "tuning_ranges":
        dpg.add_text(value, tag=tag)
    elif key == "source":
        items = ["local", "library", "huggingface"]
        dpg.add_combo(items=items, default_value=value, width=width, tag=tag)
    elif key == "data_name" and parents[-1] == "library":
        items = [
            *["news", "squad", "science", "movies", "yelp"],
            *["tiny_stories", "stackoverflow", "wikipedia"],
        ]
        dpg.add_combo(items=items, default_value=value, width=width, tag=tag)
    elif key == "token_level":
        items = ["char", "word"]
        dpg.add_combo(items=items, default_value=value, width=width, tag=tag)
    elif key == "pruner":
        items = ["median", "halving", "hyperband"]
        dpg.add_combo(items=items, default_value=value, width=width, tag=tag)
    else:
        dpg.add_input_text(default_value=value, width=width, tag=tag)
    return tag
