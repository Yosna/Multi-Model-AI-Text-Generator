"""A module for creating and managing the configuration window.

Includes:
- Command-line argument parsing for configuration
- Building and managing the configuration window
- Saving and loading configuration files

Example:
    To run the configuration window:
    python -m run config
"""

from typing import Any, Callable

import dearpygui.dearpygui as dpg

from run.helpers.widgets import add_config_item
from utils.io_utils import load_config, save_config


def run_config(path: str = "config.json", width: int = 540, height: int = 800) -> None:
    """Run the configuration window.

    Args:
        path (str): The path to the config file.
        width (int): The width of the window.
        height (int): The height of the window.
    """
    config = load_config(path)
    build_window(label=path, width=width, height=height, config=config)
    initialize_dpg(title="Config Editor", width=width, height=height)


def build_window(label: str, width: int, height: int, config: dict[str, Any]) -> None:
    """Build the config window.

    Args:
        label (str): The label for the window.
        width (int): The width of the window.
        height (int): The height of the window.
        config (dict[str, Any]): The configuration dictionary.
    """
    width -= 16
    height -= 40

    dpg.create_context()

    with dpg.window(label=label, width=width, height=height):
        padding = 40
        width = int(width * 0.97)
        button_height = 50
        config_height = height - button_height - padding

        headers, widgets = _build_config_window(config, width, config_height)
        _build_button_window(label, headers, widgets, width, button_height)


def _build_config_window(
    config: dict[str, Any], width: int, height: int
) -> tuple[list[str], dict[str, str]]:
    """Build the config window.

    Args:
        config (dict[str, Any]): The configuration dictionary.
        width (int): The width of the window.
        height (int): The height of the window.

    Returns:
        tuple[list[str], dict[str, str]]: A tuple containing the headers and widgets.
    """
    with dpg.child_window(width=width, height=height):
        headers = []
        widgets = {}

        def build_config_section(
            config: dict[str, Any], parents: list[str] = [], indent: int = 0
        ) -> None:
            """Build a config section.

            Args:
                config (dict[str, Any]): The config dictionary.
                parents (list[str]): The parents of the config.
                indent (int): The indent of the config.
            """
            for key, value in config.items():
                parents = [] if indent == 0 else parents
                spacer = "  " * indent
                label = f"{spacer}{key}:"

                if isinstance(value, dict):
                    with dpg.collapsing_header(
                        label=label, default_open=False, indent=indent
                    ) as header:
                        headers.append(header)
                        with dpg.group(indent=indent):
                            build_config_section(value, parents + [key], indent + 1)
                else:
                    tag = add_config_item(
                        label, key, value, ".".join(parents + [key]), width, parents
                    )
                    widgets[tag] = tag

        build_config_section(config)

        return headers, widgets


def _build_button_window(
    label: str, headers: list[str], widgets: dict[str, str], width: int, height: int
) -> None:
    """Build the button window.

    Args:
        label (str): The label of the window.
        headers (list[str]): The headers of the window.
        widgets (dict[str, str]): The widgets of the window.
        width (int): The width of the window.
        height (int): The height of the window.
    """
    with dpg.child_window(width=width, height=height):
        with dpg.table(header_row=False, policy=dpg.mvTable_SizingFixedFit):
            for i in range(4):
                dpg.add_table_column()

            with dpg.table_row():
                width = int(width * 0.2325)
                user_data = {
                    **{"widgets": widgets, "headers": headers},
                    **{"path": label, "config": {}},
                }

                callbacks = {
                    "Expand All": _expand_config,
                    "Collapse All": _collapse_config,
                    "Save Config": _save_config,
                    "Close": dpg.stop_dearpygui,
                }

                for button in callbacks:
                    add_config_button(button, width, callbacks[button], user_data)


def add_config_button(
    label: str, width: int, callback: Callable, user_data: dict[str, Any] | None = None
) -> None:
    """Add a save button to the window.

    Args:
        label (str): The label of the button.
        width (int): The width of the button.
        callback (Callable): The callback function.
        user_data (dict[str, Any]): The user data.
    """
    dpg.add_button(
        label=label, width=width, height=-1, callback=callback, user_data=user_data
    )


def _expand_config(_, __, user_data: dict[str, Any]) -> None:
    """Expand the config.

    Args:
        user_data (dict[str, Any]): The user data.
    """
    for tag in user_data["headers"]:
        dpg.set_value(tag, True)


def _collapse_config(_, __, user_data: dict[str, Any]) -> None:
    """Collapse the config.

    Args:
        user_data (dict[str, Any]): The user data.
    """
    for tag in user_data["headers"]:
        dpg.set_value(tag, False)


def _save_config(_, __, user_data: dict[str, Any]) -> None:
    """Save the config.

    Args:
        user_data (dict[str, Any]): The user data.
    """
    for tag in user_data["widgets"].values():
        value = dpg.get_value(tag)
        keys = tag.split(".")
        rebuilt_config = user_data["config"]

        for key in keys[:-1]:
            if key not in rebuilt_config:
                rebuilt_config[key] = {}
            rebuilt_config = rebuilt_config[key]

        rebuilt_config[keys[-1]] = _clean_config_value(keys, value)

    save_config(user_data["config"], user_data["path"])


def _clean_config_value(keys: list[str], value: Any) -> Any:
    """Clean the config value.

    Args:
        keys (list[str]): The keys of the value.
        value (Any): The value to clean.

    Returns:
        Any: The cleaned value.
    """
    if value is None:
        base_tag = ".".join(keys)
        min = dpg.get_value(f"{base_tag}.min")
        max = dpg.get_value(f"{base_tag}.max")
        step = dpg.get_value(f"{base_tag}.step")
        value = [i for i in range(min, max + 1, step)]
    elif isinstance(value, bool):
        value = value
    elif isinstance(value, float):
        value = round(value, 6)
    elif isinstance(value, int):
        value = value
    elif value.isdigit():
        value = int(value)
    return value


def initialize_dpg(title: str, width: int, height: int) -> None:
    """Initialize DearPyGui.

    Args:
        title (str): The title of the window.
        width (int): The width of the window.
        height (int): The height of the window.
    """
    dpg.create_viewport(title=title, width=width, height=height)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    run_config()
