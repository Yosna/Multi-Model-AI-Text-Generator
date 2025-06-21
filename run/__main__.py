"""Entry point for running tools in the run package.

This module allows you to run available tools using `python -m run <tool>`.

Tools:
    - config: Configuration editor
    - dashboard: Optuna dashboard
    - profiler: cProfile profiler

Examples:
    python -m run config
    python -m run dashboard
    python -m run profiler
"""

import argparse
import sys

from . import config, dashboard, profiler


def get_tool() -> str:
    """Parse and return the tool name from command-line arguments.

    Returns:
        str: The name of the tool to run.
    """
    tools = ["config", "dashboard", "profiler"]
    parser = argparse.ArgumentParser()
    parser.add_argument("tool", type=str, choices=tools)
    args = parser.parse_args()
    return args.tool


def run_tool(tool: str) -> None:
    """Run the selected tool based on the provided tool name.

    Args:
        tool (str): The name of the tool to run.

    Raises:
        ValueError: If an invalid tool name is provided.
    """
    sys.argv = [sys.argv[0][:-3]]
    if tool == "config":
        config.run_config()
    elif tool == "dashboard":
        dashboard.run_dashboard()
    elif tool == "profiler":
        profiler.run_profiler()
    else:
        raise ValueError(f"Invalid tool: {tool}")


if __name__ == "__main__":
    run_tool(get_tool())
