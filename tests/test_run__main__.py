import sys
from unittest.mock import patch

import pytest

from run.__main__ import get_tool, run_tool


@pytest.mark.parametrize("tool", ["dashboard", "profiler"])
def test_get_tool(tool):
    with patch.object(sys, "argv", ["run.py", tool]):
        assert get_tool() == tool


@pytest.mark.parametrize("tool", ["dashboard", "profiler"])
def test_run_tool(tool):
    sys.argv = ["run...", tool]
    with patch(f"run.__main__.{tool}.run_{tool}") as module:
        sys.argv = ["run...", tool]
        run_tool(tool)
        module.assert_called_once()


def test_run_tool_error():
    sys.argv = ["run...", "invalid"]
    with pytest.raises(ValueError, match="Invalid tool: invalid"):
        run_tool("invalid")
