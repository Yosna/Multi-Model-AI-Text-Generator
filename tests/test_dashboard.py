from unittest.mock import patch

import pytest

from run.dashboard import run_dashboard


def test_run_dashboard():
    with patch("subprocess.Popen") as popen, patch("webbrowser.open") as open:
        run_dashboard()
    popen.assert_called_once()
    open.assert_called_once()


def test_run_dashboard_file_not_found():
    with patch("subprocess.Popen", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            run_dashboard()
