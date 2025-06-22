"""Optuna Dashboard launcher for visualizing hyperparameter optimization studies.

This module provides a simple interface to launch the Optuna Dashboard, which
visualizes the results of hyperparameter optimization studies stored in the
SQLite database. The dashboard provides interactive visualizations of:
- Parameter importance
- Optimization history
- Parallel coordinate plots
- Parameter relationships

The dashboard is launched on http://localhost:8080 and automatically opens
in the default web browser.

Example:
    To launch the dashboard:
    python -m run dashboard
"""

import subprocess
import time
import webbrowser


def run_dashboard(
    database: str = "optuna.db", dashboard_url: str = "http://localhost:8080"
):
    """Launch the Optuna Dashboard and open it in the default web browser.

    This function:
    1. Attempts to launch the Optuna Dashboard server
    2. Waits briefly for the server to start
    3. Opens the dashboard in the default web browser
    4. Provides helpful error messages if the dashboard fails to launch

    The dashboard connects to the SQLite database at 'optuna.db' in the
    current directory, which should contain the results of previous
    hyperparameter optimization studies.

    Args:
        database: The path to the SQLite database containing the optimization results
        dashboard_url: The URL to the dashboard

    Raises:
        FileNotFoundError: If optuna-dashboard is not installed
    """
    print(f"Launching Optuna Dashboard at {dashboard_url} for database {database}")
    try:
        subprocess.Popen(["optuna-dashboard", f"sqlite:///{database}"])
        time.sleep(1)
        webbrowser.open(dashboard_url)
        print(f"Optuna Dashboard is running at {dashboard_url}")
    except FileNotFoundError:
        raise FileNotFoundError(
            """
            Optuna Dashboard is not installed.
            Please install it with:
            pip install optuna-dashboard
            Then run the dashboard with:
            python -m run dashboard
            """
        )


if __name__ == "__main__":
    run_dashboard()
