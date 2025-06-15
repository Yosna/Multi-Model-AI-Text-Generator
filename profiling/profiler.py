"""Profiler for running and saving performance profiles of the main application."""

import cProfile
import pstats
import argparse
import datetime
import os
import io
from main import main


def add_args_to_parser() -> argparse.Namespace:
    """Create and parse command-line arguments for profiling the main application.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="transformer")
    parser.add_argument("--token-level", type=str, default="word")
    parser.add_argument("--training", type=bool, default=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-model", type=bool, default=False)
    parser.add_argument("--save-tuning", type=bool, default=False)
    parser.add_argument("--save-study", type=bool, default=False)
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--save-plot", type=bool, default=False)
    parser.add_argument("--show-plot", type=bool, default=False)

    return parser.parse_args()


def run_profiler(args: argparse.Namespace | None = None) -> None:
    """Run the profiler on the main application with the given arguments.

    Args:
        args (argparse.Namespace | None): Parsed command-line arguments for main().
    """
    profiler = cProfile.Profile()
    profiler.enable()
    main(args or add_args_to_parser())
    profiler.disable()
    profile = generate_profile(profiler)
    save_profile(profile)


def generate_profile(profiler: cProfile.Profile) -> str:
    """Generate a filtered and formatted profile report from a cProfile.Profile object.

    Args:
        profiler (cProfile.Profile): The profiler object containing performance data.

    Returns:
        str: The formatted and filtered profile report as a string.
    """
    sorting = ["calls", "time", "cumulative"]
    exclude = ["venv", "Python311", "bootstrap"]
    profile = ""
    line_limit = 25

    for sort in sorting:
        # Capture the output of print_stats for each sorting method
        output = io.StringIO()
        stats = pstats.Stats(profiler, stream=output)
        stats.sort_stats(sort)
        stats.print_stats()
        output.seek(0)
        lines = output.readlines()

        # Filter out lines with excluded substrings
        filtered = [line for line in lines if not any(ex in line for ex in exclude)]
        profile += f"=== Top {line_limit} by {sort.upper()} ===\n"
        profile += "".join(filtered[:line_limit]) + "\n"

    return profile


def save_profile(profile: str, directory: str = "profiling/profiles") -> None:
    """Save the profile report to a timestamped file in the specified directory.

    Args:
        profile (str): The profile report to save.
        directory (str): Directory to save the profile file.
    """
    os.makedirs(directory, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"profile_{timestamp}.txt"
    file_path = os.path.join(directory, file_name)

    with open(file_path, "w") as f:
        f.write(profile)


if __name__ == "__main__":
    run_profiler()
