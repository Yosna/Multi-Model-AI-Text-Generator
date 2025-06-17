import cProfile
import os
import sys
from unittest.mock import patch

from run.profiler import (
    add_args_to_parser,
    generate_profile,
    run_profiler,
    save_profile,
)


def test_add_args_to_parser():
    with patch.object(sys, "argv", ["profiler.py"]):
        expected_types = {
            **{"model": str, "token_level": str, "training": bool},
            **{"steps": int, "save_model": bool, "save_tuning": bool},
            **{"save_study": bool, "n_trials": int},
            **{"save_plot": bool, "show_plot": bool},
        }
        args = add_args_to_parser()
        for arg, type in expected_types.items():
            assert isinstance(getattr(args, arg), type)


def test_run_profiler():
    with patch.object(sys, "argv", ["profiler.py"]), patch(
        "run.profiler.main"
    ) as main, patch("run.profiler.generate_profile") as generate_profile, patch(
        "run.profiler.save_profile"
    ) as save_profile:
        run_profiler()
        main.assert_called_once()
        generate_profile.assert_called_once()
        save_profile.assert_called_once()


def test_generate_profile():
    profiler = cProfile.Profile()
    profiler.enable()
    profiler.disable()
    profile = generate_profile(profiler)
    assert "call count" in profile
    assert "internal time" in profile
    assert "cumulative time" in profile


def test_save_profile(tmp_path):
    profile = "test"
    save_profile(profile, tmp_path)
    profiles = list(tmp_path.iterdir())
    with open(profiles[0], "r", encoding="utf-8") as f:
        saved_profile = f.read()
    assert os.path.exists(profiles[0])
    assert len(profiles) == 1
    assert saved_profile == profile
