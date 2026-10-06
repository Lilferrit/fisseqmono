"""Shared fixtures of the root (cross-package) tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from _fixtures import run_scenario


@pytest.fixture(scope="session")
def scenario_runs(tmp_path_factory):
    """``run(scenario) -> (run_root, pipeline_dir)``: each scenario runs once per session."""
    cache: dict[str, tuple[Path, Path]] = {}

    def run(scenario: str) -> tuple[Path, Path]:
        if scenario not in cache:
            run_root = tmp_path_factory.mktemp(scenario)
            cache[scenario] = (run_root, run_scenario(scenario, run_root))
        return cache[scenario]

    return run
