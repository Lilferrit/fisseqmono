"""Each pipeline's integration fixture still produces its saved reference outputs.

Rule: the embeddings pipeline's per-experiment outputs never change in a refactor. The data
pipeline's change only in a commit that swaps a stage to the shared implementation, together
with a diff report under docs/diff-reports/ and regenerated references
(``uv run python tests/reference/capture.py data``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _compare import compare_trees
from _fixtures import REFERENCE_DIR, SCENARIOS, published_parquets, run_scenario


@pytest.fixture(scope="module", params=SCENARIOS)
def scenario_run(request, tmp_path_factory) -> tuple[str, Path, Path]:
    scenario = request.param
    if not (REFERENCE_DIR / scenario).exists():
        pytest.fail(
            f"no reference outputs for {scenario}; run tests/reference/capture.py"
        )
    run_root = tmp_path_factory.mktemp(scenario)
    return scenario, run_root, run_scenario(scenario, run_root)


def test_outputs_match_reference(scenario_run) -> None:
    scenario, run_root, pipeline_dir = scenario_run
    problems = compare_trees(
        published_parquets(pipeline_dir, scenario), REFERENCE_DIR / scenario, run_root
    )
    assert not problems, "\n".join(problems)
