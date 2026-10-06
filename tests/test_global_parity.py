"""fisseqborn's cross-experiment aggregation reproduces the embeddings pipeline's former
global stages (GLOBAL_BLOCKLIST, GLOBAL_VARIANT_EMBEDDINGS, GLOBAL_VARIANT_DISTINGUISHABILITY,
GLOBAL_VARIANT_CP_FEATURES, GLOBAL_VARIANT_DISTINGUISHABILITY_CP_FEATURES).

Their outputs were saved under ``tests/reference/<scenario>/global/`` before the stages were
deleted. Each case runs `fisseqborn.write_global` on the same per-experiment outputs and
compares, file by file: ``<out>/<dir>/<file>`` against ``global/<dir>/<file>``.

- ``embeddings`` and ``embeddings_two``: the embeddings pipeline's integration fixture (one and
  two experiments), run fresh, with the pipeline's defaults (``min_batches_ok`` null,
  ``cumulative_variance_explained`` 0.9, ``random_seed`` 0).
- ``embeddings_global``: richer synthetic per-experiment outputs (``_global_fixture``; three
  experiments, disagreeing votes, a missing feature), saved with the reference, and run with
  both ``min_batches_ok`` settings the old stages were run with.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import _global_fixture
import fisseqborn as fb
from _compare import compare_trees
from _fixtures import REFERENCE_DIR


def _compare(
    written: dict[str, Path], reference: Path, mapping: dict[str, str]
) -> list[str]:
    """Compare ``written`` (keyed ``<dir>/<stem>``) with ``reference/global/<ref dir>/``,
    for each ``out dir -> reference dir`` of ``mapping``."""
    problems = []
    for out_dir, ref_dir in mapping.items():
        actual = {
            f"global/{ref_dir}/{Path(path).name}": path
            for key, path in written.items()
            if key.split("/")[0] == out_dir
        }
        assert actual, f"write_global wrote nothing to {out_dir}/"
        problems += compare_trees(
            actual, reference, reference, prefix=f"global/{ref_dir}/"
        )
    return problems


SAME_DIRS = {
    d: d
    for d in (
        "embeddings",
        "distinguishability",
        "cp_features",
        "distinguishability_cp_features",
    )
}


@pytest.mark.parametrize("scenario", ["embeddings", "embeddings_two"])
def test_pipeline_fixture_global_outputs(scenario, scenario_runs, tmp_path):
    _run_root, pipeline_dir = scenario_runs(scenario)
    written = fb.write_global(pipeline_dir, tmp_path / "global")
    problems = _compare(written, REFERENCE_DIR / scenario, SAME_DIRS)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "name, min_batches", list(_global_fixture.MIN_BATCHES_OK.items())
)
def test_synthetic_global_outputs(name, min_batches, tmp_path):
    reference = REFERENCE_DIR / "embeddings_global"
    written = fb.write_global(
        reference, tmp_path / "global", layout="embeddings", min_batches=min_batches
    )
    mapping = {"embeddings": name}
    if min_batches is None:
        mapping |= {
            "distinguishability": "distinguishability",
            "cp_features": "cp_features",
            "distinguishability_cp_features": "distinguishability_cp_features",
        }
    problems = _compare(written, reference, mapping)
    assert not problems, "\n".join(problems)
