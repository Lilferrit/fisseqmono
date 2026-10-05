"""Run each pipeline's integration fixture end to end, for the root reference tests.

The synthetic inputs and the ``nextflow run`` invocation come from each package's own
``tests/integration/test_integration.py`` (loaded by path, so the fixtures here stay the ones
those suites run). Every scenario is deterministic: fixed numpy seeds for the data, a seeded
torch init for the embeddings pipeline's tiny Cell-DINO checkpoint, and each pipeline's
``random_seed`` default.

Scenarios:

- ``data``: the data pipeline's two-experiment session fixture (``batch1``, ``batch2``).
- ``embeddings``: the embeddings pipeline's one-experiment session fixture (``batch1``).
- ``embeddings_global``: synthetic per-experiment outputs (``_global_fixture``) and what the
  embeddings pipeline's old GLOBAL_* stages made of them.
- ``embeddings_two``: the same fixture written twice with different tile images, as
  ``batch1`` and ``batch2`` in one run, so the cross-experiment vote and median see more
  than one experiment. Its variants include two synonymous ones
  (``_MULTI_EXPERIMENT_VARIANTS``), so the per-experiment z-scores are defined.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import yaml

ROOT = Path(__file__).parents[1]
PACKAGES = ROOT / "packages"
REFERENCE_DIR = ROOT / "tests" / "reference"

SCENARIOS = ("data", "embeddings", "embeddings_two", "embeddings_global")

# Stage output directories that hold published results (the rest of a fixture's pipeline_dir is
# synthetic input: starcall trees, raw batches, stub binaries).
_DATA_OUTPUT_DIRS = (
    "input",
    "qc_filter",
    "normalization",
    "ovwt_batchwise",
    "feature_select_batchwise",
)
_EMBEDDINGS_OUTPUT_DIRS = (
    "cell_images",
    "cell_metadata",
    "qc_filter",
    "embeddings",
    "filter_embeddings",
    "feature_select_batchwise",
    "ovwt_batchwise",
    "cp_features",
    "filter_cp_features",
    "feature_select_batchwise_cp_features",
    "ovwt_batchwise_cp_features",
    "global",
)


def _load(package: str) -> ModuleType:
    path = PACKAGES / package / "tests" / "integration" / "test_integration.py"
    name = f"_integration_{package.replace('-', '_')}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def output_dirs(scenario: str) -> tuple[str, ...]:
    return _DATA_OUTPUT_DIRS if scenario == "data" else _EMBEDDINGS_OUTPUT_DIRS


def run_scenario(scenario: str, work_dir: Path) -> Path:
    """Run ``scenario`` from scratch under ``work_dir``; return its ``pipeline_dir``."""
    if shutil.which("nextflow") is None:
        raise RuntimeError("nextflow not on PATH")
    work_dir.mkdir(parents=True, exist_ok=True)
    if scenario == "data":
        return _run_data(work_dir)
    if scenario == "embeddings":
        return _run_embeddings(work_dir, n_experiments=1)
    if scenario == "embeddings_two":
        return _run_embeddings(work_dir, n_experiments=2)
    if scenario == "embeddings_global":
        return _run_global(work_dir)
    raise ValueError(f"unknown scenario {scenario!r}")


def _run_global(work_dir: Path) -> Path:
    import _global_fixture

    pipeline_dir = work_dir / "pipeline"
    _global_fixture.write_inputs(pipeline_dir)
    _global_fixture.run_old_global_stages(pipeline_dir)
    return pipeline_dir


def _run_data(work_dir: Path) -> Path:
    it = _load("fisseq-data-pipeline")
    exp_dir = work_dir / "pipeline"
    raw_dir = work_dir / "raw"
    experiments = [
        it._stage_experiment(raw_dir, "batch1", seed=42),
        it._stage_experiment(raw_dir, "batch2", seed=99),
    ]
    result = it._run_pipeline(exp_dir, experiments)
    _check(result)
    return exp_dir


def _run_embeddings(work_dir: Path, n_experiments: int) -> Path:
    import numpy as np
    import torch

    it = _load("fisseq-embeddings-pipeline")
    exp_dir = work_dir / "pipeline"
    exp_dir.mkdir(parents=True, exist_ok=True)
    original = it._make_tile_image, it._VARIANTS
    try:
        if n_experiments > 1:
            it._VARIANTS = _MULTI_EXPERIMENT_VARIANTS
        it._write_synthetic_experiment(exp_dir)
        params_path = exp_dir / "params.yaml"
        params = yaml.safe_load(params_path.read_text())
        for i in range(2, n_experiments + 1):
            # Another starcall tree with a different (still seeded) phenotype image.
            it._make_tile_image = _seeded_tile_image(it, seed=i)
            other = work_dir / f"experiment{i}"
            other.mkdir(parents=True, exist_ok=True)
            it._write_synthetic_experiment(other)
            entries = yaml.safe_load((other / "params.yaml").read_text())["experiments"]
            params["experiments"].append({**entries[0], "batch_stem": f"batch{i}"})
    finally:
        it._make_tile_image, it._VARIANTS = original
    params_path.write_text(yaml.safe_dump(params))

    checkpoint = work_dir / "weights" / "checkpoint.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    it._write_tiny_checkpoint(checkpoint)

    result = it._run_nextflow(exp_dir, checkpoint)
    _check(result)
    return exp_dir


# The multi-experiment scenario needs at least two synonymous variants per experiment: the
# cross-experiment OvWT step z-scores each experiment's scores against its synonymous variants,
# and with one the standard deviation is undefined. Three cells per barcode (fewer leaves OvWT
# without scorable variants) and 24 cells in all, within the fixture tile's 25 cell positions.
_MULTI_EXPERIMENT_VARIANTS = {
    "WT": ("bc_wt_{i}", 2, 3),
    "A1A": ("bc_syn_{i}", 2, 3),
    "A2A": ("bc_syn2_{i}", 2, 3),
    "M1K": ("bc_mis_{i}", 2, 3),
}


def _seeded_tile_image(it: ModuleType, seed: int):
    import numpy as np

    def make(channels: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(
            0, 255, size=(1, channels, it._TILE_SIZE, it._TILE_SIZE), dtype=np.uint16
        )

    return make


def _check(result) -> None:
    if result.returncode != 0:
        raise RuntimeError(
            f"nextflow run failed ({result.returncode})\n--- stdout\n{result.stdout[-5000:]}"
            f"\n--- stderr\n{result.stderr[-5000:]}"
        )


def published_parquets(pipeline_dir: Path, scenario: str) -> dict[str, Path]:
    """Every published parquet of a run, keyed by its path relative to ``pipeline_dir``."""
    found: dict[str, Path] = {}
    for top in output_dirs(scenario):
        base = pipeline_dir / top
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.parquet")):
            found[path.relative_to(pipeline_dir).as_posix()] = path
    return found
