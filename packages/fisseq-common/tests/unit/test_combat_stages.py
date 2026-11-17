"""COMBAT_FEATURES and COMBAT_OVWT (``python -m fisseq_common.stages.combat{features,ovwt}``):
per-experiment tables in, one stacked ComBat-corrected table out."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_common.stages.combatfeatures as features_stage
import fisseq_common.stages.combatovwt as ovwt_stage

BATCHES = {"T1_R1": (0.0, 1.0), "T2_R1": (2.0, 2.0), "T3_R1": (-1.0, 0.5)}


def _write_batches(tmp_path: Path, make) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(0)
    effect = rng.normal(size=(20, 3))
    files, names = [], []
    for name, (shift, scale) in BATCHES.items():
        noise = rng.normal(scale=0.3, size=effect.shape)
        path = tmp_path / "in" / name / "table.parquet"
        path.parent.mkdir(parents=True)
        make(effect + shift + scale * noise).write_parquet(path)
        files.append(str(path))
        names.append(name)
    return files, names


def _aggregates(values: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "meta_aa_changes": [f"p.V{i}A" for i in range(len(values))],
            **{f"f{j}_median": values[:, j] for j in range(values.shape[1])},
        }
    )


def _results(values: np.ndarray) -> pl.DataFrame:
    scores = 0.5 + 0.05 * values
    return pl.DataFrame(
        {
            "meta_aa_changes": [f"p.V{i}A" for i in range(len(values))],
            "auroc_pooled": scores[:, 0],
            "auroc_median_barcode": scores[:, 1],
            "auroc_median_fold": scores[:, 2],
            "auroc_folds": [list(row) for row in scores],
            "meta_n_cells": np.arange(len(values), dtype=np.int64) + 100,
            "meta_n_barcodes": np.full(len(values), 3, dtype=np.int64),
        }
    )


def _run(stage, params, **kw) -> None:
    with patch("fisseq_common.stages.config.setup_logging"):
        stage.main.__wrapped__(OmegaConf.structured(params(**kw)))


def test_combat_features_stacks_and_corrects_every_feature(tmp_path: Path) -> None:
    files, names = _write_batches(tmp_path, _aggregates)
    _run(
        features_stage,
        features_stage.CombatFeaturesParams,
        output_dir=str(tmp_path / "out"),
        input_files=files,
        batch_names=names,
    )
    out = pl.read_parquet(tmp_path / "out" / "features.parquet")
    assert out.columns == [
        "meta_aa_changes",
        "f0_median",
        "f1_median",
        "f2_median",
        "meta_experiment",
    ]
    assert out.height == 60
    means = out.group_by("meta_experiment").agg(pl.col("f0_median").mean())
    assert np.ptp(means["f0_median"].to_numpy()) < 0.25


def test_combat_ovwt_corrects_the_scores_and_drops_the_fold_list(
    tmp_path: Path,
) -> None:
    files, names = _write_batches(tmp_path, _results)
    _run(
        ovwt_stage,
        ovwt_stage.CombatOvwtParams,
        output_dir=str(tmp_path / "out"),
        input_files=files,
        batch_names=names,
    )
    out = pl.read_parquet(tmp_path / "out" / "ovwt_scores.parquet")
    assert "auroc_folds" not in out.columns
    assert {"auroc_pooled", "meta_n_cells", "meta_experiment"} <= set(out.columns)
    raw = pl.concat(
        [
            pl.read_parquet(f).with_columns(meta_experiment=pl.lit(n))
            for f, n in zip(files, names)
        ]
    )
    joined = out.join(raw, on=["meta_aa_changes", "meta_experiment"], suffix="_raw")
    assert (joined["meta_n_cells"] == joined["meta_n_cells_raw"]).all()
    assert not np.allclose(joined["auroc_pooled"], joined["auroc_pooled_raw"])


def test_combat_ovwt_drops_variants_with_no_score(tmp_path: Path) -> None:
    """A variant OvWT skipped in one experiment (every score null) loses that row; left in
    two experiments it is still corrected."""

    def make(values: np.ndarray) -> pl.DataFrame:
        df = _results(values)
        return df.with_columns(
            pl.when(pl.col("meta_aa_changes") == "p.V0A")
            .then(None)
            .otherwise(pl.col(s))
            .alias(s)
            for s in ("auroc_pooled", "auroc_median_barcode", "auroc_median_fold")
        )

    files, names = _write_batches(tmp_path, _results)
    # the first experiment (T1_R1) skipped p.V0A
    make(np.random.default_rng(1).normal(size=(20, 3))).write_parquet(files[0])
    _run(
        ovwt_stage,
        ovwt_stage.CombatOvwtParams,
        output_dir=str(tmp_path / "out"),
        input_files=files,
        batch_names=names,
    )
    out = pl.read_parquet(tmp_path / "out" / "ovwt_scores.parquet")
    v0 = out.filter(pl.col("meta_aa_changes") == "p.V0A")
    assert sorted(v0["meta_experiment"]) == ["T2_R1", "T3_R1"]
    assert out.height == 59
    assert out["auroc_pooled"].is_finite().all()


def test_mismatched_lists_are_an_error(tmp_path: Path) -> None:
    files, names = _write_batches(tmp_path, _aggregates)
    with pytest.raises(ValueError, match="batch_names"):
        _run(
            features_stage,
            features_stage.CombatFeaturesParams,
            output_dir=str(tmp_path / "out"),
            input_files=files,
            batch_names=names[:2],
        )


def test_cli(tmp_path: Path) -> None:
    files, names = _write_batches(tmp_path, _aggregates)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_common.stages.combatfeatures",
            f"output_dir={tmp_path / 'out'}",
            f"input_files=[{','.join(files)}]",
            f"batch_names=[{','.join(names)}]",
            "mean_only=true",
            "random_seed=0",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert pl.read_parquet(tmp_path / "out" / "features.parquet").height == 60


def test_combat_features_joins_each_experiments_methods(tmp_path: Path) -> None:
    """The aggregate stage writes one file per method: an experiment's files are joined on
    the variant into one wide table before correcting."""
    median_files, names = _write_batches(tmp_path / "median", _aggregates)
    ks_files, _ = _write_batches(
        tmp_path / "ks",
        lambda values: _aggregates(values).rename(
            {f"f{j}_median": f"f{j}_KS" for j in range(3)}
        ),
    )
    _run(
        features_stage,
        features_stage.CombatFeaturesParams,
        output_dir=str(tmp_path / "out"),
        input_files=[f for pair in zip(median_files, ks_files) for f in pair],
        batch_names=[n for n in names for _ in range(2)],
    )
    out = pl.read_parquet(tmp_path / "out" / "features.parquet")
    assert sorted(c for c in out.columns if not c.startswith("meta_")) == [
        "f0_KS",
        "f0_median",
        "f1_KS",
        "f1_median",
        "f2_KS",
        "f2_median",
    ]
    assert out.height == 60


def test_combat_ovwt_takes_one_file_per_experiment(tmp_path: Path) -> None:
    files, names = _write_batches(tmp_path, _results)
    with pytest.raises(ValueError, match="One results file per experiment"):
        _run(
            ovwt_stage,
            ovwt_stage.CombatOvwtParams,
            output_dir=str(tmp_path / "out"),
            input_files=[*files, files[0]],
            batch_names=[*names, names[0]],
        )
