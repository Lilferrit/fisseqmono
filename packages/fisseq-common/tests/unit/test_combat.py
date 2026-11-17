"""fisseq_common.combat: ComBat (neuroCombat) with the variant as a covariate, and the null /
single-experiment handling around it."""

import logging

import numpy as np
import polars as pl
import pytest

from fisseq_common.combat import (
    combat,
    combat_frame,
    drop_single_batch_variants,
    stack_batches,
)

BATCHES = ("A", "B", "C")
SHIFT = {"A": 0.0, "B": 3.0, "C": -2.0}
SCALE = {"A": 1.0, "B": 2.5, "C": 0.5}


def _table(
    n_variants: int = 40,
    n_features: int = 4,
    seed: int = 0,
    batches: dict[str, tuple[float, float]] | None = None,
) -> pl.DataFrame:
    """Every variant in every batch, as ComBat models it: a per-variant effect shared by every
    batch, plus each batch's shift and its scale of the noise."""
    rng = np.random.default_rng(seed)
    effect = rng.normal(scale=2.0, size=(n_variants, n_features))
    if batches is None:
        batches = {b: (SHIFT[b], SCALE[b]) for b in BATCHES}
    rows = []
    for b, (shift, scale) in batches.items():
        noise = rng.normal(scale=0.5, size=(n_variants, n_features))
        values = effect + shift + scale * noise
        for v in range(n_variants):
            rows.append(
                {
                    "meta_aa_changes": f"V{v:03d}",
                    "meta_experiment": b,
                    **{f"f{j}_median": values[v, j] for j in range(n_features)},
                }
            )
    return pl.DataFrame(rows)


def _features(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("meta_")]


def _batch_stats(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Each batch's mean, and the std of its residuals around each variant's mean."""
    residual = pl.col(column) - pl.col(column).mean().over("meta_aa_changes")
    return (
        df.with_columns(residual.alias("_residual"))
        .group_by("meta_experiment")
        .agg(
            pl.col(column).mean().alias("mean"), pl.col("_residual").std().alias("std")
        )
    )


# --- combat ---------------------------------------------------------------------------------


def test_combat_removes_batch_location_and_scale_and_keeps_variant_effects() -> None:
    """Shifts of 5 and noise scales 10x apart. The locations line up; the scales get much
    closer but not equal: with one row per (variant, batch) the per-variant fit absorbs part
    of a noisy batch's noise, so ComBat underestimates the scale differences (less so with
    more batches)."""
    df = _table(
        batches={
            f"B{i}": (shift, scale)
            for i, (shift, scale) in enumerate(
                [(0, 1), (3, 2.5), (-2, 0.5), (1, 1.5), (-1, 0.8), (2, 2)]
            )
        }
    )
    out = combat_frame(df, _features(df), label_column="meta_aa_changes")
    for f in _features(df):
        raw, stats = _batch_stats(df, f), _batch_stats(out, f)
        assert raw["mean"].max() - raw["mean"].min() > 4
        assert stats["mean"].max() - stats["mean"].min() < 0.25
        raw_ratio = raw["std"].max() / raw["std"].min()
        assert raw_ratio > 2
        assert stats["std"].max() / stats["std"].min() < 0.75 * raw_ratio
        # within a batch the variants keep their order (the fitted variant effects are added
        # back unscaled, so the map isn't exactly affine)
        both = out.join(df, on=["meta_aa_changes", "meta_experiment"], suffix="_raw")
        for _, group in both.group_by("meta_experiment"):
            assert np.corrcoef(group[f], group[f"{f}_raw"])[0, 1] > 0.95


def test_combat_returns_float64_samples_by_features() -> None:
    """The wrapper passes neuroCombat a features × samples float64 array and transposes
    back (a DataFrame would be cast to float32)."""
    df = _table(n_variants=10, n_features=3)
    x = df.select(_features(df)).to_numpy()
    out = combat(x, df["meta_experiment"].to_numpy(), df["meta_aa_changes"].to_numpy())
    assert out.shape == x.shape
    assert out.dtype == np.float64


def test_combat_without_covariate_and_mean_only() -> None:
    df = _table()
    x = df.select(_features(df)).to_numpy()
    batch = df["meta_experiment"].to_numpy()
    # without empirical Bayes: mean_only's EB (as in sva) shrinks each shift hard toward the
    # batch's average shift across features
    out = combat(x, batch, mean_only=True, eb=False)
    means = np.array([out[batch == b].mean(axis=0) for b in BATCHES])
    assert np.ptp(means, axis=0).max() < 1e-9
    # mean_only leaves each batch's spread alone
    stds = np.array([out[batch == b].std(axis=0) for b in BATCHES])
    raw = np.array([x[batch == b].std(axis=0) for b in BATCHES])
    assert np.allclose(stds, raw)


def test_one_column_skips_empirical_bayes(caplog) -> None:
    df = _table(n_features=1)
    with caplog.at_level(logging.INFO, logger="fisseq_common.combat"):
        out = combat_frame(df, ["f0_median"], label_column="meta_aa_changes")
    assert "without empirical Bayes" in caplog.text
    assert out["f0_median"].is_finite().all()
    stats = _batch_stats(out, "f0_median")
    assert stats["mean"].max() - stats["mean"].min() < 0.1


@pytest.mark.parametrize(
    "batch, match",
    [
        (["A"] * 6, "at least 2 batches"),
        (["A"] * 5 + ["B"], "at least 2 rows"),
    ],
)
def test_combat_rejects_degenerate_batches(batch, match) -> None:
    with pytest.raises(ValueError, match=match):
        combat(np.ones((6, 2)), batch)


def test_combat_rejects_non_finite_input() -> None:
    x = np.arange(12, dtype=float).reshape(6, 2)
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="complete matrix"):
        combat(x, ["A", "A", "A", "B", "B", "B"])


def test_neurocombat_spreads_a_single_nan_everywhere() -> None:
    """Why combat_frame drops nulls first: one NaN given to neuroCombat itself turns every
    corrected value NaN, in every feature."""
    import pandas as pd
    from neuroCombat import neuroCombat

    df = _table(n_variants=10, n_features=3)
    x = df.select(_features(df)).to_numpy()
    x[0, 0] = np.nan
    out = neuroCombat(
        dat=x.T,
        covars=pd.DataFrame({"batch": df["meta_experiment"].to_numpy()}),
        batch_col="batch",
    )["data"]
    assert np.isnan(out).all()


# --- single-experiment variants ---------------------------------------------------------------


def test_variants_in_one_experiment_are_dropped_with_a_warning(caplog) -> None:
    df = pl.concat(
        [
            _table(n_variants=10),
            pl.DataFrame(
                {
                    "meta_aa_changes": ["ONLY_A"],
                    "meta_experiment": ["A"],
                    **{f"f{j}_median": [1.0] for j in range(4)},
                }
            ),
        ]
    )
    with caplog.at_level(logging.WARNING, logger="fisseq_common.combat"):
        out = combat_frame(df, _features(df), label_column="meta_aa_changes")
    assert "ONLY_A" not in out["meta_aa_changes"].to_list()
    assert out.height == 30
    assert "1 variant(s) measured in only one experiment (ONLY_A)" in caplog.text


def test_drop_single_batch_variants_keeps_shared_ones() -> None:
    df = pl.DataFrame(
        {"meta_aa_changes": ["x", "x", "y"], "meta_experiment": ["A", "B", "A"]}
    )
    out = drop_single_batch_variants(df, label_column="meta_aa_changes")
    assert out["meta_aa_changes"].to_list() == ["x", "x"]


# --- nulls ------------------------------------------------------------------------------------


@pytest.fixture(params=[None, float("nan"), float("inf")], ids=["null", "nan", "inf"])
def missing(request) -> float | None:
    return request.param


def _set(df: pl.DataFrame, row: int, columns: list[str], value) -> pl.DataFrame:
    index = pl.int_range(pl.len()) == row
    return df.with_columns(
        pl.when(index)
        .then(pl.lit(value, dtype=pl.Float64))
        .otherwise(pl.col(c))
        .alias(c)
        for c in columns
    )


def test_a_row_with_every_value_missing_is_dropped(missing, caplog) -> None:
    df = _table(n_variants=10)
    df = _set(df, 0, _features(df), missing)
    with caplog.at_level(logging.WARNING, logger="fisseq_common.combat"):
        out = combat_frame(df, _features(df), label_column="meta_aa_changes")
    assert "Dropped 1 row(s) with no value" in caplog.text
    # V000 is still in B and C, so it stays; every feature column survives
    assert out.height == 29
    assert _features(out) == _features(df)
    assert all(out[c].is_finite().all() for c in _features(out))


def test_a_column_with_a_scattered_missing_value_is_dropped(missing, caplog) -> None:
    df = _table(n_variants=10)
    df = _set(df, 3, ["f1_median"], missing)
    with caplog.at_level(logging.WARNING, logger="fisseq_common.combat"):
        out = combat_frame(df, _features(df), label_column="meta_aa_changes")
    assert "Dropped 1 of 4 column(s)" in caplog.text
    assert "f1_median" in caplog.text
    assert _features(out) == ["f0_median", "f2_median", "f3_median"]
    assert out.height == 30
    assert all(out[c].is_finite().all() for c in _features(out))


def test_dropping_empty_rows_can_leave_a_variant_in_one_experiment(caplog) -> None:
    """Rows go first, then single-experiment variants: V000 empty in B and C is left in A
    alone and is dropped, without costing any column."""
    df = _table(n_variants=10)
    rows = [
        i
        for i, (v, b) in enumerate(
            df.select("meta_aa_changes", "meta_experiment").rows()
        )
        if v == "V000" and b != "A"
    ]
    for i in rows:
        df = _set(df, i, _features(df), None)
    with caplog.at_level(logging.WARNING, logger="fisseq_common.combat"):
        out = combat_frame(df, _features(df), label_column="meta_aa_changes")
    assert "V000" not in out["meta_aa_changes"].to_list()
    assert _features(out) == _features(df)
    assert "only one experiment (V000)" in caplog.text


def test_every_column_missing_somewhere_is_an_error() -> None:
    df = _table(n_variants=10, n_features=2)
    df = _set(df, 0, ["f0_median"], None)
    df = _set(df, 1, ["f1_median"], None)
    with pytest.raises(ValueError, match="nothing left to correct"):
        combat_frame(df, _features(df), label_column="meta_aa_changes")


def test_other_columns_are_carried_along_unchanged() -> None:
    df = _table(n_variants=10).with_columns(
        meta_n_cells=pl.int_range(pl.len(), dtype=pl.Int64)
    )
    out = combat_frame(df, _features(df), label_column="meta_aa_changes")
    joined = out.join(df, on=["meta_aa_changes", "meta_experiment"], suffix="_in")
    assert (joined["meta_n_cells"] == joined["meta_n_cells_in"]).all()
    assert out.select("meta_aa_changes", "meta_experiment").rows() == sorted(
        out.select("meta_aa_changes", "meta_experiment").rows()
    )


# --- stacking ---------------------------------------------------------------------------------


def test_stack_batches_keeps_shared_columns_and_warns(caplog) -> None:
    a = pl.DataFrame({"meta_aa_changes": ["x"], "f_median": [1.0], "g_median": [2.0]})
    b = pl.DataFrame({"meta_aa_changes": ["x"], "f_median": [3.0]})
    with caplog.at_level(logging.WARNING, logger="fisseq_common.combat"):
        out = stack_batches({"A": a, "B": b})
    assert out.columns == ["meta_aa_changes", "f_median", "meta_experiment"]
    assert out["meta_experiment"].to_list() == ["A", "B"]
    assert "Experiment A: dropped 1 column(s)" in caplog.text
