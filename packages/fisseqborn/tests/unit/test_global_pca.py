"""Pooling then full-rank PCA: `full_rank_pca` and `write_global`'s per-track outputs on an
embeddings-pipeline run (formerly the GLOBAL_VARIANT_EMBEDDINGS and
GLOBAL_VARIANT_CP_FEATURES stages).

No n_components knob: every PC the pooled table supports is computed, and
``pca_reduced`` keeps the fewest leading ones reaching ``cumulative_variance_explained``.
"""

import pathlib

import numpy as np
import polars as pl
import pytest

import fisseqborn as fb
from fisseq_common.global_aggregation import median_across_batches
from fisseq_common.layout import EmbeddingsPipelineLayout
from fisseq_common.schema import (
    COMPONENT_IDX_COL,
    CONTROL_COLUMN_NAME,
    CUMULATIVE_VARIANCE_EXPLAINED_COL,
    IMPACT_SCORE_COL,
    VARIANCE_EXPLAINED_COL,
)
from fisseqborn.global_aggregate import full_rank_pca

LABEL_COLUMN = "meta_aa_changes"

TRACK_FILES = {
    "median_aggregate",
    "pca_scores",
    "pca_components",
    "pca_variance_explained",
    "pca_reduced",
}


def _batch_aggregate(
    labels: list[str], values: list[list[float]], prefix: str = "emb_"
) -> pl.DataFrame:
    """One experiment's per-variant aggregates (``aggregates/median.parquet``): label_column +
    features."""
    n_dims = len(values[0])
    return pl.DataFrame(
        {
            LABEL_COLUMN: labels,
            **{f"{prefix}{i:04d}": [row[i] for row in values] for i in range(n_dims)},
        }
    )


def _two_batches() -> list[pl.DataFrame]:
    """2 experiments, overlapping on M1K, disjoint on the rest, 4 emb dims."""
    return [
        _batch_aggregate(
            ["M1K", "M2K", "M3K"],
            [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 3.0, 2.0], [2.0, 3.0, 0.0, 1.0]],
        ),
        _batch_aggregate(
            ["M1K", "M4K", "M5K"],
            [[2.0, 3.0, 0.0, 1.0], [3.0, 2.0, 1.0, 0.0], [0.0, 2.0, 3.0, 1.0]],
        ),
    ]


def _two_batches_with_control() -> list[pl.DataFrame]:
    """Like _two_batches, but with a real synonymous ("A1A") row in each batch so
    meta_is_control/meta_impact_score have a non-trivial control population to fit
    against."""
    return [
        _batch_aggregate(
            ["A1A", "M2K", "M3K"],
            [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 3.0, 2.0], [2.0, 3.0, 0.0, 1.0]],
        ),
        _batch_aggregate(
            ["A1A", "M4K", "M5K"],
            [[2.0, 3.0, 0.0, 1.0], [3.0, 2.0, 1.0, 0.0], [0.0, 2.0, 3.0, 1.0]],
        ),
    ]


def _results_df(labels: list[str], auroc: list[float]) -> pl.DataFrame:
    """One experiment's OVWT_BATCHWISE results.parquet."""
    return pl.DataFrame(
        {
            LABEL_COLUMN: labels,
            "auroc_pooled": auroc,
            "auroc_median_barcode": auroc,
            "auroc_median_fold": auroc,
        }
    )


def _write(path: pathlib.Path, df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _emb_run(
    root: pathlib.Path,
    aggregates: list[pl.DataFrame],
    *,
    blocklists: list[pl.DataFrame] | None = None,
    cp_aggregates: list[pl.DataFrame] | None = None,
    ovwt: bool = False,
) -> pathlib.Path:
    """An embeddings-pipeline run with batches e1, e2, ...: the Cell-DINO track's
    aggregates and blocklists (default: every feature OK in every experiment), and
    optionally the CellProfiler track's aggregates and both tracks' OvWT results."""
    if blocklists is None:
        blocklists = [
            _blocklist({c: True for c in df.columns if not c.startswith("meta_")})
            for df in aggregates
        ]
    emb, cp = (
        EmbeddingsPipelineLayout("embeddings"),
        EmbeddingsPipelineLayout("cp_features"),
    )
    results = [
        _results_df(["A1A", "A2A", "A3A", "M1K"], [0.45, 0.5, 0.55, 0.9]),
        _results_df(["A1A", "A2A", "A3A", "M1K"], [0.4, 0.5, 0.6, 0.85]),
    ]
    for i, df in enumerate(aggregates):
        batch = f"e{i + 1}"
        _write(root / emb.aggregate(batch, "median"), df)
        _write(root / emb.blocklist(batch), blocklists[i])
        if cp_aggregates is not None:
            _write(root / cp.aggregate(batch, "median"), cp_aggregates[i])
        if ovwt:
            _write(root / emb.ovwt_results(batch), results[i])
            if cp_aggregates is not None:
                _write(root / cp.ovwt_results(batch), results[i])
    return root


def _write_global(run: pathlib.Path, out: pathlib.Path, **kw) -> dict:
    return fb.write_global(run, out, layout="embeddings", **kw)


def _pc_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("meta_pc_")]


# --- full_rank_pca ----------------------------------------------------------------------------


def _median() -> pl.DataFrame:
    return median_across_batches(
        [df.lazy() for df in _two_batches()], LABEL_COLUMN
    ).sort(LABEL_COLUMN)


def test_full_rank_pca_not_a_fixed_default() -> None:
    """No n_components knob -- every retained component is computed:
    min(n_variants=5, n_retained_dims=4) == 4, not a fixed default of 50 (which
    5 rows/4 dims couldn't even support)."""
    scores_df, components_df = full_rank_pca(_median(), LABEL_COLUMN, seed=0)
    assert len(_pc_cols(scores_df)) == 4
    assert components_df.height == 4


def test_full_rank_pca_rank_is_bounded_by_rows() -> None:
    """Fewer variants than features: min(n_rows=3, n_features=5) == 3."""
    df = _batch_aggregate(
        ["M1K", "M2K", "M3K"],
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 0.0, 3.0, 2.0, 5.0],
            [2.0, 3.0, 0.0, 1.0, 1.0],
        ],
    )
    scores_df, components_df = full_rank_pca(df, LABEL_COLUMN)
    assert _pc_cols(scores_df) == ["meta_pc_1", "meta_pc_2", "meta_pc_3"]
    assert components_df.height == 3


def test_full_rank_pca_scores_columns() -> None:
    scores_df, _ = full_rank_pca(_median(), LABEL_COLUMN, seed=0)
    assert scores_df.columns == [
        LABEL_COLUMN,
        "meta_pc_1",
        "meta_pc_2",
        "meta_pc_3",
        "meta_pc_4",
    ]
    assert scores_df.height == 5
    assert scores_df[LABEL_COLUMN].to_list() == ["M1K", "M2K", "M3K", "M4K", "M5K"]


def test_full_rank_pca_components_columns() -> None:
    _, components_df = full_rank_pca(_median(), LABEL_COLUMN, seed=0)
    assert components_df.columns == [
        COMPONENT_IDX_COL,
        "emb_0000",
        "emb_0001",
        "emb_0002",
        "emb_0003",
        VARIANCE_EXPLAINED_COL,
        CUMULATIVE_VARIANCE_EXPLAINED_COL,
    ]
    assert components_df[COMPONENT_IDX_COL].to_list() == [1, 2, 3, 4]
    # cumulative variance explained is monotonically non-decreasing and
    # ends at (approximately) the full retained variance.
    cumulative = components_df[CUMULATIVE_VARIANCE_EXPLAINED_COL].to_list()
    assert cumulative == sorted(cumulative)
    assert cumulative[-1] == pytest.approx(1.0, abs=1e-6)


def test_full_rank_pca_drops_all_null_columns() -> None:
    df = _median().with_columns(pl.lit(None, dtype=pl.Float64).alias("emb_9999"))
    scores_df, components_df = full_rank_pca(df, LABEL_COLUMN)
    assert "emb_9999" not in components_df.columns
    # The rank is over the retained features: still 4, not 5.
    assert len(_pc_cols(scores_df)) == 4


def test_full_rank_pca_raises_when_every_feature_is_null() -> None:
    df = pl.DataFrame(
        {LABEL_COLUMN: ["M1K", "M2K"], "emb_0000": [None, None]},
        schema={LABEL_COLUMN: pl.String, "emb_0000": pl.Float64},
    )
    with pytest.raises(ValueError, match="entirely null"):
        full_rank_pca(df, LABEL_COLUMN)


def test_full_rank_pca_random_state_is_threaded_through(monkeypatch) -> None:
    from sklearn.decomposition import PCA

    seen = []
    fit_transform = PCA.fit_transform

    def spy(self, *args, **kwargs):
        seen.append(self.random_state)
        return fit_transform(self, *args, **kwargs)

    monkeypatch.setattr(PCA, "fit_transform", spy)
    full_rank_pca(_median(), LABEL_COLUMN, seed=7)
    assert seen == [7]


def test_full_rank_pca_same_seed_reproduces_scores() -> None:
    # Deterministic solver path at these matrix sizes: the same seed reproduces the
    # same per-variant scores, up to the sign-indeterminacy floating noise (~1e-16)
    # SVD leaves on a component with essentially-zero eigenvalue at full retained
    # rank.
    scores_a, _ = full_rank_pca(_median(), LABEL_COLUMN, seed=0)
    scores_b, _ = full_rank_pca(_median(), LABEL_COLUMN, seed=0)
    assert scores_a[LABEL_COLUMN].to_list() == scores_b[LABEL_COLUMN].to_list()
    np.testing.assert_allclose(
        scores_a.select(_pc_cols(scores_a)).to_numpy(),
        scores_b.select(_pc_cols(scores_b)).to_numpy(),
        atol=1e-8,
    )


# --- write_global on an embeddings-pipeline run -----------------------------------------------


def _files(out: pathlib.Path) -> set[str]:
    return {p.relative_to(out).as_posix() for p in out.rglob("*.parquet")}


def _blocklist(ok: dict[str, bool]) -> pl.DataFrame:
    """One experiment's combined blocklist.parquet."""
    return pl.DataFrame(
        {
            "feature": list(ok),
            "median_r": [0.9 if v else 0.1 for v in ok.values()],
            "feature_ok": list(ok.values()),
        }
    )


def _random_batches() -> list[pl.DataFrame]:
    labels = ["A1B", "C2D", "E3F", "WT"]
    rng = np.random.default_rng(0)
    return [
        _batch_aggregate(labels, rng.normal(size=(4, 3)).tolist()) for _ in range(2)
    ]


def _cp_batches() -> list[pl.DataFrame]:
    return [
        pl.DataFrame(
            {
                LABEL_COLUMN: ["A1A", "M2K", "M3K"],
                "Cells_AreaShape_Area": [0.0, 1.0, 2.0],
                "Cells_Intensity_MeanIntensity_DNA": [1.0, 0.0, 3.0],
            }
        ),
        pl.DataFrame(
            {
                LABEL_COLUMN: ["A1A", "M4K", "M5K"],
                "Cells_AreaShape_Area": [0.5, 2.0, 1.0],
                "Cells_Intensity_MeanIntensity_DNA": [1.5, 1.0, 0.0],
            }
        ),
    ]


def test_write_global_writes_each_tracks_outputs(tmp_path: pathlib.Path) -> None:
    run = _emb_run(
        tmp_path / "run",
        _two_batches_with_control(),
        cp_aggregates=_cp_batches(),
        ovwt=True,
    )
    out = tmp_path / "global"
    written = _write_global(run, out)
    # The CellProfiler track has no blocklist: that track has no reproducibility
    # filtering.
    expected = (
        {f"embeddings/{f}.parquet" for f in TRACK_FILES | {"blocklist"}}
        | {f"cp_features/{f}.parquet" for f in TRACK_FILES}
        | {
            "distinguishability/global_scores.parquet",
            "distinguishability_cp_features/global_scores.parquet",
        }
    )
    assert _files(out) == expected
    assert {p.relative_to(out).as_posix() for p in written.values()} == expected

    for ovwt_dir in ("distinguishability", "distinguishability_cp_features"):
        global_scores = pl.read_parquet(out / ovwt_dir / "global_scores.parquet")
        assert set(global_scores.columns) == {
            LABEL_COLUMN,
            "meta_median_auroc_pooled",
            "meta_median_auroc_median_barcode",
            "meta_median_auroc_median_fold",
            "meta_num_experiments",
        }
        m1k = global_scores.filter(pl.col(LABEL_COLUMN) == "M1K").row(0, named=True)
        assert m1k["meta_num_experiments"] == 2


def test_median_aggregate_has_one_row_per_distinct_variant(
    tmp_path: pathlib.Path,
) -> None:
    run = _emb_run(tmp_path / "run", _two_batches())
    written = _write_global(run, tmp_path / "global")
    median_df = pl.read_parquet(written["embeddings/median_aggregate"])
    # M1K appears in both batches and collapses to one row; the rest appear
    # in exactly one batch each -- 5 distinct variants total.
    assert median_df[LABEL_COLUMN].to_list() == ["M1K", "M2K", "M3K", "M4K", "M5K"]


def test_pca_outputs_are_full_rank(tmp_path: pathlib.Path) -> None:
    run = _emb_run(tmp_path / "run", _two_batches())
    written = _write_global(run, tmp_path / "global")
    scores_df = pl.read_parquet(written["embeddings/pca_scores"])
    assert scores_df.columns == [
        LABEL_COLUMN,
        "meta_pc_1",
        "meta_pc_2",
        "meta_pc_3",
        "meta_pc_4",
    ]
    assert scores_df.height == 5
    assert pl.read_parquet(written["embeddings/pca_components"]).height == 4
    assert pl.read_parquet(written["embeddings/pca_variance_explained"]).height == 4


def test_pca_components_has_no_variance_columns(tmp_path: pathlib.Path) -> None:
    """pca_components.parquet carries loadings only -- variance-explained
    lives in its own file."""
    run = _emb_run(tmp_path / "run", _two_batches())
    written = _write_global(run, tmp_path / "global")
    components_df = pl.read_parquet(written["embeddings/pca_components"])
    assert components_df.columns[0] == COMPONENT_IDX_COL
    assert VARIANCE_EXPLAINED_COL not in components_df.columns
    assert CUMULATIVE_VARIANCE_EXPLAINED_COL not in components_df.columns
    assert "emb_0000" in components_df.columns


def test_pca_variance_explained_has_only_variance_columns(
    tmp_path: pathlib.Path,
) -> None:
    run = _emb_run(tmp_path / "run", _two_batches())
    written = _write_global(run, tmp_path / "global")
    variance_df = pl.read_parquet(written["embeddings/pca_variance_explained"])
    assert variance_df.columns == [
        COMPONENT_IDX_COL,
        VARIANCE_EXPLAINED_COL,
        CUMULATIVE_VARIANCE_EXPLAINED_COL,
    ]
    cumulative = variance_df[CUMULATIVE_VARIANCE_EXPLAINED_COL].to_list()
    assert cumulative == sorted(cumulative)
    assert cumulative[-1] == pytest.approx(1.0, abs=1e-6)


def test_pca_reduced_is_truncated_to_threshold(tmp_path: pathlib.Path) -> None:
    run = _emb_run(tmp_path / "run", _two_batches_with_control())
    written = _write_global(run, tmp_path / "global", cumulative_variance_explained=0.5)
    scores_df = pl.read_parquet(written["embeddings/pca_scores"])
    variance_df = pl.read_parquet(written["embeddings/pca_variance_explained"])
    reduced_df = pl.read_parquet(written["embeddings/pca_reduced"])
    full_pc_cols = _pc_cols(scores_df)
    reduced_pc_cols = _pc_cols(reduced_df)
    expected_n = next(
        i
        for i, c in enumerate(variance_df[CUMULATIVE_VARIANCE_EXPLAINED_COL], start=1)
        if c >= 0.5
    )
    assert 0 < len(reduced_pc_cols) < len(full_pc_cols)
    assert len(reduced_pc_cols) == expected_n
    # The reduced matrix's leading components are identical to the full
    # matrix's leading components -- a true truncation, not a re-fit.
    a = scores_df.sort(LABEL_COLUMN).select(reduced_pc_cols).to_numpy()
    b = reduced_df.sort(LABEL_COLUMN).select(reduced_pc_cols).to_numpy()
    np.testing.assert_allclose(a, b, atol=1e-8)


def test_pca_reduced_full_rank_when_threshold_is_one(tmp_path: pathlib.Path) -> None:
    run = _emb_run(tmp_path / "run", _two_batches_with_control())
    written = _write_global(run, tmp_path / "global", cumulative_variance_explained=1.0)
    full_pc_cols = _pc_cols(pl.read_parquet(written["embeddings/pca_scores"]))
    reduced_pc_cols = _pc_cols(pl.read_parquet(written["embeddings/pca_reduced"]))
    # threshold=1.0 should keep essentially every component -- allow an
    # off-by-one for the theoretical last (near-zero-eigenvalue) component,
    # whose cumulative value can round to >= 1.0 one step early due to
    # floating-point summation noise in np.cumsum.
    assert len(reduced_pc_cols) >= len(full_pc_cols) - 1


def test_pca_reduced_has_control_flag_and_impact_score(
    tmp_path: pathlib.Path,
) -> None:
    run = _emb_run(tmp_path / "run", _two_batches_with_control())
    written = _write_global(run, tmp_path / "global", cumulative_variance_explained=0.9)
    reduced_df = pl.read_parquet(written["embeddings/pca_reduced"])
    assert reduced_df.columns == [
        LABEL_COLUMN,
        *_pc_cols(reduced_df),
        CONTROL_COLUMN_NAME,
        IMPACT_SCORE_COL,
    ]
    # A1A is the only synonymous label -- the sole control row.
    control_rows = reduced_df.filter(pl.col(CONTROL_COLUMN_NAME))
    assert control_rows[LABEL_COLUMN].to_list() == ["A1A"]
    # meta_impact_score in [0, 1] for every row (up to floating noise), per
    # compute_impact_score's own contract (cosine distance / 2).
    for val in reduced_df[IMPACT_SCORE_COL].to_list():
        assert -1e-6 <= val <= 1.0 + 1e-6
    # The control row's own impact score should be ~0 (it's the reference
    # point itself).
    control_score = control_rows[IMPACT_SCORE_COL].to_list()[0]
    assert control_score == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("threshold", [0.0, 1.5])
def test_write_global_raises_on_invalid_cumulative_variance_explained(
    tmp_path: pathlib.Path, threshold: float
) -> None:
    run = _emb_run(tmp_path / "run", _two_batches_with_control())
    with pytest.raises(ValueError, match="cumulative_variance_explained"):
        _write_global(run, tmp_path / "global", cumulative_variance_explained=threshold)


def test_blocklist_drops_non_reproducible_dimensions_before_pooling(
    tmp_path: pathlib.Path,
) -> None:
    ok = {"emb_0000": True, "emb_0001": False, "emb_0002": True}
    run = _emb_run(
        tmp_path / "run",
        _random_batches(),
        blocklists=[_blocklist(ok), _blocklist(ok)],
    )
    written = _write_global(run, tmp_path / "global")
    vote = pl.read_parquet(written["embeddings/blocklist"])
    assert dict(vote.select("feature", "feature_ok").iter_rows()) == ok
    median_df = pl.read_parquet(written["embeddings/median_aggregate"])
    assert "emb_0001" not in median_df.columns
    assert "emb_0000" in median_df.columns


def test_blocked_dimensions_do_not_reach_the_pca(tmp_path: pathlib.Path) -> None:
    """The point of applying the blocklist before pooling rather than leaving it to
    the per-experiment filtered aggregates: a non-reproducible dimension must not
    contribute a loading to any principal component."""
    ok = {"emb_0000": True, "emb_0001": False, "emb_0002": True}
    run = _emb_run(
        tmp_path / "run",
        _random_batches(),
        blocklists=[_blocklist(ok), _blocklist(ok)],
    )
    written = _write_global(run, tmp_path / "global")
    components_df = pl.read_parquet(written["embeddings/pca_components"])
    assert "emb_0001" not in components_df.columns


def test_no_blocklist_keeps_every_dimension(tmp_path: pathlib.Path) -> None:
    ok = {"emb_0000": True, "emb_0001": False, "emb_0002": True}
    run = _emb_run(
        tmp_path / "run",
        _random_batches(),
        blocklists=[_blocklist(ok), _blocklist(ok)],
    )
    written = _write_global(run, tmp_path / "global", blocklist=False)
    assert "embeddings/blocklist" not in written
    median_df = pl.read_parquet(written["embeddings/median_aggregate"])
    assert {"emb_0000", "emb_0001", "emb_0002"} <= set(median_df.columns)


def test_cp_track_is_pooled_without_a_blocklist(tmp_path: pathlib.Path) -> None:
    """The CellProfiler track gets no reproducibility filtering: even with a
    blocklist.parquet in its directory, every feature is pooled."""
    run = _emb_run(
        tmp_path / "run",
        _two_batches_with_control(),
        cp_aggregates=_cp_batches(),
    )
    cp = EmbeddingsPipelineLayout("cp_features")
    stray = run / cp.batch_dir("feature_select", "e1") / "blocklist.parquet"
    _blocklist({"Cells_AreaShape_Area": False}).write_parquet(stray)
    written = _write_global(run, tmp_path / "global")
    assert "cp_features/blocklist" not in written
    median_df = pl.read_parquet(written["cp_features/median_aggregate"])
    assert {"Cells_AreaShape_Area", "Cells_Intensity_MeanIntensity_DNA"} <= set(
        median_df.columns
    )


def test_cp_track_outputs(tmp_path: pathlib.Path) -> None:
    """The CellProfiler track runs the same pooling and PCA against
    CellProfiler-shaped columns (the methods key off the non-``meta_`` columns), and
    writes the raw, pre-PCA cross-experiment median alongside the PCA outputs."""
    run = _emb_run(
        tmp_path / "run",
        _two_batches_with_control(),
        cp_aggregates=_cp_batches(),
    )
    written = _write_global(run, tmp_path / "global")
    median_df = pl.read_parquet(written["cp_features/median_aggregate"])
    scores_df = pl.read_parquet(written["cp_features/pca_scores"])
    components_df = pl.read_parquet(written["cp_features/pca_components"])
    variance_df = pl.read_parquet(written["cp_features/pca_variance_explained"])
    reduced_df = pl.read_parquet(written["cp_features/pca_reduced"])

    assert "Cells_AreaShape_Area" in median_df.columns
    assert median_df[LABEL_COLUMN].to_list() == ["A1A", "M2K", "M3K", "M4K", "M5K"]
    # A1A is in both experiments: the median of 0.0 and 0.5.
    assert median_df["Cells_AreaShape_Area"].to_list()[0] == 0.25
    assert scores_df.height == 5
    assert components_df.height == variance_df.height
    assert VARIANCE_EXPLAINED_COL not in components_df.columns
    assert VARIANCE_EXPLAINED_COL in variance_df.columns
    assert reduced_df.height == 5
    assert CONTROL_COLUMN_NAME in reduced_df.columns
    assert IMPACT_SCORE_COL in reduced_df.columns
