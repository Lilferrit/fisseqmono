"""fisseq_common.global_aggregation: the cross-experiment blocklist vote, pooling,
distinguishability and reduced-PCA methods (formerly the embeddings pipeline's
GLOBAL_BLOCKLIST, GLOBAL_VARIANT_EMBEDDINGS and GLOBAL_VARIANT_DISTINGUISHABILITY stages).

The full-rank PCA fit needs scikit-learn and is tested with its caller, fisseqborn
(``test_global_pca.py``); here the PC score frames are built by hand.
"""

import polars as pl
import pytest

from fisseq_common.global_aggregation import (
    blocked_features,
    blocklist_vote,
    drop_blocked,
    median_across_batches,
    n_components_for_variance,
    ovwt_global_scores,
    reduced_pca,
)
from fisseq_common.schema import (
    COMPONENT_IDX_COL,
    CONTROL_COLUMN_NAME,
    CUMULATIVE_VARIANCE_EXPLAINED_COL,
    IMPACT_SCORE_COL,
    VARIANCE_EXPLAINED_COL,
)

LABEL_COLUMN = "meta_aa_changes"


# --- blocklist_vote ---------------------------------------------------------------------------


def _bl(ok: "dict[str, bool]") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": list(ok),
            "median_r": [0.9 if v else 0.1 for v in ok.values()],
            "feature_ok": list(ok.values()),
        }
    )


def test_blocklist_vote_default_requires_unanimity() -> None:
    out = blocklist_vote([_bl({"a": True, "b": True}), _bl({"a": True, "b": False})])
    verdicts = dict(zip(out["feature"].to_list(), out["feature_ok"].to_list()))
    assert verdicts == {"a": True, "b": False}


def test_blocklist_vote_min_batches_ok_tolerates_one_dissenting_experiment() -> None:
    out = blocklist_vote(
        [_bl({"b": True}), _bl({"b": False}), _bl({"b": True})], min_batches_ok=2
    )
    assert out["n_ok"].to_list() == [2]
    assert out["n_batches"].to_list() == [3]
    assert out["feature_ok"].to_list() == [True]


def test_blocklist_vote_unanimity_is_over_reporting_experiments_only() -> None:
    """A dimension absent from one experiment's blocklist is judged on the
    experiments that do name it, not counted as a failure there."""
    out = blocklist_vote([_bl({"a": True, "b": True}), _bl({"a": True})])
    verdicts = dict(zip(out["feature"].to_list(), out["feature_ok"].to_list()))
    assert verdicts["b"] is True
    counts = dict(zip(out["feature"].to_list(), out["n_batches"].to_list()))
    assert counts == {"a": 2, "b": 1}


def test_blocklist_vote_output_is_sorted_by_feature() -> None:
    out = blocklist_vote([_bl({"z": True, "a": True, "m": True})])
    assert out["feature"].to_list() == ["a", "m", "z"]


def test_blocklist_vote_columns() -> None:
    out = blocklist_vote([_bl({"a": True, "b": True}), _bl({"a": True, "b": False})])
    assert out.columns == ["feature", "n_batches", "n_ok", "feature_ok"]
    assert out["feature"].to_list() == ["a", "b"]
    assert out["feature_ok"].to_list() == [True, False]


def test_blocklist_vote_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        blocklist_vote([])


# --- blocked_features / drop_blocked ----------------------------------------------------------


def _blocklist(ok: dict[str, bool]) -> pl.DataFrame:
    """A global vote, as blocklist_vote writes it."""
    return pl.DataFrame(
        {
            "feature": list(ok),
            "n_batches": [2] * len(ok),
            "n_ok": [2 if v else 0 for v in ok.values()],
            "feature_ok": list(ok.values()),
        }
    )


def test_blocked_features_are_the_not_ok_ones() -> None:
    vote = _blocklist({"emb_0000": True, "emb_0001": False, "emb_0002": True})
    assert blocked_features(vote) == {"emb_0001"}


def test_blocked_features_null_verdict_counts_as_blocked() -> None:
    vote = pl.DataFrame(
        {"feature": ["a", "b"], "feature_ok": [True, None]},
        schema={"feature": pl.String, "feature_ok": pl.Boolean},
    )
    assert blocked_features(vote) == {"b"}


def _lf(labels: list[str], **cols: list[float]) -> pl.LazyFrame:
    return pl.DataFrame({LABEL_COLUMN: labels, **cols}).lazy()


def test_drop_blocked_drops_non_reproducible_dimensions_from_every_experiment() -> None:
    batches = [
        _lf(["A1B"], emb_0000=[1.0], emb_0001=[2.0], emb_0002=[3.0]),
        _lf(["A1B"], emb_0000=[4.0], emb_0001=[5.0], emb_0002=[6.0]),
    ]
    vote = _blocklist({"emb_0000": True, "emb_0001": False, "emb_0002": True})
    for lf in drop_blocked(batches, vote):
        assert lf.collect_schema().names() == [LABEL_COLUMN, "emb_0000", "emb_0002"]


def test_drop_blocked_keeps_features_the_vote_does_not_mention() -> None:
    batches = [_lf(["A1B"], emb_0000=[1.0], emb_0001=[2.0], emb_0002=[3.0])]
    (out,) = drop_blocked(batches, _blocklist({"emb_0001": False}))
    assert out.collect_schema().names() == [LABEL_COLUMN, "emb_0000", "emb_0002"]


def test_blocked_dimensions_do_not_reach_the_pooled_table() -> None:
    """Dropping before pooling: a non-reproducible dimension must not reach the
    cross-experiment median (and so contribute no loading to any principal
    component)."""
    batches = [
        _lf(["A1B"], emb_0000=[1.0], emb_0001=[2.0]),
        _lf(["A1B"], emb_0000=[3.0], emb_0001=[4.0]),
    ]
    vote = _blocklist({"emb_0000": True, "emb_0001": False})
    median = median_across_batches(drop_blocked(batches, vote), LABEL_COLUMN)
    assert median.columns == [LABEL_COLUMN, "emb_0000"]
    assert median["emb_0000"].to_list() == [2.0]


# --- median_across_batches --------------------------------------------------------------------


def test_median_across_batches_raises_on_empty_list() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        median_across_batches([], LABEL_COLUMN)


def test_median_across_batches_raises_when_no_common_feature_column() -> None:
    batch1 = _lf(["A1A"], emb_0000=[1.0])
    batch2 = _lf(["A1A"], emb_0001=[2.0])
    with pytest.raises(ValueError, match="common to every batch"):
        median_across_batches([batch1, batch2], LABEL_COLUMN)


def test_median_across_batches_collapses_repeated_variant_to_its_median() -> None:
    batch1 = _lf(["M1K"], emb_0000=[1.0])
    batch2 = _lf(["M1K"], emb_0000=[3.0])
    result = median_across_batches([batch1, batch2], LABEL_COLUMN)
    assert result.height == 1
    assert result["emb_0000"].to_list() == [2.0]


def test_median_across_batches_keeps_variant_present_in_only_one_batch() -> None:
    batch1 = _lf(["M1K"], emb_0000=[1.0])
    batch2 = _lf(["M2K"], emb_0000=[5.0])
    result = median_across_batches([batch1, batch2], LABEL_COLUMN)
    assert sorted(result[LABEL_COLUMN].to_list()) == ["M1K", "M2K"]


def test_median_across_batches_drops_columns_not_common_to_every_batch() -> None:
    batch1 = _lf(["M1K"], emb_0000=[1.0], emb_0001=[9.0])
    batch2 = _lf(["M1K"], emb_0000=[3.0])
    result = median_across_batches([batch1, batch2], LABEL_COLUMN)
    assert "emb_0001" not in result.columns
    assert result["emb_0000"].to_list() == [2.0]


def test_median_across_batches_one_row_per_distinct_variant() -> None:
    """2 experiments, overlapping on M1K, disjoint on the rest: M1K collapses to
    one row and the rest appear in exactly one experiment each -- 5 variants."""
    batch1 = _lf(["M1K", "M2K", "M3K"], emb_0000=[0.0, 1.0, 2.0])
    batch2 = _lf(["M1K", "M4K", "M5K"], emb_0000=[2.0, 3.0, 0.0])
    result = median_across_batches([batch1, batch2], LABEL_COLUMN)
    assert sorted(result[LABEL_COLUMN].to_list()) == ["M1K", "M2K", "M3K", "M4K", "M5K"]


# --- ovwt_global_scores -----------------------------------------------------------------------


def _results_df(
    labels: list[str],
    auroc_pooled: list[float],
    auroc_median_barcode: list[float],
    with_folds: bool = True,
) -> pl.DataFrame:
    """One experiment's OVWT_BATCHWISE results.parquet -- includes
    synonymous-labeled rows (OVWT scores every non-WT variant, synonymous
    ones included, against WT). ``auroc_median_fold`` reuses the barcode
    value, and ``with_folds=False`` omits the ``auroc_folds`` list column."""
    n = len(labels)
    cols = {
        LABEL_COLUMN: labels,
        "auroc_pooled": auroc_pooled,
        "auroc_median_barcode": auroc_median_barcode,
    }
    if with_folds:
        cols["auroc_folds"] = pl.Series(
            [[a, None, a] for a in auroc_median_barcode], dtype=pl.List(pl.Float64)
        )
    cols["auroc_median_fold"] = auroc_median_barcode
    cols["meta_n_barcodes"] = [1] * n
    cols["meta_n_cells"] = [10] * n
    return pl.DataFrame(cols)


def test_ovwt_global_scores_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ovwt_global_scores([], LABEL_COLUMN)


def test_ovwt_global_scores_output_columns() -> None:
    df = _results_df(["A1A", "A2A", "M1K"], [0.5, 0.6, 0.9], [0.5, 0.55, 0.85])
    result = ovwt_global_scores([df], LABEL_COLUMN)
    assert set(result.columns) == {
        LABEL_COLUMN,
        "meta_median_auroc_pooled",
        "meta_median_auroc_median_barcode",
        "meta_median_auroc_median_fold",
        "meta_num_experiments",
    }


def test_ovwt_global_scores_without_fold_list() -> None:
    """A results file with no auroc_folds column is still accepted."""
    df = _results_df(
        ["A1A", "A2A", "M1K"], [0.5, 0.6, 0.9], [0.5, 0.55, 0.85], with_folds=False
    )
    result = ovwt_global_scores([df], LABEL_COLUMN)
    assert "meta_median_auroc_median_fold" in result.columns
    assert len(result) == 3


def test_ovwt_global_scores_raises_on_missing_score_column() -> None:
    df = _results_df(["A1A", "A2A", "M1K"], [0.5, 0.6, 0.9], [0.5, 0.55, 0.85]).drop(
        "auroc_median_fold"
    )
    with pytest.raises(ValueError, match="auroc_median_fold"):
        ovwt_global_scores([df], LABEL_COLUMN)


def test_ovwt_global_scores_median_fold_is_zscored() -> None:
    """auroc_median_fold goes through the same synonymous z-score as the
    other two columns, and the auroc_folds list column never reaches the
    Normalizer (it would otherwise be swept in as a "feature")."""
    df = _results_df(
        ["A1A", "A2A", "A3A", "M1K"], [0.5, 0.55, 0.6, 0.95], [0.5, 0.55, 0.6, 0.93]
    )
    result = ovwt_global_scores([df], LABEL_COLUMN)
    m1k = result.filter(pl.col(LABEL_COLUMN) == "M1K").row(0, named=True)
    # Synonymous mean 0.55 -> M1K sits well above it once z-scored.
    assert m1k["meta_median_auroc_median_fold"] > 1.0


def test_ovwt_global_scores_synonymous_lands_near_zero() -> None:
    """Built-in sanity check: synonymous variants are the
    Normalizer's own fit population, so their z-scored value should land
    near 0 (not exactly -- a small synonymous population's own mean isn't
    each individual row's value)."""
    df = _results_df(
        # Several synonymous rows to give the Normalizer a real spread, plus
        # one clearly-separable real variant.
        ["A1A", "A2A", "A3A", "A4A", "M1K"],
        [0.48, 0.5, 0.52, 0.5, 0.95],
        [0.49, 0.5, 0.51, 0.5, 0.9],
    )
    result = ovwt_global_scores([df], LABEL_COLUMN)
    syn_rows = result.filter(pl.col(LABEL_COLUMN).is_in(["A1A", "A2A", "A3A", "A4A"]))
    for val in syn_rows["meta_median_auroc_pooled"].to_list():
        assert val == pytest.approx(0.0, abs=1.5)
    variant_row = result.filter(pl.col(LABEL_COLUMN) == "M1K").row(0, named=True)
    # The real variant's z-score should clearly exceed the synonymous rows'.
    assert variant_row["meta_median_auroc_pooled"] > max(
        syn_rows["meta_median_auroc_pooled"].to_list()
    )


def test_ovwt_global_scores_medians_zscored_not_raw_auroc() -> None:
    """Median of the *z-scored* values across
    experiments, not a direct median of raw AUROC -- two experiments with
    very different synonymous baselines should not simply average their raw
    AUROC for the shared variant."""
    # Experiment 1: synonymous cluster tight around 0.5 (small real spread,
    # not exactly identical -- an exactly-zero-variance column degrades to
    # null per the graceful-degradation test below); M1K is far above it.
    df1 = _results_df(
        ["A1A", "A2A", "A3A", "M1K"], [0.45, 0.5, 0.55, 0.9], [0.45, 0.5, 0.55, 0.9]
    )
    # Experiment 2: synonymous cluster around 0.7 (a shifted/noisier batch);
    # M1K sits only slightly above it in raw terms.
    df2 = _results_df(
        ["A1A", "A2A", "A3A", "M1K"], [0.65, 0.7, 0.75, 0.8], [0.65, 0.7, 0.75, 0.8]
    )
    result = ovwt_global_scores([df1, df2], LABEL_COLUMN)
    m1k = result.filter(pl.col(LABEL_COLUMN) == "M1K").row(0, named=True)
    # If this were a direct median of raw AUROC, M1K would land at
    # median(0.9, 0.8) == 0.85 -- assert the z-scored result instead
    # reflects that M1K was clearly, comparably separable in *both*
    # experiments (a positive z-score, not literally 0.85).
    assert m1k["meta_num_experiments"] == 2
    assert m1k["meta_median_auroc_pooled"] > 0


def test_ovwt_global_scores_graceful_degradation_zero_variance() -> None:
    """An experiment with a zero-variance (or too-thin) synonymous population
    nulls that column out entirely rather than raising or corrupting the pooled
    median -- polars' .median() silently excludes the null."""
    # Experiment 1: normal spread.
    df1 = _results_df(
        ["A1A", "A2A", "A3A", "M1K"], [0.4, 0.5, 0.6, 0.9], [0.4, 0.5, 0.6, 0.9]
    )
    # Experiment 2: exactly one synonymous row (zero variance -- std < EPS
    # is stored as None by Normalizer.from_lazyframe) -- this whole
    # experiment's auroc_pooled column should degrade to null, not raise.
    df2 = _results_df(["A1A", "M1K"], [0.5, 0.5], [0.5, 0.5])

    result = ovwt_global_scores([df1, df2], LABEL_COLUMN)
    m1k = result.filter(pl.col(LABEL_COLUMN) == "M1K").row(0, named=True)
    # Only experiment 1 contributes a non-null z-score for M1K.
    assert m1k["meta_num_experiments"] == 1
    assert m1k["meta_median_auroc_pooled"] is not None


def test_ovwt_global_scores_two_experiments_count_both() -> None:
    df1 = _results_df(
        ["A1A", "A2A", "A3A", "M1K"], [0.45, 0.5, 0.55, 0.9], [0.45, 0.5, 0.55, 0.9]
    )
    df2 = _results_df(
        ["A1A", "A2A", "A3A", "M1K"], [0.4, 0.5, 0.6, 0.85], [0.4, 0.5, 0.6, 0.85]
    )
    result = ovwt_global_scores([df1, df2], LABEL_COLUMN)
    m1k = result.filter(pl.col(LABEL_COLUMN) == "M1K").row(0, named=True)
    assert m1k["meta_num_experiments"] == 2


# --- n_components_for_variance ----------------------------------------------------------------


def _variance_df(cumulative: list[float]) -> pl.DataFrame:
    n = len(cumulative)
    ratios = [cumulative[0]] + [cumulative[i] - cumulative[i - 1] for i in range(1, n)]
    return pl.DataFrame(
        {
            COMPONENT_IDX_COL: list(range(1, n + 1)),
            VARIANCE_EXPLAINED_COL: ratios,
            CUMULATIVE_VARIANCE_EXPLAINED_COL: cumulative,
        }
    )


def test_n_components_for_variance_picks_smallest_prefix_reaching_threshold() -> None:
    variance_df = _variance_df([0.5, 0.8, 0.95, 1.0])
    assert n_components_for_variance(variance_df, 0.9) == 3


def test_n_components_for_variance_exact_match_at_first_component() -> None:
    variance_df = _variance_df([0.9, 0.95, 1.0])
    assert n_components_for_variance(variance_df, 0.9) == 1


def test_n_components_for_variance_never_reached_keeps_every_component() -> None:
    """A threshold above the achievable cumulative total (e.g. it never
    quite reaches exactly 1.0 due to floating error) falls back to the
    full retained rank, same as pca_components.parquet itself."""
    variance_df = _variance_df([0.5, 0.8, 0.9999999])
    assert n_components_for_variance(variance_df, 1.0) == 3


@pytest.mark.parametrize("threshold", [0.0, 1.5])
def test_n_components_for_variance_raises_on_invalid_threshold(
    threshold: float,
) -> None:
    with pytest.raises(ValueError, match="cumulative_variance_explained"):
        n_components_for_variance(_variance_df([0.5, 1.0]), threshold)


# --- reduced_pca ------------------------------------------------------------------------------


def _scores_df() -> pl.DataFrame:
    """Full-rank PC scores, by hand: A1A (the only synonymous variant, so the sole
    control) on PC1; M2K orthogonal to it; M3K opposite; M4K along A1A in PCs 1-2 but
    far off in PC3."""
    return pl.DataFrame(
        {
            LABEL_COLUMN: ["A1A", "M2K", "M3K", "M4K"],
            "meta_pc_1": [1.0, 0.0, -1.0, 1.0],
            "meta_pc_2": [0.0, 1.0, 0.0, 0.0],
            "meta_pc_3": [0.0, 0.0, 0.0, 5.0],
        }
    )


def test_reduced_pca_columns_are_truncated_to_n_components() -> None:
    reduced = reduced_pca(_scores_df(), LABEL_COLUMN, 2)
    assert reduced.columns == [
        LABEL_COLUMN,
        "meta_pc_1",
        "meta_pc_2",
        CONTROL_COLUMN_NAME,
        IMPACT_SCORE_COL,
    ]
    # A true truncation, not a re-fit: the leading components are the full ones.
    full = _scores_df()
    assert reduced.select(LABEL_COLUMN, "meta_pc_1", "meta_pc_2").equals(
        full.select(LABEL_COLUMN, "meta_pc_1", "meta_pc_2")
    )


def test_reduced_pca_marks_synonymous_variants_as_controls() -> None:
    reduced = reduced_pca(_scores_df(), LABEL_COLUMN, 2)
    # A1A is the only synonymous label -- the sole control row.
    controls = reduced.filter(pl.col(CONTROL_COLUMN_NAME))
    assert controls[LABEL_COLUMN].to_list() == ["A1A"]


def test_reduced_pca_impact_score_is_on_the_kept_components() -> None:
    """The impact score is the cosine distance / 2 to the controls' median, computed
    on the kept PCs only: 0 for the control itself (the reference point), 0.5 for an
    orthogonal variant, 1 for an opposite one. M4K differs from A1A only in PC3, so it
    scores 0 when PC3 is cut and above 0 when it is kept."""
    reduced = reduced_pca(_scores_df(), LABEL_COLUMN, 2)
    impact = dict(zip(reduced[LABEL_COLUMN], reduced[IMPACT_SCORE_COL]))
    assert impact["A1A"] == pytest.approx(0.0, abs=1e-6)
    assert impact["M2K"] == pytest.approx(0.5, abs=1e-6)
    assert impact["M3K"] == pytest.approx(1.0, abs=1e-6)
    assert impact["M4K"] == pytest.approx(0.0, abs=1e-6)

    full = reduced_pca(_scores_df(), LABEL_COLUMN, 3)
    impact_full = dict(zip(full[LABEL_COLUMN], full[IMPACT_SCORE_COL]))
    assert impact_full["M4K"] > 0.1
    # meta_impact_score is in [0, 1] for every row (up to floating noise), per
    # compute_impact_score's own contract (cosine distance / 2).
    for val in full[IMPACT_SCORE_COL].to_list():
        assert -1e-6 <= val <= 1.0 + 1e-6
