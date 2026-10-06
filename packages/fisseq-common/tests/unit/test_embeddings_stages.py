"""The shared stages as the embeddings pipeline runs them.

fisseq-embeddings-pipeline's ``conf/modules.config`` calls every stage downstream of EMBED_CELLS
(``python -m fisseq_common.stages.<stage>``) with its own configuration:

- QC_FILTER reads BUILD_CELL_METADATA's ``metadata.parquet`` (already ``meta_*``-named, with a
  per-tile ``meta_cell_index``) and sorts on the row keys;
- a cell is ``EMBEDDINGS_JOIN_KEYS`` (batch, well, tile, per-tile cell index), so the same
  ``meta_cell_index`` recurs across tiles;
- the cell table is EMBED_CELLS' ``embeddings.parquet`` (``feature_selector=embeddings``) or, on
  the CellProfiler track, BUILD_CP_FEATURES' ``cp_features.parquet``
  (``feature_selector=features``); neither carries QC's pseudo-variant rows or tags;
- the normalizer is fitted on the wildtype cells; the published aggregates are z-scored against
  the synonymous variants.

These tests run each stage in that configuration, from QC_FILTER to FINALIZE_FEATURE_SELECT,
the CLI-level ones with the exact arguments the Nextflow modules pass. Moved from the embeddings
pipeline's own stage wrappers, which these entry points replaced; the stages' general behaviour
is tested in the other ``test_*.py`` files here.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fisseq_common.normalizer import Normalizer
from fisseq_common.schema import (
    CONTROL_COLUMN_NAME,
    IMPACT_SCORE_COL,
    META_VARIANT_TAG_COL,
)
from fisseq_common.stages.aggregate import AggregateConfig, run_aggregate
from fisseq_common.stages.blocklist import BlocklistParams, run_blocklist
from fisseq_common.stages.combineblocklists import (
    CombineBlocklistsParams,
    run_combine_blocklists,
)
from fisseq_common.stages.config import EMBEDDINGS_JOIN_KEYS, CellsInput, row_keys
from fisseq_common.stages.correlatefeatures import (
    CorrelateFeaturesParams,
    run_correlate_features,
)
from fisseq_common.stages.filter import (
    FilterParams,
    load_cells,
    mark_controls,
    run_filter,
)
from fisseq_common.stages.finalize import FinalizeConfig, run_finalize
from fisseq_common.stages.generatesplit import GenerateSplitParams, run_generate_split
from fisseq_common.stages.qcfilter import QcFilterParams, run_qc_filter

LABEL = "meta_aa_changes"
JOIN_KEYS = list(EMBEDDINGS_JOIN_KEYS)
ROW_KEYS = row_keys(EMBEDDINGS_JOIN_KEYS)

# conf/modules.config's arguments (without the shell quoting).
JOIN_KEYS_ARG = "join_keys=[meta_batch,meta_well,meta_tile,meta_cell_index]"
QC_SORT_ARG = (
    "sort_output_by=[meta_batch,meta_well,meta_tile,meta_cell_index,meta_variant_tag]"
)

# Per-label offset of emb_0000: the one dimension that is reproducible across bootstrap
# halves. WT is the cell-level control; A1A and A2A, the synonymous variants, are what the
# aggregates are z-scored against.
OFFSETS = {"WT": 0.0, "A1A": 0.5, "A2A": -0.5, "M1K": 4.0, "L2P": -4.0}


# ---------------------------------------------------------------------------
# Fixtures: the embeddings pipeline's inputs
# ---------------------------------------------------------------------------


def _metadata(n_per_label: int = 12, labels=tuple(OFFSETS)) -> pl.DataFrame:
    """BUILD_CELL_METADATA's ``metadata.parquet``: the seven ``meta_*`` columns, cells spread
    over two tiles, so every ``meta_cell_index`` appears twice -- only the composite key
    identifies a cell. Three barcodes per label."""
    rows = []
    next_index = {"tile0x0y": 0, "tile0x1y": 0}
    for label in labels:
        for i in range(n_per_label):
            tile = "tile0x0y" if i % 2 == 0 else "tile0x1y"
            rows.append(
                {
                    "meta_batch": "batch1",
                    "meta_well": "well1",
                    "meta_tile": tile,
                    "meta_cell_index": next_index[tile],
                    "meta_barcode": f"{label}_bc{i % 3}",
                    LABEL: label,
                    "meta_edit_distance": 0,
                }
            )
            next_index[tile] += 1
    return pl.DataFrame(rows)


def _embeddings(metadata: pl.DataFrame, n_dims: int = 4, seed: int = 0) -> pl.DataFrame:
    """EMBED_CELLS' ``embeddings.parquet``: the metadata plus ``emb_NNNN``. ``emb_0000``
    carries a per-label offset (:data:`OFFSETS`); the other dimensions are noise."""
    rng = np.random.default_rng(seed)
    n = metadata.height
    offsets = np.array([OFFSETS[label] for label in metadata[LABEL]])
    columns = {"emb_0000": offsets + rng.normal(0, 0.3, size=n)}
    for d in range(1, n_dims):
        columns[f"emb_{d:04d}"] = rng.normal(0, 1.0, size=n)
    return metadata.with_columns(**{k: pl.Series(v) for k, v in columns.items()})


def _cp_features(metadata: pl.DataFrame, seed: int = 1) -> pl.DataFrame:
    """BUILD_CP_FEATURES' ``cp_features.parquet``: the metadata plus CellProfiler columns."""
    rng = np.random.default_rng(seed)
    offsets = np.array([OFFSETS[label] for label in metadata[LABEL]])
    return metadata.with_columns(
        Cells_AreaShape_Area=pl.Series(
            100 + 10 * offsets + rng.normal(0, 1, size=metadata.height)
        ),
        Cells_Intensity_MeanIntensity_DNA=pl.Series(
            rng.normal(0, 1, size=metadata.height)
        ),
    )


def _qc_filter(tmp_path: Path, metadata: pl.DataFrame, **overrides) -> Path:
    """QC_FILTER as the embeddings pipeline runs it, every threshold at 1; returns
    ``filtered_cells.parquet``."""
    out = tmp_path / "qc_filter"
    out.mkdir(exist_ok=True)
    source = tmp_path / "metadata.parquet"
    metadata.write_parquet(source)
    fields = dict(
        output_dir=str(out),
        cell_files=[str(source)],
        bc_threshold=1,
        variant_bc_threshold=1,
        sort_output_by=ROW_KEYS,
    )
    run_qc_filter(QcFilterParams(**{**fields, **overrides}))
    return out / "filtered_cells.parquet"


def _normalize(
    tmp_path: Path, cells: pl.DataFrame, qc_passed: Path, name: str = "embeddings"
) -> CellsInput:
    """NORMALIZE (the filter stage) with the embeddings join keys and the default wildtype
    control; returns the three files the downstream stages rebuild the cells from."""
    cells_path = tmp_path / f"{name}.parquet"
    cells.write_parquet(cells_path)
    out = tmp_path / f"normalization_{name}"
    out.mkdir(exist_ok=True)
    run_filter(
        FilterParams(
            output_dir=str(out),
            cells_file=str(cells_path),
            qc_passed_file=str(qc_passed),
            join_keys=JOIN_KEYS,
        )
    )
    return CellsInput(
        cells_file=str(cells_path),
        filtered_keys_file=str(out / "filtered_keys.parquet"),
        normalizer_file=str(out / "normalizer.parquet"),
        join_keys=JOIN_KEYS,
        feature_selector="embeddings" if name == "embeddings" else "features",
    )


@pytest.fixture
def cells(tmp_path: Path) -> CellsInput:
    """The cellDINO track's normalized cells: every cell passes QC."""
    metadata = _metadata()
    return _normalize(tmp_path, _embeddings(metadata), _qc_filter(tmp_path, metadata))


def _aggregate(
    cells: CellsInput, out: Path, aggregator: str, **overrides
) -> pl.DataFrame:
    """One AGGREGATE task, in process; returns its ``<output_name>.parquet``."""
    out.mkdir(parents=True, exist_ok=True)
    cfg = AggregateConfig(
        output_dir=str(out),
        cells_file=cells.cells_file,
        filtered_keys_file=cells.filtered_keys_file,
        normalizer_file=cells.normalizer_file,
        join_keys=cells.join_keys,
        feature_selector=cells.feature_selector,
        aggregator=aggregator,
        output_name=aggregator,
        **overrides,
    )
    run_aggregate(cfg)
    return pl.read_parquet(out / f"{aggregator}.parquet")


def _run_cli(tmp_path: Path, stage: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", f"fisseq_common.stages.{stage}", *args],
        capture_output=True,
        text=True,
        # Keeps Hydra's own outputs/<date>/<time>/ dir out of the repo tree.
        cwd=tmp_path,
    )


# ---------------------------------------------------------------------------
# QC_FILTER on metadata.parquet
# ---------------------------------------------------------------------------


def test_qc_filter_cli_sorts_on_the_row_keys_and_keeps_the_per_tile_cell_index(
    tmp_path: Path,
):
    """QC doesn't reassign meta_cell_index (unlike in the data pipeline), and
    filtered_cells.parquet comes out sorted on the row keys: a stable row order for every
    seeded step downstream."""
    metadata = _metadata(n_per_label=4).sample(fraction=1.0, shuffle=True, seed=3)
    source = tmp_path / "metadata.parquet"
    metadata.write_parquet(source)
    output_dir = tmp_path / "out"

    result = _run_cli(
        tmp_path,
        "qcfilter",
        f"output_dir={output_dir}",
        f"cell_files=[{source}]",
        "bc_threshold=1",
        "variant_bc_threshold=1",
        QC_SORT_ARG,
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr

    filtered = pl.read_parquet(output_dir / "filtered_cells.parquet")
    assert set(ROW_KEYS) <= set(filtered.columns)
    assert filtered.equals(filtered.sort(ROW_KEYS, nulls_last=False))
    assert (
        filtered.select(JOIN_KEYS)
        .sort(JOIN_KEYS)
        .equals(metadata.select(JOIN_KEYS).sort(JOIN_KEYS))
    )
    for name in ("barcode_counts", "variants_per_barcode"):
        assert (output_dir / f"{name}.parquet").exists()


def test_qc_filter_n_variants_restricts_before_the_qc_thresholds(tmp_path: Path):
    """M1K alone would fail variant_bc_threshold=2, but n_variants=1 ("top") drops it before
    the thresholds run, so variants_per_barcode never sees it either."""
    metadata = pl.DataFrame(
        {
            "meta_batch": ["batch1"] * 4,
            "meta_well": ["well1"] * 4,
            "meta_tile": ["tile0x0y"] * 4,
            "meta_cell_index": [0, 1, 2, 3],
            "meta_barcode": ["bc0", "bc1", "bc2", "bc3"],
            LABEL: ["M1K", "M2L", "M2L", "M2L"],
            "meta_edit_distance": [0] * 4,
        }
    )
    filtered_path = _qc_filter(tmp_path, metadata, variant_bc_threshold=2, n_variants=1)

    assert set(pl.read_parquet(filtered_path)[LABEL]) == {"M2L"}
    variants = pl.read_parquet(filtered_path.parent / "variants_per_barcode.parquet")
    assert "M1K" not in variants[LABEL].to_list()


def test_qc_filter_allow_list_is_ignored_without_n_variants(tmp_path: Path, caplog):
    metadata = _metadata(n_per_label=3, labels=("M1K", "M2L"))
    allow_list = tmp_path / "allow_list.parquet"
    pl.DataFrame({LABEL: ["M1K"]}).write_parquet(allow_list)

    with caplog.at_level("WARNING"):
        filtered_path = _qc_filter(
            tmp_path, metadata, variant_allow_list_file=str(allow_list)
        )

    assert set(pl.read_parquet(filtered_path)[LABEL]) == {"M1K", "M2L"}
    assert "variant_allow_list_file" in caplog.text


def test_qc_pseudo_variant_rows_share_the_cell_keys_but_not_the_row_keys(
    tmp_path: Path,
):
    """A pseudo-variant row is a copy of a cell under its own label and tag: the same
    EMBEDDINGS_JOIN_KEYS, a distinct row (ROW_KEYS adds the tag)."""
    filtered = pl.read_parquet(
        _qc_filter(
            tmp_path,
            _metadata(n_per_label=4),
            downsample_amounts=1.0,
            downsample_classes=["Synonymous"],
        )
    )
    tagged = filtered.filter(pl.col(META_VARIANT_TAG_COL).is_not_null())
    assert set(tagged[LABEL]) == {"A1A:downsample-1.0", "A2A:downsample-1.0"}
    assert tagged.height == 8
    assert filtered.select(JOIN_KEYS).is_duplicated().sum() == 2 * tagged.height
    assert not filtered.select(ROW_KEYS).is_duplicated().any()


# ---------------------------------------------------------------------------
# NORMALIZE: embeddings.parquet / cp_features.parquet + QC's keys
# ---------------------------------------------------------------------------


def _filtered_keys(cells: CellsInput) -> pl.DataFrame:
    return pl.read_parquet(cells.filtered_keys_file)


def test_filtered_keys_carry_qc_metadata_and_no_features(cells: CellsInput):
    """No stage copies another's data: the keys hold every meta_* column (QC's side) and
    meta_is_control, never an emb_* column."""
    keys = _filtered_keys(cells)
    assert not any(c.startswith("emb_") for c in keys.columns)
    assert set(ROW_KEYS) | {LABEL, "meta_barcode", CONTROL_COLUMN_NAME} <= set(
        keys.columns
    )
    assert keys.equals(keys.sort(ROW_KEYS, nulls_last=False))


def test_only_qc_passed_cells_are_kept(tmp_path: Path):
    """A cell absent from QC's output never reaches the keys or the normalizer fit, however
    extreme its features."""
    metadata = _metadata()
    embeddings = _embeddings(metadata).with_columns(
        emb_0000=pl.when((pl.col(LABEL) == "WT") & (pl.col("meta_cell_index") == 0))
        .then(999.0)
        .otherwise(pl.col("emb_0000"))
    )
    qc_passed = _qc_filter(tmp_path, metadata)
    failed = (pl.col(LABEL) == "WT") & (pl.col("meta_cell_index") == 0)
    pl.read_parquet(qc_passed).filter(~failed).write_parquet(qc_passed)

    cells = _normalize(tmp_path, embeddings, qc_passed)

    keys = _filtered_keys(cells)
    assert keys.height == metadata.height - 2  # cell 0 of both tiles
    wt = embeddings.filter((pl.col(LABEL) == "WT") & ~failed)
    normalizer = Normalizer.load(cells.normalizer_file)
    assert normalizer.means["emb_0000"][0] == pytest.approx(wt["emb_0000"].mean())


def test_wildtype_cells_are_the_controls(cells: CellsInput):
    """The default control is WT; the synonymous variants are ordinary rows."""
    keys = _filtered_keys(cells)
    assert keys[CONTROL_COLUMN_NAME].to_list() == (keys[LABEL] == "WT").to_list()


def test_normalizer_is_fitted_on_the_wildtype_cells(cells: CellsInput):
    embeddings = pl.read_parquet(cells.cells_file)
    wt = embeddings.filter(pl.col(LABEL) == "WT")
    normalizer = Normalizer.load(cells.normalizer_file)
    for column in ("emb_0000", "emb_0001"):
        assert normalizer.means[column][0] == pytest.approx(wt[column].mean())
        assert normalizer.stds[column][0] == pytest.approx(wt[column].std())


def test_rebuilt_cells_are_normalized_and_keyed_by_the_composite_key(
    cells: CellsInput,
):
    """load_cells joins every cell to its own features: meta_cell_index alone recurs across
    the two tiles, so a join on it would cross cells."""
    embeddings = pl.read_parquet(cells.cells_file)
    assert embeddings["meta_cell_index"].is_duplicated().any()
    wt = embeddings.filter(pl.col(LABEL) == "WT")["emb_0000"]

    rebuilt = load_cells(cells).collect()

    assert rebuilt.height == embeddings.height
    expected = embeddings.sort(JOIN_KEYS).select(
        (pl.col("emb_0000") - wt.mean()) / wt.std()
    )
    np.testing.assert_allclose(
        rebuilt["emb_0000"].to_numpy(), expected["emb_0000"].to_numpy()
    )
    assert len(rebuilt.columns) == len(set(rebuilt.columns))
    assert not any(c.endswith("_right") for c in rebuilt.columns)
    assert rebuilt.equals(rebuilt.sort(ROW_KEYS, nulls_last=False))


def test_pseudo_variant_rows_get_their_source_cells_features(tmp_path: Path):
    """embeddings.parquet has no pseudo-variant rows: each joins on its source cell's keys and
    keeps QC's label and tag."""
    metadata = _metadata(n_per_label=4)
    qc_passed = _qc_filter(
        tmp_path, metadata, downsample_amounts=1.0, downsample_classes=["Synonymous"]
    )
    cells = _normalize(tmp_path, _embeddings(metadata), qc_passed)

    rebuilt = load_cells(cells).collect()
    tagged = rebuilt.filter(pl.col(META_VARIANT_TAG_COL).is_not_null())
    source = rebuilt.filter(pl.col(META_VARIANT_TAG_COL).is_null())
    assert tagged.height == 8
    paired = tagged.join(source, on=JOIN_KEYS, suffix="_source")
    assert paired.height == tagged.height
    assert paired["emb_0000"].to_list() == paired["emb_0000_source"].to_list()
    assert (paired[LABEL] == paired[f"{LABEL}_source"] + ":downsample-1.0").all()


def test_rebuilt_cells_equal_a_single_step_join_and_normalize(cells: CellsInput):
    """Splitting the filter stage (keys + normalizer) from the rebuild (load_cells) is
    output-equivalent to joining, marking and normalizing in one step."""
    embeddings = pl.scan_parquet(cells.cells_file).select(
        *JOIN_KEYS, pl.col("^emb_.*$")
    )
    qc = pl.scan_parquet(cells.filtered_keys_file).drop(CONTROL_COLUMN_NAME)
    one_step = mark_controls(
        qc.join(embeddings, on=JOIN_KEYS), "meta_aa_changes = 'WT'", LABEL
    )
    expected = (
        Normalizer.from_lazyframe(one_step, fit_only_on_control=True)
        .apply(one_step)
        .collect()
        .sort(ROW_KEYS, nulls_last=False)
    )

    rebuilt = load_cells(cells).collect()

    assert set(rebuilt.columns) == set(expected.columns)
    assert rebuilt.select(sorted(rebuilt.columns)).equals(
        expected.select(sorted(expected.columns))
    )


@pytest.mark.parametrize(
    "track, feature",
    [("embeddings", "emb_0000"), ("cp_features", "Cells_AreaShape_Area")],
)
def test_filter_cli_with_the_embeddings_join_keys(tmp_path: Path, track, feature):
    """NORMALIZE / NORMALIZE_CP_FEATURES, with conf/modules.config's arguments."""
    metadata = _metadata(n_per_label=4)
    cells = _embeddings(metadata) if track == "embeddings" else _cp_features(metadata)
    cells_path = tmp_path / f"{track}.parquet"
    cells.write_parquet(cells_path)
    qc_passed = _qc_filter(tmp_path, metadata)
    output_dir = tmp_path / "out"

    result = _run_cli(
        tmp_path,
        "filter",
        f"output_dir={output_dir}",
        f"cells_file={cells_path}",
        f"qc_passed_file={qc_passed}",
        f"label_column={LABEL}",
        JOIN_KEYS_ARG,
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr

    keys = pl.read_parquet(output_dir / "filtered_keys.parquet")
    assert keys.height == metadata.height
    assert feature not in keys.columns
    normalizer = Normalizer.load(output_dir / "normalizer.parquet")
    wt = cells.filter(pl.col(LABEL) == "WT")[feature]
    assert normalizer.means[feature][0] == pytest.approx(wt.mean())


# ---------------------------------------------------------------------------
# AGGREGATE: AGGREGATE_FEATURE_TYPE / _PASSTHROUGH / _HALF / _CP_FEATURES
# ---------------------------------------------------------------------------


def test_aggregate_feature_type_cli(tmp_path: Path, cells: CellsInput):
    """AGGREGATE_FEATURE_TYPE_BATCHWISE: one file per method, emb_NNNN_<method> columns, the
    wildtype cells excluded, every other variant (synonymous ones included) a row, z-scored
    against the synonymous variants."""
    output_dir = tmp_path / "aggregates"
    result = _run_cli(
        tmp_path,
        "aggregate",
        f"output_dir={output_dir}",
        f"cells_file={cells.cells_file}",
        f"filtered_keys_file={cells.filtered_keys_file}",
        f"normalizer_file={cells.normalizer_file}",
        "aggregator=median",
        f"label_column={LABEL}",
        "feature_chunk_size=32",
        "output_name=median",
        JOIN_KEYS_ARG,
        "feature_selector=embeddings",
        "downsample_wt=null",
        "normalize_to_synonymous=true",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr

    agg = pl.read_parquet(output_dir / "median.parquet")
    assert agg.columns == [LABEL] + [f"emb_{d:04d}_median" for d in range(4)]
    assert agg[LABEL].to_list() == ["A1A", "A2A", "L2P", "M1K"]
    synonymous = agg.filter(pl.col(LABEL).is_in(["A1A", "A2A"])).drop(LABEL)
    for column in synonymous.columns:
        assert synonymous[column].mean() == pytest.approx(0.0, abs=1e-9)
        assert synonymous[column].std() == pytest.approx(1.0)


def test_embeddings_selector_aggregates_only_the_emb_dimensions(
    tmp_path: Path, cells: CellsInput
):
    """feature_selector=embeddings: a non-emb, non-meta column is not a feature."""
    cells_df = pl.read_parquet(cells.cells_file).with_columns(stray=pl.lit(1.0))
    cells_df.write_parquet(cells.cells_file)
    agg = _aggregate(cells, tmp_path / "agg", "KS")
    assert agg.columns == [LABEL] + [f"emb_{d:04d}_KS" for d in range(4)]


def test_aggregate_cp_features_uses_every_non_meta_column(tmp_path: Path):
    """AGGREGATE_FEATURE_TYPE_CP_FEATURES: feature_selector=features."""
    metadata = _metadata()
    cp = _normalize(
        tmp_path, _cp_features(metadata), _qc_filter(tmp_path, metadata), "cp_features"
    )
    agg = _aggregate(cp, tmp_path / "agg", "median", normalize_to_synonymous=True)
    assert agg.columns == [
        LABEL,
        "Cells_AreaShape_Area_median",
        "Cells_Intensity_MeanIntensity_DNA_median",
    ]
    assert "WT" not in agg[LABEL].to_list()


def test_passthrough_aggregates_keep_their_own_scale(tmp_path: Path, cells: CellsInput):
    """AGGREGATE_FEATURE_TYPE_PASSTHROUGH skips the z-score: -log10 p-values stay >= 0."""
    raw = _aggregate(cells, tmp_path / "raw", "KSnegLogP")
    zscored = _aggregate(
        cells, tmp_path / "zscored", "KSnegLogP", normalize_to_synonymous=True
    )
    values = raw.drop(LABEL).to_numpy()
    assert (values >= 0).all()
    assert not np.allclose(values, zscored.drop(LABEL).to_numpy())


def test_half_aggregate_is_restricted_to_the_split_and_lean(
    tmp_path: Path, cells: CellsInput
):
    """AGGREGATE_HALF_BATCHWISE: only the half's rows, and only the label column besides the
    statistics (a half's cell counts would be misleading)."""
    keys = _filtered_keys(cells)
    half = keys.filter(pl.col(LABEL).is_in(["WT", "M1K"])).head(18)
    split = tmp_path / "half1.parquet"
    half.select(ROW_KEYS).write_parquet(split)

    agg = _aggregate(cells, tmp_path / "half", "median", split_file=str(split))

    assert agg.columns == [LABEL] + [f"emb_{d:04d}_median" for d in range(4)]
    assert agg[LABEL].to_list() == ["M1K"]
    rebuilt = load_cells(cells).collect()
    m1k = rebuilt.join(half.select(ROW_KEYS), on=ROW_KEYS, nulls_equal=True).filter(
        pl.col(LABEL) == "M1K"
    )
    assert m1k.height == 6
    assert agg["emb_0000_median"][0] == pytest.approx(m1k["emb_0000"].median())


def test_half_split_names_rows_not_cells(tmp_path: Path):
    """A split names a row by ROW_KEYS: a pseudo-variant row and its source cell share the
    cell keys, and a half naming only the source keeps the pseudo-variant row out."""
    metadata = _metadata(n_per_label=4)
    qc_passed = _qc_filter(
        tmp_path, metadata, downsample_amounts=1.0, downsample_classes=["Synonymous"]
    )
    cells = _normalize(tmp_path, _embeddings(metadata), qc_passed)
    split = tmp_path / "untagged.parquet"
    _filtered_keys(cells).filter(pl.col(META_VARIANT_TAG_COL).is_null()).select(
        ROW_KEYS
    ).write_parquet(split)

    agg = _aggregate(cells, tmp_path / "half", "mean", split_file=str(split))

    assert agg[LABEL].to_list() == ["A1A", "A2A", "L2P", "M1K"]


# ---------------------------------------------------------------------------
# GENERATE_SPLIT
# ---------------------------------------------------------------------------


def test_generate_split_cli_names_rows_by_the_row_keys(
    tmp_path: Path, cells: CellsInput
):
    output_dir = tmp_path / "splits"
    result = _run_cli(
        tmp_path,
        "generatesplit",
        f"output_dir={output_dir}",
        f"filtered_keys_file={cells.filtered_keys_file}",
        f"label_column={LABEL}",
        "bootstrap_idx=2",
        JOIN_KEYS_ARG,
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr

    half1 = pl.read_parquet(output_dir / "half1.parquet")
    half2 = pl.read_parquet(output_dir / "half2.parquet")
    keys = _filtered_keys(cells)

    # A split file names rows; it never copies their data.
    assert half1.columns == ROW_KEYS
    assert half1.height + half2.height == keys.height
    assert half1.join(half2, on=ROW_KEYS, nulls_equal=True).height == 0
    for half in (half1, half2):
        counts = (
            half.join(keys, on=ROW_KEYS, nulls_equal=True)
            .group_by(LABEL)
            .len()
            .sort(LABEL)
        )
        # 12 cells per label, split 50/50.
        assert counts["len"].to_list() == [6] * len(OFFSETS)


# ---------------------------------------------------------------------------
# CORRELATE_FEATURES / BLOCKLIST / COMBINE_BLOCKLISTS, as the Nextflow modules call them
# ---------------------------------------------------------------------------


def test_correlate_features_cli(tmp_path: Path):
    labels = ["A1A", "A2A", "L2P", "M1K"]
    for name, values in (("h1", [1.0, 2.0, 3.0, 4.0]), ("h2", [2.0, 4.0, 6.0, 8.0])):
        pl.DataFrame({LABEL: labels, "emb_0000_KS": values}).write_parquet(
            tmp_path / f"{name}.parquet"
        )
    output_dir = tmp_path / "out"

    result = _run_cli(
        tmp_path,
        "correlatefeatures",
        f"output_dir={output_dir}",
        f"half1_file={tmp_path / 'h1.parquet'}",
        f"half2_file={tmp_path / 'h2.parquet'}",
        f"label_column={LABEL}",
        "output_name=KS",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr
    out = pl.read_parquet(output_dir / "KS.parquet")
    assert out["feature"].to_list() == ["emb_0000_KS"]
    assert out["r"][0] == pytest.approx(1.0)


def test_blocklist_cli_takes_the_median_over_the_staged_replicates(tmp_path: Path):
    """BLOCKLIST_BATCHWISE stages one method's bootstrap_N.parquet files under
    correlations/."""
    staged = tmp_path / "correlations"
    staged.mkdir()
    for rep, r in enumerate([0.9, 0.8, 0.1], start=1):
        pl.DataFrame(
            {
                "feature": ["emb_0000_KS", "emb_0001_KS"],
                "r": [r, 0.0],
                "r_squared": [r**2, 0.0],
            }
        ).write_parquet(staged / f"bootstrap_{rep}.parquet")
    output_dir = tmp_path / "out"

    result = _run_cli(
        tmp_path,
        "blocklist",
        f"output_dir={output_dir}",
        "correlation_files=correlations/*.parquet",
        "output_name=KS",
        "minimum_correlation=0.5",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr

    out = pl.read_parquet(output_dir / "KS.parquet")
    assert out["feature"].to_list() == ["emb_0000_KS", "emb_0001_KS"]
    assert out["feature_ok"].to_list() == [True, False]
    assert out["median_r"].to_list() == pytest.approx([0.8, 0.0])


@pytest.mark.parametrize(
    "stage, pattern_arg",
    [
        ("blocklist", "correlation_files=correlations/*.parquet"),
        ("combineblocklists", "blocklist_files=blocklists/*.parquet"),
    ],
)
def test_an_empty_glob_is_an_error(tmp_path: Path, stage, pattern_arg):
    """The globbed files are a declared task input: matching nothing is a wiring bug."""
    result = _run_cli(tmp_path, stage, f"output_dir={tmp_path / 'out'}", pattern_arg)
    assert result.returncode != 0
    assert "No files matched glob pattern" in result.stderr


def test_combine_blocklists_cli_keeps_each_methods_verdict(tmp_path: Path):
    """emb_0000 is reproducible as a median and not as a KS statistic: the stat suffixes keep
    the verdicts apart, so a plain concatenation is correct."""
    staged = tmp_path / "blocklists"
    staged.mkdir()
    for method, ok in (("median", [True, False]), ("KS", [False, True])):
        pl.DataFrame(
            {
                "feature": [f"emb_0000_{method}", f"emb_0001_{method}"],
                "median_r": [0.9 if o else 0.1 for o in ok],
                "feature_ok": ok,
            }
        ).write_parquet(staged / f"{method}.parquet")
    output_dir = tmp_path / "out"

    result = _run_cli(
        tmp_path,
        "combineblocklists",
        f"output_dir={output_dir}",
        "blocklist_files=blocklists/*.parquet",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr

    out = pl.read_parquet(output_dir / "blocklist.parquet")
    assert out["feature"].to_list() == [
        "emb_0000_KS",
        "emb_0000_median",
        "emb_0001_KS",
        "emb_0001_median",
    ]
    assert out["feature_ok"].to_list() == [False, True, True, False]


# ---------------------------------------------------------------------------
# The whole feature-selection chain, through FINALIZE_FEATURE_SELECT
# ---------------------------------------------------------------------------


def _feature_select(tmp_path: Path, cells: CellsInput, methods, passthrough, reps=3):
    """Every task of the cellDINO track's feature selection, in process, published under
    ``tmp_path / "fs"`` in fisseq_common.layout's ``feature_select_batchwise/<batch>/`` shape;
    returns that directory."""
    base = tmp_path / "fs"
    for method in methods:
        _aggregate(cells, base / "aggregates", method, normalize_to_synonymous=True)
    for method in passthrough:
        _aggregate(cells, base / "passthrough_aggregates", method)
    for rep in range(1, reps + 1):
        split_dir = base / "splits" / f"bootstrap_{rep}"
        split_dir.mkdir(parents=True)
        run_generate_split(
            GenerateSplitParams(
                output_dir=str(split_dir),
                filtered_keys_file=cells.filtered_keys_file,
                bootstrap_idx=rep,
                join_keys=cells.join_keys,
            )
        )
        for method in methods:
            half_dir = base / "half_aggregates" / f"bootstrap_{rep}" / method
            for half in (1, 2):
                _aggregate(
                    cells,
                    half_dir / f"half{half}",
                    method,
                    split_file=str(split_dir / f"half{half}.parquet"),
                    random_seed=rep * 2 + half,
                )
            corr_dir = base / "correlations" / method
            corr_dir.mkdir(parents=True, exist_ok=True)
            run_correlate_features(
                CorrelateFeaturesParams(
                    output_dir=str(corr_dir),
                    half1_file=str(half_dir / "half1" / f"{method}.parquet"),
                    half2_file=str(half_dir / "half2" / f"{method}.parquet"),
                    output_name=f"bootstrap_{rep}",
                )
            )
    (base / "blocklists").mkdir()
    for method in methods:
        run_blocklist(
            BlocklistParams(
                output_dir=str(base / "blocklists"),
                correlation_files=str(base / "correlations" / method / "*.parquet"),
                output_name=method,
            )
        )
    run_combine_blocklists(
        CombineBlocklistsParams(
            output_dir=str(base), blocklist_files=str(base / "blocklists" / "*.parquet")
        )
    )
    run_finalize(
        FinalizeConfig(
            output_dir=str(base),
            feature_type_files=str(base / "aggregates" / "*.parquet"),
            passthrough_feature_type_files=str(
                base / "passthrough_aggregates" / "*.parquet"
            ),
            block_list_file=str(base / "blocklist.parquet"),
            filtered_keys_file=cells.filtered_keys_file,
        )
    )
    return base


def test_feature_selection_chain_end_to_end(tmp_path: Path, cells: CellsInput):
    base = _feature_select(tmp_path, cells, ["median", "KS"], ["KSnegLogP"])

    blocklist = pl.read_parquet(base / "blocklist.parquet")
    aggregates = [
        pl.read_parquet(base / "aggregates" / f"{m}.parquet") for m in ("median", "KS")
    ]
    # The blocklist judges every aggregate column by name -- and no passthrough column.
    assert set(blocklist["feature"]) == {
        c for agg in aggregates for c in agg.columns if c != LABEL
    }
    verdicts = dict(zip(blocklist["feature"], blocklist["feature_ok"]))
    assert verdicts["emb_0000_median"] and verdicts["emb_0000_KS"]
    blocked = {f for f, ok in verdicts.items() if not ok}
    assert blocked, "the noise dimensions should not all be reproducible"

    output = pl.read_parquet(base / "output.parquet")
    assert output[LABEL].to_list() == ["A1A", "A2A", "L2P", "M1K"]
    assert output[CONTROL_COLUMN_NAME].to_list() == [True, True, False, False]
    assert output["meta_num_cells"].to_list() == [12] * 4
    assert output[IMPACT_SCORE_COL].null_count() == 0
    kept = {c for c in output.columns if not c.startswith("meta_")}
    assert kept == (set(verdicts) - blocked) | {
        f"emb_{d:04d}_KSnegLogP" for d in range(4)
    }


# ---------------------------------------------------------------------------
# OVWT_BATCHWISE / OVWT_BATCHWISE_CP_FEATURES
# ---------------------------------------------------------------------------


def _ovwt_inputs(tmp_path: Path, track: str) -> CellsInput:
    """WT (one barcode, 15 cells) and M1K (two barcodes x 15 cells), separable on the first
    feature. Sized for n_folds=3: each fold's inner train/calibration split wants comfortably
    more cells per (barcode, is_wt) stratum than there are folds."""
    rng = np.random.default_rng(3)
    labels = ["WT"] * 15 + ["M1K"] * 30
    n = len(labels)
    metadata = pl.DataFrame(
        {
            "meta_batch": ["batch1"] * n,
            "meta_well": ["well1"] * n,
            "meta_tile": ["tile0x0y"] * n,
            "meta_cell_index": list(range(n)),
            "meta_barcode": ["bc_wt"] * 15 + ["bc_v0"] * 15 + ["bc_v1"] * 15,
            LABEL: labels,
            "meta_edit_distance": [0] * n,
        }
    )
    signal = np.where(np.array(labels) == "WT", 6.0, 4.0) + rng.normal(0, 0.3, n)
    feature = "emb_0000" if track == "embeddings" else "Cells_AreaShape_Area"
    cells = metadata.with_columns(pl.Series(feature, signal))
    return _normalize(tmp_path, cells, _qc_filter(tmp_path, metadata), track)


def _run_ovwt_cli(tmp_path: Path, cells: CellsInput, *args: str) -> Path:
    output_dir = tmp_path / "ovwt"
    result = _run_cli(
        tmp_path,
        "ovwt",
        f"output_dir={output_dir}",
        f"cells_file={cells.cells_file}",
        f"filtered_keys_file={cells.filtered_keys_file}",
        f"normalizer_file={cells.normalizer_file}",
        f"label_column={LABEL}",
        "wt_label=WT",
        "calibrate=true",
        "min_cells=1",
        "downsample_wt=true",
        *args,
        JOIN_KEYS_ARG,
        f"feature_selector={cells.feature_selector}",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr
    return output_dir


@pytest.mark.parametrize("track", ["embeddings", "cp_features"])
def test_ovwt_cli_scores_each_variant_against_wildtype(tmp_path: Path, track):
    output_dir = _run_ovwt_cli(
        tmp_path, _ovwt_inputs(tmp_path, track), "cv_mode=kfold", "n_folds=3"
    )

    results = pl.read_parquet(output_dir / "results.parquet")
    assert results[LABEL].to_list() == ["M1K"]
    row = results.row(0, named=True)
    assert row["meta_n_barcodes"] == 2
    assert row["meta_n_cells"] == 45  # the variant's cells and the wildtype's
    assert row["auroc_pooled"] > 0.9
    cell_scores = pl.read_parquet(output_dir / "cell_scores.parquet")
    assert cell_scores.height == 45
    assert cell_scores["score"].null_count() == 0
    with open(output_dir / "models.pkl", "rb") as f:
        assert len(pickle.load(f)["M1K"]) == 3


def test_ovwt_cli_barcode_holdout_with_null_n_folds(tmp_path: Path):
    """``n_folds=null`` is what the module emits for a null ``params.ovwt_n_folds``: one fold
    per variant barcode."""
    output_dir = _run_ovwt_cli(
        tmp_path,
        _ovwt_inputs(tmp_path, "embeddings"),
        "cv_mode=barcode_holdout",
        "n_folds=null",
    )

    with open(output_dir / "models.pkl", "rb") as f:
        assert len(pickle.load(f)["M1K"]) == 2
    results = pl.read_parquet(output_dir / "results.parquet")
    assert len(results.row(0, named=True)["auroc_folds"]) == 2
