"""GENERATE_SPLIT: stratified 50/50 splits of the filtered keys, written as cell keys."""

import polars as pl
import pytest

from fisseq_common.stages.generatesplit import (
    GenerateSplitParams,
    run_generate_split,
    split_keys,
)
from fisseq_common.utils.splits import filter_by_split_file

KEYS = ["meta_cell_index", "meta_variant_tag"]


def _keys(n_per_label: int = 6) -> pl.DataFrame:
    labels = ["WT", "A1A", "M1K"]
    rows = [(i, None, label) for i, label in enumerate(labels * n_per_label)]
    # a pseudo-variant row sharing cell 0's index under a tag
    rows.append((0, "downsample-1", "WT"))
    return pl.DataFrame(
        rows,
        schema={
            "meta_cell_index": pl.Int64,
            "meta_variant_tag": pl.String,
            "meta_aa_changes": pl.String,
        },
        orient="row",
    )


def test_halves_partition_the_cells():
    h1, h2 = split_keys(_keys(), "meta_aa_changes", KEYS, seed=1)
    assert h1.height + h2.height == _keys().height
    assert h1.join(h2, on=KEYS, how="inner", nulls_equal=True).height == 0


def test_split_is_stratified():
    h1, _ = split_keys(_keys(), "meta_aa_changes", KEYS, seed=1)
    labelled = h1.join(_keys(), on=KEYS, nulls_equal=True)
    counts = labelled.group_by("meta_aa_changes").len()
    assert counts.filter(pl.col("meta_aa_changes") != "WT")["len"].to_list() == [3, 3]


def test_same_seed_reproduces_and_another_seed_differs():
    a, _ = split_keys(_keys(20), "meta_aa_changes", KEYS, seed=1)
    b, _ = split_keys(_keys(20), "meta_aa_changes", KEYS, seed=1)
    c, _ = split_keys(_keys(20), "meta_aa_changes", KEYS, seed=2)
    assert a.equals(b)
    assert not a.equals(c)


def test_singleton_label_raises():
    keys = pl.concat(
        [
            _keys(),
            pl.DataFrame([(99, None, "Q9Z")], schema=_keys().schema, orient="row"),
        ]
    )
    with pytest.raises(ValueError, match="fewer than 2"):
        split_keys(keys, "meta_aa_changes", KEYS, seed=1)


def test_run_writes_halves_that_select_cells_by_key(tmp_path):
    keys_path = tmp_path / "filtered_keys.parquet"
    _keys().write_parquet(keys_path)
    cfg = GenerateSplitParams(
        output_dir=str(tmp_path), filtered_keys_file=str(keys_path), bootstrap_idx=2
    )
    run_generate_split(cfg, KEYS)
    half1 = pl.read_parquet(tmp_path / "half1.parquet")
    assert half1.columns == KEYS
    # the semi-join matches untagged cells (null tag) too
    cells = _keys().with_columns(pl.lit(1.0).alias("f1")).lazy()
    selected = filter_by_split_file(cells, tmp_path / "half1.parquet", KEYS).collect()
    assert selected.height == half1.height


def test_seed_is_random_seed_plus_bootstrap_idx(tmp_path):
    keys_path = tmp_path / "filtered_keys.parquet"
    _keys(20).write_parquet(keys_path)
    outs = []
    for seed, idx in ((0, 3), (2, 1)):
        out = tmp_path / f"s{seed}"
        out.mkdir()
        run_generate_split(
            GenerateSplitParams(
                output_dir=str(out),
                filtered_keys_file=str(keys_path),
                random_seed=seed,
                bootstrap_idx=idx,
            ),
            KEYS,
        )
        outs.append(pl.read_parquet(out / "half1.parquet"))
    assert outs[0].equals(outs[1])
