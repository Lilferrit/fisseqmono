"""GENERATE_SPLIT: one stratified 50/50 pseudo-replicate split of an experiment's cells.

Splits the QC-passed cells (the filter stage's ``filtered_keys.parquet``) into two halves,
stratified by variant label, for one bootstrap replicate. Each half is written as the cells'
join keys (:func:`fisseq_common.utils.splits.write_split`), and AGGREGATE_HALF selects a half
with a semi-join on those keys.

The split is seeded with ``random_seed + bootstrap_idx``, so every replicate draws a distinct,
reproducible split from the one shared seed.
"""

import dataclasses
import logging
import pathlib
from typing import Sequence

import polars as pl
import sklearn.model_selection
from omegaconf import MISSING

from fisseq_common.utils.splits import write_split

from .config import AppConfig


@dataclasses.dataclass
class GenerateSplitParams(AppConfig):
    """
    GENERATE_SPLIT settings shared by both pipelines.

    Attributes
    ----------
    filtered_keys_file : str
        The filter stage's ``filtered_keys.parquet``.
    label_column : str
        Variant label column the split is stratified on. Defaults to ``"meta_aa_changes"``.
    bootstrap_idx : int
        Bootstrap replicate number; the split is seeded with ``random_seed + bootstrap_idx``.
        Defaults to ``1``.
    """

    filtered_keys_file: str = MISSING
    label_column: str = "meta_aa_changes"
    bootstrap_idx: int = 1


def split_keys(
    keys_df: pl.DataFrame, label_column: str, join_keys: Sequence[str], seed: int
) -> "tuple[pl.DataFrame, pl.DataFrame]":
    """
    Split ``keys_df``'s cells 50/50, stratified by ``label_column``.

    Returns the two halves' ``join_keys`` columns, each in ``keys_df``'s row order.

    Raises
    ------
    ValueError
        If a label has fewer than two cells: it cannot be in both halves, and failing loudly
        beats silently biasing one half's aggregate.
    """
    singletons = (
        keys_df.group_by(label_column)
        .len()
        .filter(pl.col("len") < 2)[label_column]
        .to_list()
    )
    if singletons:
        raise ValueError(
            f"Cannot stratify a 50/50 split: {len(singletons)} label(s) have "
            f"fewer than 2 QC-passed cells, e.g. {sorted(singletons)[:5]}. "
            "Raise QC_FILTER's barcode/variant thresholds so every retained "
            "variant has at least two cells."
        )

    labels = keys_df[label_column].to_list()
    idx = list(range(keys_df.height))
    half1_idx, half2_idx = sklearn.model_selection.train_test_split(
        idx, stratify=labels, random_state=seed, test_size=0.5
    )
    key_only = keys_df.select(list(join_keys))
    return key_only[sorted(half1_idx)], key_only[sorted(half2_idx)]


def run_generate_split(cfg: GenerateSplitParams, join_keys: Sequence[str]) -> None:
    """
    Run the stage with ``cfg`` (``output_dir`` must exist): write ``{prefix}half1.parquet``
    and ``{prefix}half2.parquet``, with ``prefix`` = ``{output_root}.`` when set.
    """
    output_dir = pathlib.Path(cfg.output_dir)
    prefix = f"{cfg.output_root}." if cfg.output_root is not None else ""
    seed = cfg.random_seed + cfg.bootstrap_idx

    logging.info("Reading filtered keys from %s", cfg.filtered_keys_file)
    keys_df = pl.read_parquet(cfg.filtered_keys_file).select(
        list(join_keys) + [cfg.label_column]
    )
    half1, half2 = split_keys(keys_df, cfg.label_column, join_keys, seed)
    logging.info(
        "Bootstrap replicate %d (seed %d): %d + %d cells",
        cfg.bootstrap_idx,
        seed,
        half1.height,
        half2.height,
    )
    for name, half in (("half1", half1), ("half2", half2)):
        out_path = output_dir / f"{prefix}{name}.parquet"
        logging.info("Writing %s", out_path)
        write_split(half, out_path)

    logging.info("Done")
