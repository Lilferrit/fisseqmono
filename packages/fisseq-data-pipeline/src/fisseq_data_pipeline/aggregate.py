"""Standalone per-variant aggregation (``python -m fisseq_data_pipeline.aggregate``).

The aggregators (mean, median, MAD, std, KS, signedKS, QQ, AUROC, KSnegLogP, AUROCnegLogP)
live in :mod:`fisseq_common.stages.aggregate` and are re-exported here. This entry point
aggregates with one of them, z-scores the result against the synonymous variants, attaches
per-variant metadata and an impact score. It isn't wired into the Nextflow workflow; the
feature-selection branch aggregates with :mod:`.aggregatefeaturetype`.
"""

import dataclasses
import logging
import pathlib
from typing import Optional

import hydra
import polars as pl
import polars.selectors as cs
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from fisseq_common.normalizer import Normalizer
from fisseq_common.schema import FEATURE_SELECTOR
from fisseq_common.stages.aggregate import (  # noqa: F401 (re-exported)
    _AGGREGATORS,
    DEFAULT_FEATURE_CHUNK_SIZE,
    AUROCAggregator,
    AUROCNegLogPValueAggregator,
    BaseAggregator,
    KSAggregator,
    KSNegLogPValueAggregator,
    MADAggregator,
    MeanAggregator,
    MedianAggregator,
    QQCorrelationAggregator,
    ReferenceBasedAggregator,
    SignedKSAggregator,
    StdAggregator,
    downsample_control,
)
from fisseq_common.stages.aggregate import aggregate as _aggregate
from fisseq_common.stages.filter import (
    variant_classification,  # noqa: F401 (re-exported)
)
from fisseq_common.utils.batches import load_batches
from fisseq_common.utils.log import setup_logging
from fisseq_common.utils.metadata import get_aggregate_meta_data
from fisseq_common.utils.vectors import compute_impact_score

from .config import LabeledInputConfig


@dataclasses.dataclass
class AggregateConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the aggregation entry point.

    Attributes
    ----------
    aggregator : str
        Aggregation method. One of: ``mean``, ``median``, ``MAD``, ``std``,
        ``KS``, ``signedKS``, ``QQ``, ``AUROC``, ``KSnegLogP``,
        ``AUROCnegLogP``. Required.
    save_normalizer : bool
        If ``True``, persist the fitted :class:`fisseq_common.normalizer.Normalizer` alongside
        the output. Defaults to ``True``.
    block_list_file : str or None
        Optional path to a parquet file with at least ``feature`` (str) and
        ``feature_ok`` (bool) columns. Features where ``feature_ok`` is
        ``False`` are excluded from aggregation. Defaults to ``None`` (no
        features blocked).
    feature_chunk_size : int or None
        Number of feature columns aggregated per Polars query. ``None``
        disables chunking (every feature in one query). Defaults to
        :data:`DEFAULT_FEATURE_CHUNK_SIZE`.
    """

    aggregator: str = MISSING
    save_normalizer: bool = True
    block_list_file: Optional[str] = None
    compute_impact_score: bool = True
    feature_chunk_size: Optional[int] = DEFAULT_FEATURE_CHUNK_SIZE


_cs = ConfigStore.instance()
_cs.store(name="aggregate_main", node=AggregateConfig)


def aggregate(
    lf: pl.LazyFrame,
    label_col: str,
    aggregator_name: str,
    block_list: Optional[set[str]] = None,
    feature_chunk_size: Optional[int] = DEFAULT_FEATURE_CHUNK_SIZE,
) -> pl.LazyFrame:
    """
    :func:`fisseq_common.stages.aggregate.aggregate` over the CellProfiler features, skipping
    block-listed statistics. See :func:`fisseq_common.stages.aggregate.aggregate` for the
    parameters other than ``block_list``.

    Parameters
    ----------
    block_list : set[str] or None
        Aggregated output column names to skip (e.g. ``"f1_KS"``). A blocked statistic is
        not computed and does not appear in the output; names matching no output are ignored.
    """
    if aggregator_name not in _AGGREGATORS:
        raise ValueError(
            f"Unknown aggregator {aggregator_name!r}. Choose from: {sorted(_AGGREGATORS)}"
        )
    selector = FEATURE_SELECTOR
    if block_list:
        suffix = _AGGREGATORS[aggregator_name]._stat_suffix
        blocked = [
            f
            for f in lf.select(FEATURE_SELECTOR).collect_schema().names()
            if f"{f}{suffix}" in block_list
        ]
        if blocked:
            selector = FEATURE_SELECTOR - cs.by_name(blocked)
    return _aggregate(
        lf,
        label_col,
        aggregator_name,
        feature_selector=selector,
        feature_chunk_size=feature_chunk_size,
    )


@hydra.main(version_base=None, config_path=None, config_name="aggregate_main")
def main(cfg: DictConfig) -> None:
    """
    Aggregate cell-level features, z-score normalize to synonymous baseline,
    and attach per-variant metadata.

    ``input_file`` is interpreted as a glob pattern via
    :func:`.utils.load_batches`; each matching file becomes one batch, with
    ``meta_batch`` set to the filename stem. A concrete (non-glob) path is
    treated as a single-file pattern.

    Runs the configured aggregator to produce one row per variant, marks
    synonymous variants as the normalization reference via
    :func:`variant_classification`, fits a :class:`fisseq_common.normalizer.Normalizer` on
    those rows, applies it, joins per-variant metadata from
    :func:`get_aggregate_meta_data`, and writes the result.

    Output path
    -----------
    - Glob input: ``{output_root}.output.parquet`` or ``{output_dir}/output.parquet``
    - Single-file input: ``{output_root}.{stem}.{ext}`` or
      ``{output_dir}/{filename}`` (same name as the input file)

    If ``save_normalizer`` is ``True``, the fitted :class:`fisseq_common.normalizer.Normalizer`
    is also written alongside the output as ``normalizer.parquet``.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.aggregate \\
            output_dir=./out \\
            'input_file=data/batches/*.parquet' \\
            aggregator=KS
    """
    agg_cfg: AggregateConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(agg_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg_cfg.output_dir = output_dir
    setup_logging(agg_cfg, "aggregate")

    logging.info("Loading input from %s", agg_cfg.input_file)
    lf, output_stem = load_batches(agg_cfg.input_file)

    block_list: Optional[set[str]] = None
    if agg_cfg.block_list_file is not None:
        logging.info("Loading block list from %s", agg_cfg.block_list_file)
        bl_df = pl.read_parquet(agg_cfg.block_list_file)
        block_list = set(bl_df.filter(~pl.col("feature_ok"))["feature"].to_list())

    logging.info(
        "Running %s aggregator (feature_chunk_size=%s)",
        agg_cfg.aggregator,
        agg_cfg.feature_chunk_size,
    )
    logging.info(
        "Classifying variants and marking synonymous as normalization reference"
    )
    agg_lf = variant_classification(
        aggregate(
            lf,
            label_col=agg_cfg.label_column,
            aggregator_name=agg_cfg.aggregator,
            block_list=block_list,
            feature_chunk_size=agg_cfg.feature_chunk_size,
        ),
        agg_cfg.label_column,
    )

    logging.info("Fitting normalizer on synonymous rows")
    normalizer = Normalizer.from_lazyframe(agg_lf, fit_only_on_control=True)

    logging.info("Applying normalizer")
    normalized_lf = normalizer.apply(agg_lf)

    if agg_cfg.output_root is not None:
        out_path = pathlib.Path(f"{agg_cfg.output_root}.{output_stem}.parquet")
    else:
        out_path = output_dir / f"{output_stem}.parquet"

    logging.info("Adding queries to retrieve metadata")
    meta_lf = get_aggregate_meta_data(lf, agg_cfg.label_column)
    normalized_lf = normalized_lf.join(meta_lf, on=agg_cfg.label_column)

    if cfg.compute_impact_score:
        logging.info("Computing impact scores")
        normalized_lf = compute_impact_score(normalized_lf)

    logging.info("Writing output to %s", out_path)
    normalized_lf.sink_parquet(out_path)

    if agg_cfg.save_normalizer:
        if agg_cfg.output_root is not None:
            norm_path = pathlib.Path(f"{agg_cfg.output_root}.normalizer.parquet")
        else:
            norm_path = output_dir / "normalizer.parquet"
        logging.info("Saving normalizer to %s", norm_path)
        normalizer.save(norm_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
