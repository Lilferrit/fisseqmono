"""AGGREGATE_EMBEDDINGS: per-variant aggregation of the QC-passed, normalized embeddings.

The aggregators and :func:`~fisseq_common.stages.aggregate.aggregate_methods` live in
:mod:`fisseq_common.stages.aggregate`. This module is the Hydra entry point: it rebuilds the
normalized embeddings (:func:`.filter.load_filtered_embeddings`), runs every method in
``aggregators`` over the embedding dimensions (``EMBEDDING_SELECTOR``), joins per-variant
metadata, and writes ``aggregate.parquet``.

Reproducibility filtering is the separate chain GENERATE_SPLIT -> AGGREGATE_HALF ->
CORRELATE_FEATURES -> BLOCKLIST -> COMBINE_BLOCKLISTS -> FILTER_AGGREGATE downstream of this
stage. The two ``*negLogP`` methods are available but left out of ``params.aggregate_methods``'
default: they cost ~5.6x (KS) and ~2.9x (AUROC) their base statistic.
"""

import dataclasses
import logging
import pathlib
from typing import List, Optional, Sequence

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from fisseq_common.normalizer import Normalizer
from fisseq_common.schema import EMBEDDING_SELECTOR
from fisseq_common.stages.aggregate import (  # noqa: F401 (re-exported)
    _AGGREGATORS,
    DEFAULT_FEATURE_CHUNK_SIZE,
    AUROCAggregator,
    AUROCNegLogPValueAggregator,
    BaseAggregator,
    KSAggregator,
    KSNegLogPValueAggregator,
    MeanAggregator,
    MedianAggregator,
    ReferenceBasedAggregator,
    aggregate_methods,
)
from fisseq_common.utils.log import setup_logging

from .config import AppConfig
from .filter import load_filtered_embeddings


def aggregate_embeddings(
    filtered_lf: pl.LazyFrame,
    label_column: str,
    aggregators: Sequence[str] = ("median",),
    feature_selector: pl.Expr = EMBEDDING_SELECTOR,
    feature_chunk_size: Optional[int] = DEFAULT_FEATURE_CHUNK_SIZE,
    include_metadata: bool = True,
    bare_median: bool = True,
) -> pl.DataFrame:
    """
    :func:`fisseq_common.stages.aggregate.aggregate_methods` over the embedding dimensions.

    Same parameters; only ``feature_selector`` defaults to ``EMBEDDING_SELECTOR``.
    """
    return aggregate_methods(
        filtered_lf,
        label_column,
        aggregators,
        feature_selector=feature_selector,
        feature_chunk_size=feature_chunk_size,
        include_metadata=include_metadata,
        bare_median=bare_median,
    )


@dataclasses.dataclass
class AggregateEmbeddingsConfig(AppConfig):
    """
    Hydra structured configuration for AGGREGATE_EMBEDDINGS.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    AGGREGATE_EMBEDDINGS's own logic doesn't consume random_seed itself
    (every ported aggregator is deterministic), but every stage config
    inherits it uniformly.

    Attributes
    ----------
    embeddings_file : str
        Path to EMBED_CELLS' embeddings.parquet. Required.
    filtered_keys_file : str
        Path to FILTER_EMBEDDINGS' filtered_keys.parquet. Required.
    normalizer_file : str
        Path to FILTER_EMBEDDINGS' normalizer.parquet. Required.
    label_column : str
        Name of the variant label column. Defaults to ``"meta_aa_changes"``.
    aggregators : List[str]
        Aggregation method(s) to run, in order. One or more of
        ``"mean"``, ``"median"``, ``"KS"``, ``"AUROC"``, ``"KSnegLogP"``,
        ``"AUROCnegLogP"``. Defaults to
        ``["median", "KS", "AUROC"]`` (the two ``*negLogP`` methods are
        available but off by default -- see the module docstring for their
        cost) -- note this means output columns are
        suffixed by method (``emb_0000_median``, ``emb_0000_KS``,
        ``emb_0000_AUROC``, ...) by default; only the exact single-element
        selection ``["median"]`` produces bare ``emb_0000..emb_{D-1}``
        columns (see :func:`aggregate_embeddings`). Contrast
        AGGREGATE_CP_FEATURES' ``AggregateCpFeaturesConfig.aggregators``,
        whose default stays ``["median"]``.
    feature_chunk_size : int or None
        Embedding dimensions evaluated per Polars query -- a memory dial
        only, identical output at every value. Lower it if the task is
        OOM-killed; ``None`` disables chunking. Defaults to
        :data:`~fisseq_common.stages.aggregate.DEFAULT_FEATURE_CHUNK_SIZE`.
    """

    embeddings_file: str = MISSING
    filtered_keys_file: str = MISSING
    normalizer_file: str = MISSING
    label_column: str = "meta_aa_changes"
    aggregators: List[str] = dataclasses.field(
        default_factory=lambda: ["median", "KS", "AUROC"]
    )
    feature_chunk_size: Optional[int] = DEFAULT_FEATURE_CHUNK_SIZE


_cs = ConfigStore.instance()
_cs.store(name="aggregate_main", node=AggregateEmbeddingsConfig)


@hydra.main(version_base=None, config_path=None, config_name="aggregate_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: aggregate QC-passed, synonymous-corrected embeddings per variant.

    Reads ``embeddings_file``, ``filtered_keys_file``, and
    ``normalizer_file``, reconstructs the QC-passed, synonymous-corrected
    embedding table via :func:`fisseq_embeddings_pipeline.filter.load_filtered_embeddings`,
    calls :func:`aggregate_embeddings`, and writes
    ``{prefix}aggregate.parquet`` to ``output_dir``. No other file is
    written -- never a materialized copy of the QC-filtered or normalized
    embedding matrix itself.

    Output file
    ------------
    - ``{prefix}aggregate.parquet``

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.aggregate \\
            output_dir=./out \\
            embeddings_file=embeddings.parquet \\
            filtered_keys_file=filtered_keys.parquet \\
            normalizer_file=normalizer.parquet \\
            'aggregators=[mean,median]'
    """
    agg_cfg: AggregateEmbeddingsConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(agg_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg_cfg.output_dir = str(output_dir)
    setup_logging(agg_cfg, "aggregate")

    prefix = f"{agg_cfg.output_root}." if agg_cfg.output_root is not None else ""

    logging.info("Reading embeddings from %s", agg_cfg.embeddings_file)
    embeddings_lf = pl.scan_parquet(agg_cfg.embeddings_file)
    logging.info("Reading filtered keys from %s", agg_cfg.filtered_keys_file)
    filtered_keys_lf = pl.scan_parquet(agg_cfg.filtered_keys_file)
    logging.info("Loading normalizer from %s", agg_cfg.normalizer_file)
    normalizer = Normalizer.load(agg_cfg.normalizer_file)

    logging.info("Reconstructing QC-passed, synonymous-corrected embeddings")
    filtered_lf = load_filtered_embeddings(embeddings_lf, filtered_keys_lf, normalizer)

    logging.info(
        "Aggregating via %s (feature_chunk_size=%s)",
        agg_cfg.aggregators,
        agg_cfg.feature_chunk_size,
    )
    agg_df = aggregate_embeddings(
        filtered_lf,
        agg_cfg.label_column,
        agg_cfg.aggregators,
        feature_chunk_size=agg_cfg.feature_chunk_size,
    )

    out_path = output_dir / f"{prefix}aggregate.parquet"
    logging.info("Writing %s", out_path)
    agg_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
