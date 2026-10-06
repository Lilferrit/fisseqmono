"""Lean per-feature-type cell-level aggregation.

Hydra entry point backing the Nextflow processes ``AGGREGATE_FEATURE_TYPE`` and
``AGGREGATE_HALF`` — shared by the feature-selection pipeline's stage 1 (full
aggregation) and stage 2b (per-pseudo-replicate-half aggregation). Also supports
optionally downsampling control (wildtype) rows before aggregation via
``downsample_wt``/``seed`` — see :func:`fisseq_common.stages.aggregate.downsample_control` —
and optionally z-scoring the per-variant output against the synonymous variants
via ``normalize_to_synonymous``.
"""

import dataclasses
import logging
import pathlib
from typing import Optional, Union

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from fisseq_common.schema import FEATURE_SELECTOR
from fisseq_common.stages.aggregate import DEFAULT_FEATURE_CHUNK_SIZE, aggregate_cells
from fisseq_common.utils.log import setup_logging

from .cells import JOIN_KEYS, CellsInput, load_cells
from .config import LabeledInputConfig
from .utils.splits import filter_by_index_file

_cs = ConfigStore.instance()


@dataclasses.dataclass
class FeatureTypeAggregateConfig(CellsInput, LabeledInputConfig):
    """
    Hydra structured configuration for the lean per-feature-type aggregation
    entry point.

    Shared by the feature-selection pipeline's stage 1 (full aggregation)
    and stage 2b (per-pseudo-replicate-half aggregation).

    Attributes
    ----------
    aggregator : str
        A concrete key in ``fisseq_common.stages.aggregate._AGGREGATORS``
        (``mean``, ``median``, ``MAD``, ``std``, ``KS``, ``signedKS``, ``QQ``,
        ``AUROC``). Required.
    index_file : str or None
        Optional path to a single-column ``TMP_IDX_COL`` parquet file (as
        written by :func:`fisseq_data_pipeline.generatesplit.main`)
        naming a subset of cell-level rows to aggregate over (e.g. one
        pseudo-replicate half). When ``None``, all rows are aggregated.
        Defaults to ``None``.
    downsample_wt : float, int, or None
        Optional downsampling of control (wildtype) rows before aggregation.
        A float in ``(0, 1)`` keeps that fraction of control rows; an int
        keeps that many. ``None`` disables downsampling. Defaults to
        ``None``.
    feature_chunk_size : int or None
        Number of feature columns aggregated per Polars query. Lower it if a
        task is OOM-killed; ``None`` disables chunking entirely (every feature
        in one query). Defaults to
        :data:`fisseq_common.stages.aggregate.DEFAULT_FEATURE_CHUNK_SIZE`.
    normalize_to_synonymous : bool
        If ``True``, z-score every output stat column against the synonymous
        variants' rows (see Notes). Defaults to ``False``.

    Notes
    -----
    The ``downsample_wt`` draw is seeded from
    :attr:`~fisseq_data_pipeline.config.app.AppConfig.random_seed`. AGGREGATE_HALF
    passes ``random_seed + bootstrap_idx * 2 + half_num``, so a bootstrap
    replicate's two halves draw independent wildtype subsamples off the one
    shared seed.

    ``normalize_to_synonymous`` marks synonymous variants with
    :func:`fisseq_common.stages.filter.variant_classification`, fits a
    :class:`fisseq_common.normalizer.Normalizer` on those rows only (mean and ``ddof=1``
    std), applies it, and drops the ``meta_is_control`` column again so the
    output stays ``[label_column] + stat columns``. It needs at least two
    synonymous variants: with fewer the std is undefined and every stat
    column comes out null. AGGREGATE_FEATURE_TYPE sets it for
    ``feature_select_types`` and leaves it off for
    ``feature_select_passthrough_types`` (e.g. p-values, which must stay on
    their own scale); AGGREGATE_HALF leaves it off.
    """

    aggregator: str = MISSING
    index_file: Optional[str] = None
    downsample_wt: Optional[Union[float, int]] = None
    feature_chunk_size: Optional[int] = DEFAULT_FEATURE_CHUNK_SIZE
    normalize_to_synonymous: bool = False


_cs.store(name="aggregate_feature_type_main", node=FeatureTypeAggregateConfig)


@hydra.main(
    version_base=None, config_path=None, config_name="aggregate_feature_type_main"
)
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: aggregate cell-level features for one feature type.

    ``input_file`` is interpreted as a glob pattern via :func:`load_batches`
    (a concrete non-glob path is a single-file pattern). Rows are optionally
    filtered to ``index_file`` via :func:`.utils.splits.filter_by_index_file`.
    Runs the configured single aggregator via
    :func:`fisseq_common.stages.aggregate.aggregate_cells` and writes a lean output
    containing only ``[label_column] + <feature type's stat columns>`` — no
    metadata join, no impact score. With ``normalize_to_synonymous`` the stat
    columns are z-scored against the synonymous variants' rows first.

    The aggregator's reference/control group is the ``meta_is_control``
    column already present on the input (set upstream by ``normalize.py``'s
    WT-based ``control_sample_query``). The synonymous baseline used by
    ``normalize_to_synonymous`` is a different population, computed on the
    aggregated output.

    Output path
    -----------
    - Glob input: ``{output_root}.output.parquet`` or ``{output_dir}/output.parquet``
    - Single-file input: ``{output_root}.{stem}.parquet`` or
      ``{output_dir}/{stem}.parquet``

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.aggregatefeaturetype \\
            output_dir=./out \\
            input_file=data/normalized.parquet \\
            aggregator=mean \\
            index_file=./half1.parquet \\
            downsample_wt=0.5 \\
            seed=1
    """
    ft_cfg: FeatureTypeAggregateConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(ft_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ft_cfg.output_dir = output_dir
    setup_logging(ft_cfg, "aggregate_feature_type")

    logging.info("Loading input from %s", ft_cfg.input_file)
    lf, output_stem = load_cells(ft_cfg)

    logging.info("Filtering by index_file=%s", ft_cfg.index_file)
    lf = filter_by_index_file(lf, ft_cfg.index_file)

    agg_df = aggregate_cells(
        lf,
        ft_cfg.label_column,
        ft_cfg.aggregator,
        join_keys=JOIN_KEYS,
        feature_selector=FEATURE_SELECTOR,
        feature_chunk_size=ft_cfg.feature_chunk_size,
        downsample_controls=ft_cfg.downsample_wt,
        seed=ft_cfg.random_seed,
        normalize_to_synonymous=ft_cfg.normalize_to_synonymous,
    )

    if ft_cfg.output_root is not None:
        out_path = pathlib.Path(f"{ft_cfg.output_root}.{output_stem}.parquet")
    else:
        out_path = output_dir / f"{output_stem}.parquet"

    logging.info("Writing output to %s", out_path)
    agg_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
