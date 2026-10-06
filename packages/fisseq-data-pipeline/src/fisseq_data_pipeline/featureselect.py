"""Final feature-selection stage of the bootstrap pseudo-replicate pipeline.

Hydra entry point backing the Nextflow process ``FINALIZE_FEATURE_SELECT``: joins
per-feature-type aggregates (per-feature-type aggregation itself lives in
:mod:`.aggregatefeaturetype`, run as ``AGGREGATE_HALF``), applies the combined
blocklist (from :mod:`.combineblocklists`), and z-scores the result against the synonymous
variants. The blocklist is the only feature selection: there is no pycytominer filtering and no
PCA/UMAP (cross-experiment PCA is ``fisseqborn-global``'s).
"""

import dataclasses
import glob
import logging
import pathlib
from typing import Optional

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from fisseq_common.normalizer import Normalizer
from fisseq_common.stages.filter_aggregate import apply_blocklist, join_passthrough
from fisseq_common.utils.log import setup_logging
from fisseq_common.utils.metadata import get_aggregate_meta_data
from fisseq_common.utils.vectors import compute_impact_score

from .aggregate import variant_classification
from .cells import CellsInput, load_cells
from .config import LabeledInputConfig
from .utils.featuretypes import join_feature_type_files

_cs = ConfigStore.instance()


@dataclasses.dataclass
class FinalizeFeatureSelectConfig(CellsInput, LabeledInputConfig):
    """
    Hydra structured configuration for the final feature-selection entry
    point: joins per-feature-type aggregates, applies the combined
    blocklist, and z-scores the result against the synonymous variants.

    ``input_file`` (inherited) is the raw/normalized cell-level input, used
    only to derive per-variant metadata (:func:`.utils.metadata.get_aggregate_meta_data`).

    Attributes
    ----------
    feature_type_files : str
        Glob pattern matching the per-feature-type full aggregate parquet
        files produced by :func:`fisseq_data_pipeline.aggregatefeaturetype.main`.
        Each file contains ``[label_column] + <feature type's stat
        columns>``; all matching files are joined on ``label_column`` to
        reconstruct the combined per-variant aggregate table. Required.
    block_list_file : str
        Path to the combined blocklist parquet, with ``feature`` and
        ``feature_ok`` columns. Required.
    compute_impact_score : bool
        If ``True``, compute per-variant impact score (cosine distance vs
        synonymous baseline) after feature selection and normalization.
        Defaults to ``True``.
    passthrough_feature_type_files : str or None
        Optional glob pattern matching per-feature-type aggregate parquet
        files to join onto the output *without* running them through feature
        selection. Defaults to ``None`` (no passthrough columns).

    Notes
    -----
    ``passthrough_feature_type_files`` backs ``params.feature_select_passthrough_types``:
    aggregates that are wanted in the output but must not influence which
    features are kept -- p-value statistics (``KSnegLogP``, ``AUROCnegLogP``)
    above all, which must stay on their own scale. Those files are
    joined last, after the blocklist, normalization and impact score, so
    every one of those steps -- each of which picks its inputs with
    ``FEATURE_SELECTOR`` -- is blind to them. Unlike ``feature_type_files``, a
    glob matching nothing is a warning rather than an error: an empty
    passthrough list is the default, and it reaches this stage as an empty
    staging directory.
    """

    feature_type_files: str = MISSING
    block_list_file: str = MISSING
    compute_impact_score: bool = True
    passthrough_feature_type_files: Optional[str] = None


_cs.store(name="feature_select_main", node=FinalizeFeatureSelectConfig)


@hydra.main(version_base=None, config_path=None, config_name="feature_select_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: final feature-selection stage.

    Steps
    -----
    1. Load raw cell-level input at ``input_file`` (for metadata join only).
    2. Glob ``feature_type_files`` and join each per-feature-type aggregate
       parquet on ``label_column``, reconstructing the combined per-variant
       aggregate table.
    3. Load ``block_list_file`` and drop blocked feature columns.
    4. Mark synonymous variants as the normalization reference
       (:func:`fisseq_common.stages.filter.variant_classification`), fit a
       :class:`fisseq_common.normalizer.Normalizer` on those rows, and apply it — the
       output features are z-score normalized to the synonymous baseline.
    5. Optionally compute impact score on the normalized features
       (:func:`fisseq_common.utils.vectors.compute_impact_score`).
    6. Join per-variant metadata via :func:`fisseq_common.utils.metadata.get_aggregate_meta_data`.
    7. Join any ``passthrough_feature_type_files`` aggregates -- last, so that
       none of steps 4-5 ever saw them.
    8. Write output.

    Output path
    -----------
    - Glob input: ``{output_root}.output.parquet`` or ``{output_dir}/output.parquet``
    - Single-file input: ``{output_root}.{stem}.parquet`` or
      ``{output_dir}/{stem}.parquet``

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.featureselect \\
            output_dir=./out \\
            input_file=data/normalized.parquet \\
            'feature_type_files=./ft/*.parquet' \\
            block_list_file=./blocklist.parquet
    """
    feat_cfg: FinalizeFeatureSelectConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(feat_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_cfg.output_dir = output_dir
    setup_logging(feat_cfg, "features")

    logging.info("Loading raw input from %s", feat_cfg.input_file)
    lf, output_stem = load_cells(feat_cfg)

    logging.info(
        "Loading per-feature-type aggregates from %s", feat_cfg.feature_type_files
    )
    ft_paths = sorted(glob.glob(feat_cfg.feature_type_files))
    if not ft_paths:
        raise ValueError(
            f"No files matched glob pattern: {feat_cfg.feature_type_files!r}"
        )
    agg_df = join_feature_type_files(ft_paths, feat_cfg.label_column)

    logging.info("Loading block list from %s", feat_cfg.block_list_file)
    agg_df = apply_blocklist(agg_df, pl.read_parquet(feat_cfg.block_list_file))

    logging.info(
        "Classifying variants and marking synonymous as normalization reference"
    )
    selected_lf = variant_classification(agg_df.lazy(), feat_cfg.label_column)

    logging.info("Fitting normalizer on synonymous rows")
    normalizer = Normalizer.from_lazyframe(selected_lf, fit_only_on_control=True)
    logging.info("Applying normalizer")
    normalized_lf = normalizer.apply(selected_lf)

    if feat_cfg.compute_impact_score:
        logging.info("Computing impact scores")
        normalized_lf = compute_impact_score(normalized_lf)

    logging.info("Adding queries to retrieve metadata")
    meta_lf = get_aggregate_meta_data(lf, feat_cfg.label_column)
    selected_lf = normalized_lf.join(meta_lf, on=feat_cfg.label_column)

    # Deliberately last. Every step above -- the synonymous-baseline Normalizer and the
    # impact score -- picks its inputs with FEATURE_SELECTOR, so joining here is what keeps
    # the passthrough aggregates out of them. Moving this join earlier silently turns them
    # back into ordinary features.
    if feat_cfg.passthrough_feature_type_files:
        selected_lf = join_passthrough(
            selected_lf.collect(),
            feat_cfg.passthrough_feature_type_files,
            feat_cfg.label_column,
        ).lazy()

    if feat_cfg.output_root is not None:
        out_path = pathlib.Path(f"{feat_cfg.output_root}.{output_stem}.parquet")
    else:
        out_path = output_dir / f"{output_stem}.parquet"

    # One row per variant, sorted: the joins above don't preserve an order.
    logging.info("Writing output to %s", out_path)
    selected_lf.sort(feat_cfg.label_column).sink_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
