"""Hydra structured-config base classes every stage config extends, and the entry-point helper.

:class:`AppConfig` supplies ``output_dir``, ``output_root``, ``log_level`` and ``random_seed``;
:class:`InputConfig` adds ``input_file`` and :class:`LabeledInputConfig` adds ``label_column``.
:class:`CellsInput` is the three files a stage downstream of the filter stage rebuilds the
normalized cell table from, plus the pipeline's cell identity (``join_keys``) and which columns are
features (``feature_selector``).

Every shared stage is its own Hydra entry point, ``python -m fisseq_common.stages.<stage>``, made
by :func:`stage_main`. What differs between the pipelines is a config field, set by each
pipeline's ``conf/modules.config``.
"""

import dataclasses
import pathlib
from typing import Any, Callable, List, Optional

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from fisseq_common.schema import (
    EMBEDDING_SELECTOR,
    FEATURE_SELECTOR,
    META_CELL_INDEX_COL,
    META_VARIANT_TAG_COL,
)
from fisseq_common.utils.log import setup_logging

#: The data pipeline's cell identity: QC_FILTER's ``meta_cell_index`` (from the raw input row
#: order) and the variant tag, which tells a pseudo-variant row from its source cell. The
#: default of every ``join_keys`` field.
DATA_JOIN_KEYS: tuple[str, ...] = (META_CELL_INDEX_COL, META_VARIANT_TAG_COL)

#: The embeddings pipeline's cell identity: ``meta_cell_index`` is the cell's index within its
#: tile, unique only together with the batch, well and tile.
EMBEDDINGS_JOIN_KEYS: tuple[str, ...] = (
    "meta_batch",
    "meta_well",
    "meta_tile",
    META_CELL_INDEX_COL,
)

#: ``feature_selector`` values: every non-``meta_`` column, or only ``emb_NNNN`` dimensions.
FEATURE_SELECTORS: dict[str, pl.Expr] = {
    "features": FEATURE_SELECTOR,
    "embeddings": EMBEDDING_SELECTOR,
}


def feature_selector(name: str) -> pl.Expr:
    """The column selector named ``name`` (a key of :data:`FEATURE_SELECTORS`)."""
    try:
        return FEATURE_SELECTORS[name]
    except KeyError:
        raise ValueError(
            f"Unknown feature_selector {name!r}; expected one of {sorted(FEATURE_SELECTORS)}"
        ) from None


def row_keys(join_keys: "list[str] | tuple[str, ...]") -> list[str]:
    """
    The columns identifying one row of a cell table: ``join_keys`` plus the variant tag.

    ``join_keys`` find a cell's features. A QC pseudo-variant row is a copy of a cell under its
    own tag, so a row is identified by the cell *and* its tag: that is what the stages sort on
    and what a split file names.
    """
    keys = list(join_keys)
    if META_VARIANT_TAG_COL not in keys:
        keys.append(META_VARIANT_TAG_COL)
    return keys


@dataclasses.dataclass
class AppConfig:
    """
    Shared application-level configuration.

    Attributes
    ----------
    output_dir : str
        Directory for outputs produced by the current run (e.g. per-experiment
        results, normalized data, model artifacts). Required.
    output_root : str or None
        If set, every output file is prefixed ``{output_root}.{name}`` instead
        of being placed under ``output_dir``. Optional, defaults to ``None``.
    log_level : str
        Logging verbosity. One of ``debug``, ``info``, ``warning``, ``error``,
        ``critical``. Defaults to ``info``.
    random_seed : int
        The one seed every stochastic pipeline stage reads from -- QC
        pseudo-variant downsampling, the feature-selection bootstrap splits and
        their wildtype subsampling, OvWT's ``StratifiedKFold`` shuffle / inner
        calibration split / XGBoost ``seed``, PCA's solver, and UMAP's fit.
        Defaults to ``0``.

        Never add a stage-local ``random_state``/``seed`` field to a config
        that extends this one: a stage that must differ from its siblings
        derives a fixed offset from this seed instead (e.g. GENERATE_SPLIT uses
        ``random_seed + bootstrap_idx``), so changing this single value moves
        every stage coherently. ``tests/unit/test_config.py`` enforces this.
    """

    output_dir: str = MISSING
    output_root: Optional[str] = None
    log_level: str = "info"
    random_seed: int = 0


@dataclasses.dataclass
class InputConfig(AppConfig):
    """
    Extends AppConfig with a required input file path.

    Attributes
    ----------
    input_file : str
        Path to the input file. Required.
    """

    input_file: str = MISSING


@dataclasses.dataclass
class LabeledInputConfig(InputConfig):
    """
    Extends InputConfig for steps that operate on variant-labeled data.

    Attributes
    ----------
    label_column : str
        Name of the column identifying variant labels. Defaults to
        ``"meta_aa_changes"``.
    """

    label_column: str = "meta_aa_changes"


@dataclasses.dataclass
class CellsInput:
    """
    The inputs of a stage that reads the normalized cell table
    (:func:`fisseq_common.stages.filter.load_cells`).

    Attributes
    ----------
    cells_file : str
        The cell feature table: the data pipeline's QC_FILTER ``filtered_cells.parquet``, the
        embeddings pipeline's ``embeddings.parquet`` or ``cp_features.parquet``. Required.
    filtered_keys_file : str
        The filter stage's ``filtered_keys.parquet``. Required.
    normalizer_file : str
        The filter stage's ``normalizer.parquet``. Required.
    join_keys : list of str
        The columns ``cells_file`` and ``filtered_keys_file`` identify a cell by. Defaults to
        :data:`DATA_JOIN_KEYS`; the embeddings pipeline sets :data:`EMBEDDINGS_JOIN_KEYS`.
    feature_selector : str
        Which columns are features: ``"features"`` (every non-``meta_`` column) or
        ``"embeddings"`` (``emb_NNNN`` only). Defaults to ``"features"``.
    """

    cells_file: str = MISSING
    filtered_keys_file: str = MISSING
    normalizer_file: str = MISSING
    join_keys: List[str] = dataclasses.field(
        default_factory=lambda: list(DATA_JOIN_KEYS)
    )
    feature_selector: str = "features"


def prepare_output(cfg: Any, log_name: str) -> None:
    """Create ``cfg.output_dir`` (stored back as a string) and set up the stage's logging."""
    output_dir = pathlib.Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(output_dir)
    setup_logging(cfg, log_name)


def stage_main(name: str, node: type, run: Callable[[Any], None]) -> Callable[[], None]:
    """
    The Hydra entry point of a stage: registers ``node`` (the stage's config dataclass) as
    ``name`` and returns a ``main()`` that resolves the command-line overrides, prepares the
    output directory and logging (:func:`prepare_output`) and calls ``run(cfg)``.

    A stage module ends with::

        main = stage_main("<stage>_main", StageConfig, run_stage)

        if __name__ == "__main__":
            main()

    ``main.__wrapped__(dict_config)`` runs the stage without Hydra's command-line parsing.
    """
    ConfigStore.instance().store(name=name, node=node)
    log_name = name.removesuffix("_main")

    @hydra.main(version_base=None, config_path=None, config_name=name)
    def main(cfg: DictConfig) -> None:
        stage_cfg = OmegaConf.to_object(cfg)
        prepare_output(stage_cfg, log_name)
        run(stage_cfg)

    return main
