"""Z-score normalization of cell-level features against a control baseline.

Hydra entry point (``python -m fisseq_data_pipeline.normalize``) / Nextflow process ``NORMALIZE`` (second
pipeline stage). Fits the :class:`Normalizer` on rows matching a configurable SQL
control-sample query (default: WT cells) and applies it to every feature column,
producing normalized cell-level Parquet output plus an optional serialized
normalizer.
"""

import dataclasses
import logging
import pathlib

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from fisseq_common.normalizer import Normalizer
from fisseq_common.schema import (
    CONTROL_COLUMN_NAME,
)
from fisseq_common.utils.log import setup_logging

from .config import InputConfig


@dataclasses.dataclass
class NormalizeConfig(InputConfig):
    """
    Hydra structured configuration for the normalization entry point.

    Attributes
    ----------
    control_sample_query : str
        SQL-like WHERE clause identifying control rows used to fit the
        normalizer (e.g. ``"meta_aa_changes = 'WT'"``).
    save_normalizer : bool
        If ``True``, persist the fitted :class:`Normalizer` to a parquet file
        alongside the normalized output.
    """

    control_sample_query: str = "meta_aa_changes = 'WT'"
    save_normalizer: bool = True


_cs = ConfigStore.instance()
_cs.store(name="normalize_main", node=NormalizeConfig)


def add_control_indicator_column(
    lf: pl.LazyFrame, cfg: NormalizeConfig
) -> pl.LazyFrame:
    """
    Append a boolean ``CONTROL_COLUMN`` to a LazyFrame using a SQL predicate.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input LazyFrame to annotate.
    cfg : NormalizeConfig
        Configuration supplying ``control_sample_query``, a SQL-like WHERE
        clause evaluated against the frame (e.g. ``"meta_aa_changes = 'WT'"``).

    Returns
    -------
    pl.LazyFrame
        The input frame with an additional boolean ``CONTROL_COLUMN`` column
        that is ``True`` for rows matching the query.
    """
    return lf.with_columns(
        pl.sql_expr(cfg.control_sample_query).alias(CONTROL_COLUMN_NAME)
    )


@hydra.main(version_base=None, config_path=None, config_name="normalize_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: fit and apply z-score normalization to a parquet file.

    Reads the input file at ``input_file``, adds a control indicator column
    via :func:`add_control_indicator_column`, fits a :class:`Normalizer` on
    the control rows, applies it, and writes the result.

    Output path
    -----------
    - If ``output_root`` is set: ``{output_root}.{stem}.{ext}``
    - Otherwise: ``{output_dir}/{filename}`` (same name as the input file)

    If ``save_normalizer`` is ``True``, the fitted :class:`Normalizer` is also
    written alongside the output using the same root/dir convention with the
    name ``normalizer.parquet``.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.normalize \\
            output_dir=./out \\
            input_file=data/cells.parquet
    """
    norm_cfg: NormalizeConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(norm_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_cfg.output_dir = output_dir
    setup_logging(norm_cfg, "normalize")

    input_path = pathlib.Path(norm_cfg.input_file)
    logging.info("Loading input from %s", input_path)
    lf = pl.scan_parquet(input_path)
    lf = add_control_indicator_column(lf, norm_cfg)

    logging.info("Fitting normalizer")
    normalizer = Normalizer.from_lazyframe(lf)
    logging.info("Applying normalizer")
    lf = normalizer.apply(lf)

    stem = input_path.stem
    ext = input_path.suffix.lstrip(".")
    if norm_cfg.output_root is not None:
        out_path = pathlib.Path(f"{norm_cfg.output_root}.{stem}.{ext}")
    else:
        out_path = output_dir / input_path.name

    logging.info("Writing output to %s", out_path)
    lf.sink_parquet(out_path)

    if norm_cfg.save_normalizer:
        if norm_cfg.output_root is not None:
            norm_path = pathlib.Path(f"{norm_cfg.output_root}.normalizer.parquet")
        else:
            norm_path = output_dir / "normalizer.parquet"
        logging.info("Saving normalizer to %s", norm_path)
        normalizer.save(norm_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
