"""CORRELATE_FEATURES: per-feature Pearson correlation between two pseudo-replicate halves.

For one bootstrap replicate and one aggregation method, correlates every aggregate column across
the variants of the replicate's two halves. A correlation that is undefined (a constant column,
NaN) is stored as null, and an aggregate with no feature columns gives an empty table.
"""

import dataclasses
import logging
import pathlib

import polars as pl
from omegaconf import MISSING

from fisseq_common.schema import FEATURE_SELECTOR

from .config import AppConfig


def compute_feature_correlations(
    df1: pl.DataFrame, df2: pl.DataFrame, label_col: str
) -> pl.DataFrame:
    """
    Per-dimension Pearson correlation between two half-aggregates.

    Parameters
    ----------
    df1, df2 : pl.DataFrame
        The two halves' aggregates for one method: ``label_col`` plus one
        column per embedding dimension. Same schema.
    label_col : str
        Column the two frames are aligned on (one row per variant).

    Returns
    -------
    pl.DataFrame
        One row per dimension: ``feature``, ``r``, ``r_squared``. ``r`` is
        **null** for a dimension that is constant in either half -- Polars'
        ``corr`` of a zero-variance column is NaN, which is normalized to
        null here on purpose. NaN would propagate through BLOCKLIST's
        median and condemn a dimension outright on the strength of one
        degenerate replicate; null is skipped by ``median``, so the
        dimension is judged on the replicates that actually produced a
        number. A dimension that was degenerate in *every* replicate gets a
        null median, which fails the threshold comparison and is blocked --
        the right verdict, since it carried no variant-to-variant signal at
        all.

    Notes
    -----
    Uses ``FEATURE_SELECTOR`` (exclude ``meta_*``), not
    ``EMBEDDING_SELECTOR``: AGGREGATE_HALF's columns are stat-suffixed
    (``emb_0000_KS``), which ``EMBEDDING_SELECTOR``'s ``^emb_\\d+$`` does
    not match.
    """
    df1 = df1.select(FEATURE_SELECTOR, pl.col(label_col))
    df2 = df2.select(FEATURE_SELECTOR, pl.col(label_col))
    df_joined = df1.join(df2, on=label_col, suffix="_right")

    features = [c for c in df1.columns if c != label_col]
    if not features:
        return pl.DataFrame(
            schema={"feature": pl.String, "r": pl.Float64, "r_squared": pl.Float64}
        )
    corrs = df_joined.select(
        # nan_to_null: a zero-variance column makes pl.corr return NaN, and
        # NaN survives BLOCKLIST's median where null is skipped. See the
        # Returns section above.
        pl.corr(f, f"{f}_right").fill_nan(None).alias(f)
        for f in features
    ).row(0)

    return pl.DataFrame(
        [
            {
                "feature": feature,
                "r": corr,
                "r_squared": None if corr is None else corr**2,
            }
            for feature, corr in zip(features, corrs)
        ],
        schema=["feature", "r", "r_squared"],
    )


@dataclasses.dataclass
class CorrelateFeaturesParams(AppConfig):
    """
    Hydra structured configuration for CORRELATE_FEATURES.

    Attributes
    ----------
    half1_file : str
        Path to the first half's AGGREGATE_HALF output. Required.
    half2_file : str
        Path to the second half's AGGREGATE_HALF output, same method.
        Required.
    label_column : str
        Name of the variant label column. Defaults to ``"meta_aa_changes"``.
    output_name : str
        Basename (without ``.parquet``) of the single output file. The Nextflow module
        sets it to the aggregation method, so one experiment's methods can
        share a per-replicate directory. Defaults to ``"correlations"``.
    """

    half1_file: str = MISSING
    half2_file: str = MISSING
    label_column: str = "meta_aa_changes"
    output_name: str = "correlations"


def run_correlate_features(cfg: CorrelateFeaturesParams) -> None:
    """Run the stage with ``cfg`` (``output_dir`` must exist)."""
    output_dir = pathlib.Path(cfg.output_dir)

    prefix = f"{cfg.output_root}." if cfg.output_root is not None else ""

    logging.info("Reading halves %s / %s", cfg.half1_file, cfg.half2_file)
    df1 = pl.read_parquet(cfg.half1_file)
    df2 = pl.read_parquet(cfg.half2_file)
    corr_df = compute_feature_correlations(df1, df2, cfg.label_column)

    out_path = output_dir / f"{prefix}{cfg.output_name}.parquet"
    logging.info("Writing %s (%d dimension(s))", out_path, corr_df.height)
    corr_df.write_parquet(out_path)

    logging.info("Done")
