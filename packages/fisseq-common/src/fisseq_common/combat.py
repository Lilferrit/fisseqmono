"""ComBat batch correction of per-variant tables stacked across experiments.

ComBat itself is `neuroCombat <https://github.com/Jfortin1/neuroCombat>`_ (parametric empirical
Bayes, Johnson et al. 2007), from the ``combat`` extra (``fisseq-common[combat]``, part of
``stages``); it is imported only when :func:`combat` runs, so the base install doesn't need it.

The design is the batch plus the variant as a categorical covariate: each variant's own effect
is kept and only each experiment's location and scale are removed. Batch effects are therefore
estimated from the variants experiments share, and :func:`combat_frame` drops a variant
measured in only one experiment.

neuroCombat takes no missing values: a single NaN turns every corrected value into NaN (its
pooled variance and priors use plain ``np.var``). :func:`combat_frame` therefore drops, in
order, rows whose values are all null, variants left in a single experiment, and value
columns still holding a null. NaN and ±inf count as null.

Used by the ``combatfeatures`` and ``combatovwt`` stages and by fisseqborn's
``Profiles.combat`` / ``OvwtScores.combat``.
"""

import logging
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

#: How many names a warning lists before eliding the rest.
_SHOW = 5


def _names(values: Sequence[Any]) -> str:
    shown = ", ".join(str(v) for v in values[:_SHOW])
    return shown + (", ..." if len(values) > _SHOW else "")


def combat(
    x: np.ndarray,
    batch: Sequence[Any] | np.ndarray,
    covariate: Optional[Sequence[Any] | np.ndarray] = None,
    *,
    mean_only: bool = False,
    eb: bool = True,
) -> np.ndarray:
    """ComBat-correct ``x`` (samples × features) for ``batch``, keeping ``covariate``.

    ``covariate`` is categorical (one-hot, first level dropped); ``None`` corrects for the
    batch alone. ``mean_only`` removes only the location effect. ``eb=False`` uses each
    batch's least-squares location and scale without the empirical-Bayes shrinkage, which is
    also what happens with a single feature (the priors need at least two).

    Raises
    ------
    ValueError
        On a non-finite value, fewer than two batches or a batch with fewer than two rows.
    """
    try:
        import pandas as pd
        from neuroCombat import neuroCombat
    except ImportError as err:
        raise ImportError(
            "ComBat needs neuroCombat and pandas: install fisseq-common[combat]"
        ) from err

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D (samples × features), got shape {x.shape}")
    batch = np.asarray(batch, dtype=object)
    if len(batch) != x.shape[0]:
        raise ValueError(f"batch has {len(batch)} entries for {x.shape[0]} samples")
    if not np.isfinite(x).all():
        raise ValueError(
            "x contains null, NaN or infinite values; ComBat needs a complete matrix "
            "(see combat_frame)"
        )
    levels, counts = np.unique(batch.astype(str), return_counts=True)
    if len(levels) < 2:
        raise ValueError(f"ComBat needs at least 2 batches, got {len(levels)}")
    small = levels[counts < 2]
    if len(small):
        raise ValueError(
            f"Every batch needs at least 2 rows; these have fewer: {_names(list(small))}"
        )
    if eb and x.shape[1] < 2:
        logger.info("A single column: ComBat runs without empirical Bayes")
        eb = False

    covars = {"batch": batch.astype(str)}
    categorical: list[str] = []
    if covariate is not None:
        covariate = np.asarray(covariate, dtype=object)
        if len(covariate) != x.shape[0]:
            raise ValueError(
                f"covariate has {len(covariate)} entries for {x.shape[0]} samples"
            )
        covars["covariate"] = covariate.astype(str)
        categorical.append("covariate")
    with warnings.catch_warnings():
        if x.shape[1] < 2:
            # neuroCombat computes the (unused) priors' variance over a single feature
            warnings.simplefilter("ignore", RuntimeWarning)
        # A numpy array, not a DataFrame: neuroCombat casts DataFrames to float32.
        out = neuroCombat(
            dat=x.T,
            covars=pd.DataFrame(covars),
            batch_col="batch",
            categorical_cols=categorical,
            eb=eb,
            parametric=True,
            mean_only=mean_only,
        )["data"]
    return np.asarray(out, dtype=np.float64).T


def stack_batches(
    frames: Mapping[str, pl.DataFrame], *, batch_column: str = "meta_experiment"
) -> pl.DataFrame:
    """Stack per-experiment tables, tagging each row with its experiment in ``batch_column``.

    Only the columns every table has are kept; a warning names each table's dropped ones.
    """
    if not frames:
        raise ValueError("No tables to stack")
    common = set.intersection(*(set(df.columns) for df in frames.values()))
    for name, df in frames.items():
        dropped = [c for c in df.columns if c not in common]
        if dropped:
            logger.warning(
                "Experiment %s: dropped %d column(s) not in every experiment (%s)",
                name,
                len(dropped),
                _names(dropped),
            )
    first = next(iter(frames.values()))
    columns = [c for c in first.columns if c in common]
    return pl.concat(
        [
            df.select(columns).with_columns(pl.lit(name).alias(batch_column))
            for name, df in frames.items()
        ],
        how="vertical_relaxed",
    )


def drop_single_batch_variants(
    df: pl.DataFrame, *, label_column: str, batch_column: str = "meta_experiment"
) -> pl.DataFrame:
    """Drop the variants measured in only one experiment, with a warning naming them."""
    n_batches = df.group_by(label_column).agg(
        pl.col(batch_column).n_unique().alias("n")
    )
    single = n_batches.filter(pl.col("n") < 2).get_column(label_column).sort().to_list()
    if single:
        logger.warning(
            "Dropped %d variant(s) measured in only one experiment (%s): ComBat needs each "
            "variant in at least two",
            len(single),
            _names(single),
        )
        df = df.filter(~pl.col(label_column).is_in(single))
    return df


def _missing(column: str) -> pl.Expr:
    value = pl.col(column).cast(pl.Float64)
    return value.is_null() | ~value.is_finite()


def combat_frame(
    df: pl.DataFrame,
    columns: Sequence[str],
    *,
    label_column: str,
    batch_column: str = "meta_experiment",
    mean_only: bool = False,
) -> pl.DataFrame:
    """ComBat-correct ``columns`` of a table stacked across experiments (one row per variant
    and experiment), with the variant as a covariate. Values are replaced in place.

    Before correcting, each step logging a warning: rows whose ``columns`` are all null
    (NaN and ±inf included) are dropped, then variants left in only one experiment, then the
    ``columns`` still holding a null (left out of the result). Other columns are carried
    along unchanged. Rows are sorted by variant, then experiment.

    Raises
    ------
    ValueError
        If no column survives, or as :func:`combat` does.
    """
    columns = list(columns)
    if not columns:
        raise ValueError("No columns to correct")
    missing = [c for c in (label_column, batch_column, *columns) if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {', '.join(missing)}")

    all_null = pl.all_horizontal(_missing(c) for c in columns)
    n_empty = df.select(all_null.sum()).item()
    if n_empty:
        logger.warning(
            "Dropped %d row(s) with no value in any of the %d column(s) to correct",
            n_empty,
            len(columns),
        )
        df = df.filter(~all_null)

    df = drop_single_batch_variants(
        df, label_column=label_column, batch_column=batch_column
    )

    counts = df.select(_missing(c).sum().alias(c) for c in columns).row(0, named=True)
    holey = [c for c in columns if counts[c]]
    if holey:
        logger.warning(
            "Dropped %d of %d column(s) holding a null, NaN or infinite value (%s)",
            len(holey),
            len(columns),
            _names(holey),
        )
        df = df.drop(holey)
        columns = [c for c in columns if c not in holey]
    if not columns:
        raise ValueError(
            "Every column to correct holds a null; nothing left to correct"
        )
    if df.height == 0:
        raise ValueError("No rows left to correct")

    df = df.sort(label_column, batch_column, maintain_order=True)
    corrected = combat(
        df.select(columns).cast(pl.Float64).to_numpy(),
        df.get_column(batch_column).to_numpy(),
        df.get_column(label_column).to_numpy(),
        mean_only=mean_only,
    )
    return df.with_columns(
        pl.Series(c, corrected[:, i], dtype=pl.Float64) for i, c in enumerate(columns)
    )
