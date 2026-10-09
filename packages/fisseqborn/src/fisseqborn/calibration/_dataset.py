"""`Dataset.calibrate` and `Dataset.apply_calibration`: the polars side of calibration."""

import dataclasses
from os import PathLike
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import polars as pl

from . import _core, _inputs
from ._result import Calibration

if TYPE_CHECKING:
    from ..dataset import Dataset

D = TypeVar("D", bound="Dataset")

PREFIX = "meta_excalibr"


def _check_score(ds: "Dataset", score: str) -> None:
    ds._require(ds.variant_col, score)
    if not ds.schema[score].is_numeric():
        raise ValueError(
            f"Score column {score!r} must be numeric, got {ds.schema[score]}"
        )


def _check_unique(df: pl.DataFrame, variant_col: str) -> None:
    dups = df.filter(pl.col(variant_col).is_duplicated())[variant_col].unique().sort()
    if dups.len():
        shown = ", ".join(map(str, dups.head(5).to_list()))
        more = " ..." if dups.len() > 5 else ""
        raise ValueError(
            f"calibrate needs one row per {variant_col!r}, but {dups.len()} variant(s) "
            f"appear more than once ({shown}{more}). Pool them first: "
            ".per_variant() on OvwtScores or .median_across_batches() on Profiles (or "
            "filter to a single batch to calibrate it alone)."
        )


def _scored_columns(
    calibration: Calibration,
    scores: np.ndarray,
    prefix: str,
    oob: np.ndarray | None = None,
) -> list[pl.Series]:
    """LR, posterior and points; ``oob`` points (NaN: none) replace the in-bag ones."""
    points = calibration.points(scores).astype(float)
    points[np.isnan(scores)] = np.nan
    if oob is not None:
        points = np.where(np.isnan(oob), points, oob)
    return [
        pl.Series(f"{prefix}_lr", calibration.lr(scores), nan_to_null=True),
        pl.Series(
            f"{prefix}_posterior", calibration.posterior(scores), nan_to_null=True
        ),
        pl.Series(f"{prefix}_points", points, nan_to_null=True).cast(pl.Int8),
    ]


def _scores(df: pl.DataFrame, score: str) -> np.ndarray:
    return df[score].cast(pl.Float64).fill_nan(None).to_numpy().astype(float)


def calibrate(
    ds: D,
    score: str,
    *,
    gnomad: "str | PathLike | pl.DataFrame",
    clinvar: "str | PathLike | pl.DataFrame | pl.LazyFrame | None" = None,
    direction: str = "auto",
    benign_method: str = "avg",
    n_components: int | str = "auto",
    n_bootstrap: int = 1000,
    n_restarts: int = 8,
    prior: float | None = None,
    splice_max: float | None = None,
    min_stars: int | None = 1,
    strict_constraint: bool = False,
    out_of_bag: bool = True,
    seed: int = 0,
    n_jobs: int | None = -1,
    prefix: str = PREFIX,
) -> D:
    _check_score(ds, score)
    df = ds.df
    _check_unique(df, ds.variant_col)
    notes: list[str] = []
    gnomad_table = _inputs.read_gnomad(gnomad, splice_max=splice_max, notes=notes)
    clinvar_table = (
        None
        if clinvar is None
        else _inputs.read_clinvar(clinvar, min_stars=min_stars, notes=notes)
    )
    groups = _inputs.assign_groups(
        df, ds.variant_col, gnomad_table, clinvar_table, notes
    )
    scores = _scores(df, score)
    member = (
        groups.select(f"in_{g}" for g in _inputs.GROUP_LABELS).to_numpy().astype(bool)
    )
    calibration, oob = _core.fit_calibration(
        scores,
        member,
        direction=direction,
        benign_method=benign_method,
        n_components=n_components,
        n_bootstrap=n_bootstrap,
        n_restarts=n_restarts,
        prior=prior,
        strict_constraint=strict_constraint,
        out_of_bag=out_of_bag,
        seed=seed,
        n_jobs=n_jobs,
        extra_warnings=notes,
    )
    calibration = _with_settings(
        calibration,
        score=score,
        clinvar="table" if clinvar is not None else "gnomad",
        splice_max=splice_max,
        min_stars=min_stars,
    )
    out = df.with_columns(
        groups["group"].alias(f"{prefix}_group"),
        groups["groups"].alias(f"{prefix}_groups"),
        *_scored_columns(calibration, scores, prefix, oob),
        pl.Series(f"{prefix}_oob", ~np.isnan(oob)),
    )
    new = ds._replace(out)
    new.calibration = calibration
    return new


def _with_settings(calibration: Calibration, **settings: Any) -> Calibration:
    return dataclasses.replace(
        calibration, settings={**calibration.settings, **settings}
    )


def apply_calibration(
    ds: D,
    calibration: "Calibration | str | PathLike",
    score: str | None = None,
    *,
    prefix: str = PREFIX,
) -> D:
    if not isinstance(calibration, Calibration):
        calibration = Calibration.from_json(calibration)
    score = score or calibration.settings.get("score")
    if score is None:
        raise ValueError("Pass score=: the calibration does not record a score column")
    _check_score(ds, score)
    df = ds.df
    out = df.with_columns(_scored_columns(calibration, _scores(df, score), prefix))
    new = ds._replace(out)
    new.calibration = calibration
    return new
