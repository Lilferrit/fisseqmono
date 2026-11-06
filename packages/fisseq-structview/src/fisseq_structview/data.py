"""Load a pipeline run's distinguishability scores once, at launch.

`load_distinguishability` reads every batch's OvWT ``results.parquet`` with
`fisseqborn.OvwtScores.from_pipeline` and pools each score column into one value per variant,
both batch-corrected (`OvwtScores.correct`, which keeps the AUROC scale) and raw, taking the
median across batches. The app then only filters and groups these small in-memory tables.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from typing import TYPE_CHECKING

import polars as pl

import fisseqborn as fb

if TYPE_CHECKING:
    from .params import ClinvarSpec

logger = logging.getLogger(__name__)

#: The score the app shows first, when the run has it.
DEFAULT_SCORE = "auroc_median_fold"

#: Columns of every per-variant table in `RunScores.tables`.
VARIANT = "variant"
VARIANT_TYPE = "variant_type"
POSITION = "position"
WT = "wt"
SCORE = "score"
N_EXPERIMENTS = "n_experiments"


@dataclass(frozen=True)
class RunScores:
    """A run's per-variant distinguishability scores, pooled across batches.

    Attributes
    ----------
    source : str
        The run directory as given.
    batches : tuple[str, ...]
        The batches read.
    tables : Mapping[tuple[str, bool], pl.DataFrame]
        ``(score, corrected)`` -> one row per variant with `VARIANT`, `VARIANT_TYPE`,
        `POSITION` (null for labels other than single substitutions), `WT` (the wild-type
        residue of a single substitution), `SCORE` and `N_EXPERIMENTS`.
    batch_rows : Mapping[str, int]
        Batch -> its number of OvWT result rows.
    layout, track, variant_col : str | None
        How the run was read (``layout`` ``None``: detected).
    """

    source: str
    batches: tuple[str, ...]
    tables: Mapping[tuple[str, bool], pl.DataFrame] = field(repr=False)
    batch_rows: Mapping[str, int] = field(default_factory=dict)
    layout: str | None = None
    track: str | None = None
    variant_col: str | None = None

    @property
    def scores(self) -> list[str]:
        """The score columns, in the order the run has them."""
        return list(dict.fromkeys(score for score, _ in self.tables))

    def table(self, score: str, corrected: bool = True) -> pl.DataFrame:
        """The per-variant table of ``score``."""
        try:
            return self.tables[(score, corrected)]
        except KeyError:
            raise ValueError(
                f"No score {score!r} in {self.source}; choose from {self.scores}"
            ) from None


def default_score(scores: Sequence[str]) -> str:
    """`DEFAULT_SCORE` if present, else the first of ``scores``."""
    if not scores:
        raise ValueError("No score columns")
    return DEFAULT_SCORE if DEFAULT_SCORE in scores else scores[0]


def score_columns(scores: fb.OvwtScores) -> list[str]:
    """The numeric score columns of a run's results: `fisseqborn.ovwt.SCORES` in their order,
    then any other numeric non-``meta_`` column (``test_auroc`` and friends of older runs)."""
    schema = scores.schema
    numeric = [
        c
        for c, dtype in schema.items()
        if dtype.is_numeric() and not c.startswith("meta_")
    ]
    known = [s for s in fb.ovwt.SCORES if s in numeric]
    return known + [c for c in numeric if c not in known]


def _per_variant(scores: fb.OvwtScores, score: str, corrected: bool) -> pl.DataFrame:
    if corrected:
        pooled = scores.correct(score).per_variant(f"{score}_corrected")
        value = f"{score}_corrected"
    else:
        pooled = scores.per_variant(score)
        value = score
    variant_col = scores.variant_col
    label = pl.col(variant_col)
    return (
        pooled.variant_type(output_col=VARIANT_TYPE, control_col=None)
        .position(strict=True, output_col=POSITION)
        .lazy()
        .select(
            label.alias(VARIANT),
            pl.col(VARIANT_TYPE),
            pl.col(POSITION),
            label.str.extract(r"^([A-Z])\d+[A-Z]$", 1).alias(WT),
            pl.col(value).cast(pl.Float64).fill_nan(None).alias(SCORE),
            pl.col("meta_n_experiments").alias(N_EXPERIMENTS),
        )
        .collect()
    )


def load_distinguishability(
    run_dir: "str | PathLike[str]",
    *,
    layout: str | None = None,
    track: str = "embeddings",
    batches: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    cache: "bool | str | PathLike[str]" = False,
    refresh: bool = False,
    variant_col: str = "meta_aa_changes",
) -> RunScores | None:
    """Read and pool a run's OvWT scores; ``None`` (with a warning) if the run has none.

    Parameters
    ----------
    run_dir : str | PathLike
        A pipeline output directory, or ``"user@host:/path"`` for a remote one.
    layout, track, batches, exclude, cache, refresh, variant_col
        Passed to `fisseqborn.OvwtScores.from_pipeline`.

    Returns
    -------
    RunScores | None
        ``None`` when the run has no OvWT batch directories or results files.
    """
    try:
        scores = fb.OvwtScores.from_pipeline(
            run_dir,
            layout=layout,
            track=track,
            batches=batches,
            exclude=exclude,
            cache=cache,
            refresh=refresh,
            variant_col=variant_col,
        )
        columns = score_columns(scores)
    except FileNotFoundError as err:
        logger.warning(
            "No distinguishability scores in %s (%s); showing the structure only",
            run_dir,
            err,
        )
        return None
    if not columns:
        logger.warning(
            "The OvWT results in %s have no numeric score columns; showing the structure only",
            run_dir,
        )
        return None
    if variant_col not in scores.columns:
        logger.warning(
            "The OvWT results in %s have no %r column (pass --variant-col); showing the "
            "structure only",
            run_dir,
            variant_col,
        )
        return None
    counts = (
        scores.lazy()
        .group_by(scores.batch_col, maintain_order=True)
        .len()
        .collect()
        .rows()
    )
    batch_names = tuple(batch for batch, _ in counts)
    tables = {
        (score, corrected): _per_variant(scores, score, corrected)
        for score in columns
        for corrected in (True, False)
    }
    logger.info(
        "Loaded %d score columns for %d variants from %d batches of %s",
        len(columns),
        tables[(columns[0], True)].height,
        len(batch_names),
        run_dir,
    )
    return RunScores(
        source=str(run_dir),
        batches=batch_names,
        tables=tables,
        batch_rows=dict(counts),
        layout=layout,
        track=track,
        variant_col=variant_col,
    )


#: Columns of `load_clinvar`'s table.
CLINVAR_CLASS = "clinvar_class"
SIGNIFICANCE = "clinical_significance"
STARS = "stars"
CONDITIONS = "conditions"

#: ClinVar classes the app marks, most severe first, with the notebooks' colors.
CLINVAR_CLASSES = {
    "pathogenic": ("Pathogenic/Likely pathogenic", "#d62728"),
    "vus": ("Uncertain significance", "#e6b800"),
    "benign": ("Benign/Likely benign", "#4a7fa8"),
}

#: ClinVar calls counted as benign (as in the variant-effect notebook).
BENIGN_LIKE = ("Benign", "Likely benign", "Benign/Likely benign")


@dataclass(frozen=True)
class ClinvarCalls:
    """ClinVar calls for a run's variants.

    Attributes
    ----------
    table : pl.DataFrame
        One row per annotated variant: `VARIANT`, `CLINVAR_CLASS` (a key of
        `CLINVAR_CLASSES`), `SIGNIFICANCE`, `STARS` and `CONDITIONS` (null unless fetched).
    source, release : str | None
        Where the calls came from, and the ClinVar release used.
    """

    table: pl.DataFrame
    source: str
    release: str | None


def load_clinvar(run: RunScores, spec: "ClinvarSpec") -> ClinvarCalls | None:
    """ClinVar calls for ``run``'s variants via `fisseqborn.Dataset.clinvar`; ``None``
    (with a warning) if they can't be read."""
    import fisseqborn.fisseq as fisseq

    labels = run.table(run.scores[0], True)[VARIANT].unique().sort()
    source = str(spec.path) if spec.path is not None else f"gnomAD ({spec.gene})"
    try:
        annotated = (
            fb.Dataset(pl.DataFrame({"meta_aa_changes": labels}))
            .variant_type()
            .clinvar(
                spec.path,
                gene=spec.gene,
                min_stars=spec.min_stars,
                release=spec.release,
                conditions=spec.conditions,
            )
            .df
        )
    except Exception as err:  # a bad table fails in many ways (I/O, schema, network)
        logger.warning("Couldn't read ClinVar calls from %s: %s", source, err)
        return None
    annotation = pl.col("meta_clinvar_annotation")
    significance = pl.col("meta_clinvar_clinical_significance")
    klass = (
        pl.when(annotation == fisseq.PATHOGENIC)
        .then(pl.lit("pathogenic"))
        .when(annotation == fisseq.UNCERTAIN)
        .then(pl.lit("vus"))
        .when(significance.is_in(BENIGN_LIKE))
        .then(pl.lit("benign"))
    )
    conditions = (
        pl.col("meta_clinvar_conditions").list.join("; ")
        if "meta_clinvar_conditions" in annotated.columns
        else pl.lit(None, dtype=pl.String)
    )
    table = annotated.select(
        pl.col("meta_aa_changes").alias(VARIANT),
        klass.alias(CLINVAR_CLASS),
        significance.alias(SIGNIFICANCE),
        pl.col("meta_clinvar_stars").alias(STARS),
        conditions.alias(CONDITIONS),
    ).filter(pl.col(CLINVAR_CLASS).is_not_null())
    release = next(iter(annotated["meta_clinvar_release"].drop_nulls()), None)
    counts = dict(table.group_by(CLINVAR_CLASS).len().rows())
    logger.info(
        "ClinVar (%s, release %s): %s",
        source,
        release,
        ", ".join(f"{counts.get(k, 0)} {k}" for k in CLINVAR_CLASSES),
    )
    return ClinvarCalls(table, source, release)
