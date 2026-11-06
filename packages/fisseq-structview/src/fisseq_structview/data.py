"""Load a pipeline run's distinguishability scores and feature profiles once, at launch.

`load_distinguishability` reads every batch's OvWT ``results.parquet`` with
`fisseqborn.OvwtScores.from_pipeline` and pools each score column into one value per variant,
both batch-corrected (`OvwtScores.correct`, which keeps the AUROC scale) and raw, taking the
median across batches.

`load_features` reads every batch's feature-selection aggregates with
`fisseqborn.Profiles.from_pipeline`, z-scores each batch against its own synonymous variants
and takes each feature's median across batches.

The app then only filters and groups these in-memory tables.
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

    @property
    def variants(self) -> pl.Series:
        """Every variant label, sorted."""
        return self.tables[(self.scores[0], True)][VARIANT].unique().sort()

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


def _label_columns(variant_col: str) -> list[pl.Expr]:
    """`VARIANT`, `VARIANT_TYPE`, `POSITION` and `WT` of a table with ``variant_col``,
    `VARIANT_TYPE` and `POSITION`."""
    label = pl.col(variant_col)
    return [
        label.alias(VARIANT),
        pl.col(VARIANT_TYPE),
        pl.col(POSITION),
        label.str.extract(r"^([A-Z])\d+[A-Z]$", 1).alias(WT),
    ]


def _per_variant(scores: fb.OvwtScores, score: str, corrected: bool) -> pl.DataFrame:
    if corrected:
        pooled = scores.correct(score).per_variant(f"{score}_corrected")
        value = f"{score}_corrected"
    else:
        pooled = scores.per_variant(score)
        value = score
    return (
        pooled.variant_type(output_col=VARIANT_TYPE, control_col=None)
        .position(strict=True, output_col=POSITION)
        .lazy()
        .select(
            *_label_columns(scores.variant_col),
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


@dataclass(frozen=True)
class FeatureProfiles:
    """A run's per-variant feature profiles, z-scored per batch and pooled across batches.

    Attributes
    ----------
    source : str
        The run directory as given.
    batches : tuple[str, ...]
        The batches read.
    wide : pl.DataFrame
        One row per variant: `VARIANT`, `VARIANT_TYPE`, `POSITION`, `WT`, `N_EXPERIMENTS`
        and one Float32 column per feature (``<feature>_<type>``, e.g.
        ``AreaShape_Area_median``), each a z-score against the synonymous variants.
    features : tuple[str, ...]
        The feature columns: the gene's landmark features first, then the rest in the run's
        order.
    labels : Mapping[str, str]
        Feature column -> readable name, for the landmark features.
    batch_rows : Mapping[str, int]
        Batch -> its number of variants.
    gene : str | None
        The gene whose landmark features come first.
    layout, track, variant_col : str | None
        How the run was read (``layout`` ``None``: detected).
    """

    source: str
    batches: tuple[str, ...]
    wide: pl.DataFrame = field(repr=False)
    features: tuple[str, ...] = ()
    labels: Mapping[str, str] = field(default_factory=dict)
    batch_rows: Mapping[str, int] = field(default_factory=dict)
    gene: str | None = None
    layout: str | None = None
    track: str | None = None
    variant_col: str | None = None

    @property
    def landmarks(self) -> list[str]:
        """The landmark feature columns, in `features` order."""
        return [f for f in self.features if f in self.labels]

    @property
    def types(self) -> dict[str, int]:
        """Aggregate type (the column suffix, e.g. ``median``) -> its number of features."""
        counts: dict[str, int] = {}
        for f in self.features:
            kind = f.rpartition("_")[2]
            counts[kind] = counts.get(kind, 0) + 1
        return counts

    @property
    def variants(self) -> pl.Series:
        """Every variant label, sorted."""
        return self.wide[VARIANT].unique().sort()

    def label(self, feature: str) -> str:
        """The feature's readable name, else its column name."""
        return self.labels.get(feature, feature)

    def table(self, feature: str) -> pl.DataFrame:
        """The per-variant table of ``feature``, with the columns of `RunScores.table`."""
        if feature not in self.features:
            raise ValueError(f"No feature {feature!r} in {self.source}")
        return self.wide.select(
            VARIANT,
            VARIANT_TYPE,
            POSITION,
            WT,
            pl.col(feature).cast(pl.Float64).fill_nan(None).alias(SCORE),
            N_EXPERIMENTS,
        )


def landmark_columns(columns: Sequence[str], gene: str | None) -> dict[str, str]:
    """Column -> readable name of ``gene``'s landmark features among ``columns``: each
    landmark key's ``<key>_<type>`` columns, in the gene's order (``median`` first), named
    ``"<name> (<type>)"``.

    Raises
    ------
    ValueError
        If ``gene`` isn't a gene `fisseqborn.genes.resolve` knows.
    """
    if gene is None:
        return {}
    from fisseqborn import genes

    landmarks = genes.resolve(gene).landmark_features or {}
    out: dict[str, str] = {}
    for key, name in landmarks.items():
        prefix = f"{key}_"
        matches = [
            (col, col[len(prefix) :])
            for col in columns
            if col.startswith(prefix) and col[len(prefix) :].isalnum()
        ]
        # The median (what the notebooks plot) first, then the run's order.
        matches.sort(key=lambda m: m[1] != "median")
        out.update((col, f"{name} ({kind})") for col, kind in matches)
    return out


def load_features(
    run_dir: "str | PathLike[str]",
    *,
    layout: str | None = None,
    track: str = "embeddings",
    batches: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    cache: "bool | str | PathLike[str]" = False,
    refresh: bool = False,
    variant_col: str = "meta_aa_changes",
    gene: str | None = None,
) -> FeatureProfiles | None:
    """Read, z-score and pool a run's feature profiles; ``None`` if the run has none.

    Every aggregate type the run wrote (``feature_select_batchwise/<batch>/aggregates/
    <type>.parquet``) is read with `fisseqborn.Profiles.from_pipeline`, each batch is
    z-scored against its own synonymous variants (``normalize(by=<batch>)``) and each
    feature's median is taken across batches (``median_across_batches``). Features without
    any value are dropped.

    Parameters
    ----------
    run_dir : str | PathLike
        A pipeline output directory, or ``"user@host:/path"`` for a remote one.
    layout, track, batches, exclude, cache, refresh, variant_col
        Passed to `fisseqborn.Profiles.from_pipeline`.
    gene : str | None
        A gene of `fisseqborn.genes` whose landmark features are listed first.

    Raises
    ------
    ValueError
        If ``gene`` is unknown.
    """
    landmark_columns([], gene)  # an unknown gene fails before the slow read
    try:
        profiles = fb.Profiles.from_pipeline(
            run_dir,
            types=None,
            layout=layout,
            track=track,
            batches=batches,
            exclude=exclude,
            cache=cache,
            refresh=refresh,
            variant_col=variant_col,
        )
        columns = profiles.columns
    except FileNotFoundError as err:
        logger.info("No feature profiles in %s (%s)", run_dir, err)
        return None
    if variant_col not in columns:
        logger.warning(
            "The feature aggregates in %s have no %r column (pass --variant-col); "
            "feature coloring is off",
            run_dir,
            variant_col,
        )
        return None
    if not profiles.values:
        logger.warning(
            "The feature aggregates in %s have no feature columns; feature coloring is off",
            run_dir,
        )
        return None
    counts = (
        profiles.lazy()
        .group_by(profiles.batch_col, maintain_order=True)
        .len()
        .collect()
        .rows()
    )
    pooled = (
        profiles.variant_type().normalize(by=profiles.batch_col).median_across_batches()
    )
    values = pooled.values  # before adding the (non-meta_) label columns
    pooled = pooled.variant_type(output_col=VARIANT_TYPE, control_col=None).position(
        strict=True, output_col=POSITION
    )
    wide = (
        pooled.lazy()
        .select(
            *_label_columns(variant_col),
            pl.col("meta_n_experiments").alias(N_EXPERIMENTS),
            *(pl.col(c).cast(pl.Float32).fill_nan(None).alias(c) for c in values),
        )
        .collect()
    )
    empty = [c for c in values if wide[c].null_count() == wide.height]
    if empty:
        logger.info("Dropping %d features without any value", len(empty))
        wide = wide.drop(empty)
    kept = [c for c in values if c not in set(empty)]
    if not kept:
        logger.warning(
            "No feature in %s has a value after pooling; feature coloring is off",
            run_dir,
        )
        return None
    labels = landmark_columns(kept, gene)
    features = (*labels, *(c for c in kept if c not in labels))
    batch_names = tuple(batch for batch, _ in counts)
    logger.info(
        "Loaded %d features for %d variants from %d batches of %s",
        len(features),
        wide.height,
        len(batch_names),
        run_dir,
    )
    return FeatureProfiles(
        source=str(run_dir),
        batches=batch_names,
        wide=wide,
        features=features,
        labels=labels,
        batch_rows=dict(counts),
        gene=gene,
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


def load_clinvar(
    run: "RunScores | FeatureProfiles", spec: "ClinvarSpec"
) -> ClinvarCalls | None:
    """ClinVar calls for ``run``'s variants (its scores' or its features') via
    `fisseqborn.Dataset.clinvar`; ``None`` (with a warning) if they can't be read."""
    import fisseqborn.fisseq as fisseq

    labels = run.variants
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
