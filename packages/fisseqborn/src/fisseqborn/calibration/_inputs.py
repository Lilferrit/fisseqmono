"""Control groups for calibration: gnomAD and ClinVar inputs keyed to ``meta_aa_changes``.

fisseqborn keys variants by one-letter amino-acid change (``"R123W"``, ``"R123*"``).
gnomAD is nucleotide-level: its browser export's ``Protein Consequence`` (``p.Arg123Trp``)
is converted to that key, and nucleotide variants with the same protein change are
collapsed into one. Only single-codon substitutions (missense, synonymous, nonsense) can
match; multi-codon labels and ``:<tag>`` pseudo-variants are never controls.
"""

import logging
from os import PathLike

import polars as pl

from .. import _data, _variants

logger = logging.getLogger(__name__)

#: Display names of the control groups, in `_core.GROUPS` order.
GROUP_LABELS: tuple[str, ...] = ("P/LP", "B/LB", "gnomAD", "Synonymous")
#: ``meta_excalibr_group`` takes the first group a variant is in, in this order.
GROUP_PRIORITY: tuple[str, ...] = ("P/LP", "B/LB", "Synonymous", "gnomAD")

THREE_TO_ONE: dict[str, str] = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
    "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
    "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V", "Ter": "*",
    "Sec": "U",
}  # fmt: skip

# Exact ClinVar calls counted as controls; everything else (uncertain, conflicting, not
# provided, risk alleles, ...) is not a control.
PATHOGENIC_CALL = r"(?i)^(likely )?pathogenic(/(likely )?pathogenic)*$"
BENIGN_CALL = r"(?i)^(likely )?benign(/(likely )?benign)*$"

# ClinVar review status -> stars (ClinVar's own mapping).
REVIEW_STARS: dict[str, int] = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
    "no assertion criteria provided": 0,
    "no classification provided": 0,
    "no assertion provided": 0,
    "no classification for the individual variant": 0,
}

GNOMAD_PROTEIN = "Protein Consequence"
GNOMAD_CLINVAR = "ClinVar Germline Classification"
GNOMAD_FILTER = "Filters - joint"
GNOMAD_AC, GNOMAD_AN = "Allele Count", "Allele Number"
GNOMAD_SPLICE = "spliceai_ds_max"

_HGVS = r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|=)$"


def key_expr(column: str | pl.Expr) -> pl.Expr:
    """The join key of a one-letter label: single substitutions only (``"A12V"``,
    ``"A12A"``, ``"A12*"``; ``X`` is read as ``*``), else null."""
    col = pl.col(column) if isinstance(column, str) else column
    return col.str.replace(r"X$", "*").str.extract(r"^([A-Z]\d+[A-Z*])$", 1)


def hgvs_key_expr(column: str) -> pl.Expr:
    """``p.Arg123Trp`` -> ``R123W`` (``p.Arg123Ter`` -> ``R123*``, ``p.Arg123=`` ->
    ``R123R``); null for anything that is not a single-codon substitution."""
    col = pl.col(column)
    ref = col.str.extract(_HGVS, 1).replace_strict(THREE_TO_ONE, default=None)
    alt3 = col.str.extract(_HGVS, 3)
    alt = (
        pl.when(alt3 == "=")
        .then(ref)
        .otherwise(alt3.replace_strict(THREE_TO_ONE, default=None))
    )
    return pl.concat_str(ref, col.str.extract(_HGVS, 2), alt)


def _class_expr(significance: pl.Expr) -> pl.Expr:
    return (
        pl.when(significance.str.contains(PATHOGENIC_CALL))
        .then(pl.lit("pathogenic"))
        .when(significance.str.contains(BENIGN_CALL))
        .then(pl.lit("benign"))
    )


def _collapse_classes(
    frame: pl.LazyFrame, source: str, notes: list[str]
) -> pl.DataFrame:
    """One ClinVar class per key; keys with both pathogenic and benign calls are dropped."""
    classes = (
        frame.filter(
            pl.col("key").is_not_null() & pl.col("clinvar_class").is_not_null()
        )
        .group_by("key")
        .agg(pl.col("clinvar_class").unique().alias("classes"))
        .collect()
    )
    conflicting = (
        classes.filter(pl.col("classes").list.len() > 1)["key"].sort().to_list()
    )
    if conflicting:
        notes.append(
            f"{len(conflicting)} variant(s) with both pathogenic and benign ClinVar calls "
            f"in {source} were left out of the controls: {', '.join(conflicting[:10])}"
            + (" ..." if len(conflicting) > 10 else "")
        )
    return classes.filter(pl.col("classes").list.len() == 1).select(
        "key", pl.col("classes").list.first().alias("clinvar_class")
    )


def read_gnomad(
    gnomad: "str | PathLike | pl.DataFrame",
    *,
    splice_max: float | None = None,
    notes: list[str] | None = None,
) -> pl.DataFrame:
    """The gnomAD browser CSV export, one row per amino-acid key.

    Columns: ``key``, ``in_gnomad`` (at least one nucleotide variant passes
    ``Filters - joint == PASS`` and, with ``splice_max``, has ``spliceai_ds_max`` at most
    ``splice_max`` or missing), ``allele_count`` and ``allele_number`` (summed over the
    passing nucleotide variants / their largest allele number), ``allele_frequency``, and
    ``clinvar_class`` (``"pathogenic"``, ``"benign"`` or null, from
    ``ClinVar Germline Classification`` over every nucleotide variant).
    """
    notes = [] if notes is None else notes
    if isinstance(gnomad, pl.DataFrame):
        table = gnomad
    else:
        table = pl.read_csv(gnomad, infer_schema_length=0)
    required = [GNOMAD_PROTEIN, GNOMAD_FILTER, GNOMAD_AC, GNOMAD_AN, GNOMAD_CLINVAR]
    if splice_max is not None:
        required.append(GNOMAD_SPLICE)
    _data.require_columns(table, *required)
    passing = pl.col(GNOMAD_FILTER) == "PASS"
    if splice_max is not None:
        splice = pl.col(GNOMAD_SPLICE).cast(pl.Float64, strict=False)
        passing = passing & (splice.is_null() | (splice <= splice_max))
    rows = table.lazy().select(
        hgvs_key_expr(GNOMAD_PROTEIN).alias("key"),
        passing.fill_null(False).alias("passing"),
        pl.col(GNOMAD_AC).cast(pl.Int64, strict=False).alias("ac"),
        pl.col(GNOMAD_AN).cast(pl.Int64, strict=False).alias("an"),
        _class_expr(pl.col(GNOMAD_CLINVAR)).alias("clinvar_class"),
    )
    n_rows = table.height
    n_unkeyed = rows.select(pl.col("key").is_null().sum()).collect().item()
    if n_unkeyed:
        logger.info(
            "gnomAD: %d of %d rows are not single-codon substitutions (intronic, UTR, "
            "frameshift, ...) and were dropped",
            n_unkeyed,
            n_rows,
        )
    counts = (
        rows.filter(pl.col("key").is_not_null())
        .group_by("key")
        .agg(
            pl.col("passing").any().alias("in_gnomad"),
            pl.col("ac").filter(pl.col("passing")).sum().alias("allele_count"),
            pl.col("an").filter(pl.col("passing")).max().alias("allele_number"),
        )
        .with_columns(
            (pl.col("allele_count") / pl.col("allele_number")).alias("allele_frequency")
        )
        .collect()
    )
    classes = _collapse_classes(rows, "the gnomAD export", notes)
    return counts.join(classes, on="key", how="left").sort("key")


def _stars_expr(schema: pl.Schema) -> pl.Expr | None:
    if "clinvar_stars" in schema:
        return pl.col("clinvar_stars").cast(pl.Int64, strict=False)
    if "clinvar_review_status" in schema:
        return (
            pl.col("clinvar_review_status")
            .str.to_lowercase()
            .str.strip_chars()
            .replace_strict(REVIEW_STARS, default=None, return_dtype=pl.Int64)
        )
    return None


def read_clinvar(
    clinvar: "str | PathLike | pl.DataFrame | pl.LazyFrame",
    *,
    min_stars: int | None = 1,
    notes: list[str] | None = None,
) -> pl.DataFrame:
    """ClinVar controls from the converted ClinVar table `Dataset.clinvar` reads: a
    ``variant`` column in the ``"A12V"`` format and ``clinvar_clinical_significance``.

    With ``min_stars``, records below that many review stars are dropped, read from a
    ``clinvar_stars`` column or a ``clinvar_review_status`` column (ClinVar's review status
    text). A table with neither column is used unfiltered, with a warning. Returns
    ``key`` and ``clinvar_class`` (``"pathogenic"`` or ``"benign"``).
    """
    notes = [] if notes is None else notes
    if isinstance(clinvar, (str, PathLike)):
        table = pl.scan_parquet(clinvar)
    else:
        table = clinvar.lazy()
    schema = table.collect_schema()
    _data.require_columns(
        pl.DataFrame(schema=schema), "variant", "clinvar_clinical_significance"
    )
    if min_stars is not None:
        stars = _stars_expr(schema)
        if stars is None:
            notes.append(
                "The ClinVar table has no clinvar_stars or clinvar_review_status column, "
                f"so the {min_stars}-star review filter was not applied"
            )
        else:
            table = table.filter(stars.fill_null(0) >= min_stars)
    rows = table.select(
        key_expr("variant").alias("key"),
        _class_expr(pl.col("clinvar_clinical_significance")).alias("clinvar_class"),
    )
    return _collapse_classes(rows, "the ClinVar table", notes)


def assign_groups(
    df: pl.DataFrame,
    variant_col: str,
    gnomad: pl.DataFrame,
    clinvar: pl.DataFrame | None,
    notes: list[str],
) -> pl.DataFrame:
    """Group membership for each row of ``df`` (columns ``in_<group>`` in `GROUP_LABELS`
    order, plus ``group`` and ``groups``).

    - P/LP and B/LB: the ClinVar table's class, or without one, gnomAD's
      ``ClinVar Germline Classification``.
    - gnomAD: in the gnomAD export and passing its filters.
    - Synonymous: the control rule (`fisseq_common.variant.control_expr`).
    """
    if clinvar is None:
        notes.append(
            "No ClinVar table given: P/LP and B/LB controls come from the gnomAD export's "
            "ClinVar column, which covers only variants gnomAD observed and has no "
            "review-status (star) filter"
        )
        clinvar = gnomad.select("key", "clinvar_class").filter(
            pl.col("clinvar_class").is_not_null()
        )
    variant_type = _variants.variant_type_expr(variant_col)
    frame = (
        df.select(variant_col)
        .with_columns(
            key_expr(pl.col(variant_col).str.split(":").list.first()).alias("__key"),
            _variants.control_expr(variant_type, variant_col)
            .fill_null(False)
            .alias("in_Synonymous"),
            pl.col(variant_col).str.contains(r"[:|]").fill_null(True).alias("__tagged"),
        )
        .with_columns(
            pl.when(pl.col("__tagged"))
            .then(None)
            .otherwise(pl.col("__key"))
            .alias("__key")
        )
        .join(
            gnomad.select("key", pl.col("in_gnomad").alias("in_gnomAD")),
            left_on="__key",
            right_on="key",
            how="left",
            maintain_order="left",
        )
        .join(
            clinvar.rename({"clinvar_class": "__class"}),
            left_on="__key",
            right_on="key",
            how="left",
            maintain_order="left",
        )
        .with_columns(
            (pl.col("__class") == "pathogenic").fill_null(False).alias("in_P/LP"),
            (pl.col("__class") == "benign").fill_null(False).alias("in_B/LB"),
            pl.col("in_gnomAD").fill_null(False),
        )
    )
    groups = pl.concat_list(
        [pl.when(pl.col(f"in_{g}")).then(pl.lit(g)) for g in GROUP_PRIORITY]
    ).list.drop_nulls()
    return frame.select(
        *(f"in_{g}" for g in GROUP_LABELS),
        groups.list.first().alias("group"),
        groups.alias("groups"),
    )
