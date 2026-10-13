"""Control groups for calibration: gnomAD and ClinVar inputs keyed to ``meta_aa_changes``.

fisseqborn keys variants by one-letter amino-acid change (``"R123W"``, ``"R123*"``).
gnomAD and ClinVar are nucleotide-level: their HGVS protein consequence (``p.Arg123Trp``)
is converted to that key, and nucleotide variants with the same protein change are
collapsed into one. Only single-codon substitutions (missense, synonymous, nonsense) can
match; multi-codon labels and ``:<tag>`` pseudo-variants are never controls.

`Dataset.gnomad` and `Dataset.clinvar` put the per-key results on the dataset
(``meta_gnomad_*``, ``meta_clinvar_*``); `assign_groups` reads the groups from there.
"""

import logging
import re
from os import PathLike

import polars as pl

from .. import _cache, _data, _gnomad_api, _variants

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

# Columns of the gnomAD browser's "Export variants to CSV".
GNOMAD_ID = "gnomAD ID"
GNOMAD_PROTEIN = "Protein Consequence"
GNOMAD_CONSEQUENCE = "VEP Annotation"
GNOMAD_CLINVAR = "ClinVar Germline Classification"
GNOMAD_FILTER = "Filters - joint"
GNOMAD_AC, GNOMAD_AN = "Allele Count", "Allele Number"
GNOMAD_SPLICE = "spliceai_ds_max"

#: The normalised ClinVar record columns `Dataset.clinvar` joins (with a ``meta_`` prefix).
CLINVAR_SCHEMA = {
    "clinvar_clinical_significance": pl.String,
    "clinvar_review_status": pl.String,
    "clinvar_stars": pl.Int64,
    "clinvar_variation_id": pl.String,
    "clinvar_release": pl.String,
}

#: ``clinvar.<YYYYMM>.<field>``: the columns bbi-lab/variant-annotation's ClinVar step adds.
VARIANT_ANNOTATION_CLINVAR = re.compile(r"^clinvar\.(\d{6})\.(\w+)$")

_HGVS = r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|=)$"


def key_expr(column: str | pl.Expr) -> pl.Expr:
    """The join key of a one-letter label: single substitutions only (``"A12V"``,
    ``"A12A"``, ``"A12*"``; ``X`` is read as ``*``), else null."""
    col = pl.col(column) if isinstance(column, str) else column
    return col.str.replace(r"X$", "*").str.extract(r"^([A-Z]\d+[A-Z*])$", 1)


def dataset_key_expr(variant_col: str) -> pl.Expr:
    """The key of a dataset's variant labels; null for ``:<tag>`` / ``|`` labels, which
    are never matched to gnomAD or ClinVar."""
    col = pl.col(variant_col)
    return (
        pl.when(col.str.contains(r"[:|]").fill_null(True))
        .then(None)
        .otherwise(key_expr(col))
    )


def hgvs_key_expr(column: str | pl.Expr) -> pl.Expr:
    """``p.Arg123Trp`` -> ``R123W`` (``p.Arg123Ter`` -> ``R123*``, ``p.Arg123=`` ->
    ``R123R``); null for anything that is not a single-codon substitution. A leading
    accession (``NP_000268.1:p.Arg123Trp``) and predicted-change parentheses
    (``p.(Arg123Trp)``) are ignored."""
    col = pl.col(column) if isinstance(column, str) else column
    col = col.str.replace(r"^[^:]*:", "").str.replace(r"^p\.\((.*)\)$", "p.$1")
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


def _read_table(source: "str | PathLike | pl.DataFrame | pl.LazyFrame") -> pl.DataFrame:
    """A frame as-is, a parquet file, or a CSV / TSV (optionally gzipped) read as strings."""
    if isinstance(source, pl.LazyFrame):
        return source.collect()
    if isinstance(source, pl.DataFrame):
        return source
    name = str(source).lower().removesuffix(".gz")
    if name.endswith(".parquet"):
        return pl.read_parquet(source)
    separator = "\t" if name.endswith((".tsv", ".txt")) else ","
    return pl.read_csv(source, separator=separator, infer_schema_length=0)


def _blank_to_null(table: pl.DataFrame) -> pl.DataFrame:
    strings = [c for c, t in table.schema.items() if t == pl.String]
    return table.with_columns(
        pl.when(pl.col(c).str.strip_chars() == "")
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in strings
    )


def gnomad_rows(
    gnomad: "str | PathLike | pl.DataFrame | None",
    *,
    gene: str = "LMNA",
    dataset: str = "gnomad_r4",
    splice_max: float | None = None,
    cache: "_cache.Cache" = False,
    refresh: bool = False,
) -> tuple[pl.DataFrame, str]:
    """gnomAD's nucleotide variants in `_gnomad_api.VARIANT_SCHEMA`, and where they came
    from.

    ``gnomad`` is the gnomAD browser's "Export variants to CSV" for the gene (a path or a
    frame of strings), a frame already in `VARIANT_SCHEMA` (e.g. from
    `_gnomad_api.fetch_variants`), or ``None`` to download ``gene`` from the gnomAD API.
    """
    schema = _gnomad_api.VARIANT_SCHEMA
    if gnomad is None:
        rows = _gnomad_api.fetch_variants(gene, dataset, cache=cache, refresh=refresh)
        return rows, f"gnomAD API ({dataset}, {gene})"
    source = "table" if isinstance(gnomad, pl.DataFrame) else str(gnomad)
    table = _read_table(gnomad)
    if set(schema) <= set(table.columns):
        return table.select(pl.col(c).cast(t) for c, t in schema.items()), source
    required = [GNOMAD_PROTEIN, GNOMAD_FILTER, GNOMAD_AC, GNOMAD_AN]
    if splice_max is not None:
        required.append(GNOMAD_SPLICE)
    _data.require_columns(table, *required)

    def optional(column: str) -> pl.Expr:
        return pl.col(column) if column in table.columns else pl.lit(None)

    rows = table.select(
        optional(GNOMAD_ID).alias("variant_id"),
        pl.col(GNOMAD_PROTEIN).alias("hgvsp"),
        optional(GNOMAD_CONSEQUENCE).alias("consequence"),
        pl.col(GNOMAD_FILTER).alias("filters"),
        pl.col(GNOMAD_AC).cast(pl.Int64, strict=False).alias("allele_count"),
        pl.col(GNOMAD_AN).cast(pl.Int64, strict=False).alias("allele_number"),
        optional(GNOMAD_SPLICE).cast(pl.Float64, strict=False).alias("spliceai_ds_max"),
        optional(GNOMAD_CLINVAR).alias("clinvar_significance"),
    )
    return rows.select(pl.col(c).cast(t) for c, t in schema.items()), source


def gnomad_table(
    rows: pl.DataFrame,
    *,
    splice_max: float | None = None,
    notes: list[str] | None = None,
) -> pl.DataFrame:
    """gnomAD per amino-acid key, from `gnomad_rows`.

    Columns: ``key``; ``in_gnomad`` (at least one nucleotide variant passes the joint
    filters (``PASS``) and, with ``splice_max``, has ``spliceai_ds_max`` at most
    ``splice_max`` or missing); ``allele_count`` and ``allele_number`` (summed over the
    passing nucleotide variants / their largest allele number); ``allele_frequency``;
    ``spliceai_ds_max`` (the largest over every nucleotide variant); and
    ``clinvar_class`` (``"pathogenic"``, ``"benign"`` or null, from the browser export's
    ClinVar column over every nucleotide variant; null for API downloads).
    """
    notes = [] if notes is None else notes
    passing = pl.col("filters") == "PASS"
    if splice_max is not None:
        splice = pl.col("spliceai_ds_max")
        passing = passing & (splice.is_null() | (splice <= splice_max))
    keyed = rows.lazy().select(
        hgvs_key_expr("hgvsp").alias("key"),
        passing.fill_null(False).alias("passing"),
        pl.col("allele_count").alias("ac"),
        pl.col("allele_number").alias("an"),
        "spliceai_ds_max",
        _class_expr(pl.col("clinvar_significance")).alias("clinvar_class"),
    )
    n_unkeyed = keyed.select(pl.col("key").is_null().sum()).collect().item()
    if n_unkeyed:
        logger.info(
            "gnomAD: %d of %d rows are not single-codon substitutions (intronic, UTR, "
            "frameshift, ...) and were dropped",
            n_unkeyed,
            rows.height,
        )
    counts = (
        keyed.filter(pl.col("key").is_not_null())
        .group_by("key")
        .agg(
            pl.col("passing").any().alias("in_gnomad"),
            pl.col("ac").filter(pl.col("passing")).sum().alias("allele_count"),
            pl.col("an").filter(pl.col("passing")).max().alias("allele_number"),
            pl.col("spliceai_ds_max").max(),
        )
        .with_columns(
            (pl.col("allele_count") / pl.col("allele_number")).alias("allele_frequency")
        )
        .collect()
    )
    classes = _collapse_classes(keyed, "the gnomAD export", notes)
    return (
        counts.join(classes, on="key", how="left")
        .select(
            "key",
            "in_gnomad",
            "allele_count",
            "allele_number",
            "allele_frequency",
            "spliceai_ds_max",
            "clinvar_class",
        )
        .sort("key")
    )


def _stars_expr(schema: pl.Schema) -> pl.Expr | None:
    status = None
    if "clinvar_review_status" in schema:
        status = (
            pl.col("clinvar_review_status")
            .str.to_lowercase()
            .str.strip_chars()
            .replace_strict(REVIEW_STARS, default=None, return_dtype=pl.Int64)
        )
    if "clinvar_stars" in schema:
        stars = pl.col("clinvar_stars").cast(pl.Int64, strict=False)
        return stars if status is None else pl.coalesce(stars, status)
    return status


def variant_annotation_records(
    table: pl.DataFrame,
    *,
    protein_col: str = "mapped_hgvs_p",
    release: str | None = None,
) -> pl.DataFrame:
    """ClinVar records from the output of bbi-lab/variant-annotation
    (https://github.com/bbi-lab/variant-annotation): its ``clinvar.<YYYYMM>.<field>``
    columns become ``clinvar_<field>`` (with `CLINVAR_SCHEMA`'s types), the release goes in
    ``clinvar_release`` (``"YYYY-MM"``), and ``key`` comes from ``protein_col``. Rows
    without a ClinVar record are dropped. With several releases in the table the latest is
    used, unless ``release`` (``"YYYYMM"`` or ``"YYYY-MM"``) picks one.
    """
    fields: dict[str, dict[str, str]] = {}
    for column in table.columns:
        match = VARIANT_ANNOTATION_CLINVAR.match(column)
        if match:
            fields.setdefault(match[1], {})[column] = f"clinvar_{match[2]}"
    if not fields:
        raise ValueError(
            "No clinvar.<YYYYMM>.<field> columns: expected the output of "
            "variant-annotation's ClinVar step (src/scripts/run_annotate_clinvar.sh)"
        )
    if release is None:
        release = max(fields)
    release = release.replace("-", "")
    if release not in fields:
        raise ValueError(
            f"ClinVar release {release} is not in the table; it has {', '.join(sorted(fields))}"
        )
    _data.require_columns(table, protein_col)
    columns = fields[release]
    if "clinvar_clinical_significance" not in columns.values():
        raise ValueError(f"No clinvar.{release}.clinical_significance column")
    records = _blank_to_null(
        table.select(
            hgvs_key_expr(protein_col).alias("key"),
            *(pl.col(c).alias(new) for c, new in columns.items()),
        )
    ).with_columns(pl.lit(f"{release[:4]}-{release[4:]}").alias("clinvar_release"))
    for column, dtype in CLINVAR_SCHEMA.items():
        if column not in records.columns:
            records = records.with_columns(pl.lit(None, dtype=dtype).alias(column))
        else:
            records = records.with_columns(pl.col(column).cast(dtype, strict=False))
    return records.filter(pl.col("clinvar_clinical_significance").is_not_null())


def clinvar_records(
    clinvar: "str | PathLike | pl.DataFrame | pl.LazyFrame | None",
    *,
    gene: str = "LMNA",
    protein_col: str = "mapped_hgvs_p",
    release: str | None = None,
    cache: "_cache.Cache" = False,
    refresh: bool = False,
) -> tuple[pl.DataFrame, str]:
    """ClinVar records with a ``key`` and the `CLINVAR_SCHEMA` columns (plus any other
    ``clinvar_*`` fields of the source), and where they came from.

    ``clinvar`` is variant-annotation's output (see `variant_annotation_records`): a TSV,
    CSV or parquet path, or a frame; ``None`` downloads ``gene``'s ClinVar variants from
    the gnomAD API.
    """
    if clinvar is None:
        records = _gnomad_api.fetch_clinvar(gene, cache=cache, refresh=refresh)
        records = records.select(
            hgvs_key_expr("hgvsp").alias("key"), pl.exclude("hgvsp")
        )
        return records, f"gnomAD API ({gene})"
    source = (
        "table" if isinstance(clinvar, (pl.DataFrame, pl.LazyFrame)) else str(clinvar)
    )
    records = variant_annotation_records(
        _read_table(clinvar), protein_col=protein_col, release=release
    )
    return records, source


def clinvar_classes(
    records: pl.DataFrame,
    *,
    min_stars: int | None = 1,
    notes: list[str] | None = None,
) -> pl.DataFrame:
    """The ExCALIBR class of each key of `clinvar_records`: ``key`` and ``clinvar_class``
    (``"pathogenic"`` or ``"benign"``).

    With ``min_stars``, records below that many review stars are dropped first (stars from
    ``clinvar_stars``, else ``clinvar_review_status``, ClinVar's own mapping). Keys with
    both pathogenic and benign calls are left out (see `_collapse_classes`).
    """
    notes = [] if notes is None else notes
    table = records.lazy()
    if min_stars is not None:
        stars = _stars_expr(records.schema)
        if stars is None:
            notes.append(
                "The ClinVar records have no clinvar_stars or clinvar_review_status, "
                f"so the {min_stars}-star review filter was not applied"
            )
        else:
            table = table.filter(stars.fill_null(0) >= min_stars)
    rows = table.select(
        "key",
        _class_expr(pl.col("clinvar_clinical_significance")).alias("clinvar_class"),
    )
    return _collapse_classes(rows, "ClinVar", notes)


def assign_groups(
    df: pl.DataFrame,
    variant_col: str,
    notes: list[str],
    *,
    gnomad_col: str = "meta_gnomad_in",
    clinvar_col: str = "meta_clinvar_class",
    fallback_col: str = "meta_gnomad_clinvar_class",
) -> tuple[pl.DataFrame, str]:
    """Group membership for each row of ``df`` (columns ``in_<group>`` in `GROUP_LABELS`
    order, plus ``group`` and ``groups``), and which column the ClinVar classes came from.

    - P/LP and B/LB: ``clinvar_col`` (`Dataset.clinvar`), or without it, gnomAD's
      browser-export ClinVar column ``fallback_col`` (`Dataset.gnomad` with a CSV).
    - gnomAD: ``gnomad_col`` (`Dataset.gnomad`).
    - Synonymous: the control rule (`fisseq_common.variant.control_expr`).
    """
    if gnomad_col not in df.columns:
        raise ValueError(
            f"No {gnomad_col} column: run .gnomad() before .excalibr() to add gnomAD"
        )
    if clinvar_col in df.columns:
        class_col = clinvar_col
    elif fallback_col in df.columns and df[fallback_col].is_not_null().any():
        class_col = fallback_col
        notes.append(
            "No .clinvar() annotation: P/LP and B/LB controls come from the gnomAD "
            "export's ClinVar column, which covers only variants gnomAD observed and has "
            "no review-status (star) filter"
        )
    else:
        raise ValueError(
            f"No {clinvar_col} column: run .clinvar() before .excalibr() to add the "
            "pathogenic and benign controls"
        )
    variant_type = _variants.variant_type_expr(variant_col)
    tagged = pl.col(variant_col).str.contains(r"[:|]").fill_null(True)
    frame = df.select(
        (~tagged & (pl.col(class_col) == "pathogenic"))
        .fill_null(False)
        .alias("in_P/LP"),
        (~tagged & (pl.col(class_col) == "benign")).fill_null(False).alias("in_B/LB"),
        (~tagged & pl.col(gnomad_col)).fill_null(False).alias("in_gnomAD"),
        _variants.control_expr(variant_type, variant_col)
        .fill_null(False)
        .alias("in_Synonymous"),
    )
    groups = pl.concat_list(
        [pl.when(pl.col(f"in_{g}")).then(pl.lit(g)) for g in GROUP_PRIORITY]
    ).list.drop_nulls()
    return (
        frame.select(
            *(f"in_{g}" for g in GROUP_LABELS),
            groups.list.first().alias("group"),
            groups.alias("groups"),
        ),
        class_col,
    )
