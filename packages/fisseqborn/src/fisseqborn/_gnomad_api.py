"""Per-gene downloads from the gnomAD browser's GraphQL API
(https://gnomad.broadinstitute.org/api): the gene's variants, and the ClinVar variants
gnomAD mirrors.

The API allows a few dozen requests per minute per IP; a gene is one request.
"""

from typing import Any

import polars as pl

from . import _cache, _http

URL = "https://gnomad.broadinstitute.org/api"

_VARIANTS = """
query ($gene: String!, $dataset: DatasetId!) {
  gene(gene_symbol: $gene, reference_genome: GRCh38) {
    variants(dataset: $dataset) {
      variant_id hgvsp consequence transcript_id
      joint { ac an filters }
      exome { ac an filters }
      genome { ac an filters }
      in_silico_predictors { id value }
    }
  }
}
"""

_CLINVAR = """
query ($gene: String!) {
  meta { clinvar_release_date }
  gene(gene_symbol: $gene, reference_genome: GRCh38) {
    clinvar_variants {
      variant_id clinvar_variation_id hgvsp major_consequence
      clinical_significance review_status gold_stars
    }
  }
}
"""

#: The columns of `fetch_variants`; the gnomAD browser's CSV export is converted to them.
VARIANT_SCHEMA = {
    "variant_id": pl.String,
    "hgvsp": pl.String,
    "consequence": pl.String,
    "filters": pl.String,
    "allele_count": pl.Int64,
    "allele_number": pl.Int64,
    "spliceai_ds_max": pl.Float64,
    "clinvar_significance": pl.String,
}


def _check(payload: Any) -> None:
    errors = payload.get("errors")
    if not errors:
        return
    messages = "; ".join(str(e.get("message", e)) for e in errors)
    if "rate limit" in messages.lower():
        raise _http.RateLimited(messages)
    raise RuntimeError(f"gnomAD API error: {messages}")


def query(text: str, **variables: Any) -> dict[str, Any]:
    payload = _http.request_json(
        URL, body={"query": text, "variables": variables}, check=_check
    )
    return payload["data"]


def _gene(data: dict[str, Any], gene: str) -> dict[str, Any]:
    if data.get("gene") is None:
        raise ValueError(
            f"gnomAD has no gene {gene!r} (use the HGNC symbol, e.g. LMNA)"
        )
    return data["gene"]


def _callset(variant: dict[str, Any]) -> dict[str, Any] | None:
    """The joint (exome + genome) call, else whichever callset has the variant."""
    return variant.get("joint") or variant.get("exome") or variant.get("genome")


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def variant_rows(variants: list[dict[str, Any]]) -> pl.DataFrame:
    """`VARIANT_SCHEMA` rows of the API's variants: ``filters`` is ``"PASS"`` when the
    joint callset has no filters, else the filters joined by ``,``."""
    rows = []
    for v in variants:
        call = _callset(v) or {}
        filters = call.get("filters")
        predictors = {p["id"]: p["value"] for p in v.get("in_silico_predictors") or []}
        rows.append(
            {
                "variant_id": v["variant_id"],
                "hgvsp": v.get("hgvsp"),
                "consequence": v.get("consequence"),
                "filters": None
                if filters is None
                else ",".join(filters)
                if filters
                else "PASS",
                "allele_count": call.get("ac"),
                "allele_number": call.get("an"),
                "spliceai_ds_max": _float(predictors.get("spliceai_ds_max")),
                "clinvar_significance": None,
            }
        )
    return pl.DataFrame(rows, schema=VARIANT_SCHEMA)


def fetch_variants(
    gene: str,
    dataset: str = "gnomad_r4",
    *,
    cache: "_cache.Cache" = False,
    refresh: bool = False,
) -> pl.DataFrame:
    """Every gnomAD variant of ``gene`` in ``dataset``, as `VARIANT_SCHEMA` rows (HGVS
    protein consequence on the canonical transcript), cached as
    ``<cache>/gnomad/<dataset>/<gene>.parquet``."""

    def fetch() -> pl.DataFrame:
        data = query(_VARIANTS, gene=gene, dataset=dataset)
        return variant_rows(_gene(data, gene)["variants"])

    return _http.cached_frame(
        cache, ("gnomad", dataset, f"{gene}.parquet"), fetch, refresh=refresh
    )


def clinvar_rows(variants: list[dict[str, Any]], release: str | None) -> pl.DataFrame:
    """The API's ClinVar variants in the normalised ClinVar columns (see
    `fisseqborn.calibration._inputs.CLINVAR_SCHEMA`), keyed by ``hgvsp``."""
    from .calibration import _inputs

    rows = [
        {
            "hgvsp": v.get("hgvsp"),
            "clinvar_clinical_significance": v.get("clinical_significance"),
            "clinvar_review_status": v.get("review_status"),
            "clinvar_stars": v.get("gold_stars"),
            "clinvar_variation_id": v.get("clinvar_variation_id"),
            "clinvar_variant_id": v.get("variant_id"),
            "clinvar_consequence": v.get("major_consequence"),
            "clinvar_release": release,
        }
        for v in variants
    ]
    return pl.DataFrame(
        rows,
        schema={
            "hgvsp": pl.String,
            **_inputs.CLINVAR_SCHEMA,
            "clinvar_variant_id": pl.String,
            "clinvar_consequence": pl.String,
        },
    )


def fetch_clinvar(
    gene: str, *, cache: "_cache.Cache" = False, refresh: bool = False
) -> pl.DataFrame:
    """The ClinVar variants gnomAD mirrors for ``gene`` (GRCh38), with the ClinVar
    release in ``clinvar_release``, cached as ``<cache>/clinvar/gnomad/<gene>.parquet``."""

    def fetch() -> pl.DataFrame:
        data = query(_CLINVAR, gene=gene)
        release = (data.get("meta") or {}).get("clinvar_release_date")
        return clinvar_rows(_gene(data, gene)["clinvar_variants"], release)

    return _http.cached_frame(
        cache, ("clinvar", "gnomad", f"{gene}.parquet"), fetch, refresh=refresh
    )
