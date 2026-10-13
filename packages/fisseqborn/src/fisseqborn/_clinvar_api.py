"""The conditions (diseases) of ClinVar variations, from NCBI E-utilities
(``esummary`` on the ``clinvar`` database).

NCBI allows 3 requests per second, or 10 with an API key in the ``NCBI_API_KEY``
environment variable; a request summarises up to `BATCH` variations.
"""

import datetime
import logging
import os
import time
import urllib.parse
from collections.abc import Iterable
from typing import Any

import polars as pl

from . import _cache, _http

logger = logging.getLogger(__name__)

URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
BATCH = 200

#: Trait names that say nothing about a disease.
UNINFORMATIVE = frozenset({"not provided", "not specified", "see cases"})

SCHEMA = {
    "clinvar_variation_id": pl.String,
    "clinvar_conditions": pl.List(pl.String),
    "fetched": pl.Date,
}


def _conditions(summary: dict[str, Any]) -> list[str]:
    traits = (summary.get("germline_classification") or {}).get("trait_set") or []
    names = (t.get("trait_name") for t in traits)
    return list(dict.fromkeys(n for n in names if n and n.lower() not in UNINFORMATIVE))


def _check(payload: Any) -> None:
    error = payload.get("error") or payload.get("esummaryresult")
    if error and "rate" in str(error).lower():
        raise _http.RateLimited(str(error))
    if "result" not in payload:
        raise RuntimeError(f"NCBI esummary error: {error or payload}")


def _fetch(ids: list[str]) -> pl.DataFrame:
    key = os.environ.get("NCBI_API_KEY")
    pause = 0.11 if key else 0.34
    today = datetime.date.today()
    rows = []
    for start in range(0, len(ids), BATCH):
        batch = ids[start : start + BATCH]
        params = {"db": "clinvar", "id": ",".join(batch), "retmode": "json"}
        if key:
            params["api_key"] = key
        if start:
            time.sleep(pause)
        result = _http.request_json(
            f"{URL}?{urllib.parse.urlencode(params)}", check=_check
        )["result"]
        for uid in batch:
            summary = result.get(uid)
            if summary is None or "error" in summary:
                continue
            rows.append(
                {
                    "clinvar_variation_id": uid,
                    "clinvar_conditions": _conditions(summary),
                    "fetched": today,
                }
            )
    logger.info("Fetched the conditions of %d ClinVar variations", len(rows))
    return pl.DataFrame(rows, schema=SCHEMA)


def fetch_conditions(
    variation_ids: Iterable[str | None],
    *,
    cache: "_cache.Cache" = False,
    refresh: bool = False,
) -> pl.DataFrame:
    """``clinvar_variation_id`` → ``clinvar_conditions`` (the condition names of its
    germline classification, deduplicated, "not provided" / "not specified" left out).

    With a ``cache``, every variation fetched is kept in ``<cache>/clinvar/conditions.parquet``
    and only those missing from it are fetched (all of them with ``refresh``).
    """
    ids = sorted({str(i) for i in variation_ids if i not in (None, "")})
    path = _cache.cache_dir(cache, "clinvar", "conditions.parquet")
    known = pl.DataFrame(schema=SCHEMA)
    if path is not None and path.is_file():
        known = pl.read_parquet(path)
    if refresh:
        known = known.filter(~pl.col("clinvar_variation_id").is_in(ids))
    missing = sorted(set(ids) - set(known["clinvar_variation_id"]))
    if missing:
        known = pl.concat([known, _fetch(missing)])
        if path is not None:
            _http.write_atomic(known, path)
    return known.filter(pl.col("clinvar_variation_id").is_in(ids)).select(
        "clinvar_variation_id", "clinvar_conditions"
    )
