import io
import json
import urllib.error
import urllib.parse

import polars as pl
import pytest

import fisseqborn as fb
from fisseqborn import _clinvar_api, _gnomad_api, _http

VARIANTS = [
    {
        "variant_id": "1-156114922-G-C",
        "hgvsp": "p.Ala1Val",
        "consequence": "missense_variant",
        "transcript_id": "ENST00000368300",
        "joint": {"ac": 3, "an": 1000, "filters": []},
        "exome": {"ac": 3, "an": 900, "filters": []},
        "genome": None,
        "in_silico_predictors": [
            {"id": "cadd", "value": "23.3"},
            {"id": "spliceai_ds_max", "value": "0.05"},
        ],
    },
    {
        "variant_id": "1-156114924-G-T",
        "hgvsp": "p.Ala2Val",
        "consequence": "missense_variant",
        "transcript_id": "ENST00000368300",
        "joint": {"ac": 0, "an": 1000, "filters": ["AC0"]},
        "exome": None,
        "genome": None,
        "in_silico_predictors": None,
    },
    {
        "variant_id": "1-156114000-C-T",
        "hgvsp": None,
        "consequence": "intron_variant",
        "transcript_id": "ENST00000368300",
        "joint": {"ac": 7, "an": 1000, "filters": []},
        "exome": None,
        "genome": None,
        "in_silico_predictors": [],
    },
]

CLINVAR = [
    {
        "variant_id": "1-156114922-G-C",
        "clinvar_variation_id": "14498",
        "hgvsp": "p.Ala1Val",
        "major_consequence": "missense_variant",
        "clinical_significance": "Pathogenic",
        "review_status": "reviewed by expert panel",
        "gold_stars": 3,
    },
    {
        "variant_id": "1-156114924-G-T",
        "clinvar_variation_id": "1362553",
        "hgvsp": "p.Ala2Val",
        "major_consequence": "missense_variant",
        "clinical_significance": "Benign",
        "review_status": "no assertion criteria provided",
        "gold_stars": 0,
    },
]

ESUMMARY = {
    "result": {
        "uids": ["14498", "1362553"],
        "14498": {
            "germline_classification": {
                "trait_set": [
                    {"trait_name": "Dilated cardiomyopathy 1A"},
                    {"trait_name": "not provided"},
                    {"trait_name": "Dilated cardiomyopathy 1A"},
                    {"trait_name": "Hutchinson-Gilford syndrome"},
                ]
            }
        },
        "1362553": {"germline_classification": {"trait_set": []}},
    }
}


class FakeApi:
    """Stands in for `urllib.request.urlopen`: answers the gnomAD GraphQL queries and
    NCBI esummary, after ``fail`` rate-limited responses."""

    def __init__(self):
        self.requests: list[tuple[str, dict | None]] = []
        self.fail = 0

    def __call__(self, request, timeout=None):
        body = json.loads(request.data) if request.data else None
        self.requests.append((request.full_url, body))
        if self.fail:
            self.fail -= 1
            return io.BytesIO(
                json.dumps(
                    {"errors": [{"message": "Query rate limit exceeded"}]}
                ).encode()
            )
        if request.full_url.startswith(_clinvar_api.URL):
            ids = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)
            ids = ids["id"][0].split(",")
            result = {u: ESUMMARY["result"][u] for u in ids if u in ESUMMARY["result"]}
            return io.BytesIO(json.dumps({"result": {"uids": ids, **result}}).encode())
        if body["variables"]["gene"] == "NOPE":
            payload = {"data": {"gene": None, "meta": {}}}
        elif "clinvar_variants" in body["query"]:
            payload = {
                "data": {
                    "meta": {"clinvar_release_date": "2026-09-28"},
                    "gene": {"clinvar_variants": CLINVAR},
                }
            }
        else:
            payload = {"data": {"gene": {"variants": VARIANTS}}}
        return io.BytesIO(json.dumps(payload).encode())


@pytest.fixture
def api(monkeypatch):
    fake = FakeApi()
    monkeypatch.setattr(_http.urllib.request, "urlopen", fake)
    monkeypatch.setattr(_http, "RETRY_DELAYS", (0.0, 0.0))
    monkeypatch.setattr(_clinvar_api.time, "sleep", lambda s: None)
    return fake


def test_fetch_variants(api):
    rows = _gnomad_api.fetch_variants("LMNA")
    assert rows.schema == pl.Schema(_gnomad_api.VARIANT_SCHEMA)
    assert rows["filters"].to_list() == ["PASS", "AC0", "PASS"]
    assert rows["allele_count"].to_list() == [3, 0, 7]
    assert rows["spliceai_ds_max"].to_list() == [0.05, None, None]
    assert api.requests[0][1]["variables"] == {"gene": "LMNA", "dataset": "gnomad_r4"}
    with pytest.raises(ValueError, match="NOPE"):
        _gnomad_api.fetch_variants("NOPE")


def test_rate_limited_requests_are_retried(api):
    api.fail = 2
    assert len(_gnomad_api.fetch_variants("LMNA")) == 3
    assert len(api.requests) == 3
    api.fail = 3
    with pytest.raises(_http.RateLimited):
        _gnomad_api.fetch_variants("LMNA")


def test_http_errors(monkeypatch):
    calls = []

    def urlopen(request, timeout=None):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(_http.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        _http.request_json("https://example.org")
    assert len(calls) == 1  # not retried


def test_downloads_are_cached(api, tmp_path):
    first = _gnomad_api.fetch_variants("LMNA", cache=tmp_path)
    assert (tmp_path / "gnomad" / "gnomad_r4" / "LMNA.parquet").is_file()
    assert _gnomad_api.fetch_variants("LMNA", cache=tmp_path).equals(first)
    assert len(api.requests) == 1
    _gnomad_api.fetch_variants("LMNA", cache=tmp_path, refresh=True)
    assert len(api.requests) == 2
    _gnomad_api.fetch_variants("LMNA")
    assert len(api.requests) == 3  # no cache: fetched every time


def test_fetch_clinvar(api, tmp_path):
    records = _gnomad_api.fetch_clinvar("LMNA", cache=tmp_path)
    assert records["clinvar_stars"].to_list() == [3, 0]
    assert records["clinvar_release"].to_list() == ["2026-09-28"] * 2
    assert (tmp_path / "clinvar" / "gnomad" / "LMNA.parquet").is_file()


def test_fetch_conditions(api, tmp_path, monkeypatch):
    monkeypatch.setattr(_clinvar_api, "BATCH", 1)
    out = _clinvar_api.fetch_conditions(["14498", "1362553", None], cache=tmp_path)
    assert dict(out.iter_rows()) == {
        "14498": ["Dilated cardiomyopathy 1A", "Hutchinson-Gilford syndrome"],
        "1362553": [],
    }
    assert len(api.requests) == 2  # one per batch
    again = _clinvar_api.fetch_conditions(["14498"], cache=tmp_path)
    assert again["clinvar_variation_id"].to_list() == ["14498"]
    assert len(api.requests) == 2
    _clinvar_api.fetch_conditions(["14498"], cache=tmp_path, refresh=True)
    assert len(api.requests) == 3


def test_dataset_downloads(api, tmp_path):
    ds = (
        fb.Dataset(pl.DataFrame({"meta_aa_changes": ["A1V", "A2V", "A3A"]}))
        .variant_type()
        .gnomad(cache=tmp_path)
        .clinvar(min_stars=0, cache=tmp_path)
    )
    df = ds.df
    assert df["meta_gnomad_in"].to_list() == [True, False, None]
    assert df["meta_gnomad_clinvar_class"].to_list() == [None, None, None]
    assert df["meta_clinvar_stars"].to_list() == [3, 0, None]
    assert df["meta_clinvar_class"].to_list() == ["pathogenic", "benign", None]
    assert df["meta_clinvar_conditions"].to_list() == [
        ["Dilated cardiomyopathy 1A", "Hutchinson-Gilford syndrome"],
        [],
        None,
    ]
    assert ds.annotations["gnomad"]["source"] == "gnomAD API (gnomad_r4, LMNA)"
    assert ds.annotations["clinvar"]["release"] == "2026-09-28"
