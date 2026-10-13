from datetime import date

import polars as pl
import pytest
from fisseqborn_calibration_data import variant_annotation

import fisseqborn as fb
from fisseqborn import _clinvar_api, fisseq
from fisseqborn._variants import classify_variant

LABELS = [
    "A12A",
    "A12V",
    "A12fs",
    "A12-",
    "A12-|L13-",
    "A12-|L15-",
    "A12-|L13-|K14-",
    "A12*",
    "A12X",
    "WT",
    "A12A:downsampled",
    "A12V|L13L",
    "12V",
    "a12a",
]


@pytest.fixture
def labels() -> fb.Dataset:
    return fb.Dataset(
        pl.DataFrame({"meta_aa_changes": LABELS, "feature_0": range(len(LABELS))})
    )


def test_rejects_non_polars():
    with pytest.raises(TypeError, match="polars"):
        fb.Dataset({"a": [1]})


def test_variant_type_matches_reference(labels):
    df = labels.variant_type().df
    assert df["meta_variant_type"].to_list() == [classify_variant(v) for v in LABELS]
    assert dict(zip(LABELS, df["meta_variant_type"])) == {
        "A12A": "Synonymous",
        "A12V": "Single Missense",
        "A12fs": "Frameshift",
        "A12-": "3nt Deletion",
        "A12-|L13-": "3nt Deletion",
        "A12-|L15-": "Other",
        "A12-|L13-|K14-": "Other",
        "A12*": "Nonsense",
        "A12X": "Nonsense",
        "WT": "WT",
        "A12A:downsampled": "Synonymous",
        "A12V|L13L": "Single Missense",
        "12V": "Other",
        "a12a": "Other",
    }


def test_control_excludes_tagged_labels(labels):
    df = labels.variant_type().df
    controls = df.filter("meta_is_control")["meta_aa_changes"].to_list()
    assert controls == ["A12A"]


def test_null_label_stays_null():
    df = fb.Dataset(pl.DataFrame({"meta_aa_changes": ["A1A", None]})).variant_type().df
    assert df["meta_variant_type"].to_list() == ["Synonymous", None]


def test_chaining_returns_copies(labels):
    typed = labels.variant_type()
    assert "meta_variant_type" not in labels.columns
    assert "meta_variant_type" in typed.columns
    assert typed is not labels


def test_position_domain_and_tile():
    ds = fb.Dataset(
        pl.DataFrame(
            {"meta_aa_changes": ["A5V", "A95fs", "A222V", "A300-|L301-", "WT", "A700V"]}
        )
    )
    df = ds.position().position(strict=True, output_col="strict").domain().tile().df
    assert df["meta_position"].to_list() == [5, 95, 222, 300, None, 700]
    assert df["strict"].to_list() == [5, None, 222, None, None, 700]
    # overlapping ranges: the first listed wins
    assert df["meta_domain"].to_list() == ["Head", None, "Coil 1B", None, None, None]
    assert df["meta_tile"].to_list() == ["T1", "T1", "T3", "T4", None, None]
    both = ds.tile(allow_multiple=True).df["meta_tile"].to_list()
    assert both == ["T1", "T1,T2", "T3", "T4", None, None]


def test_domain_custom_regions():
    ds = fb.Dataset(pl.DataFrame({"meta_aa_changes": ["A1V", "A5V"]}))
    df = ds.domain({"first": (1, 2)}, output_col="region").df
    assert df["region"].to_list() == ["first", None]


def test_clinvar(tmp_path, monkeypatch):
    clinvar = variant_annotation(
        [f"NP_000268.1:p.Ala{v[1:-1]}Val" for v in
         ["A2V", "A2V", "A3V", "A4V", "A5V", "A5V", "A6V", "A7V", "A8V"]],
        [
            "Benign",
            "Pathogenic",
            "Uncertain significance",
            "Likely pathogenic/Pathogenic",
            "Uncertain significance",
            "Likely pathogenic",
            "Conflicting classifications of pathogenicity",
            "Benign",
            "Uncertain significance/Uncertain risk allele",
        ],
        ["criteria provided, single submitter"] * 9,
        allele_id=["a", "b", "c", "d", "e", "f", "g", "h", "i"],
    )  # fmt: skip
    # a protein variant with two nucleotide variants, as variant-annotation lists them
    listed = {
        "clinical_significance": "|Benign",
        "review_status": "|criteria provided, single submitter",
        "allele_id": "x|h",
        "variation_id": "|1007",
    }
    clinvar = clinvar.with_columns(
        pl.when(pl.int_range(pl.len()) == 7)
        .then(pl.lit(value))
        .otherwise(pl.col(f"clinvar.202601.{field}"))
        .alias(f"clinvar.202601.{field}")
        for field, value in listed.items()
    )
    path = tmp_path / "annotated.tsv"
    clinvar.write_csv(path, separator="\t")
    requested = []

    def fetch(ids):
        requested.append(ids)
        return pl.DataFrame(
            {
                "clinvar_variation_id": ids,
                "clinvar_conditions": [[f"disease {i}"] for i in ids],
                "fetched": [date.today()] * len(ids),
            },
            schema=_clinvar_api.SCHEMA,
        )

    monkeypatch.setattr(_clinvar_api, "_fetch", fetch)
    variants = ["A1A", "A2V", "A3V", "A4V", "A5V", "A6V", "A7V", "A8V", "A2V:tag"]
    ds = fb.Dataset(pl.DataFrame({"meta_aa_changes": variants})).variant_type()
    out = ds.clinvar(path, cache=tmp_path / "cache")
    df = out.df
    assert df["meta_clinvar_annotation"].to_list() == [
        "Synonymous",
        fisseq.PATHOGENIC,
        fisseq.UNCERTAIN,
        fisseq.PATHOGENIC,
        fisseq.PATHOGENIC,
        "Single Missense",
        "Single Missense",
        fisseq.UNCERTAIN,
        "Single Missense",  # tagged labels are not matched
    ]
    assert df["meta_clinvar_allele_id"].to_list() == [
        None, "b", "c", "d", "f", "g", "h", "i", None
    ]  # fmt: skip
    assert df["meta_clinvar_clinical_significance"][6] == "Benign"
    assert df["meta_clinvar_release"][1] == "2026-01"
    assert df["meta_clinvar_conditions"][1].to_list() == ["disease 1001"]
    # A2V has pathogenic and benign records: no ExCALIBR class
    assert df["meta_clinvar_class"].to_list() == [
        None, None, None, "pathogenic", "pathogenic", None, "benign", None, None
    ]  # fmt: skip
    assert not any(c.startswith("__") for c in df.columns)
    assert out.annotations["clinvar"]["release"] == "2026-01"
    assert ds.annotations == {}
    # the conditions are cached; a frame source gives the same result
    assert ds.clinvar(clinvar, cache=tmp_path / "cache").df.equals(df)
    assert len(requested) == 1
    assert "meta_clinvar_conditions" not in ds.clinvar(path, conditions=False).columns
    # running it again replaces the columns
    assert out.clinvar(path, conditions=False).df.height == len(variants)


def test_clinvar_needs_variant_type():
    ds = fb.Dataset(pl.DataFrame({"meta_aa_changes": ["A1A"]}))
    with pytest.raises(ValueError, match="meta_variant_type"):
        ds.clinvar(variant_annotation(["p.Ala1Ala"], ["Benign"]), conditions=False)


def test_gnomad():
    gnomad = pl.DataFrame(
        {
            "Protein Consequence": ["p.Ala1Val", "p.Ala1Val", "p.Ala2Val", "p.Ala3Ala"],
            "Filters - joint": ["PASS", "PASS", "AC0", "PASS"],
            "Allele Count": ["2", "3", "0", "4"],
            "Allele Number": ["1000", "1200", "1000", "1000"],
            "ClinVar Germline Classification": ["Benign", "", "", ""],
            "spliceai_ds_max": ["0.0", "0.3", "0.0", "0.5"],
        }
    )
    variants = ["A1V", "A2V", "A3A", "A4V", "A1V:tag"]
    ds = fb.Dataset(pl.DataFrame({"meta_aa_changes": variants})).gnomad(gnomad)
    df = ds.df
    assert df["meta_gnomad_in"].to_list() == [True, False, True, None, None]
    assert df["meta_gnomad_allele_count"].to_list() == [5, 0, 4, None, None]
    assert df["meta_gnomad_allele_frequency"][0] == pytest.approx(5 / 1200)
    assert df["meta_gnomad_spliceai_ds_max"].to_list() == [0.3, 0.0, 0.5, None, None]
    assert df["meta_gnomad_clinvar_class"].to_list() == [
        "benign", None, None, None, None
    ]  # fmt: skip
    assert ds.annotations["gnomad"] == {"source": "table", "splice_max": None}
    spliced = ds.gnomad(gnomad, splice_max=0.2).df
    assert spliced["meta_gnomad_in"].to_list() == [True, False, False, None, None]
    assert spliced["meta_gnomad_allele_count"].to_list() == [2, 0, 0, None, None]
    assert spliced.columns == df.columns


def test_polars_passthroughs(labels):
    ds = (
        labels.filter(pl.col("feature_0") < 3)
        .with_columns(double=pl.col("feature_0") * 2)
        .rename({"double": "meta_double"})
        .sort("feature_0", descending=True)
    )
    assert ds.df["meta_double"].to_list() == [4, 2, 0]
    assert ds.features == ["feature_0"]
    assert ds.meta == ["meta_aa_changes", "meta_double"]
    joined = ds.join(ds.select("meta_aa_changes", other=pl.lit(1)))
    assert joined.df["other"].to_list() == [1, 1, 1]
    assert ds.pipe(lambda lf, k: lf.head(k), 1).df.height == 1
    assert repr(ds) == "Dataset(2 meta columns, 1 feature columns)"


def test_save_and_read(labels, tmp_path):
    out = tmp_path / "nested" / "labels.parquet"
    labels.save(out)
    assert fb.Dataset.read(out).df.equals(labels.df)


def test_plots_accept_datasets(profiles):
    ds = fb.Dataset(profiles)
    plot = fb.BoxPlot(ds, x="meta_variant_type", y="meta_impact_score")
    assert plot.data.equals(profiles)
    plot.plot()
    fb.Heatmap.correlation(ds, ["feature_0", "feature_1"]).plot()
