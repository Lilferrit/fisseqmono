import logging

import numpy as np
import polars as pl
import pytest
from fisseqborn_calibration_data import dataset_frames

import fisseqborn as fb
from fisseqborn.calibration import _inputs

FAST = {"n_bootstrap": 8, "n_restarts": 2, "n_components": 2, "n_jobs": 1}


@pytest.fixture(scope="module")
def frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return dataset_frames(0)


@pytest.fixture(scope="module")
def calibrated(frames) -> fb.Dataset:
    data, gnomad, clinvar = frames
    return fb.Dataset(data).calibrate("score", gnomad=gnomad, clinvar=clinvar, **FAST)


# ----- inputs ------------------------------------------------------------------------


def test_hgvs_key():
    df = pl.DataFrame(
        {
            "p": [
                "p.Arg123Trp",
                "p.Arg123Ter",
                "p.Thr3Thr",
                "p.Ala5=",
                "p.Leu4fs",
                "",
                None,
            ]
        }
    )
    keys = df.select(_inputs.hgvs_key_expr("p"))["p"].to_list()
    assert keys == ["R123W", "R123*", "T3T", "A5A", None, None, None]


def test_key_of_dataset_labels():
    df = pl.DataFrame(
        {"v": ["A12V", "A12A", "A12*", "A12X", "A12fs", "A12V|L13L", "WT"]}
    )
    assert df.select(_inputs.key_expr("v"))["v"].to_list() == [
        "A12V", "A12A", "A12*", "A12*", None, None, None,
    ]  # fmt: skip


def _gnomad(rows):
    cols = ["Protein Consequence", "Filters - joint", "Allele Count", "Allele Number",
            "ClinVar Germline Classification", "spliceai_ds_max"]  # fmt: skip
    return pl.DataFrame([dict(zip(cols, r, strict=True)) for r in rows], orient="row")


def test_gnomad_collapses_nucleotide_variants():
    table = _gnomad(
        [
            ("p.Arg10Trp", "PASS", "2", "1000", "", "0.0"),
            ("p.Arg10Trp", "PASS", "3", "1200", "Pathogenic", "0.0"),
            ("p.Arg11Gln", "AC0", "0", "1000", "", "0.0"),
            ("p.Arg12Gln", "AC0", "0", "1000", "", "0.0"),
            ("p.Arg12Gln", "PASS", "1", "900", "", "0.0"),
            ("", "PASS", "5", "1000", "", ""),  # intronic: dropped
        ]
    )
    out = _inputs.read_gnomad(table)
    rows = {r["key"]: r for r in out.iter_rows(named=True)}
    assert set(rows) == {"R10W", "R11Q", "R12Q"}
    assert rows["R10W"]["in_gnomad"] and rows["R10W"]["allele_count"] == 5
    assert rows["R10W"]["allele_number"] == 1200
    assert rows["R10W"]["clinvar_class"] == "pathogenic"
    assert not rows["R11Q"]["in_gnomad"]
    assert rows["R12Q"]["in_gnomad"] and rows["R12Q"]["allele_count"] == 1


def test_gnomad_splice_filter():
    table = _gnomad(
        [
            ("p.Arg10Trp", "PASS", "2", "1000", "", "0.5"),
            ("p.Arg11Gln", "PASS", "2", "1000", "", "0.1"),
            ("p.Arg12Gln", "PASS", "2", "1000", "", ""),
        ]
    )
    keep = (
        _inputs.read_gnomad(table, splice_max=0.2).filter("in_gnomad")["key"].to_list()
    )
    assert keep == ["R11Q", "R12Q"]
    with pytest.raises(ValueError, match="spliceai_ds_max"):
        _inputs.read_gnomad(table.drop("spliceai_ds_max"), splice_max=0.2)


def test_conflicting_clinvar_classes_are_dropped():
    table = _gnomad(
        [
            ("p.Arg10Trp", "PASS", "2", "1000", "Likely pathogenic", "0"),
            ("p.Arg10Trp", "PASS", "2", "1000", "Likely benign", "0"),
            ("p.Arg11Gln", "PASS", "2", "1000", "Benign/Likely benign", "0"),
            ("p.Arg12Gln", "PASS", "2", "1000", "Conflicting classifications of pathogenicity", "0"),
        ]
    )  # fmt: skip
    notes: list[str] = []
    out = _inputs.read_gnomad(table, notes=notes)
    classes = dict(out.select("key", "clinvar_class").iter_rows())
    assert classes == {"R10W": None, "R11Q": "benign", "R12Q": None}
    assert any("R10W" in n for n in notes)


def test_clinvar_table_star_filter():
    table = pl.DataFrame(
        {
            "variant": ["A1V", "A2V", "A3V", "A4X"],
            "clinvar_clinical_significance": ["Pathogenic", "Benign", "Likely benign", "Pathogenic"],
            "clinvar_review_status": [
                "criteria provided, single submitter",
                "no assertion criteria provided",
                "reviewed by expert panel",
                "criteria provided, multiple submitters, no conflicts",
            ],
        }
    )  # fmt: skip
    out = dict(_inputs.read_clinvar(table).iter_rows())
    assert out == {"A1V": "pathogenic", "A3V": "benign", "A4*": "pathogenic"}
    assert len(_inputs.read_clinvar(table, min_stars=None)) == 4
    notes: list[str] = []
    _inputs.read_clinvar(table.drop("clinvar_review_status"), notes=notes)
    assert any("star" in n for n in notes)


def test_assign_groups_skips_tagged_and_multi_codon():
    df = pl.DataFrame(
        {"meta_aa_changes": ["A1V", "A1V:tag", "A1V|L2L", "A3A", "A3A:tag"]}
    )
    gnomad = pl.DataFrame(
        {
            "key": ["A1V", "A3A"],
            "in_gnomad": [True, True],
            "clinvar_class": ["benign", None],
        }
    )
    out = _inputs.assign_groups(df, "meta_aa_changes", gnomad, None, [])
    assert out["group"].to_list() == ["B/LB", None, None, "Synonymous", None]
    assert out["groups"].to_list()[0] == ["B/LB", "gnomAD"]
    assert out["groups"].to_list()[3] == ["Synonymous", "gnomAD"]


# ----- Dataset.calibrate -------------------------------------------------------------


def test_calibrate_adds_columns(calibrated, frames):
    df = calibrated.df
    assert df.height == frames[0].height
    for col in ("group", "groups", "lr", "posterior", "points", "oob"):
        assert f"meta_excalibr_{col}" in df.columns
    assert df.schema["meta_excalibr_points"] == pl.Int8
    cal = calibrated.calibration
    assert isinstance(cal, fb.Calibration)
    assert cal.settings["score"] == "score" and cal.settings["clinvar"] == "table"
    assert cal.group_counts == {
        "pathogenic": 60,
        "benign": 60,
        "gnomad": 310,
        "synonymous": 60,
    }
    groups = df.group_by("meta_excalibr_group").len()
    assert dict(groups.iter_rows())[None] == 40  # unlabelled missense variants
    # Low scores (pathogenic) get pathogenic points, high scores benign points.
    low = df.filter(pl.col("score") < -2.5)["meta_excalibr_points"]
    high = df.filter(pl.col("score") > 2)["meta_excalibr_points"]
    assert (low > 0).all() and (high < 0).all()


def test_calibration_survives_chaining(calibrated):
    assert calibrated.filter(pl.col("score") > 0).calibration is calibrated.calibration


def test_calibrate_rejects_duplicates(frames):
    data, gnomad, _ = frames
    dup = pl.concat([data, data.head(2)])
    with pytest.raises(ValueError, match=r"per_variant\(\).*median_across_batches"):
        fb.Dataset(dup).calibrate("score", gnomad=gnomad, **FAST)


def test_calibrate_validates_score_column(frames):
    data, gnomad, _ = frames
    with pytest.raises(ValueError, match="did you mean 'score'"):
        fb.Dataset(data).calibrate("scor", gnomad=gnomad, **FAST)
    with pytest.raises(ValueError, match="numeric"):
        fb.Dataset(data).calibrate("meta_aa_changes", gnomad=gnomad, **FAST)


def test_calibrate_without_clinvar_warns(frames, caplog):
    data, gnomad, _ = frames
    with caplog.at_level(logging.WARNING):
        out = fb.Dataset(data).calibrate("score", gnomad=gnomad, **FAST)
    assert "No ClinVar table given" in caplog.text
    assert out.calibration.settings["clinvar"] == "gnomad"
    assert out.calibration.group_counts["pathogenic"] == 60


def test_apply_calibration(calibrated, frames, tmp_path):
    path = tmp_path / "cal.json"
    calibrated.calibration.to_json(path)
    data = frames[0].with_columns(pl.col("score") + 0.0)
    applied = fb.Dataset(data).apply_calibration(path)
    np.testing.assert_allclose(
        applied.df["meta_excalibr_lr"], calibrated.df["meta_excalibr_lr"]
    )
    assert "meta_excalibr_group" not in applied.columns
    assert applied.calibration == calibrated.calibration
    with pytest.raises(ValueError, match="'score'"):
        fb.Dataset(data.rename({"score": "other"})).apply_calibration(path)


# ----- CalibrationPlot ---------------------------------------------------------------


def test_calibration_plot_renders(calibrated, tmp_path):
    plot = fb.CalibrationPlot(calibrated).refline(x=0).set(title="LMNA")
    plot.save(tmp_path / "cal.png")
    assert (tmp_path / "cal.png").stat().st_size > 0
    fig, ax = fb.CalibrationPlot(calibrated, kind="kde", fit=False, bands=False).plot()
    labels = ax.get_legend_handles_labels()[1]
    assert any(label.startswith("P/LP (n=60)") for label in labels)


def test_calibration_plot_needs_calibration(calibrated):
    with pytest.raises(ValueError, match="No calibration"):
        fb.CalibrationPlot(calibrated.df)
    fb.CalibrationPlot(calibrated.df, calibration=calibrated.calibration).plot()
    with pytest.raises(ValueError, match="Unknown groups"):
        fb.CalibrationPlot(calibrated, groups=["VUS"])
