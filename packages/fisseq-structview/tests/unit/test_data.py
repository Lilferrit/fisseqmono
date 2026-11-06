import logging

import polars as pl
import pytest
from structview_testdata import RUN

import fisseqborn as fb
from fisseq_structview.data import (
    CLINVAR_CLASS,
    DEFAULT_SCORE,
    POSITION,
    SCORE,
    VARIANT,
    WT,
    default_score,
    load_clinvar,
    load_distinguishability,
)
from fisseq_structview.info import data_info
from fisseq_structview.params import ClinvarSpec
from fisseq_structview.structure import load_structure


def test_loads_every_score_corrected_and_raw(run_dir):
    run = load_distinguishability(run_dir)
    assert run.batches == ("batch1", "batch2")
    assert run.scores == ["auroc_pooled", "auroc_median_barcode", "auroc_median_fold"]
    assert set(run.tables) == {(s, c) for s in run.scores for c in (True, False)}
    raw = run.table("auroc_median_fold", corrected=False)
    a4v = raw.filter(pl.col(VARIANT) == "A4V").row(0, named=True)
    assert a4v[SCORE] == pytest.approx(0.88)  # median of 0.90, 0.86
    assert a4v[POSITION] == 4 and a4v[WT] == "A"
    nonsense = raw.filter(pl.col(VARIANT) == "A4*").row(0, named=True)
    assert nonsense[POSITION] is None  # strict positions: single substitutions only


def test_corrected_matches_fisseqborn(run_dir):
    run = load_distinguishability(run_dir)
    expected = (
        fb.OvwtScores.from_pipeline(run_dir)
        .correct("auroc_median_fold")
        .per_variant("auroc_median_fold_corrected")
        .df
    )
    got = run.table("auroc_median_fold").select(VARIANT, SCORE)
    joined = got.join(expected, left_on=VARIANT, right_on="meta_aa_changes")
    assert joined.height == expected.height
    assert joined[SCORE].to_list() == pytest.approx(
        joined["auroc_median_fold_corrected"].to_list()
    )


def test_missing_ovwt_warns_and_returns_none(tmp_path, caplog):
    (tmp_path / "qc_filter").mkdir()
    with caplog.at_level(logging.WARNING):
        assert load_distinguishability(tmp_path) is None
    assert "No distinguishability scores" in caplog.text


def test_unknown_score(run_dir):
    run = load_distinguishability(run_dir)
    with pytest.raises(ValueError, match="No score 'test_auroc'"):
        run.table("test_auroc")


def test_default_score():
    assert default_score(["auroc_pooled", DEFAULT_SCORE]) == DEFAULT_SCORE
    assert default_score(["test_auroc", "val_auroc"]) == "test_auroc"
    with pytest.raises(ValueError):
        default_score([])


def test_records_batches_and_how_the_run_was_read(run_dir):
    run = load_distinguishability(run_dir, track="embeddings")
    assert run.batch_rows == {b: len(v) for b, v in RUN.items()}
    assert (run.layout, run.track, run.variant_col) == (
        None,
        "embeddings",
        "meta_aa_changes",
    )


def test_data_info_lists_batches_and_structures(run_dir, pdb_path):
    run = load_distinguishability(run_dir)
    structure = load_structure(pdb_path)
    info = data_info(
        run,
        [structure, structure],
        plddt=[True, False],
        plddt_forced=False,
        labels=["A", "B"],
        offsets=[0, 3],
    )
    assert info.batches == [(b, len(v)) for b, v in RUN.items()]
    run_rows = dict(info.sections["Run"])
    assert run_rows["Batches"] == "2" and run_rows["Layout"] == "detected"
    assert "auroc_median_fold" in run_rows["Scores"]
    assert [(r[0], r[3], r[4], r[5]) for r in info.structures] == [
        ("A", "1–8 (8)", "yes", "0"),
        ("B", "1–8 (8)", "no", "3"),
    ]
    bare = data_info(None, [structure], plddt=[False], plddt_forced=True)
    assert bare.batches == [] and dict(bare.sections["Run"]) == {
        "Scores": "none loaded"
    }
    assert bare.structures[0][4] == "no (--plddt)"


def _annotation_table(path):
    """A bbi-lab/variant-annotation table with ClinVar calls for A4V, K2E and K2R."""
    pl.DataFrame(
        {
            "mapped_hgvs_p": [
                "NP_000268.1:p.Ala4Val",
                "NP_000268.1:p.Lys2Glu",
                "NP_000268.1:p.Lys2Arg",
            ],
            "clinvar.202601.variation_id": ["1", "2", "3"],
            "clinvar.202601.clinical_significance": [
                "Pathogenic",
                "Benign",
                "Uncertain significance",
            ],
            "clinvar.202601.review_status": ["criteria provided, single submitter"] * 3,
        }
    ).write_csv(path, separator="\t")
    return path


def test_load_clinvar(run_dir, tmp_path):
    run = load_distinguishability(run_dir)
    calls = load_clinvar(run, ClinvarSpec(path=_annotation_table(tmp_path / "a.tsv")))
    assert dict(calls.table.select(VARIANT, CLINVAR_CLASS).rows()) == {
        "A4V": "pathogenic",
        "K2E": "benign",
        "K2R": "vus",
    }
    assert calls.release is not None and calls.source.endswith("a.tsv")


def test_unreadable_clinvar_warns(run_dir, tmp_path, caplog):
    run = load_distinguishability(run_dir)
    with caplog.at_level(logging.WARNING):
        assert load_clinvar(run, ClinvarSpec(path=tmp_path / "missing.tsv")) is None
    assert "ClinVar" in caplog.text
