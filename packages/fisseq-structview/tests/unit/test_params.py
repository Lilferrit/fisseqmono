from pathlib import Path

import pytest

from fisseq_structview.params import (
    ClinvarSpec,
    ParamsError,
    StructureSpec,
    load_params,
    merge,
)


def _write(tmp_path, text, name="parameters.yaml"):
    path = tmp_path / "share" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    return path


def test_relative_paths_are_relative_to_the_file(tmp_path):
    path = _write(
        tmp_path,
        """
pipeline_dir: data/run
structures:
  - structures/a.pdb
  - {path: /abs/b.cif, label: B, chain: B, offset: 3}
cache: cache
clinvar: {path: annotations/clinvar.tsv, min_stars: 2}
""",
    )
    params = merge(load_params(path), {}, path)
    base = path.parent
    assert params.pipeline_dir == str(base / "data/run")
    assert params.structures == (
        StructureSpec(base / "structures/a.pdb"),
        StructureSpec(Path("/abs/b.cif"), "B", "B", 3),
    )
    assert params.cache == str(base / "cache")
    assert params.clinvar == ClinvarSpec(
        path=base / "annotations/clinvar.tsv", min_stars=2
    )
    assert params.structures[1].name == "B" and params.structures[0].name == "a.pdb"


def test_remote_pipeline_dir_is_kept(tmp_path):
    path = _write(tmp_path, "pipeline_dir: me@host:/runs/x\nstructures: [a.pdb]\n")
    assert merge(load_params(path), {}, path).pipeline_dir == "me@host:/runs/x"


def test_cli_values_override_whole_keys(tmp_path):
    path = _write(
        tmp_path,
        "pipeline_dir: run\nstructures: [a.pdb, b.pdb]\nport: 9000\n"
        "clinvar: {path: c.tsv, gene: LMNA}\n",
    )
    params = merge(
        load_params(path),
        {"port": 9100, "structures": [Path("z.pdb")], "clinvar": Path("/x.tsv")},
        path,
    )
    assert params.port == 9100
    assert [s.path for s in params.structures] == [Path("z.pdb")]
    # --clinvar replaces only the table, keeping the rest of the file's ClinVar settings.
    assert params.clinvar == ClinvarSpec(path=Path("/x.tsv"), gene="LMNA")


def test_regions(tmp_path):
    path = _write(
        tmp_path,
        """
pipeline_dir: run
structures: [a.pdb]
regions:
  - label: Tile
    regions: {T1: [1, 96], T2: [90, 185]}
  - label: Domain
    regions: {Head: [1, 28]}
    palette: {Head: "#ff0000"}
""",
    )
    tile, domain = merge(load_params(path), {}, path).regions
    assert tile.label == "Tile" and tile.regions == {"T1": (1, 96), "T2": (90, 185)}
    assert domain.palette == {"Head": "#ff0000"}


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (
            "pipeline_dir: run\nstructures: [a.pdb]\nport: high\n",
            "port: must be an integer",
        ),
        (
            "pipeline_dir: run\nstructures: [a.pdb]\nplddt: maybe\n",
            "plddt: must be one of",
        ),
        (
            "pipeline_dir: run\nstructures: [a.pdb]\n"
            "regions: [{label: T, regions: {T1: [9, 1]}}]\n",
            r"regions\[0\]\.regions\.T1: first \(9\) is after last \(1\)",
        ),
        (
            "pipeline_dir: run\nstructures: [{file: a.pdb}]\n",
            r"structures\[0\]: unknown key\(s\) file",
        ),
        (
            "pipeline_dir: run\nstructures: [a.pdb]\nclinvar: {min_stars: 2}\n",
            "needs a path or a gene",
        ),
        (
            "pipeline_dir: run\nstructures: [a.pdb]\ngene: [LMNA]\n",
            "gene: must be a string",
        ),
        ("structures: [a.pdb]\n", "no pipeline_dir"),
        ("pipeline_dir: run\n", "no structures"),
    ],
)
def test_validation_errors_name_the_file_and_key(tmp_path, text, match):
    path = _write(tmp_path, text)
    with pytest.raises(ParamsError, match=match) as err:
        merge(load_params(path), {}, path)
    assert str(path) in str(err.value)


def test_unreadable_files(tmp_path):
    with pytest.raises(ParamsError, match="not found"):
        load_params(tmp_path / "nope.yaml")
    with pytest.raises(ParamsError, match="not valid YAML"):
        load_params(_write(tmp_path, "a: [1, 2\n"))
    with pytest.raises(ParamsError, match="must be a mapping"):
        load_params(_write(tmp_path, "- a\n- b\n"))
    with pytest.raises(ParamsError, match="unknown parameter"):
        load_params(_write(tmp_path, "pipline_dir: x\n"))


def test_gene(tmp_path):
    path = _write(tmp_path, "pipeline_dir: run\nstructures: [a.pdb]\ngene: LMNA\n")
    assert merge(load_params(path), {}, path).gene == "LMNA"
    assert merge(load_params(path), {"gene": "KRAS"}, path).gene == "KRAS"
    assert merge({"pipeline_dir": "r", "structures": ["a.pdb"]}, {}).gene is None
