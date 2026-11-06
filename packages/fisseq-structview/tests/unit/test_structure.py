import gzip

import pytest
from structview_testdata import SEQUENCE, pdb_text, write_pdb

from fisseq_structview.structure import StructureError, load_structure


def test_reads_residues_and_bfactors(tmp_path):
    path = write_pdb(tmp_path / "m.pdb", start=10, bfactors=[50 + i for i in range(8)])
    st = load_structure(path)
    assert st.chain == "A" and st.fmt == "pdb"
    assert st.residues == {10 + i: aa for i, aa in enumerate(SEQUENCE)}
    assert st.bfactors[10] == pytest.approx(50)


def test_gzipped(tmp_path):
    path = tmp_path / "m.pdb.gz"
    with gzip.open(path, "wt") as f:
        f.write(pdb_text())
    assert len(load_structure(path).residues) == 8


def test_cif(tmp_path):
    import gemmi

    st = gemmi.read_pdb_string(pdb_text())
    st.setup_entities()
    path = tmp_path / "m.cif"
    st.make_mmcif_document().write_file(str(path))
    loaded = load_structure(path)
    assert loaded.fmt == "cif" and "".join(loaded.residues.values()) == SEQUENCE


def test_missing_file(tmp_path):
    with pytest.raises(StructureError, match="not found"):
        load_structure(tmp_path / "nope.pdb")


def test_unknown_suffix(tmp_path):
    (tmp_path / "m.txt").write_text("x")
    with pytest.raises(StructureError, match="Can't tell the format"):
        load_structure(tmp_path / "m.txt")


def test_garbage_pdb(tmp_path):
    (tmp_path / "m.pdb").write_text("this is not a structure\n")
    with pytest.raises(StructureError, match=r"m\.pdb"):
        load_structure(tmp_path / "m.pdb")


def test_garbage_cif(tmp_path):
    (tmp_path / "m.cif").write_text("not a cif {\n")
    with pytest.raises(StructureError, match="Can't parse"):
        load_structure(tmp_path / "m.cif")


def test_binary_file(tmp_path):
    (tmp_path / "m.pdb").write_bytes(b"\xff\xfe\x00\x01" * 10)
    with pytest.raises(StructureError, match="Can't read"):
        load_structure(tmp_path / "m.pdb")


def test_chain_choice(tmp_path):
    path = tmp_path / "m.pdb"
    path.write_text(
        pdb_text(chain="A").replace("END\n", "") + pdb_text("GG", chain="B")
    )
    assert load_structure(path, "B").residues == {1: "G", 2: "G"}
    assert load_structure(path).chains == ("A", "B")
    with pytest.raises(StructureError, match=r"no protein chain 'C'.*\['A', 'B'\]"):
        load_structure(path, "C")


def test_plddt_detection(tmp_path):
    plddt = write_pdb(tmp_path / "a.pdb", bfactors=[95, 90, 40, 30, 88, 91, 70, 65])
    assert load_structure(plddt).has_plddt
    crystal = write_pdb(tmp_path / "b.pdb", per_atom_noise=True)
    assert not load_structure(crystal).has_plddt
    out_of_range = write_pdb(tmp_path / "c.pdb", bfactors=[150] * 8)
    assert not load_structure(out_of_range).has_plddt
    nmr = write_pdb(
        tmp_path / "e.pdb", bfactors=[0] * 8
    )  # NMR models: every B-factor 0
    assert not load_structure(nmr).has_plddt
    # An AlphaFold title wins over the B-factor heuristic.
    titled = write_pdb(
        tmp_path / "d.pdb",
        per_atom_noise=True,
        title="ALPHAFOLD MONOMER V2.0 PREDICTION",
    )
    assert load_structure(titled).has_plddt
