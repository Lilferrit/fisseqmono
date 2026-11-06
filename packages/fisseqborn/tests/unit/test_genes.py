import polars as pl
import pytest

import fisseqborn as fb
from fisseqborn import fisseq, genes


def test_lmna_constants():
    lmna = genes.LMNA
    assert lmna.name == "LMNA"
    assert list(lmna.domains)[0] == "Head"
    assert lmna.tiles[0] == ("T1", 1, 96)
    protein = lmna.protein_sequence
    assert len(protein) == 664
    assert protein.startswith("METPSQRRATRSGAQ") and protein.endswith("CSIM")
    assert lmna.reference_sequence.endswith("TAG")


def test_gene_fields_are_optional():
    gene = fb.Gene("KRAS")
    assert gene.protein_sequence is None
    with pytest.raises(ValueError, match="KRAS.*no domains"):
        gene.require("domains")


def test_gene_validates_ranges():
    with pytest.raises(ValueError, match="invalid range"):
        fb.Gene("X", domains={"a": (5, 2)})
    with pytest.raises(ValueError, match="past the 2-residue protein"):
        fb.Gene("X", domains={"a": (1, 3)}, reference_sequence="ATGGCC")
    with pytest.raises(ValueError, match="divisible by 3"):
        fb.Gene("X", reference_sequence="ATGG")


def test_reference_sequence_is_uppercased():
    assert fb.Gene("X", reference_sequence="atggcctaa").protein_sequence == "MA"


def test_resolve():
    assert genes.resolve("LMNA") is genes.LMNA
    custom = fb.Gene("X")
    assert genes.resolve(custom) is custom
    with pytest.raises(ValueError, match="unknown gene 'NOPE'"):
        genes.resolve("NOPE")


def test_custom_gene_through_dataset():
    gene = fb.Gene("X", domains={"first": (1, 2)}, tiles=[("A", 1, 3), ("B", 3, 5)])
    ds = fb.Dataset(pl.DataFrame({"meta_aa_changes": ["A1V", "A3V", "A9V"]}))
    df = ds.domain(gene).tile(gene, allow_multiple=True).df
    assert df["meta_domain"].to_list() == ["first", None, None]
    assert df["meta_tile"].to_list() == ["A", "A,B", None]


def test_domain_and_tile_need_a_gene():
    ds = fb.Dataset(pl.DataFrame({"meta_aa_changes": ["A1V"]}))
    with pytest.raises(ValueError, match="pass a gene"):
        ds.domain()
    with pytest.raises(ValueError, match="pass a gene"):
        ds.tile()
    with pytest.raises(ValueError, match="no tiles"):
        ds.tile(fb.Gene("X", domains={"a": (1, 2)}))


def test_download_needs_a_gene():
    ds = fb.Dataset(pl.DataFrame({"meta_aa_changes": ["A1V"]}))
    with pytest.raises(ValueError, match="pass gene="):
        ds.gnomad()
    with pytest.raises(ValueError, match="pass gene="):
        ds.variant_type().clinvar()


def test_register_adds_a_domain_order():
    gene = fb.Gene("REG_TEST", domains={"d1": (1, 2), "d2": (3, 4)})
    try:
        genes.register(gene)
        assert genes.resolve("REG_TEST") is gene
        assert ["d1", "d2"] in fisseq.ORDERS
    finally:
        genes.BUILTIN.pop("REG_TEST", None)
        fisseq.ORDERS.remove(["d1", "d2"])


def test_variant_effect_map_regions_from_gene():
    variants = pl.DataFrame({"meta_aa_changes": ["A1V", "A5V"], "z": [1.0, 2.0]})
    assert fb.VariantEffectMap(variants, "z").regions is None
    plot = fb.VariantEffectMap(variants, "z", gene="LMNA")
    assert plot.regions == dict(genes.LMNA.domains)
