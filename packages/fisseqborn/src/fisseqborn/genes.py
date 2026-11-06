"""Per-gene reference constants: domain regions, library tiles, landmark features and the
wild-type sequence.

`Gene` bundles what the dataset methods and plots need to know about a protein. Every
field but ``name`` is optional; a method that needs a missing one raises an error saying
so. Built-in genes are module constants (`LMNA`); define your own with `Gene`:

>>> kras = Gene("KRAS", domains={"G domain": (1, 166), "HVR": (167, 188)})
>>> ds.domain(kras)
"""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

_BASES = "TCAG"
_AMINO_ACIDS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
_CODONS = {
    a + b + c: _AMINO_ACIDS[i]
    for i, (a, b, c) in enumerate(
        (a, b, c) for a in _BASES for b in _BASES for c in _BASES
    )
}


def translate(cds: str) -> str:
    """Translate a coding sequence to one-letter amino acids (``*`` for a stop)."""
    cds = cds.upper()
    if len(cds) % 3 or set(cds) - set("ACGT"):
        raise ValueError(
            "a coding sequence must be ACGT only, with a length divisible by 3"
        )
    return "".join(_CODONS[cds[i : i + 3]] for i in range(0, len(cds), 3))


@dataclasses.dataclass(frozen=True)
class Gene:
    """Reference constants of one gene's protein.

    Attributes
    ----------
    name : str
        HGNC symbol, e.g. ``"LMNA"``; used to download gnomAD and ClinVar variants.
    domains : Mapping[str, tuple[int, int]] | None
        Protein regions, name -> inclusive 1-based ``(first, last)`` amino-acid range.
    tiles : Sequence[tuple[str, int, int]] | None
        Library tiles as ``(name, first, last)`` inclusive amino-acid ranges; neighbors
        may overlap.
    landmark_features : Mapping[str, str] | None
        Feature column -> readable name of the features that summarize this gene's
        phenotype.
    reference_sequence : str | None
        Wild-type coding sequence, ``ATG`` through the stop codon (uppercased).
    """

    name: str
    domains: Mapping[str, tuple[int, int]] | None = None
    tiles: Sequence[tuple[str, int, int]] | None = None
    landmark_features: Mapping[str, str] | None = None
    reference_sequence: str | None = None

    def __post_init__(self) -> None:
        ranges = list((self.domains or {}).items())
        ranges += [(name, (first, last)) for name, first, last in self.tiles or ()]
        for name, (first, last) in ranges:
            if not 1 <= first <= last:
                raise ValueError(
                    f"{self.name}: region {name!r} has an invalid range ({first}, {last})"
                )
        if self.reference_sequence is not None:
            object.__setattr__(
                self, "reference_sequence", self.reference_sequence.upper()
            )
            protein = self.protein_sequence
            assert protein is not None
            for name, (_, last) in ranges:
                if last > len(protein):
                    raise ValueError(
                        f"{self.name}: region {name!r} ends at {last}, past the "
                        f"{len(protein)}-residue protein"
                    )

    @property
    def protein_sequence(self) -> str | None:
        """`reference_sequence` translated, without the stop codon (position ``i`` is
        index ``i - 1``)."""
        if self.reference_sequence is None:
            return None
        return translate(self.reference_sequence).removesuffix("*")

    def require(self, field: str) -> Any:
        """The value of ``field``; a ``ValueError`` naming the gene when it is unset."""
        value = getattr(self, field)
        if value is None:
            raise ValueError(
                f"gene {self.name!r} has no {field}; set it on the Gene or pass it explicitly"
            )
        return value


LMNA = Gene(
    "LMNA",
    domains={
        "Head": (1, 28),
        "Coil 1A": (29, 67),
        "L1": (68, 77),
        "Coil 1B": (78, 222),
        "Linker 12": (222, 241),
        "Coil 2": (242, 386),
        "Tail": (387, 416),
        "NLS": (417, 420),
        "Ig-fold": (421, 544),
        "Unfolded": (545, 664),
    },
    tiles=[
        ("T1", 1, 96),
        ("T2", 90, 185),
        ("T3", 178, 273),
        ("T4", 266, 361),
        ("T5", 354, 449),
        ("T6", 442, 535),
        ("T7", 530, 623),
        ("T8", 607, 664),
    ],
    landmark_features={
        "Mean_NucleiExpanded_Intensity_MeanIntensity_CH1": "Lamin A intensity in the nucleus",
        "Mean_NuclearBoundary_Intensity_MeanIntensity_CH1": "Lamin A intensity at the nuclear boundary",
        "Mean_NucleiExpanded_Granularity_1_CH1": "Lamin A nuclear granularity",
        "Mean_Nuclei_AreaShape_FormFactor": "Nuclear circularity",
        "AreaShape_Area": "Cell Size",
    },
    reference_sequence=(
        "ATGGAGACCCCGTCCCAGCGGCGCGCCACCCGCAGCGGGGCGCAGGCCAGCTCCACTCCGCTGTCGCCCACCCGCATCAC"
        "CCGGCTGCAGGAGAAGGAGGACCTGCAGGAGCTCAATGATCGCTTGGCGGTCTACATCGACCGTGTGCGCTCGCTGGAAA"
        "CGGAGAACGCAGGGCTGCGCCTTCGCATCACCGAGTCTGAAGAGGTGGTCAGCCGCGAGGTGTCCGGCATCAAGGCCGCC"
        "TACGAGGCCGAGCTCGGGGATGCCCGCAAGACCCTTGACTCAGTAGCCAAGGAGCGCGCCCGCCTGCAGCTGGAGCTGAG"
        "CAAAGTGCGTGAGGAGTTTAAGGAGCTGAAAGCGCGCAATACCAAGAAGGAGGGTGACCTGATAGCTGCTCAGGCTCGGC"
        "TGAAGGACCTGGAGGCTCTGCTGAACTCCAAGGAGGCCGCACTGAGCACTGCTCTCAGTGAGAAGCGCACGCTGGAGGGC"
        "GAGCTGCATGATCTGCGGGGCCAGGTGGCCAAGCTTGAGGCAGCCCTAGGTGAGGCCAAGAAACAGCTGCAGGACGAAAT"
        "GCTGCGGCGGGTGGATGCTGAGAACAGGCTGCAGACCATGAAGGAGGAACTGGACTTCCAGAAGAACATCTACAGTGAGG"
        "AGCTGCGTGAGACCAAGCGCCGTCATGAGACCCGACTGGTGGAGATTGACAATGGGAAGCAGCGTGAGTTTGAGAGCCGG"
        "CTGGCGGATGCGCTGCAGGAACTGCGGGCCCAGCATGAGGACCAGGTGGAGCAGTATAAGAAGGAGCTGGAGAAGACTTA"
        "TTCTGCCAAGCTGGACAATGCCAGGCAGTCTGCTGAGAGGAACAGCAACCTGGTGGGGGCTGCCCACGAGGAGCTGCAGC"
        "AGTCGCGCATCCGCATCGACAGCCTCTCTGCCCAGCTCAGCCAGCTCCAGAAGCAGCTGGCAGCCAAGGAGGCGAAGCTT"
        "CGAGACCTGGAGGACTCACTGGCCCGTGAGCGGGACACCAGCCGGCGGCTGCTGGCGGAAAAGGAGCGGGAGATGGCCGA"
        "GATGCGGGCAAGGATGCAGCAGCAGCTGGACGAGTACCAGGAGCTTCTGGACATCAAGCTGGCCCTGGACATGGAGATCC"
        "ACGCCTACCGCAAGCTCTTGGAGGGCGAGGAGGAGAGGCTACGCCTGTCCCCCAGCCCTACCTCGCAGCGCAGCCGTGGC"
        "CGTGCTTCCTCTCACTCATCCCAGACACAGGGTGGGGGCAGCGTCACCAAAAAGCGCAAACTGGAGTCCACTGAGAGCCG"
        "CAGCAGCTTCTCACAGCACGCACGCACTAGCGGGCGCGTGGCCGTGGAGGAGGTGGATGAGGAGGGCAAGTTTGTCCGGC"
        "TGCGCAACAAGTCCAATGAGGACCAGTCCATGGGCAATTGGCAGATCAAGCGCCAGAATGGAGACGACCCACTGCTCACC"
        "TACCGGTTCCCACCAAAGTTCACCCTGAAGGCTGGGCAGGTGGTGACGATCTGGGCTGCAGGAGCTGGGGCCACCCACAG"
        "CCCCCCTACCGACCTGGTGTGGAAGGCACAGAACACCTGGGGCTGCGGGAACAGCCTGCGTACGGCTCTCATCAACTCCA"
        "CTGGGGAAGAAGTGGCCATGCGCAAGCTGGTGCGCTCAGTGACTGTGGTTGAGGACGACGAGGATGAGGATGGAGATGAC"
        "CTGCTCCATCACCACCACGGCTCCCACTGCAGCAGCTCGGGGGACCCCGCTGAGTACAACCTGCGCTCGCGCACCGTGCT"
        "GTGCGGGACCTGCGGGCAGCCTGCCGACAAGGCATCTGCCAGCGGCTCAGGAGCCCAGGTGGGCGGACCCATCTCCTCTG"
        "GCTCTTCTGCCTCCAGTGTCACGGTCACTCGCAGCTACCGCAGTGTGGGGGGCAGTGGGGGTGGCAGCTTCGGGGACAAT"
        "CTGGTCACCCGCTCCTACCTCCTGGGCAACTCCAGCCCCCGAACCCAGAGCCCCCAGAACTGCAGCATCATGTAG"
    ),
)

#: The genes `resolve` knows by name.
BUILTIN: dict[str, Gene] = {"LMNA": LMNA}


def resolve(gene: "Gene | str") -> Gene:
    """``gene`` itself, or the built-in gene of that name."""
    if isinstance(gene, Gene):
        return gene
    try:
        return BUILTIN[gene]
    except KeyError:
        raise ValueError(
            f"unknown gene {gene!r}; built-in genes: {sorted(BUILTIN)}. "
            "Pass a Gene to use your own."
        ) from None


def register(gene: Gene) -> Gene:
    """Make ``gene`` available to `resolve` by name, and its domain names a known axis
    order for plots (`fisseq.ORDERS`)."""
    from . import fisseq

    BUILTIN[gene.name] = gene
    if gene.domains:
        fisseq.ORDERS.append(list(gene.domains))
    return gene
