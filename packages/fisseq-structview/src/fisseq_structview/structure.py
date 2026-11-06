"""Read a protein structure (PDB or mmCIF, optionally gzipped) with gemmi.

`load_structure` returns the file's text (for the viewer) together with one chain's amino-acid
residues, keyed by author residue number (the numbering the file prints, which is what scores
map onto), and each residue's B-factor. AlphaFold models store pLDDT in the B-factor column;
`Structure.has_plddt` says whether this file looks like one.
"""

import gzip
import logging
import pathlib
from dataclasses import dataclass, field
from os import PathLike

import gemmi

logger = logging.getLogger(__name__)

#: File suffixes read as PDB and as mmCIF (a trailing ``.gz`` is allowed on either).
PDB_SUFFIXES = (".pdb", ".ent")
CIF_SUFFIXES = (".cif", ".mmcif")


class StructureError(ValueError):
    """A structure file that can't be read or has no usable protein chain."""


@dataclass(frozen=True)
class Structure:
    """One chain of a structure file, ready for the viewer.

    Attributes
    ----------
    path : pathlib.Path
        The file read.
    text : str
        Its (decompressed) contents, passed to the viewer as is.
    fmt : {"pdb", "cif"}
        Its format.
    chain : str
        The author chain name of the chain the scores are painted on.
    chains : tuple[str, ...]
        Every chain with amino-acid residues, in file order.
    residues : dict[int, str]
        ``chain``'s amino-acid residues: author residue number -> one-letter code (``"X"`` for
        a modified residue without a standard parent).
    bfactors : dict[int, float]
        Mean B-factor of each residue's atoms, by author residue number.
    alphafold : bool
        Whether the file says it is an AlphaFold model.
    """

    path: pathlib.Path
    text: str
    fmt: str
    chain: str
    chains: tuple[str, ...]
    residues: dict[int, str]
    bfactors: dict[int, float] = field(repr=False)
    alphafold: bool = False

    @property
    def has_plddt(self) -> bool:
        """Whether the B-factor column holds pLDDT.

        True for files that say they are AlphaFold models, and for files whose B-factors
        are all within [0, 100], the same for every atom of a residue and not the same for
        every residue (how AlphaFold, ColabFold and ESMFold write pLDDT; a crystal
        structure's B-factors vary by atom, and NMR models often have none, all 0).
        """
        if self.alphafold:
            return True
        values = self.bfactors.values()
        return (
            self._per_residue_bfactors
            and bool(self.bfactors)
            and max(values) - min(values) > 1e-3
        )

    # Set in load_structure: whether each residue's atoms share one B-factor in [0, 100].
    _per_residue_bfactors: bool = field(default=False, repr=False)


def _format(path: pathlib.Path) -> tuple[str, bool]:
    """``("pdb" | "cif", gzipped)`` from the file name."""
    suffixes = [s.lower() for s in path.suffixes]
    gz = bool(suffixes) and suffixes[-1] == ".gz"
    if gz:
        suffixes = suffixes[:-1]
    suffix = suffixes[-1] if suffixes else ""
    if suffix in PDB_SUFFIXES:
        return "pdb", gz
    if suffix in CIF_SUFFIXES:
        return "cif", gz
    raise StructureError(
        f"Can't tell the format of {path}: expected a name ending in "
        f"{', '.join(PDB_SUFFIXES + CIF_SUFFIXES)} (optionally .gz)"
    )


def _read_text(path: pathlib.Path, gz: bool) -> str:
    try:
        if gz:
            with gzip.open(path, "rt") as f:
                return f.read()
        return path.read_text()
    except FileNotFoundError:
        raise StructureError(f"Structure file not found: {path}") from None
    except (OSError, UnicodeDecodeError) as err:
        raise StructureError(f"Can't read structure file {path}: {err}") from None


def _is_alphafold(text: str, fmt: str) -> bool:
    """AlphaFold DB files carry ``_ma_qa_metric_local`` (mmCIF) or an ALPHAFOLD title line."""
    if fmt == "cif":
        return "_ma_qa_metric_local" in text
    head = "\n".join(text.splitlines()[:50]).upper()
    return "ALPHAFOLD" in head


def _one_letter(name: str) -> str | None:
    """One-letter code of an amino-acid residue name; ``None`` for anything else."""
    info = gemmi.find_tabulated_residue(name)
    if info is None or not info.is_amino_acid():
        return None
    code = info.one_letter_code.upper()
    return code if code.isalpha() and code != " " else "X"


def load_structure(path: str | PathLike, chain: str | None = None) -> Structure:
    """Read a structure and pick the chain to paint.

    Parameters
    ----------
    path : str | PathLike
        A ``.pdb``/``.ent`` or ``.cif``/``.mmcif`` file, optionally gzipped.
    chain : str | None
        Author chain name. By default, the first chain with amino-acid residues.

    Returns
    -------
    Structure

    Raises
    ------
    StructureError
        If the file is missing, unreadable, not parseable as its format, has no amino-acid
        residues, or has no chain named ``chain``.
    """
    path = pathlib.Path(path)
    fmt, gz = _format(path)
    text = _read_text(path, gz)
    try:
        if fmt == "pdb":
            st = gemmi.read_pdb_string(text)
        else:
            st = gemmi.make_structure_from_block(
                gemmi.cif.read_string(text).sole_block()
            )
    except (RuntimeError, ValueError) as err:
        raise StructureError(f"Can't parse {path} as {fmt.upper()}: {err}") from None
    if len(st) == 0:
        raise StructureError(f"{path} has no models (is it a {fmt.upper()} file?)")

    model = st[0]
    chains: dict[str, gemmi.Chain] = {}
    for ch in model:
        if any(_one_letter(res.name) for res in ch) and ch.name not in chains:
            chains[ch.name] = ch
    if not chains:
        raise StructureError(f"{path} has no amino-acid residues")
    if chain is None:
        chain = next(iter(chains))
    elif chain not in chains:
        raise StructureError(
            f"{path} has no protein chain {chain!r}; its chains are {sorted(chains)}"
        )

    residues: dict[int, str] = {}
    bfactors: dict[int, float] = {}
    per_residue = True
    skipped_icodes = 0
    for res in chains[chain]:
        code = _one_letter(res.name)
        if code is None:
            continue
        number = res.seqid.num
        if res.seqid.icode.strip() or number in residues:
            skipped_icodes += 1
            continue
        b = [atom.b_iso for atom in res]
        if not b:
            continue
        residues[number] = code
        bfactors[number] = sum(b) / len(b)
        if max(b) - min(b) > 1e-3 or not 0 <= b[0] <= 100:
            per_residue = False
    if skipped_icodes:
        logger.warning(
            "Chain %s of %s: skipped %d residues with insertion codes or repeated numbers",
            chain,
            path,
            skipped_icodes,
        )
    return Structure(
        path=path,
        text=text,
        fmt=fmt,
        chain=chain,
        chains=tuple(chains),
        residues=residues,
        bfactors=bfactors,
        alphafold=_is_alphafold(text, fmt),
        _per_residue_bfactors=per_residue,
    )
