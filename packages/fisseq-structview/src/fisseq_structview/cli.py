"""``fisseq-structview``: serve a run's distinguishability scores on one or more structures.

    fisseq-structview parameters.yaml [--port 8051 ...]     (or --params parameters.yaml)
    fisseq-structview <pipeline_dir> --structure model_a.cif model_b.pdb [--port 8050]

Launch parameters come from a YAML file (`params`), and any flag given overrides the file's
value. The run's OvWT scores are read and pooled at launch (see `data`); the score,
aggregation and the rest are chosen in the browser. A run without OvWT outputs still serves
the structures, gray, with a warning.
"""

import argparse
import logging
import pathlib
import sys
from collections.abc import Sequence

from .app import ViewSettings, create_app
from .data import load_clinvar, load_distinguishability
from .mapping import MappingError
from .params import ParamsError, load_params, merge, resolve_paths
from .regions import track
from .structure import StructureError, load_structure

logger = logging.getLogger(__name__)

S = argparse.SUPPRESS


def build_parser() -> argparse.ArgumentParser:
    """The parser. No flag has a default: what isn't given comes from the parameters file,
    else from `params.Params`."""
    parser = argparse.ArgumentParser(
        prog="fisseq-structview",
        description=(
            "Serve a web app showing a pipeline run's per-position distinguishability "
            "scores on protein structures. Flags override the parameters file."
        ),
        argument_default=S,
    )
    parser.add_argument(
        "pipeline_dir",
        nargs="?",
        help="pipeline output directory, or user@host:/path for a remote one; or a "
        "parameters .yaml file (the same as --params)",
    )
    parser.add_argument(
        "--params",
        "-p",
        metavar="FILE",
        help="parameters YAML; its relative paths are relative to the file",
    )
    parser.add_argument(
        "--structure",
        dest="structures",
        nargs="+",
        action="extend",
        metavar="FILE",
        help="PDB (.pdb/.ent) or mmCIF (.cif/.mmcif) files, optionally gzipped; replaces "
        "the parameters file's structures",
    )
    data = parser.add_argument_group("reading the run")
    data.add_argument(
        "--layout",
        choices=["data", "embeddings"],
        help="the pipeline that wrote the run (default: detected)",
    )
    data.add_argument(
        "--track",
        choices=["embeddings", "cp_features"],
        help="embeddings pipeline track (default: embeddings)",
    )
    data.add_argument("--batches", nargs="+", help="read only these batches")
    data.add_argument(
        "--exclude", nargs="+", help="skip batches matching these patterns"
    )
    data.add_argument(
        "--variant-col", help="variant label column (default: meta_aa_changes)"
    )
    data.add_argument(
        "--cache",
        nargs="?",
        const=True,
        help="cache directory for a remote run (no value: the user cache directory)",
    )
    data.add_argument(
        "--refresh",
        action="store_const",
        const=True,
        help="download a remote run's files again",
    )
    data.add_argument(
        "--clinvar",
        metavar="TSV",
        help="bbi-lab/variant-annotation table with ClinVar calls to mark",
    )
    mapping = parser.add_argument_group("matching scores to the structure")
    mapping.add_argument(
        "--chain", help="structure chain to color (default: the first protein chain)"
    )
    mapping.add_argument(
        "--offset",
        type=int,
        help="numbering offset: structure residue = score position + offset "
        "(also adjustable in the app; default: 0)",
    )
    mapping.add_argument(
        "--no-check-wt",
        dest="check_wt",
        action="store_const",
        const=False,
        help="don't check the variant labels' wild-type residues against the structure",
    )
    mapping.add_argument(
        "--plddt",
        choices=["auto", "yes", "no"],
        help="whether the B-factors are pLDDT, enabling the fade toggle (default: auto)",
    )
    server = parser.add_argument_group("server")
    server.add_argument("--host", help="default: 127.0.0.1")
    server.add_argument("--port", type=int, help="default: 8050")
    server.add_argument(
        "--debug", action="store_const", const=True, help="Dash debug mode"
    )
    return parser


def parse_params(argv: Sequence[str] | None = None):
    """The parser and the merged `params.Params` of ``argv``.

    Raises
    ------
    ParamsError
    """
    parser = build_parser()
    cli = vars(parser.parse_args(argv))
    source = cli.pop("params", None)
    # `fisseq-structview parameters.yaml` means --params: a pipeline directory is never YAML.
    positional = str(cli.get("pipeline_dir", ""))
    if positional.lower().endswith((".yaml", ".yml")):
        if source is not None:
            raise ParamsError(
                f"two parameters files: {positional} and --params {source}; give one"
            )
        source = cli.pop("pipeline_dir")
    file_values = load_params(source) if source else {}
    if "structures" in cli:
        cli["structures"] = list(cli["structures"])
    cli = resolve_paths(cli, None)
    path = pathlib.Path(source).expanduser().resolve() if source else None
    return parser, merge(file_values, cli, path)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = build_parser()
    try:
        parser, params = parse_params(argv)
    except ParamsError as err:
        parser.exit(2, f"error: {err}\n")

    structures = []
    for spec in params.structures:
        try:
            structure = load_structure(spec.path, spec.chain or params.chain)
        except StructureError as err:
            parser.exit(2, f"error: {err}\n")
        logger.info(
            "Structure %s: chain %s, residues %d-%d",
            structure.path,
            structure.chain,
            min(structure.residues),
            max(structure.residues),
        )
        structures.append(structure)

    try:
        run = load_distinguishability(
            params.pipeline_dir,
            layout=params.layout,
            track=params.track,
            batches=params.batches,
            exclude=params.exclude,
            cache=params.cache,
            refresh=params.refresh,
            variant_col=params.variant_col,
        )
    except ValueError as err:
        parser.exit(2, f"error: {err}\n")
    notes = []
    if run is None:
        notes.append(
            f"No distinguishability (OvWT) scores found in {params.pipeline_dir}: "
            "showing the structure only."
        )

    clinvar = None
    if params.clinvar is not None and run is not None:
        clinvar = load_clinvar(run, params.clinvar)
        if clinvar is None:
            notes.append(
                f"Couldn't read ClinVar calls from {params.clinvar.path or params.clinvar.gene}"
                " (see the log): ClinVar markers are off."
            )
    try:
        tracks = [track(spec) for spec in params.regions]
    except ValueError as err:
        parser.exit(2, f"error: {params.source or 'parameters'}: regions: {err}\n")

    plddt = {"auto": None, "yes": True, "no": False}[params.plddt]
    first = structures[0]
    settings = ViewSettings(
        offset=params.offset,
        check_wt=params.check_wt,
        fade=bool(plddt if plddt is not None else first.has_plddt),
    )
    try:
        app = create_app(
            run,
            structures,
            settings,
            plddt=plddt,
            notes=notes,
            labels=[spec.name for spec in params.structures],
            offsets=[spec.offset for spec in params.structures],
            tracks=tracks,
            clinvar=clinvar,
            params_source=params.source,
        )
    except MappingError as err:
        parser.exit(2, f"error: {err}\n")
    logger.info("Serving on http://%s:%d", params.host, params.port)
    app.run(host=params.host, port=params.port, debug=params.debug)


if __name__ == "__main__":
    sys.exit(main())
