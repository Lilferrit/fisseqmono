"""The launch parameters: a YAML file, overridden by command-line flags.

`load_params` reads a ``parameters.yaml``; relative paths in it are relative to the file's
directory, so a folder holding the file and its data can be shared or moved whole. `merge`
lays command-line values over the file's (each given flag replaces the whole key) and
validates the result into a `Params`. Every key is optional; see the package README for the
schema.
"""

import pathlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from typing import Any

import yaml

from fisseqborn import _remote


class ParamsError(ValueError):
    """An invalid parameters file or combination of flags."""


@dataclass(frozen=True)
class StructureSpec:
    """One structure file; ``chain`` and ``offset`` default to the top-level ones."""

    path: pathlib.Path
    label: str | None = None
    chain: str | None = None
    offset: int | None = None

    @property
    def name(self) -> str:
        return self.label or self.path.name


@dataclass(frozen=True)
class ClinvarSpec:
    """Where ClinVar calls come from, for `fisseqborn.Dataset.clinvar`."""

    #: A bbi-lab/variant-annotation table; ``None`` downloads ``gene``'s from gnomAD.
    path: pathlib.Path | None = None
    gene: str | None = None
    min_stars: int | None = 1
    release: str | None = None
    #: Fetch each variant's conditions from NCBI (needs the network).
    conditions: bool = False


@dataclass(frozen=True)
class RegionSpec:
    """One region bar over the effect map, like `fisseqborn.RegionTrack`."""

    regions: dict[str, tuple[int, int]]
    label: str | None = None
    palette: Mapping[str, str] | str | None = None


@dataclass(frozen=True)
class Params:
    pipeline_dir: str
    structures: tuple[StructureSpec, ...]
    layout: str | None = None
    track: str = "embeddings"
    batches: tuple[str, ...] | None = None
    exclude: tuple[str, ...] | None = None
    variant_col: str = "meta_aa_changes"
    #: A `fisseqborn.genes` gene whose landmark features are listed first.
    gene: str | None = None
    cache: bool | str = False
    refresh: bool = False
    chain: str | None = None
    offset: int = 0
    check_wt: bool = True
    plddt: str = "auto"
    clinvar: ClinvarSpec | None = None
    regions: tuple[RegionSpec, ...] = ()
    host: str = "127.0.0.1"
    port: int = 8050
    debug: bool = False
    #: The parameters file, if any.
    source: pathlib.Path | None = field(default=None, compare=False)


#: The keys of a parameters file.
KEYS = tuple(f for f in Params.__dataclass_fields__ if f != "source")
STRUCTURE_KEYS = ("path", "label", "chain", "offset")
CLINVAR_KEYS = tuple(ClinvarSpec.__dataclass_fields__)
REGION_KEYS = ("label", "regions", "palette")
CHOICES = {
    "layout": (None, "data", "embeddings"),
    "track": ("embeddings", "cp_features"),
    "plddt": ("auto", "yes", "no"),
}


def _resolve(value: "str | PathLike[str]", base: pathlib.Path | None) -> pathlib.Path:
    """``value`` with ``~`` expanded, relative to ``base`` (the file's directory) when not
    absolute; relative to the working directory without a ``base`` (command-line values)."""
    path = pathlib.Path(value).expanduser()
    return path if path.is_absolute() or base is None else base / path


def _is_remote(value: Any) -> bool:
    return _remote.parse(value) is not None


def resolve_paths(
    values: Mapping[str, Any], base: pathlib.Path | None
) -> dict[str, Any]:
    """``values`` (a parameters mapping) with its paths resolved against ``base``: the
    structures, the ClinVar table, a cache directory and a local pipeline directory."""
    out = dict(values)
    if isinstance(out.get("pipeline_dir"), (str, PathLike)) and not _is_remote(
        out["pipeline_dir"]
    ):
        out["pipeline_dir"] = str(_resolve(out["pipeline_dir"], base))
    if isinstance(out.get("cache"), str):
        out["cache"] = str(_resolve(out["cache"], base))
    if isinstance(out.get("structures"), list):
        out["structures"] = [
            {**s, "path": _resolve(s["path"], base)}
            if isinstance(s, Mapping) and isinstance(s.get("path"), (str, PathLike))
            else _resolve(s, base)
            if isinstance(s, (str, PathLike))
            else s
            for s in out["structures"]
        ]
    clinvar = out.get("clinvar")
    if isinstance(clinvar, (str, PathLike)):
        out["clinvar"] = {"path": _resolve(clinvar, base)}
    elif isinstance(clinvar, Mapping) and isinstance(
        clinvar.get("path"), (str, PathLike)
    ):
        out["clinvar"] = {**clinvar, "path": _resolve(clinvar["path"], base)}
    return out


def load_params(path: "str | PathLike[str]") -> dict[str, Any]:
    """The values of a parameters file, paths resolved against its directory.

    Raises
    ------
    ParamsError
        If the file is missing, isn't YAML or isn't a mapping of known keys.
    """
    path = pathlib.Path(path).expanduser()
    if not path.is_file():
        raise ParamsError(f"Parameters file not found: {path}")
    try:
        values = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as err:
        raise ParamsError(f"{path} is not valid YAML: {err}") from None
    if not isinstance(values, Mapping):
        raise ParamsError(f"{path} must be a mapping of parameter: value")
    unknown = sorted(set(values) - set(KEYS))
    if unknown:
        raise ParamsError(
            f"{path}: unknown parameter(s) {', '.join(unknown)}; valid: {', '.join(KEYS)}"
        )
    return resolve_paths(values, path.resolve().parent)


def _check(cond: bool, where: str, message: str) -> None:
    if not cond:
        raise ParamsError(f"{where}: {message}")


def _str_list(value: Any, where: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    _check(
        isinstance(value, Sequence) and all(isinstance(v, str) for v in value),
        where,
        "must be a list of strings",
    )
    return tuple(value)


def _int(value: Any, where: str, *, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    _check(
        isinstance(value, int) and not isinstance(value, bool),
        where,
        f"must be an integer, got {value!r}",
    )
    return value


def _bool(value: Any, where: str) -> bool:
    _check(isinstance(value, bool), where, f"must be true or false, got {value!r}")
    return value


def _optional_str(value: Any, where: str) -> str | None:
    _check(
        value is None or isinstance(value, str),
        where,
        f"must be a string, got {value!r}",
    )
    return value


def _structure(value: Any, where: str) -> StructureSpec:
    if isinstance(value, (str, PathLike)):
        return StructureSpec(pathlib.Path(value))
    _check(isinstance(value, Mapping), where, "must be a path or a mapping with a path")
    unknown = sorted(set(value) - set(STRUCTURE_KEYS))
    _check(
        not unknown,
        where,
        f"unknown key(s) {', '.join(unknown)}; valid: {', '.join(STRUCTURE_KEYS)}",
    )
    _check("path" in value, where, "needs a path")
    chain = value.get("chain")
    return StructureSpec(
        path=pathlib.Path(value["path"]),
        label=_optional_str(value.get("label"), f"{where}.label"),
        chain=None if chain is None else str(chain),
        offset=_int(value.get("offset"), f"{where}.offset", optional=True),
    )


def _clinvar(value: Any, where: str) -> ClinvarSpec | None:
    if value is None:
        return None
    if isinstance(value, (str, PathLike)):
        return ClinvarSpec(path=pathlib.Path(value))
    _check(isinstance(value, Mapping), where, "must be a path or a mapping")
    unknown = sorted(set(value) - set(CLINVAR_KEYS))
    _check(
        not unknown,
        where,
        f"unknown key(s) {', '.join(unknown)}; valid: {', '.join(CLINVAR_KEYS)}",
    )
    path = value.get("path")
    spec = ClinvarSpec(
        path=None if path is None else pathlib.Path(path),
        gene=_optional_str(value.get("gene"), f"{where}.gene"),
        min_stars=_int(value.get("min_stars", 1), f"{where}.min_stars", optional=True),
        release=None if value.get("release") is None else str(value["release"]),
        conditions=_bool(value.get("conditions", False), f"{where}.conditions"),
    )
    _check(
        spec.path is not None or spec.gene is not None, where, "needs a path or a gene"
    )
    return spec


def _region(value: Any, where: str) -> RegionSpec:
    _check(
        isinstance(value, Mapping), where, "must be a mapping with label and regions"
    )
    unknown = sorted(set(value) - set(REGION_KEYS))
    _check(
        not unknown,
        where,
        f"unknown key(s) {', '.join(unknown)}; valid: {', '.join(REGION_KEYS)}",
    )
    regions = value.get("regions")
    _check(
        isinstance(regions, Mapping) and bool(regions),
        f"{where}.regions",
        "must be a non-empty mapping of name: [first, last]",
    )
    ranges = {}
    for name, bounds in regions.items():
        at = f"{where}.regions.{name}"
        _check(
            isinstance(bounds, Sequence)
            and not isinstance(bounds, str)
            and len(bounds) == 2,
            at,
            f"must be [first, last], got {bounds!r}",
        )
        first, last = (_int(b, at) for b in bounds)
        _check(first <= last, at, f"first ({first}) is after last ({last})")
        ranges[str(name)] = (first, last)
    palette = value.get("palette")
    _check(
        palette is None or isinstance(palette, (str, Mapping)),
        f"{where}.palette",
        "must be a palette name or a mapping of region: color",
    )
    return RegionSpec(
        regions=ranges,
        label=None if value.get("label") is None else str(value["label"]),
        palette=dict(palette) if isinstance(palette, Mapping) else palette,
    )


def merge(
    file_values: Mapping[str, Any],
    cli_values: Mapping[str, Any],
    source: pathlib.Path | None = None,
) -> Params:
    """`Params` from a parameters file's values with command-line values over them.

    A command-line value replaces the file's whole key (``--structure`` replaces the
    structure list; ``--clinvar`` replaces only the ClinVar table path).
    """
    where = str(source) if source else "parameters"
    values = dict(file_values)
    cli = dict(cli_values)
    if "clinvar" in cli:
        old = values.get("clinvar")
        old = old if isinstance(old, Mapping) else {}
        new = cli.pop("clinvar")
        path = new.get("path") if isinstance(new, Mapping) else new
        values["clinvar"] = {**old, "path": path}
    values.update(cli)

    if values.get("pipeline_dir") in (None, ""):
        raise ParamsError(
            f"{where}: no pipeline_dir (give it in the parameters file or on the command line)"
        )
    structures = values.get("structures") or []
    _check(isinstance(structures, list), f"{where}: structures", "must be a list")
    if not structures:
        raise ParamsError(
            f"{where}: no structures (give them in the parameters file or with --structure)"
        )
    for key, choices in CHOICES.items():
        if key in values:
            _check(
                values[key] in choices,
                f"{where}: {key}",
                f"must be one of {', '.join(str(c) for c in choices)}, got {values[key]!r}",
            )
    cache = values.get("cache", False)
    _check(
        isinstance(cache, (bool, str)),
        f"{where}: cache",
        "must be true, false or a directory",
    )
    regions = values.get("regions") or []
    _check(isinstance(regions, list), f"{where}: regions", "must be a list of tracks")
    chain = values.get("chain")
    return Params(
        pipeline_dir=str(values["pipeline_dir"]),
        structures=tuple(
            _structure(s, f"{where}: structures[{i}]") for i, s in enumerate(structures)
        ),
        layout=values.get("layout"),
        track=values.get("track", "embeddings"),
        batches=_str_list(values.get("batches"), f"{where}: batches"),
        exclude=_str_list(values.get("exclude"), f"{where}: exclude"),
        variant_col=str(values.get("variant_col", "meta_aa_changes")),
        gene=_optional_str(values.get("gene"), f"{where}: gene"),
        cache=cache,
        refresh=_bool(values.get("refresh", False), f"{where}: refresh"),
        chain=None if chain is None else str(chain),
        offset=_int(values.get("offset", 0), f"{where}: offset"),
        check_wt=_bool(values.get("check_wt", True), f"{where}: check_wt"),
        plddt=values.get("plddt", "auto"),
        clinvar=_clinvar(values.get("clinvar"), f"{where}: clinvar"),
        regions=tuple(
            _region(r, f"{where}: regions[{i}]") for i, r in enumerate(regions)
        ),
        host=str(values.get("host", "127.0.0.1")),
        port=_int(values.get("port", 8050), f"{where}: port"),
        debug=_bool(values.get("debug", False), f"{where}: debug"),
        source=source,
    )
