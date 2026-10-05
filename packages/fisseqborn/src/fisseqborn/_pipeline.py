"""Locating files in a fisseq-data-pipeline output directory.

See the pipeline's ``docs/architecture.md`` ("Output layout")::

    <pipeline_dir>/
      feature_select_batchwise/<batch>/
        aggregates/<type>.parquet              # already z-scored to each batch's synonymous controls
        passthrough_aggregates/<type>.parquet  # raw (p-values such as KSnegLogP)
        blocklists/<type>.parquet              # feature, median_r, feature_ok
        output.parquet                         # selected profile + meta_num_cells etc.
      ovwt_batchwise/<batch>/results.parquet

The pipeline no longer writes a ``global/`` directory: cross-experiment aggregation is done
here (see `fisseqborn.write_global` and the ``fisseqborn-global`` command).

A run can also be remote (``"user@host:/path/to/run"``): `source` then fetches only the
files a loader reads, over scp (see `_remote`).
"""

import fnmatch
import pathlib
import re
from collections.abc import Sequence
from os import PathLike

import polars as pl

from . import _data, _remote

FEATURE_SELECT = "feature_select_batchwise"
OVWT = "ovwt_batchwise"

#: Batch-name patterns: ``fnmatch`` globs (``"T10_*"``) or compiled regexes.
Patterns = str | re.Pattern[str] | Sequence[str | re.Pattern[str]]


def matches(name: str, patterns: Patterns) -> bool:
    """Whether ``name`` matches any of ``patterns``: strings are ``fnmatch`` globs over
    the whole name, compiled `re.Pattern` objects are searched for anywhere in it."""
    if isinstance(patterns, (str, re.Pattern)):
        patterns = [patterns]
    return any(
        p.search(name) is not None
        if isinstance(p, re.Pattern)
        else fnmatch.fnmatchcase(name, p)
        for p in patterns
    )


def select_batches(
    found: Sequence[str],
    where: str,
    batches: Sequence[str] | None = None,
    exclude: Patterns | None = None,
) -> list[str]:
    """Batch names from those ``found`` in ``where``: all of them in natural order (T2 <
    T10), or ``batches`` in that order (a missing one raises `FileNotFoundError`), minus
    those matching ``exclude`` (see `matches`; excluding every batch raises `ValueError`)."""
    if batches is None:
        names = sorted(found, key=_data._natural_key)
        if not names:
            raise FileNotFoundError(f"No batch directories in {where}")
    else:
        names = list(batches)
        missing = [b for b in names if b not in set(found)]
        if missing:
            raise FileNotFoundError(f"Batch(es) {missing} not found in {where}")
    if exclude is not None:
        names = [b for b in names if not matches(b, exclude)]
        if not names:
            raise ValueError(f"exclude={exclude!r} excludes every batch in {where}")
    return names


def batch_dirs(
    pipeline_dir: str | PathLike,
    stage: str,
    batches: Sequence[str] | None = None,
    exclude: Patterns | None = None,
) -> list[pathlib.Path]:
    """The local ``<pipeline_dir>/<stage>/<batch>`` directories chosen by `select_batches`."""
    root = pathlib.Path(pipeline_dir) / stage
    if not root.is_dir():
        raise FileNotFoundError(
            f"No {stage!r} directory in {pipeline_dir}: {root} does not exist"
        )
    found = [d.name for d in root.iterdir() if d.is_dir()]
    return [root / b for b in select_batches(found, str(root), batches, exclude)]


class Source:
    """Where a run's files come from: a local directory, or a remote one mirrored file by
    file into a local download directory (``root``)."""

    def __init__(
        self, root: pathlib.Path, remote: "_remote.Remote | None" = None
    ) -> None:
        self.root = root
        self.remote = remote

    def __str__(self) -> str:
        return str(self.remote or self.root)

    def batches(
        self,
        stage: str,
        batches: Sequence[str] | None = None,
        exclude: Patterns | None = None,
    ) -> list[str]:
        """The batch names of ``stage`` (see `select_batches`). Remote runs are listed
        with one ``ssh ls``; nothing is downloaded."""
        if self.remote is None:
            return [d.name for d in batch_dirs(self.root, stage, batches, exclude)]
        try:
            found = [
                pathlib.PurePosixPath(p).name for p in self.remote.glob([f"{stage}/*/"])
            ]
        except FileNotFoundError as err:
            raise FileNotFoundError(
                f"No batch directories in {self.remote}/{stage}"
            ) from err
        return select_batches(found, f"{self.remote}/{stage}", batches, exclude)

    def glob(self, patterns: Sequence[str]) -> list[str]:
        """The files (paths relative to the run) matching any of ``patterns``; empty if none."""
        if self.remote is None:
            return sorted(
                p.relative_to(self.root).as_posix()
                for pattern in patterns
                for p in self.root.glob(pattern)
                if p.is_file()
            )
        try:
            return self.remote.glob(patterns)
        except FileNotFoundError:
            return []

    def files(self, rel_paths: Sequence[str]) -> list[pathlib.Path]:
        """Local paths of these files (relative to the run), downloading the missing ones
        of a remote run first, all at once."""
        if self.remote is None:
            return [self.root / rel for rel in rel_paths]
        return self.remote.fetch(rel_paths)


def source(
    pipeline_dir: "str | PathLike | Source",
    download_dir: str | PathLike | None = None,
    refresh: bool = False,
) -> Source:
    """A `Source` for a local run directory or an scp-style ``"[user@]host:path"`` spec.

    Remote files go to ``download_dir`` (mirroring the run's layout), or without one to a
    temporary directory reused for that run for the rest of the session. Files already
    there are reused unless ``refresh``.
    """
    if isinstance(pipeline_dir, Source):
        if download_dir is not None:
            raise ValueError("download_dir can't be combined with an existing Source")
        return pipeline_dir
    spec = _remote.parse(pipeline_dir)
    if spec is None:
        if download_dir is not None:
            raise ValueError(
                f"download_dir is only used for remote runs ('user@host:/path'), but "
                f"{pipeline_dir!s} is local"
            )
        return Source(pathlib.Path(pipeline_dir))
    host, root = spec
    local = (
        pathlib.Path(download_dir)
        if download_dir is not None
        else _remote.temp_dir(host, root)
    )
    return Source(local, _remote.Remote(host, root, local, refresh=refresh))


def scan(path: pathlib.Path) -> pl.LazyFrame:
    """`polars.scan_parquet`, failing early with the full path if the file is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline output not found: {path}")
    return pl.scan_parquet(path)


def tag(lf: pl.LazyFrame, batch: str, batch_col: str) -> pl.LazyFrame:
    return lf.with_columns(pl.lit(batch, dtype=pl.String).alias(batch_col))
