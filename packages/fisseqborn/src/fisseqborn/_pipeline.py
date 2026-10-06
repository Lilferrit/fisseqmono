"""Locating files in a pipeline output directory.

Both pipelines' per-experiment output paths come from `fisseq_common.layout`: a run of
fisseq-data-pipeline is read through ``DataPipelineLayout``, one of
fisseq-embeddings-pipeline through ``EmbeddingsPipelineLayout`` (``track="embeddings"`` or
``"cp_features"``). The layout is detected from the run's top-level directories, or passed as
``layout=``.

Neither pipeline writes a ``global/`` directory: cross-experiment aggregation is done here
(see `fisseqborn.write_global` and the ``fisseqborn-global`` command).

A run can also be remote (``"user@host:/path/to/run"``): `source` then fetches only the
files a loader reads, over scp (see `_remote`).
"""

import fnmatch
import logging
import pathlib
import re
from collections.abc import Sequence
from os import PathLike

import polars as pl

from fisseq_common import layout as _layout
from fisseq_common.layout import (
    DataPipelineLayout,
    EmbeddingsPipelineLayout,
    PipelineLayout,
    Stage,
    Track,
)

from . import _data, _remote

#: A layout, or the pipeline that wrote the run: ``"data"`` or ``"embeddings"``.
LayoutSpec = PipelineLayout | str | None

logger = logging.getLogger(__name__)

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


def resolve_layout(
    spec: LayoutSpec, top_level: "Sequence[str] | None", track: Track
) -> PipelineLayout:
    """The layout for ``spec``: itself if a layout; ``"data"`` or ``"embeddings"`` (with
    ``track``); or, for ``None``, the one `fisseq_common.layout.detect_from_names` finds
    among the run's ``top_level`` directory names. A run with neither pipeline's marker
    directories (e.g. only ``feature_select_batchwise/`` and ``ovwt_batchwise/`` copied out)
    is read as a data-pipeline run."""
    if isinstance(spec, PipelineLayout):
        return spec
    if spec == "data":
        return DataPipelineLayout()
    if spec == "embeddings":
        return EmbeddingsPipelineLayout(track)
    if spec is not None:
        raise ValueError(
            f"layout must be 'data', 'embeddings' or a PipelineLayout, got {spec!r}"
        )
    try:
        return _layout.detect_from_names(top_level or (), track)
    except ValueError as err:
        if "neither pipeline" not in str(err):
            raise
        logger.debug("No pipeline marker directories; reading as a data-pipeline run")
        return DataPipelineLayout()


class Source:
    """Where a run's files come from: a local directory, or a remote one mirrored file by
    file into a local download directory (``root``), and the run's `PipelineLayout`."""

    def __init__(
        self,
        root: pathlib.Path,
        remote: "_remote.Remote | None" = None,
        layout: LayoutSpec = None,
        track: Track = "embeddings",
    ) -> None:
        self.root = root
        self.remote = remote
        self._layout_spec = layout
        self._track = track
        self._layout: PipelineLayout | None = None
        self._remote_dirs: list[str] | None = None

    def __str__(self) -> str:
        return str(self.remote or self.root)

    @property
    def layout(self) -> PipelineLayout:
        """The run's layout, detected on first use."""
        if self._layout is None:
            top_level = None
            if self._layout_spec is None:
                if self.remote is None:
                    top_level = (
                        [d.name for d in self.root.iterdir() if d.is_dir()]
                        if self.root.is_dir()
                        else []
                    )
                else:
                    top_level = [p for p in self._remote_listing() if "/" not in p]
            self._layout = resolve_layout(self._layout_spec, top_level, self._track)
        return self._layout

    def with_layout(self, layout: PipelineLayout) -> "Source":
        """The same run read through another layout (e.g. the embeddings pipeline's other
        track), sharing this one's remote listing."""
        other = Source(self.root, self.remote, layout)
        other._remote_dirs = self._remote_dirs
        return other

    def has_stage(self, stage: Stage) -> bool:
        """Whether the run has ``stage``'s directory (a remote run: from its listing)."""
        stage_dir = self.layout.stage_dir(stage)
        if self.remote is None:
            return (self.root / stage_dir).is_dir()
        return stage_dir in self._remote_listing()

    def _remote_listing(self) -> list[str]:
        """A remote run's directories two levels deep (``<stage>`` and
        ``<stage>/<batch>``), listed once with one ``ssh ls``: enough to detect the layout
        and list every stage's batches."""
        if self._remote_dirs is None:
            assert self.remote is not None
            try:
                found = self.remote.glob(["*/", "*/*/"])
            except FileNotFoundError:
                found = []
            self._remote_dirs = [p.rstrip("/") for p in found]
        return self._remote_dirs

    def batches(
        self,
        stage: Stage,
        batches: Sequence[str] | None = None,
        exclude: Patterns | None = None,
    ) -> list[str]:
        """The batch names of ``stage`` (``"feature_select"``, ``"ovwt"``, ...; see
        `fisseq_common.layout.Stage` and `select_batches`). A remote run is listed once, with
        one ``ssh ls`` (`_remote_listing`); nothing is downloaded."""
        stage_dir = self.layout.stage_dir(stage)
        if self.remote is None:
            return [d.name for d in batch_dirs(self.root, stage_dir, batches, exclude)]
        found = [
            pathlib.PurePosixPath(p).name
            for p in self._remote_listing()
            if pathlib.PurePosixPath(p).parent.as_posix() == stage_dir
        ]
        if not found:
            raise FileNotFoundError(
                f"No batch directories in {self.remote}/{stage_dir}"
            )
        return select_batches(found, f"{self.remote}/{stage_dir}", batches, exclude)

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
    layout: LayoutSpec = None,
    track: Track = "embeddings",
) -> Source:
    """A `Source` for a local run directory or an scp-style ``"[user@]host:path"`` spec.

    Remote files go to ``download_dir`` (mirroring the run's layout), or without one to a
    temporary directory reused for that run for the rest of the session. Files already
    there are reused unless ``refresh``.

    ``layout`` is the run's `PipelineLayout`, or ``"data"`` / ``"embeddings"``; by default
    it is detected (see `resolve_layout`). ``track`` picks the embeddings pipeline's track
    (``"embeddings"`` or ``"cp_features"``).
    """
    if isinstance(pipeline_dir, Source):
        if download_dir is not None:
            raise ValueError("download_dir can't be combined with an existing Source")
        if layout is not None or track != "embeddings":
            raise ValueError(
                "layout and track can't be combined with an existing Source"
            )
        return pipeline_dir
    spec = _remote.parse(pipeline_dir)
    if spec is None:
        if download_dir is not None:
            raise ValueError(
                f"download_dir is only used for remote runs ('user@host:/path'), but "
                f"{pipeline_dir!s} is local"
            )
        return Source(pathlib.Path(pipeline_dir), layout=layout, track=track)
    host, root = spec
    local = (
        pathlib.Path(download_dir)
        if download_dir is not None
        else _remote.temp_dir(host, root)
    )
    return Source(
        local, _remote.Remote(host, root, local, refresh=refresh), layout, track
    )


def scan(path: pathlib.Path) -> pl.LazyFrame:
    """`polars.scan_parquet`, failing early with the full path if the file is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline output not found: {path}")
    return pl.scan_parquet(path)


def tag(lf: pl.LazyFrame, batch: str, batch_col: str) -> pl.LazyFrame:
    return lf.with_columns(pl.lit(batch, dtype=pl.String).alias(batch_col))
