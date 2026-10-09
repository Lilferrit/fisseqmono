"""Where each pipeline publishes its per-experiment outputs, relative to the run's ``pipeline_dir``.

Both pipelines publish one directory per experiment (``batch``) under a few stage directories,
in the same layout: :class:`PipelineLayout`. The embeddings pipeline adds its cellDINO outputs
(cell images, metadata, embeddings) and a second track, the CellProfiler features, whose stage
directories carry a ``_cp_features`` suffix. Each pipeline's Nextflow ``publishDir`` settings
(``conf/modules.config``) are written to match :class:`DataPipelineLayout` /
:class:`EmbeddingsPipelineLayout` (each pipeline has a unit test checking this), and fisseqborn
reads every run through these classes.

Paths are relative POSIX strings, so the same layout serves a local directory and a remote one
read over scp. Join them onto the run directory to open a file::

    layout = detect(run_dir)
    for batch in layout.list_batches(run_dir, "ovwt"):
        results = pl.read_parquet(pathlib.Path(run_dir) / layout.ovwt_results(batch))

A layout method returns ``None`` for a file its pipeline (or track) never writes, e.g. the
embeddings pipeline's CellProfiler track has no blocklist.

Aggregates are one file per method, ``aggregates/<method>.parquet``, columns
``<feature>_<method>``, z-scored against the experiment's synonymous variants. Passthrough
aggregates (methods aggregated but never blocklisted) are one raw file per method.
"""

import abc
import os
import pathlib
from collections.abc import Iterable
from typing import Literal, Optional

#: The stage directories a batch can be listed from: the logical names every layout accepts.
Stage = Literal["qc_filter", "filter", "ovwt", "feature_select"]

#: The embeddings pipeline's two tracks.
Track = Literal["embeddings", "cp_features"]


class PipelineLayout(abc.ABC):
    """The per-experiment output paths both pipelines share, for one pipeline (and track)."""

    #: ``"data"`` or ``"embeddings"``.
    pipeline: str
    #: Every top-level directory of a run this layout reads (for :func:`detect`).
    stage_dirs: dict[str, str]

    def stage_dir(self, stage: Stage) -> str:
        """The directory under ``pipeline_dir`` whose subdirectories are the batches of ``stage``."""
        try:
            return self.stage_dirs[stage]
        except KeyError:
            raise ValueError(
                f"Unknown stage {stage!r}; expected one of {sorted(self.stage_dirs)}"
            ) from None

    def batch_dir(self, stage: Stage, batch: str) -> str:
        """``<stage dir>/<batch>``."""
        return f"{self.stage_dir(stage)}/{batch}"

    def list_batches(
        self, pipeline_dir: "str | os.PathLike[str]", stage: Stage
    ) -> list[str]:
        """The batch directories of ``stage`` in a local run, sorted by name; empty if none."""
        root = pathlib.Path(pipeline_dir) / self.stage_dir(stage)
        if not root.is_dir():
            return []
        return sorted(d.name for d in root.iterdir() if d.is_dir())

    # --- QC_FILTER ------------------------------------------------------------------------

    def filtered_cells(self, batch: str) -> str:
        """QC_FILTER's QC-passed cells."""
        return f"{self.batch_dir('qc_filter', batch)}/filtered_cells.parquet"

    def barcode_counts(self, batch: str) -> str:
        """QC_FILTER's per-barcode cell counts."""
        return f"{self.batch_dir('qc_filter', batch)}/barcode_counts.parquet"

    def variants_per_barcode(self, batch: str) -> str:
        """QC_FILTER's per-barcode variant counts."""
        return f"{self.batch_dir('qc_filter', batch)}/variants_per_barcode.parquet"

    # --- the filter (normalization) stage ---------------------------------------------------

    def filtered_keys(self, batch: str) -> str:
        """The filter stage's QC-passed cell keys, with ``meta_is_control``."""
        return f"{self.batch_dir('filter', batch)}/filtered_keys.parquet"

    def normalizer(self, batch: str) -> str:
        """The filter stage's control-fitted normalizer."""
        return f"{self.batch_dir('filter', batch)}/normalizer.parquet"

    # --- OVWT_BATCHWISE ---------------------------------------------------------------------

    def ovwt_results(self, batch: str) -> str:
        """Per-variant OvWT AUROCs."""
        return f"{self.batch_dir('ovwt', batch)}/results.parquet"

    def ovwt_cell_scores(self, batch: str) -> str:
        """Per-cell out-of-fold OvWT scores."""
        return f"{self.batch_dir('ovwt', batch)}/cell_scores.parquet"

    def ovwt_models(self, batch: str) -> str:
        """The pickled per-fold OvWT models."""
        return f"{self.batch_dir('ovwt', batch)}/models.pkl"

    # --- aggregation and feature selection --------------------------------------------------

    def _feature_select(self) -> bool:
        """Whether this layout's track runs the bootstrap feature selection."""
        return True

    def aggregate(self, batch: str, method: str) -> str:
        """One method's per-variant aggregates over every cell (z-scored against the
        synonymous variants)."""
        return f"{self.batch_dir('feature_select', batch)}/aggregates/{method}.parquet"

    def passthrough_aggregate(self, batch: str, method: str) -> Optional[str]:
        """One passthrough method's aggregates (never blocklisted; raw values)."""
        if not self._feature_select():
            return None
        return f"{self.batch_dir('feature_select', batch)}/passthrough_aggregates/{method}.parquet"

    def blocklist(self, batch: str) -> Optional[str]:
        """The combined blocklist of every method: ``feature``, ``median_r``, ``feature_ok``."""
        if not self._feature_select():
            return None
        return f"{self.batch_dir('feature_select', batch)}/blocklist.parquet"

    def method_blocklist(self, batch: str, method: str) -> Optional[str]:
        """One method's blocklist."""
        if not self._feature_select():
            return None
        return f"{self.batch_dir('feature_select', batch)}/blocklists/{method}.parquet"

    def split(self, batch: str, rep: int, half: int) -> Optional[str]:
        """One bootstrap replicate's half (cell keys)."""
        if not self._feature_select():
            return None
        return f"{self.batch_dir('feature_select', batch)}/splits/bootstrap_{rep}/half{half}.parquet"

    def half_aggregate(
        self, batch: str, rep: int, half: int, method: str
    ) -> Optional[str]:
        """One method's aggregates over one bootstrap half."""
        if not self._feature_select():
            return None
        return (
            f"{self.batch_dir('feature_select', batch)}/half_aggregates/bootstrap_{rep}/"
            f"{method}/half{half}_agg.parquet"
        )

    def correlations(self, batch: str, rep: int, method: str) -> Optional[str]:
        """One method's per-feature correlation between one replicate's two halves."""
        if not self._feature_select():
            return None
        return f"{self.batch_dir('feature_select', batch)}/correlations/{method}/bootstrap_{rep}.parquet"

    def selected(self, batch: str) -> Optional[str]:
        """FINALIZE_FEATURE_SELECT's per-variant table: the aggregates after the blocklist,
        z-scored to the synonymous variants, with the impact score, per-variant metadata and
        the passthrough aggregates."""
        if not self._feature_select():
            return None
        return f"{self.batch_dir('feature_select', batch)}/output.parquet"


class DataPipelineLayout(PipelineLayout):
    """fisseq-data-pipeline's per-experiment outputs."""

    pipeline = "data"
    stage_dirs = {
        "input": "input",
        "qc_filter": "qc_filter",
        "filter": "normalization",
        "ovwt": "ovwt_batchwise",
        "feature_select": "feature_select_batchwise",
    }

    def __repr__(self) -> str:
        return "DataPipelineLayout()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DataPipelineLayout)

    def __hash__(self) -> int:
        return hash(type(self))

    def input(self, batch: str) -> str:
        """INPUT's combined cell table."""
        return f"input/{batch}.parquet"


class EmbeddingsPipelineLayout(PipelineLayout):
    """fisseq-embeddings-pipeline's per-experiment outputs, for one track.

    ``track="embeddings"`` is the Cell-DINO track; ``"cp_features"`` the CellProfiler-feature
    track, which shares QC_FILTER, publishes under ``*_cp_features`` stage directories and has
    no bootstrap feature selection (no splits, half aggregates, correlations, blocklists,
    passthrough aggregates or ``output.parquet``).
    """

    pipeline = "embeddings"

    def __init__(self, track: Track = "embeddings") -> None:
        if track not in ("embeddings", "cp_features"):
            raise ValueError(
                f"track must be 'embeddings' or 'cp_features', got {track!r}"
            )
        self.track = track
        suffix = "" if track == "embeddings" else "_cp_features"
        self.stage_dirs = {
            "cell_images": "cell_images",
            "cell_metadata": "cell_metadata",
            "qc_filter": "qc_filter",
            "features": track,
            "filter": f"normalization{suffix}",
            "ovwt": f"ovwt_batchwise{suffix}",
            "feature_select": f"feature_select_batchwise{suffix}",
        }

    def __repr__(self) -> str:
        return f"EmbeddingsPipelineLayout(track={self.track!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EmbeddingsPipelineLayout) and other.track == self.track

    def __hash__(self) -> int:
        return hash((type(self), self.track))

    def _feature_select(self) -> bool:
        return self.track == "embeddings"

    def cell_table(self, batch: str) -> str:
        """BUILD_CELL_IMAGES' cell table."""
        return f"cell_images/{batch}/cell_table.parquet"

    def shards(self, batch: str) -> str:
        """BUILD_CELL_IMAGES' WebDataset shard index."""
        return f"cell_images/{batch}/shards.parquet"

    def metadata(self, batch: str) -> str:
        """BUILD_CELL_METADATA's per-cell ``meta_*`` table (QC_FILTER's input)."""
        return f"cell_metadata/{batch}/metadata.parquet"

    def features(self, batch: str) -> str:
        """The track's cell-level features: EMBED_CELLS' embeddings or BUILD_CP_FEATURES'
        CellProfiler features."""
        name = self.stage_dirs["features"]
        return f"{name}/{batch}/{name}.parquet"


#: Top-level directories only one pipeline writes, used by :func:`detect`.
_MARKERS = {
    "data": {"input"},
    "embeddings": {
        "cell_images",
        "cell_metadata",
        "embeddings",
        "cp_features",
    },
}


def detect_from_names(
    names: Iterable[str], track: Track = "embeddings"
) -> PipelineLayout:
    """The layout of a run whose top-level directory names are ``names``.

    Raises
    ------
    ValueError
        If ``names`` hold markers of both pipelines or of neither.
    """
    names = set(names)
    found = [p for p, markers in _MARKERS.items() if names & markers]
    if found == ["data"]:
        return DataPipelineLayout()
    if found == ["embeddings"]:
        return EmbeddingsPipelineLayout(track)
    what = "both pipelines" if found else "neither pipeline"
    raise ValueError(
        f"Can't tell which pipeline wrote this run: its directories {sorted(names)} match "
        f"{what} (data pipeline: {sorted(_MARKERS['data'])}; embeddings pipeline: "
        f"{sorted(_MARKERS['embeddings'])})"
    )


def detect(
    pipeline_dir: "str | os.PathLike[str]", track: Track = "embeddings"
) -> PipelineLayout:
    """The layout of the local run in ``pipeline_dir`` (see :func:`detect_from_names`).

    ``track`` selects the embeddings pipeline's track; it is ignored for a data-pipeline run.
    """
    root = pathlib.Path(pipeline_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"No pipeline run at {root}")
    return detect_from_names((d.name for d in root.iterdir() if d.is_dir()), track)
