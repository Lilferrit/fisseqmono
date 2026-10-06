"""Where each pipeline publishes its per-experiment outputs, relative to the run's ``pipeline_dir``.

Both pipelines publish one directory per experiment (``batch``) under a few stage directories.
The data pipeline's Nextflow ``publishDir`` settings (``conf/modules.config``) are written to
match :class:`DataPipelineLayout`, the embeddings pipeline's to match
:class:`EmbeddingsPipelineLayout` (each pipeline has a unit test checking this), and fisseqborn
reads every run through these classes.

Paths are relative POSIX strings, so the same layout serves a local directory and a remote one
read over scp. Join them onto the run directory to open a file::

    layout = detect(run_dir)
    for batch in layout.list_batches(run_dir, "ovwt"):
        results = pl.read_parquet(pathlib.Path(run_dir) / layout.ovwt_results(batch))

A layout method returns ``None`` for a file its pipeline (or track) never writes, e.g. the
embeddings pipeline's CellProfiler track has no blocklist.

Aggregate files differ in shape between the pipelines:

- data pipeline: one file per method, ``aggregates/<method>.parquet``, columns ``<feature>_<method>``
  (z-scored against the experiment's synonymous variants);
- embeddings pipeline: one ``aggregate.parquet`` holding every method of the run, columns
  ``<feature>_<method>``, except that a run whose only method is ``median`` writes bare
  ``<feature>`` columns. :attr:`PipelineLayout.aggregate_per_method` tells them apart.

Passthrough aggregates (methods aggregated but never blocklisted) are one file per method in
both pipelines.
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
    """The per-experiment output paths of one pipeline (and track)."""

    #: ``"data"`` or ``"embeddings"``.
    pipeline: str
    #: Whether aggregates are one file per method (:meth:`aggregate`) or one file for all.
    aggregate_per_method: bool
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

    @abc.abstractmethod
    def aggregate(self, batch: str, method: Optional[str] = None) -> str:
        """The per-variant aggregates: ``method``'s file when :attr:`aggregate_per_method`,
        else the one file holding every method (``method`` is ignored)."""

    def passthrough_aggregate(self, batch: str, method: str) -> Optional[str]:
        """One passthrough method's aggregates (never blocklisted; raw values)."""
        return f"{self.batch_dir('feature_select', batch)}/passthrough_aggregates/{method}.parquet"

    def blocklist(self, batch: str) -> Optional[str]:
        """The combined blocklist of every method: ``feature``, ``median_r``, ``feature_ok``."""
        return f"{self.batch_dir('feature_select', batch)}/blocklist.parquet"

    def method_blocklist(self, batch: str, method: str) -> Optional[str]:
        """One method's blocklist."""
        return f"{self.batch_dir('feature_select', batch)}/blocklists/{method}.parquet"

    @abc.abstractmethod
    def split(self, batch: str, rep: int, half: int) -> Optional[str]:
        """One bootstrap replicate's half (cell keys)."""

    @abc.abstractmethod
    def half_aggregate(
        self, batch: str, rep: int, half: int, method: str
    ) -> Optional[str]:
        """One method's aggregates over one bootstrap half."""

    @abc.abstractmethod
    def correlations(self, batch: str, rep: int, method: str) -> Optional[str]:
        """One method's per-feature correlation between one replicate's two halves."""

    @abc.abstractmethod
    def selected(self, batch: str) -> Optional[str]:
        """The aggregates after the blocklist (and, for the data pipeline, pycytominer
        selection, impact score and per-variant metadata)."""


class DataPipelineLayout(PipelineLayout):
    """fisseq-data-pipeline's per-experiment outputs."""

    pipeline = "data"
    aggregate_per_method = True
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

    def aggregate(self, batch: str, method: Optional[str] = None) -> str:
        if method is None:
            raise ValueError("The data pipeline writes one aggregate file per method")
        return f"{self.batch_dir('feature_select', batch)}/aggregates/{method}.parquet"

    def split(self, batch: str, rep: int, half: int) -> str:
        return f"{self.batch_dir('feature_select', batch)}/splits/bootstrap_{rep}/half{half}.parquet"

    def half_aggregate(self, batch: str, rep: int, half: int, method: str) -> str:
        return (
            f"{self.batch_dir('feature_select', batch)}/half_aggregates/bootstrap_{rep}/"
            f"{method}/half{half}_agg.parquet"
        )

    def correlations(self, batch: str, rep: int, method: str) -> str:
        return f"{self.batch_dir('feature_select', batch)}/correlations/{method}/bootstrap_{rep}.parquet"

    def selected(self, batch: str) -> str:
        return f"{self.batch_dir('feature_select', batch)}/output.parquet"

    def pca_components(self, batch: str) -> str:
        """FINALIZE_FEATURE_SELECT's PCA loadings (only with ``run_pca``)."""
        return f"{self.batch_dir('feature_select', batch)}/pca_components.parquet"


class EmbeddingsPipelineLayout(PipelineLayout):
    """fisseq-embeddings-pipeline's per-experiment outputs, for one track.

    ``track="embeddings"`` is the Cell-DINO track; ``"cp_features"`` the CellProfiler-feature
    track, which shares QC_FILTER and has no reproducibility filtering (no splits, half
    aggregates, correlations, blocklists or passthrough aggregates).
    """

    pipeline = "embeddings"
    aggregate_per_method = False

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
            "filter": "filter_embeddings"
            if track == "embeddings"
            else "filter_cp_features",
            "ovwt": f"ovwt_batchwise{suffix}",
            "feature_select": f"feature_select_batchwise{suffix}",
        }
        if track == "embeddings":
            self.stage_dirs["features"] = "embeddings"
        else:
            self.stage_dirs["features"] = "cp_features"

    def __repr__(self) -> str:
        return f"EmbeddingsPipelineLayout(track={self.track!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EmbeddingsPipelineLayout) and other.track == self.track

    def __hash__(self) -> int:
        return hash((type(self), self.track))

    def _reproducibility(self) -> bool:
        return self.track == "embeddings"

    def cell_table(self, batch: str) -> str:
        """BUILD_CELL_IMAGES' cell table."""
        return f"cell_images/{batch}/cell_table.parquet"

    def tiles(self, batch: str) -> str:
        """BUILD_CELL_IMAGES' per-tile shard index."""
        return f"cell_images/{batch}/tiles.parquet"

    def metadata(self, batch: str) -> str:
        """BUILD_CELL_METADATA's per-cell ``meta_*`` table (QC_FILTER's input)."""
        return f"cell_metadata/{batch}/metadata.parquet"

    def features(self, batch: str) -> str:
        """The track's cell-level features: EMBED_CELLS' embeddings or BUILD_CP_FEATURES'
        CellProfiler features."""
        name = self.stage_dirs["features"]
        return f"{name}/{batch}/{name}.parquet"

    def aggregate(self, batch: str, method: Optional[str] = None) -> str:
        return f"{self.batch_dir('feature_select', batch)}/aggregate.parquet"

    def passthrough_aggregate(self, batch: str, method: str) -> Optional[str]:
        return (
            super().passthrough_aggregate(batch, method)
            if self._reproducibility()
            else None
        )

    def blocklist(self, batch: str) -> Optional[str]:
        return super().blocklist(batch) if self._reproducibility() else None

    def method_blocklist(self, batch: str, method: str) -> Optional[str]:
        return (
            super().method_blocklist(batch, method) if self._reproducibility() else None
        )

    def split(self, batch: str, rep: int, half: int) -> Optional[str]:
        if not self._reproducibility():
            return None
        return f"{self.batch_dir('feature_select', batch)}/splits/rep{rep}/half{half}.parquet"

    def half_aggregate(
        self, batch: str, rep: int, half: int, method: str
    ) -> Optional[str]:
        if not self._reproducibility():
            return None
        return (
            f"{self.batch_dir('feature_select', batch)}/half_aggregates/rep{rep}/"
            f"half{half}/{method}.parquet"
        )

    def correlations(self, batch: str, rep: int, method: str) -> Optional[str]:
        if not self._reproducibility():
            return None
        return f"{self.batch_dir('feature_select', batch)}/correlations/rep{rep}/{method}.parquet"

    def selected(self, batch: str) -> Optional[str]:
        """FILTER_AGGREGATE's blocklist-filtered ``aggregate.parquet``."""
        if not self._reproducibility():
            return None
        return f"{self.batch_dir('feature_select', batch)}/filtered_aggregate.parquet"

    def aggregate_with_passthrough(self, batch: str) -> Optional[str]:
        """FILTER_AGGREGATE's filtered aggregates with the passthrough methods joined on."""
        if not self._reproducibility():
            return None
        return f"{self.batch_dir('feature_select', batch)}/aggregate_with_passthrough.parquet"


#: Top-level directories only one pipeline writes, used by :func:`detect`.
_MARKERS = {
    "data": {"normalization", "input"},
    "embeddings": {
        "filter_embeddings",
        "filter_cp_features",
        "embeddings",
        "cell_metadata",
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
