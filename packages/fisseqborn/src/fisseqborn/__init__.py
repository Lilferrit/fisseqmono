"""Seaborn-style, object-oriented plots of polars DataFrames for the fisseq project,
plus chainable datasets that load and prepare the fisseq pipeline's outputs.

>>> import fisseqborn as fb
>>> profiles = fb.Profiles.from_pipeline(run_dir).variant_type()  # already z-scored per batch
>>> (fb.BoxPlot(profiles.median_across_batches().impact_score(),
...             x="meta_variant_type", y="meta_impact_score")
...    .annotate_pairs()
...    .save("vis/impact.png"))
"""

from . import fisseq, genes
from ._base import Config, FigurePlot, Plot, config
from .batch_correlation import BatchCorrelationGrid, BatchCorrelationHeatmap
from .blocklist import Blocklists
from .box import BoxPlot
from .calibration import Calibration, CalibrationPlot
from .clustermap import ClusterMap, ClusterMapAxes, FeatureGroup
from .correlation import CorrelationPlot
from .dataset import Dataset
from .embedding import EmbeddingPlot
from .explained_variance import ExplainedVariancePlot
from .genes import Gene
from .global_aggregate import write_global
from .heatmap import Heatmap
from .ovwt import OvwtScores
from .pairplot import PairPlot
from .profiles import Profiles
from .roc import RocPlot
from .summary import ClusterSummary
from .variant_map import RegionTrack, VariantEffectMap, VariantEffectMapAxes
from .volcano import VolcanoPlot

__all__ = [
    "BatchCorrelationGrid",
    "BatchCorrelationHeatmap",
    "Blocklists",
    "BoxPlot",
    "Calibration",
    "CalibrationPlot",
    "ClusterMap",
    "ClusterMapAxes",
    "ClusterSummary",
    "Config",
    "CorrelationPlot",
    "Dataset",
    "EmbeddingPlot",
    "ExplainedVariancePlot",
    "FeatureGroup",
    "FigurePlot",
    "Gene",
    "Heatmap",
    "OvwtScores",
    "PairPlot",
    "Plot",
    "Profiles",
    "RegionTrack",
    "RocPlot",
    "VariantEffectMap",
    "VariantEffectMapAxes",
    "VolcanoPlot",
    "config",
    "fisseq",
    "genes",
    "write_global",
]
