"""Seaborn-style, object-oriented plots of polars DataFrames for the fisseq project.

>>> import fisseqborn as fb
>>> (fb.BoxPlot(df, x="meta_variant_type", y="meta_impact_score")
...    .annotate_pairs()
...    .save("vis/impact.png"))
"""

from . import fisseq
from ._base import Config, FigurePlot, Plot, config
from .box import BoxPlot
from .clustermap import ClusterMap, ClusterMapAxes, FeatureGroup
from .correlation import CorrelationPlot
from .embedding import EmbeddingPlot
from .heatmap import Heatmap
from .roc import RocPlot
from .volcano import VolcanoPlot

__all__ = [
    "BoxPlot",
    "ClusterMap",
    "ClusterMapAxes",
    "Config",
    "CorrelationPlot",
    "EmbeddingPlot",
    "FeatureGroup",
    "FigurePlot",
    "Heatmap",
    "Plot",
    "RocPlot",
    "VolcanoPlot",
    "config",
    "fisseq",
]
