"""ExCALIBR calibration of per-variant scores into ACMG/AMP evidence points.

`Dataset.calibrate` fits a `Calibration` against gnomAD and ClinVar controls;
`CalibrationPlot` draws it. The statistical core (`_core`, `_mixture`, `_skewnorm`,
`_result`) works on plain arrays and imports neither polars nor matplotlib.
"""

from ._plot import CalibrationPlot
from ._result import Calibration

__all__ = ["Calibration", "CalibrationPlot"]
