"""Constants shared by conftest.py's synthetic pipeline_dir fixture and the tests that read it.

A plain module rather than conftest.py: test modules can't import conftest under
--import-mode=importlib. Its unique name keeps it from colliding with other packages' tests.
"""

PIPELINE_VARIANTS = ["A1A", "C2C", "G6G", "D3V", "E4K", "F5fs", "H7*"]
PIPELINE_FEATURES = [
    "AreaShape_Area",
    "Mean_Nuclei_Intensity_MeanIntensity_CH1",
    "Constant",
]
# Written out of order to check that batches come back in natural order.
PIPELINE_BATCHES = ["T2_R1", "T10_R1", "T1_R1"]
