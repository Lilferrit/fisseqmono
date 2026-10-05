"""Per-experiment pipeline stages shared by fisseq-data-pipeline and fisseq-embeddings-pipeline.

Each module holds a stage's algorithm and its Hydra structured-config base class. The pipelines'
own modules are the entry points (``python -m fisseq_<pipeline>.<stage>``): they subclass the
config to set that pipeline's defaults and inputs, and call the functions here.

Needs the ``stages`` extra: ``pip install "fisseq-common[stages]"``.
"""

import importlib.util

_MISSING = [
    name
    for name in ("hydra", "sklearn", "xgboost", "scipy")
    if importlib.util.find_spec(name) is None
]
if _MISSING:
    raise ImportError(
        "fisseq_common.stages needs the 'stages' extra (missing: "
        f"{', '.join(_MISSING)}); install fisseq-common[stages]"
    )
