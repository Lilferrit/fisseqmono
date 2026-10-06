"""Per-experiment pipeline stages shared by fisseq-data-pipeline and fisseq-embeddings-pipeline.

Each module holds a stage's algorithm, its Hydra structured config and its entry point,
``python -m fisseq_common.stages.<stage>`` (:func:`.config.stage_main`). Both pipelines run the
same entry points, from the same Nextflow modules (``packages/fisseq-common/nextflow``); what
differs between them is a config field each pipeline's ``conf/modules.config`` sets.

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
