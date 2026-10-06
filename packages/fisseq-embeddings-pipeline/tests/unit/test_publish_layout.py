"""conf/modules.config publishes every shared process's outputs where
fisseq_common.layout.EmbeddingsPipelineLayout (and so fisseqborn) expects them."""

import pathlib
import posixpath
import re

import pytest

from fisseq_common.layout import EmbeddingsPipelineLayout

CONFIG = pathlib.Path(__file__).parents[2] / "conf" / "modules.config"
EMB = EmbeddingsPipelineLayout("embeddings")
CP = EmbeddingsPipelineLayout("cp_features")
# Nextflow placeholders, as the layout's arguments: the publish paths use the process inputs.
B, REP, HALF, METHOD = "${batch_stem}", "${rep}", "${half}", "${method}"

EXPECTED = {
    "QC_FILTER": [
        EMB.filtered_cells(B),
        EMB.barcode_counts(B),
        EMB.variants_per_barcode(B),
    ],
    "NORMALIZE": [EMB.filtered_keys(B), EMB.normalizer(B)],
    "NORMALIZE_CP_FEATURES": [CP.filtered_keys(B), CP.normalizer(B)],
    "OVWT_BATCHWISE": [EMB.ovwt_results(B), EMB.ovwt_cell_scores(B)],
    "OVWT_BATCHWISE_CP_FEATURES": [CP.ovwt_results(B), CP.ovwt_cell_scores(B)],
    "AGGREGATE_FEATURE_TYPE_BATCHWISE": [EMB.aggregate(B, METHOD)],
    "AGGREGATE_FEATURE_TYPE_CP_FEATURES": [CP.aggregate(B, METHOD)],
    "AGGREGATE_FEATURE_TYPE_PASSTHROUGH": [EMB.passthrough_aggregate(B, METHOD)],
    "AGGREGATE_HALF_BATCHWISE": [EMB.half_aggregate(B, REP, HALF, METHOD)],
    "GENERATE_SPLIT_BATCHWISE": [EMB.split(B, REP, 1), EMB.split(B, REP, 2)],
    "CORRELATE_FEATURES_BATCHWISE": [EMB.correlations(B, REP, METHOD)],
    "BLOCKLIST_BATCHWISE": [EMB.method_blocklist(B, METHOD)],
    "COMBINE_BLOCKLISTS_BATCHWISE": [EMB.blocklist(B)],
    "FINALIZE_FEATURE_SELECT_BATCHWISE": [EMB.selected(B)],
}


def _publish_settings() -> dict[str, tuple[str, str | None]]:
    """``{process: (publish dir relative to pipeline_dir, saveAs file name or None)}``."""
    text = CONFIG.read_text()
    settings = {}
    for match in re.finditer(r"withName: '(\w+)' \{(.*?)\n    \}", text, re.S):
        name, body = match.groups()
        path = re.search(r'path: \{ "\$\{params\.pipeline_dir\}/(.*?)" \}', body)
        if path is None:
            continue
        save_as = re.search(r'saveAs: \{ _filename -> "(.*?)" \}', body)
        settings[name] = (path.group(1), save_as.group(1) if save_as else None)
    return settings


def test_every_published_process_is_checked():
    assert set(_publish_settings()) == set(EXPECTED)


@pytest.mark.parametrize("process", sorted(EXPECTED))
def test_publish_path_matches_layout(process):
    publish_dir, save_as = _publish_settings()[process]
    for path in EXPECTED[process]:
        assert posixpath.dirname(path) == publish_dir
        if save_as is not None:
            assert posixpath.basename(path) == save_as


# This pipeline's own modules set publishDir in the module file.
OWN_MODULES = pathlib.Path(__file__).parents[2] / "modules" / "local"


@pytest.mark.parametrize(
    "module, path",
    [
        ("build_cell_metadata", EMB.metadata(B)),
        ("build_cp_features", CP.features(B)),
        ("embed_cells", EMB.features(B)),
    ],
)
def test_own_module_publish_path_matches_layout(module, path):
    text = (OWN_MODULES / module / "main.nf").read_text()
    publish = re.search(r'publishDir \{ "\$\{params\.pipeline_dir\}/(.*?)" \}', text)
    assert publish is not None and publish.group(1) == posixpath.dirname(path)
