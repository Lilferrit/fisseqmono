"""conf/modules.config publishes every shared process's outputs where
fisseq_common.layout.DataPipelineLayout (and so fisseqborn) expects them."""

import pathlib
import posixpath
import re

import pytest

from fisseq_common.layout import DataPipelineLayout

CONFIG = pathlib.Path(__file__).parents[2] / "conf" / "modules.config"
LAYOUT = DataPipelineLayout()
# Nextflow placeholders, as the layout's arguments: the publish paths use the process inputs.
B, REP, HALF, METHOD = "${batch_stem}", "${rep}", "${half}", "${method}"

EXPECTED = {
    "QC_FILTER": [
        LAYOUT.filtered_cells(B),
        LAYOUT.barcode_counts(B),
        LAYOUT.variants_per_barcode(B),
    ],
    "NORMALIZE": [LAYOUT.filtered_keys(B), LAYOUT.normalizer(B)],
    "OVWT_BATCHWISE": [LAYOUT.ovwt_results(B), LAYOUT.ovwt_cell_scores(B)],
    "AGGREGATE_FEATURE_TYPE_BATCHWISE": [LAYOUT.aggregate(B, METHOD)],
    "AGGREGATE_FEATURE_TYPE_PASSTHROUGH": [LAYOUT.passthrough_aggregate(B, METHOD)],
    "AGGREGATE_HALF_BATCHWISE": [LAYOUT.half_aggregate(B, REP, HALF, METHOD)],
    "GENERATE_SPLIT_BATCHWISE": [LAYOUT.split(B, REP, 1), LAYOUT.split(B, REP, 2)],
    "CORRELATE_FEATURES_BATCHWISE": [LAYOUT.correlations(B, REP, METHOD)],
    "BLOCKLIST_BATCHWISE": [LAYOUT.method_blocklist(B, METHOD)],
    "COMBINE_BLOCKLISTS_BATCHWISE": [LAYOUT.blocklist(B)],
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
