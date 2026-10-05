"""All four workspace packages install together and share one fisseq_common and one version."""

import importlib.metadata

import pytest

PACKAGES = ["fisseq-common", "fisseq-data-pipeline", "fisseq-embeddings-pipeline", "fisseqborn"]


@pytest.mark.parametrize("package", PACKAGES)
def test_one_version(package: str) -> None:
    assert importlib.metadata.version(package) == importlib.metadata.version("fisseq-common")


@pytest.mark.parametrize(
    "module", ["fisseq_common", "fisseq_data_pipeline", "fisseq_embeddings_pipeline", "fisseqborn"]
)
def test_imports(module: str) -> None:
    __import__(module)
