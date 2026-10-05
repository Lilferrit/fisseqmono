"""fisseq_common imports with only its base dependencies installed."""

import importlib.metadata


def test_import() -> None:
    import fisseq_common  # noqa: F401

    assert importlib.metadata.version("fisseq-common")
