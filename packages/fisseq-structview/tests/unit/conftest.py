import pytest
from structview_testdata import RUN, write_feature_run, write_pdb, write_run


@pytest.fixture
def pdb_path(tmp_path):
    return write_pdb(tmp_path / "model.pdb")


@pytest.fixture
def run_dir(tmp_path):
    return write_run(tmp_path / "run", RUN)


@pytest.fixture
def feature_run_dir(tmp_path):
    """A run with OvWT scores and feature profiles, over the same variants."""
    return write_feature_run(write_run(tmp_path / "both", RUN), RUN)
