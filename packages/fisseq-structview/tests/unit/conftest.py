import pytest
from structview_testdata import RUN, write_pdb, write_run


@pytest.fixture
def pdb_path(tmp_path):
    return write_pdb(tmp_path / "model.pdb")


@pytest.fixture
def run_dir(tmp_path):
    return write_run(tmp_path / "run", RUN)
