import json
import shutil
from pathlib import Path

import pytest
import yaml

from fisseq_structview import cli
from fisseq_structview.params import ParamsError


def test_only_pipeline_dir_and_structure_are_required(run_dir, pdb_path):
    _, params = cli.parse_params([str(run_dir), "--structure", str(pdb_path)])
    assert params.offset == 0 and params.port == 8050 and params.layout is None
    assert [s.path for s in params.structures] == [pdb_path]
    with pytest.raises(ParamsError, match="no structures"):
        cli.parse_params([str(run_dir)])
    with pytest.raises(ParamsError, match="no pipeline_dir"):
        cli.parse_params(["--structure", str(pdb_path)])


def _params_file(tmp_path, run_dir, pdb_path, **extra):
    """A parameters file in its own folder, with the run and structure next to it."""
    folder = tmp_path / "share"
    (folder / "structures").mkdir(parents=True)
    shutil.copytree(run_dir, folder / "run")
    shutil.copy(pdb_path, folder / "structures" / "a.pdb")
    shutil.copy(pdb_path, folder / "structures" / "b.pdb")
    values = {
        "pipeline_dir": "run",
        "structures": [
            "structures/a.pdb",
            {"path": "structures/b.pdb", "label": "Model B", "offset": 0},
        ],
        "port": 8200,
        "regions": [{"label": "Tile", "regions": {"T1": [1, 5], "T2": [4, 8]}}],
        **extra,
    }
    path = folder / "parameters.yaml"
    path.write_text(yaml.safe_dump(values))
    return path


def test_params_file_with_overrides(tmp_path, run_dir, pdb_path, monkeypatch):
    path = _params_file(tmp_path, run_dir, pdb_path)
    monkeypatch.chdir(tmp_path)  # relative paths in the file don't depend on the cwd
    served = _run(["--params", str(path), "--port", "8300"], monkeypatch)
    assert served["port"] == 8300
    layout = json.dumps(served["app"].layout.to_plotly_json(), default=str)
    assert "Model B" in layout and "a.pdb" in layout
    # --structure replaces the file's list (relative to the cwd).
    _, params = cli.parse_params(["-p", str(path), "--structure", "x.pdb", "y.pdb"])
    assert [s.path for s in params.structures] == [Path("x.pdb"), Path("y.pdb")]
    assert params.pipeline_dir == str(path.parent / "run")
    # The file can also be the positional argument.
    _, positional = cli.parse_params([str(path), "--port", "8400"])
    assert positional.port == 8400 and len(positional.structures) == 2
    assert positional.source == path.resolve()
    with pytest.raises(ParamsError, match="two parameters files"):
        cli.parse_params([str(path), "--params", str(path)])


def test_params_file_errors_exit(tmp_path, run_dir, pdb_path, capsys, monkeypatch):
    path = _params_file(tmp_path, run_dir, pdb_path, colour="blue")
    with pytest.raises(SystemExit) as err:
        _run(["--params", str(path)], monkeypatch)
    assert err.value.code == 2
    assert "unknown parameter(s) colour" in capsys.readouterr().err


def test_bad_clinvar_still_serves(tmp_path, run_dir, pdb_path, monkeypatch):
    path = _params_file(tmp_path, run_dir, pdb_path, clinvar="missing.tsv")
    served = _run(["--params", str(path)], monkeypatch)
    layout = json.dumps(served["app"].layout.to_plotly_json(), default=str)
    assert "ClinVar markers are off" in layout


def _run(argv, monkeypatch):
    served = {}
    monkeypatch.setattr("dash.Dash.run", lambda self, **kw: served.update(kw, app=self))
    cli.main(argv)
    return served


def test_serves(run_dir, pdb_path, monkeypatch):
    served = _run(
        [str(run_dir), "--structure", str(pdb_path), "--port", "8123"], monkeypatch
    )
    assert served["port"] == 8123


def test_bad_structure_exits_with_a_message(run_dir, tmp_path, capsys, monkeypatch):
    (tmp_path / "bad.pdb").write_text("garbage\n")
    with pytest.raises(SystemExit) as err:
        _run([str(run_dir), "--structure", str(tmp_path / "bad.pdb")], monkeypatch)
    assert err.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_wrong_offset_exits_with_a_message(run_dir, pdb_path, capsys, monkeypatch):
    with pytest.raises(SystemExit) as err:
        _run(
            [str(run_dir), "--structure", str(pdb_path), "--offset", "100"], monkeypatch
        )
    assert err.value.code == 2
    assert "--offset" in capsys.readouterr().err


def test_missing_ovwt_still_serves(tmp_path, pdb_path, monkeypatch):
    (tmp_path / "empty_run").mkdir()
    served = _run(
        [str(tmp_path / "empty_run"), "--structure", str(pdb_path)], monkeypatch
    )
    assert "app" in served
