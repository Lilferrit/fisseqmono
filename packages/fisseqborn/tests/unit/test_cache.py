import pathlib

import pytest

from fisseqborn import _cache, _remote


def test_cache_root(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _cache.platformdirs, "user_cache_dir", lambda app: str(tmp_path / app)
    )
    assert _cache.cache_root(False) is None
    assert _cache.cache_root(None) is None
    assert _cache.cache_root(True) == tmp_path / "fisseqborn"
    assert _cache.cache_root(tmp_path / "c") == tmp_path / "c"
    assert _cache.cache_root(str(tmp_path)) == tmp_path
    assert _cache.cache_dir(tmp_path, "gnomad", "x") == tmp_path / "gnomad" / "x"
    assert _cache.cache_dir(False, "gnomad") is None
    with pytest.raises(TypeError, match="cache"):
        _cache.cache_root(3)


def test_remote_local_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(_remote, "_TEMP_DIRS", {})
    assert _remote.local_dir("h", "/runs/a", tmp_path) == pathlib.Path(
        tmp_path, "remote", "h", "runs", "a"
    )
    assert _remote.local_dir("h", "runs/a", tmp_path) == pathlib.Path(
        tmp_path, "remote", "h", "~", "runs", "a"
    )
    temp = _remote.local_dir("h", "/runs/a", False)
    assert temp == _remote.local_dir("h", "/runs/a", False)
    assert temp.is_dir()
