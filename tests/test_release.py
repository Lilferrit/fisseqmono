"""scripts/release.py sets one version across the workspace."""

from __future__ import annotations

import importlib.util
import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location(
    "release", ROOT / "scripts" / "release.py"
)
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A copy of the workspace's pyproject.toml files."""
    for path in release.pyprojects(ROOT):
        target = tmp_path / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, target)
    return tmp_path


def _versions(root: Path) -> set[str]:
    return {
        re.search(r'^version\s*=\s*"([^"]*)"', p.read_text(), re.M).group(1)
        for p in release.pyprojects(root)
    }


def _url_tags(root: Path) -> list[str]:
    return [
        tag
        for p in release.pyprojects(root)
        for tag in re.findall(
            r"fisseqmono@(v[^#]+)#subdirectory=packages/fisseq-common", p.read_text()
        )
    ]


def test_current_versions_agree():
    """The workspace is released as one version, and every package pins fisseq-common to it."""
    (version,) = _versions(ROOT)
    assert _url_tags(ROOT) == [f"v{version}"] * 3


def test_release_sets_every_version_and_url(workspace):
    changed = release.release("9.8.7", root=workspace, lock=False)
    assert len(changed) == 5
    assert _versions(workspace) == {"9.8.7"}
    assert _url_tags(workspace) == ["v9.8.7"] * 3
    # Only the version and the tags changed.
    for path in release.pyprojects(ROOT):
        before = path.read_text().splitlines()
        after = (workspace / path.relative_to(ROOT)).read_text().splitlines()
        diff = [(a, b) for a, b in zip(before, after) if a != b]
        assert len(before) == len(after)
        assert all("version" in a or "fisseqmono@" in a for a, _ in diff)


def test_release_is_idempotent(workspace):
    release.release("9.8.7", root=workspace, lock=False)
    assert release.release("9.8.7", root=workspace, lock=False) == []


@pytest.mark.parametrize("bad", ["2.1", "v2.1.0", "2.1.0rc1", ""])
def test_release_rejects_bad_versions(workspace, bad):
    with pytest.raises(ValueError, match="X.Y.Z"):
        release.release(bad, root=workspace, lock=False)
