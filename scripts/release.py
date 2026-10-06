"""Set the workspace's one version everywhere, ready to tag.

    uv run python scripts/release.py 2.1.0

Sets ``version`` in the root pyproject.toml and the four packages', and points the three
packages' ``fisseq-common @ git+https://github.com/Lilferrit/fisseqmono@vX.Y.Z#...`` dependency
at the new tag, then runs ``uv lock``. Commit the result and tag it ``vX.Y.Z``: the git URLs
(used when a package is installed outside the workspace) then resolve to that commit.
``--no-lock`` skips ``uv lock``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "fisseq-common",
    "fisseq-data-pipeline",
    "fisseq-embeddings-pipeline",
    "fisseqborn",
)

_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
#: ``version = "..."`` in the [project] table (the first one in the file).
_PROJECT_VERSION = re.compile(r'^(version\s*=\s*")[^"]*(")', re.M)
#: The tag in a fisseq-common git URL.
_COMMON_URL = re.compile(
    r"(github\.com/Lilferrit/fisseqmono@)v[^#\"]+(#subdirectory=packages/fisseq-common)"
)


def pyprojects(root: Path = ROOT) -> list[Path]:
    """The root pyproject.toml and every package's."""
    return [
        root / "pyproject.toml",
        *(root / "packages" / p / "pyproject.toml" for p in PACKAGES),
    ]


def set_version(text: str, version: str) -> str:
    """``text`` (a pyproject.toml) with its project version and fisseq-common URL tag set."""
    text, n = _PROJECT_VERSION.subn(rf"\g<1>{version}\g<2>", text, count=1)
    if n != 1:
        raise ValueError('no `version = "..."` line')
    return _COMMON_URL.sub(rf"\g<1>v{version}\g<2>", text)


def release(version: str, root: Path = ROOT, lock: bool = True) -> list[Path]:
    """Set ``version`` in every pyproject.toml under ``root``; return the files changed."""
    if not _VERSION.match(version):
        raise ValueError(f"version must look like X.Y.Z, got {version!r}")
    changed = []
    for path in pyprojects(root):
        text = path.read_text()
        new = set_version(text, version)
        if new != text:
            path.write_text(new)
            changed.append(path)
    if lock:
        subprocess.run(["uv", "lock"], cwd=root, check=True)
    return changed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="the new version, X.Y.Z (tag it vX.Y.Z)")
    parser.add_argument("--no-lock", action="store_true", help="don't run uv lock")
    args = parser.parse_args(argv)
    try:
        changed = release(args.version, lock=not args.no_lock)
    except ValueError as err:
        parser.error(str(err))
    for path in changed:
        print(path.relative_to(ROOT))
    print(f"Now commit, then: git tag v{args.version}", file=sys.stderr)


if __name__ == "__main__":
    main()
