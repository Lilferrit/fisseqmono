"""Fetching individual pipeline files from a remote host with ssh / scp.

Only the files a loader asks for are copied, never whole directories: a run directory also
holds pseudo-replicate aggregates and other outputs worth tens of GB.
"""

import logging
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from os import PathLike

logger = logging.getLogger(__name__)

#: ``[user@]host:path``, as scp writes it. ``host`` can't contain ``/``, so ``/abs/path`` and
#: ``rel/path`` are local.
_SPEC_RE = re.compile(r"^(?P<host>(?:[^@/:\s]+@)?[^@/:\s]+):(?P<path>.*)$")

#: Share one ssh connection between the listing and every scp, so a run prompts for a
#: password / 2FA at most once. ``%C`` is a hash of the connection, keeping the socket path short.
SSH_OPTIONS: tuple[str, ...] = (
    "-o",
    "ControlMaster=auto",
    "-o",
    f"ControlPath={'/tmp' if os.path.isdir('/tmp') else tempfile.gettempdir()}/fisseqborn-%C",
    "-o",
    "ControlPersist=60",
)

#: Download dirs created for remote runs without a ``download_dir``, reused for the session.
_TEMP_DIRS: dict[tuple[str, str], pathlib.Path] = {}


def parse(spec: object) -> tuple[str, str] | None:
    """``(host, path)`` for an scp-style ``"[user@]host:path"`` string, else ``None``.

    Only strings can be remote; a `pathlib.Path` is always local.
    """
    if not isinstance(spec, str):
        return None
    match = _SPEC_RE.match(spec)
    if match is None or not match["path"]:
        return None
    return match["host"], match["path"].rstrip("/") or "/"


def temp_dir(host: str, root: str) -> pathlib.Path:
    """The session's download dir for ``host:root``, created on first use. It is not
    deleted: the loaders' lazy frames read from it."""
    key = (host, root)
    if key not in _TEMP_DIRS:
        _TEMP_DIRS[key] = pathlib.Path(tempfile.mkdtemp(prefix="fisseqborn-"))
        logger.info("Downloading %s:%s to %s", host, root, _TEMP_DIRS[key])
    return _TEMP_DIRS[key]


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run ssh / scp, capturing their output. ssh asks for passwords and 2FA codes on the
    terminal (``/dev/tty``), so prompts still work from a shell; without a terminal (e.g. a
    notebook) it needs a key or a running agent."""
    return subprocess.run(list(argv), capture_output=True, text=True, check=False)


def _quote_glob(path: str) -> str:
    """``path`` quoted for the remote shell, leaving its ``*`` wildcards unquoted."""
    return "*".join(shlex.quote(part) if part else "" for part in path.split("*"))


def _quote_root(root: str) -> str:
    """``root`` quoted for the remote shell, keeping a leading ``~`` expandable."""
    if root == "~" or root.startswith("~/"):
        rest = root[2:]
        return "~/" + (shlex.quote(rest) if rest else "")
    return shlex.quote(root)


class Remote:
    """A run directory ``root`` on ``host``, mirrored file by file into ``local_root``."""

    def __init__(
        self, host: str, root: str, local_root: str | PathLike, *, refresh: bool = False
    ) -> None:
        self.host = host
        self.root = root
        self.local_root = pathlib.Path(local_root)
        self.refresh = refresh

    def __str__(self) -> str:
        return f"{self.host}:{self.root}"

    def _remote(self, rel: str) -> str:
        return f"{self.root}/{rel}" if rel else self.root

    def glob(self, patterns: Sequence[str]) -> list[str]:
        """The paths (relative to the run) matching any of ``patterns``, which may use
        ``*``; a pattern ending in ``/`` matches directories only. Runs one ``ssh ls``.
        Raises `FileNotFoundError` if nothing matches."""
        command = f"cd {_quote_root(self.root)} && ls -1d -- " + " ".join(
            _quote_glob(p) for p in patterns
        )
        result = _run(["ssh", *SSH_OPTIONS, self.host, command])
        lines = [line.rstrip("/") for line in result.stdout.splitlines() if line]
        if not lines:
            shown = ", ".join(f"{self.host}:{self._remote(p)}" for p in patterns)
            detail = result.stderr.strip()
            raise FileNotFoundError(
                f"Nothing matches {shown}" + (f" ({detail})" if detail else "")
            )
        return lines

    def fetch(self, rel_paths: Sequence[str]) -> list[pathlib.Path]:
        """Download these files (paths relative to the run) unless they are already in
        `local_root` (or `refresh` is set), and return their local paths.

        Files are copied into a staging directory first and moved into place when their
        scp finishes, so an interrupted download is never mistaken for a cached file. One
        scp runs per remote directory.
        """
        local = [self.local_root / rel for rel in rel_paths]
        todo = [
            (rel, path)
            for rel, path in dict(zip(rel_paths, local)).items()
            if self.refresh or not path.is_file()
        ]
        if not todo:
            return local
        logger.info(
            "Downloading %d file(s) from %s to %s (%d already present)",
            len(todo),
            self,
            self.local_root,
            len(set(rel_paths)) - len(todo),
        )
        by_dir: dict[str, list[tuple[str, pathlib.Path]]] = defaultdict(list)
        for rel, path in todo:
            by_dir[str(pathlib.PurePosixPath(rel).parent)].append((rel, path))
        self.local_root.mkdir(parents=True, exist_ok=True)
        staging = pathlib.Path(tempfile.mkdtemp(prefix=".part-", dir=self.local_root))
        try:
            for i, files in enumerate(by_dir.values()):
                target = staging / str(i)
                target.mkdir()
                sources = [f"{self.host}:{self._remote(rel)}" for rel, _ in files]
                result = _run(["scp", "-q", *SSH_OPTIONS, *sources, f"{target}/"])
                if result.returncode != 0:
                    message = f"scp from {self} failed: {result.stderr.strip()}"
                    if "No such file" in result.stderr:
                        raise FileNotFoundError(message)
                    raise RuntimeError(message)
                for rel, path in files:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(target / pathlib.PurePosixPath(rel).name, path)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return local
