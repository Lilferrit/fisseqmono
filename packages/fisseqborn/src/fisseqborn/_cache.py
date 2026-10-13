"""Where downloads are cached.

Every function that downloads takes ``cache``: ``False`` (the default) keeps nothing
between sessions, a path caches under that directory, and ``True`` caches under the
user cache directory (``platformdirs.user_cache_dir("fisseqborn")``, e.g.
``~/Library/Caches/fisseqborn`` on macOS, ``~/.cache/fisseqborn`` on Linux). Each kind of
download has its own subdirectory: ``remote/``, ``gnomad/``, ``clinvar/``.
"""

import pathlib
from os import PathLike

import platformdirs

#: The type of a ``cache`` argument.
Cache = bool | str | PathLike


def cache_root(cache: Cache | None) -> pathlib.Path | None:
    """The cache directory for ``cache``, or ``None`` when nothing is cached."""
    if cache is None or cache is False:
        return None
    if cache is True:
        return pathlib.Path(platformdirs.user_cache_dir("fisseqborn"))
    if isinstance(cache, (str, PathLike)):
        return pathlib.Path(cache).expanduser()
    raise TypeError(f"cache must be a bool or a path, got {type(cache).__name__}")


def cache_dir(cache: Cache | None, *parts: str) -> pathlib.Path | None:
    """``cache_root(cache) / parts``, or ``None`` when nothing is cached."""
    root = cache_root(cache)
    return None if root is None else root.joinpath(*parts)
