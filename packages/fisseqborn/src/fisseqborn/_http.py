"""JSON over HTTP with retries, and the parquet files downloads are cached in."""

import json
import logging
import os
import pathlib
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import polars as pl

from . import _cache

logger = logging.getLogger(__name__)

USER_AGENT = "fisseqborn (https://github.com/Lilferrit/fisseqmono)"

#: Seconds to wait before each retry of a rate-limited or failed request.
RETRY_DELAYS: tuple[float, ...] = (2.0, 10.0, 30.0, 60.0)

_RETRY_STATUS = {429, 500, 502, 503, 504}


class RateLimited(Exception):
    """Raised by a response check to retry the request after a delay."""


def request_json(
    url: str,
    *,
    body: Any = None,
    check: Callable[[Any], None] | None = None,
    timeout: float = 120.0,
) -> Any:
    """GET ``url`` (or POST ``body`` as JSON) and decode the JSON response.

    HTTP 429 / 5xx, network errors and `RateLimited` raised by ``check(response)`` are
    retried after each of `RETRY_DELAYS`; the last failure is raised.
    """
    data = None if body is None else json.dumps(body).encode()
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    for attempt, delay in enumerate((*RETRY_DELAYS, None)):
        request = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            if check is not None:
                check(payload)
            return payload
        except (RateLimited, urllib.error.URLError, TimeoutError) as err:
            retryable = not isinstance(err, urllib.error.HTTPError) or (
                err.code in _RETRY_STATUS
            )
            if not retryable or delay is None:
                raise
            logger.info(
                "Request to %s failed (%s); retry %d in %.0f s",
                url,
                err,
                attempt + 1,
                delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def cached_frame(
    cache: "_cache.Cache",
    parts: tuple[str, ...],
    fetch: Callable[[], pl.DataFrame],
    *,
    refresh: bool = False,
) -> pl.DataFrame:
    """``fetch()``, stored as ``<cache>/<parts>`` and read from there on later calls unless
    ``refresh``. Without a ``cache`` it is fetched every time."""
    path = _cache.cache_dir(cache, *parts)
    if path is None:
        return fetch()
    if path.is_file() and not refresh:
        logger.info("Reading %s from the cache", path)
        return pl.read_parquet(path)
    frame = fetch()
    write_atomic(frame, path)
    return frame


def write_atomic(frame: pl.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".part-{os.getpid()}-{path.name}")
    frame.write_parquet(part)
    os.replace(part, path)
