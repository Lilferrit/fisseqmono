"""Grow every starcall rule's memory request with each retry.

Imported by ``Snakefile`` (next to this file), so it runs in the ops env's
Python 3.10 snakemake interpreter: standard library only. Unit tests load
it by path.

With ``--retries N`` (BUILD_CELL_IMAGES passes ``starcall_retries``),
snakemake reruns a failed job up to N more times with ``attempt`` 2, 3, ...
Upstream starcall-workflow's newer rules scale some ``mem_mb`` themselves
(``(...) * attempt``); the pinned commit's don't, so a retry of an
OOM-killed job would ask for the same memory and die again.
:func:`scale_mem_by_attempt` doubles the request on every attempt (1x, 2x,
4x, ...), for every rule, whatever its ``mem_mb`` is: one of starcall's
lambdas, a profile's ``default-resources``/``set-resources`` value, or a
number.
"""

import inspect


def _call(func, wildcards, candidates):
    """Call a snakemake resource function with the keyword arguments it
    declares, the way snakemake does (``get_input_function_aux_params``)."""
    params = inspect.signature(func).parameters
    return func(wildcards, **{k: v for k, v in candidates.items() if k in params})


def scaled_mem_mb(base):
    """``base`` (a ``mem_mb`` resource: number, numeric string or function)
    as a resource function returning ``base * 2 ** (attempt - 1)``. A value that isn't
    a number (e.g. ``"16G"``) is returned unscaled."""

    def mem_mb(wildcards, input, attempt, threads, rulename):
        value = base
        if callable(base):
            value = _call(
                base,
                wildcards,
                dict(input=input, attempt=attempt, threads=threads, rulename=rulename),
            )
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return value
        if value is None:
            return None
        return int(round(value * 2 ** (attempt - 1)))

    return mem_mb


def scale_mem_by_attempt(rules):
    """Replace the ``mem_mb`` of every rule in ``rules`` (snakemake
    ``Rule`` objects) that has one with :func:`scaled_mem_mb`."""
    for rule in rules:
        if "mem_mb" in rule.resources:
            rule.resources["mem_mb"] = scaled_mem_mb(rule.resources["mem_mb"])
