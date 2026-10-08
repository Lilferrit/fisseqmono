"""Tests for snakemake/fisseq_resources.py: every rule's mem_mb grows with
snakemake's `attempt` (doubling each time), so a starcall job retried after an OOM kill asks
for more memory.

The module runs inside the ops env's snakemake interpreter, not this
package, so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).resolve().parents[2] / "snakemake" / "fisseq_resources.py"
_spec = importlib.util.spec_from_file_location("fisseq_resources", _PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

_INPUT = SimpleNamespace(size_mb=100)


def _eval(base, attempt, threads=4):
    return mod.scaled_mem_mb(base)({}, _INPUT, attempt, threads, "some_rule")


@pytest.mark.parametrize("attempt, expected", [(1, 16000), (2, 32000), (3, 64000), (4, 128000)])
def test_number_doubles_each_attempt(attempt, expected):
    assert _eval(16000, attempt) == expected


def test_starcall_style_lambda_gets_only_the_arguments_it_declares():
    # starcall's own form: wildcards positional, then input/attempt by name.
    base = lambda wildcards, input, attempt: input.size_mb * 2 + 5000  # noqa: E731
    assert _eval(base, 1) == 5200
    assert _eval(base, 2) == 10400
    assert _eval(base, 3) == 20800


def test_lambda_with_threads_and_float_result_rounds():
    base = lambda wildcards, threads: threads * 1000.4  # noqa: E731
    assert _eval(base, 2, threads=2) == 4002


def test_numeric_string_scales_and_other_strings_pass_through():
    assert _eval("1000", 3) == 4000
    assert _eval("16G", 3) == "16G"


def test_none_stays_none():
    assert _eval(lambda wildcards: None, 2) is None


def test_scale_mem_by_attempt_wraps_only_rules_with_mem_mb():
    with_mem = SimpleNamespace(resources={"mem_mb": 500, "_cores": 1})
    without = SimpleNamespace(resources={"_cores": 1})
    mod.scale_mem_by_attempt([with_mem, without])
    assert callable(with_mem.resources["mem_mb"])
    assert with_mem.resources["mem_mb"]({}, _INPUT, 2, 1, "r") == 1000
    assert without.resources == {"_cores": 1}
