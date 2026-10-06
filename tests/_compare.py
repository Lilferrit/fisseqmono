"""Compare a run's published parquets against saved reference outputs.

Two outputs count as equal when their contents are, up to the things a rerun of unchanged code
is free to change:

- Row order. Several stages write the result of an unordered ``group_by``, so rows are sorted
  on their key (non-float) columns first, then on the float columns.
- Floating-point summation order: floats, including floats inside lists, compare with a small
  relative tolerance.
- Element order of ``*_counts`` lists (per-variant barcode/batch frequencies). The data
  pipeline builds them with ``value_counts()``, whose order Polars does not define; their
  elements compare as a set.
- The run's own directory, which appears in path columns (``meta_origin_file``,
  ``shard_tar``). :func:`load` replaces it with ``<RUN>``, both when references are captured
  and when a run is compared with them.

Column sets and dtypes must match exactly.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

REL_TOL = 1e-6
ABS_TOL = 1e-9
RUN_PLACEHOLDER = "<RUN>"


def _sortable(dtype: pl.DataType) -> bool:
    return not isinstance(dtype, (pl.List, pl.Array, pl.Struct, pl.Object))


def load(path: Path, run_root: Path | None = None) -> pl.DataFrame:
    """Read ``path``, replacing ``run_root`` in string columns with ``<RUN>``."""
    df = pl.read_parquet(path)
    if run_root is None:
        return df
    roots = sorted({str(run_root), os.path.realpath(run_root)}, key=len, reverse=True)
    exprs = []
    for col, dtype in df.schema.items():
        if dtype == pl.String:
            expr = pl.col(col)
            for root in roots:
                expr = expr.str.replace_all(root, RUN_PLACEHOLDER, literal=True)
            exprs.append(expr)
    return df.with_columns(exprs) if exprs else df


def canonical(df: pl.DataFrame) -> pl.DataFrame:
    """``df`` with columns in name order, ``*_counts`` lists sorted and rows sorted."""
    df = df.with_columns(
        pl.col(c).list.sort()
        for c, t in df.schema.items()
        if c.endswith("_counts") and isinstance(t, pl.List)
    )
    df = df.select(sorted(df.columns))
    keys = [c for c, t in df.schema.items() if _sortable(t) and not t.is_float()]
    keys += [c for c, t in df.schema.items() if t.is_float()]
    return df.sort(keys, nulls_last=True) if keys else df


def assert_frames_equal(
    actual: pl.DataFrame, expected: pl.DataFrame, name: str
) -> None:
    assert sorted(actual.columns) == sorted(expected.columns), (
        f"{name}: columns differ"
        f"\n  only in actual: {sorted(set(actual.columns) - set(expected.columns))}"
        f"\n  only in reference: {sorted(set(expected.columns) - set(actual.columns))}"
    )
    assert_frame_equal(
        canonical(actual),
        canonical(expected),
        check_exact=False,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
    )


def compare_trees(
    actual: dict[str, Path],
    reference_dir: Path,
    run_root: Path,
    prefix: str = "",
) -> list[str]:
    """Problems found comparing ``actual`` (relative path -> file) with ``reference_dir``.

    Only reference files under ``prefix`` are considered.
    """
    expected = {
        rel: p
        for p in reference_dir.rglob("*.parquet")
        if (rel := p.relative_to(reference_dir).as_posix()).startswith(prefix)
    }
    actual = {k: v for k, v in actual.items() if k.startswith(prefix)}
    problems = [f"missing from run: {k}" for k in sorted(set(expected) - set(actual))]
    problems += [f"not in reference: {k}" for k in sorted(set(actual) - set(expected))]
    for key in sorted(set(actual) & set(expected)):
        try:
            assert_frames_equal(load(actual[key], run_root), load(expected[key]), key)
        except AssertionError as err:
            problems.append(f"{key}: {err}")
    return problems
