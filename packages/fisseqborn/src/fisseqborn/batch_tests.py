"""Batch-vs-batch PERMANOVA (profiles) and ANOVA (one score) over shared variants.

Each pair of batches A and B is tested on the variants both measured: 2n observations,
experiment the factor and the variant a block (each variant is measured once in each
batch). With ``SS_T`` the total sum of squares, ``SS_V`` the variants' and ``SS_E`` the
experiments', the residual is ``SS_R = SS_T - SS_V - SS_E`` and

    F = SS_E / (SS_R / (n - 1))

A large F means the two batches differ by more than their replicate noise. PERMANOVA
computes the sums of squares from a distance matrix (any `fisseqborn.metrics` metric) and
gets its p-value by swapping A and B within random variants; the ANOVA on one score is the
same F with Euclidean distances (the paired t², on 1 and n - 1 degrees of freedom), with
its parametric p-value.
"""

import dataclasses
import json
import pathlib
from collections.abc import Sequence
from itertools import combinations, combinations_with_replacement
from os import PathLike
from typing import Any, Literal, Self

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from matplotlib.axes import Axes
from scipy import stats

from . import _data
from .batch_correlation import check_unique, pivot_by_batch
from .heatmap import Heatmap, with_cbar_label
from .metrics import Metric, MetricMatrix, Precomputed, _Normalized

Test = Literal["permanova", "anova"]
Value = Literal["f", "p", "-log10p"]

_METADATA_KEY = b"fisseqborn.batch_test"
#: Permutations evaluated per matrix product (bounds memory to chunk × n floats).
_CHUNK = 256


# ----- the statistics ----------------------------------------------------------------------


def _blocked_f(ss_e: np.ndarray, ss_t: float, ss_v: float, n: int) -> np.ndarray:
    """F of the experiment factor in the variant-blocked design, for each ``ss_e``."""
    ss_r = ss_t - ss_v - ss_e
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ss_r > 0, ss_e / (ss_r / (n - 1)), np.nan)


def _permanova_pair(
    d: np.ndarray, n: int, permutations: int, rng: np.random.Generator
) -> tuple[float, float]:
    """``(F, p)`` for a ``(2n, 2n)`` distance matrix whose rows are batch A's variants
    then batch B's, in the same variant order.

    With ``D²`` the squared distances and ``c`` the ±1 experiment contrast,
    ``SS_E = -cᵀD²c / 4n`` (``c`` sums to 0, so ``D²`` needn't be centred). Swapping A and
    B for the variants where ``s = -1`` turns ``c`` into ``[s, -s]``, so
    ``SS_E(s) = -sᵀKs / 4n`` with ``K = AA + BB - AB - BA`` (the blocks of ``D²``);
    ``SS_T`` and ``SS_V`` don't change.
    """
    n_obs = 2 * n
    d2 = np.asarray(d, dtype=float) ** 2
    aa, bb, ab = d2[:n, :n], d2[n:, n:], d2[:n, n:]
    k = aa + bb - ab - ab.T
    ss_t = d2.sum() / (2 * n_obs)
    ss_v = ss_t - np.trace(ab) / 2
    ones = np.ones(n)
    f = float(
        _blocked_f(np.array([-(ones @ k @ ones) / (2 * n_obs)]), ss_t, ss_v, n)[0]
    )
    if not np.isfinite(f):
        return f, np.nan
    exceed = 0
    for start in range(0, permutations, _CHUNK):
        size = min(_CHUNK, permutations - start)
        s = rng.choice(np.array([-1.0, 1.0]), size=(size, n))
        ss_e = -((s @ k) * s).sum(axis=1) / (2 * n_obs)
        f_perm = _blocked_f(ss_e, ss_t, ss_v, n)
        exceed += int((f_perm >= f * (1 - 1e-12)).sum())
    return f, (1 + exceed) / (1 + permutations)


def _anova_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """``(F, p)`` of the paired one-way ANOVA (the paired t²) of ``a`` against ``b``."""
    n = len(a)
    diff = np.asarray(a, float) - np.asarray(b, float)
    ss_e = n * diff.mean() ** 2 / 2
    ss_r = ((diff - diff.mean()) ** 2).sum() / 2
    if ss_r <= 0:
        return np.nan, np.nan
    f = ss_e / (ss_r / (n - 1))
    return float(f), float(stats.f.sf(f, 1, n - 1))


# ----- the result --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class BatchTestMatrix:
    """F and p of a test between every pair of batches, from `permanova` or `anova` (or
    `load`). Computing PERMANOVA can take a while: `save` the result and `load` it later,
    or pass it to `BatchPermanovaHeatmap` / `BatchAnovaHeatmap` instead of the data.

    Attributes
    ----------
    batches : list[str]
        Row and column labels of the matrices, in natural order.
    f, p : np.ndarray
        ``(n_batches, n_batches)``, symmetric; NaN on the diagonal and for pairs with
        fewer than ``min_shared`` shared variants.
    n_shared : np.ndarray
        Shared variants per pair; on the diagonal, the batch's own variant count.
    test : {"permanova", "anova"}
    params : dict
        How it was computed: ``batch``, ``label``, ``min_shared``, and ``metric``,
        ``sqrt``, ``permutations``, ``seed``, ``n_features`` (PERMANOVA) or ``score`` (ANOVA).
    """

    batches: list[str]
    f: np.ndarray
    p: np.ndarray
    n_shared: np.ndarray
    test: Test
    params: dict[str, Any] = dataclasses.field(default_factory=dict)

    # ----- computing -----------------------------------------------------------------

    @classmethod
    def permanova(
        cls,
        data: Any,
        *,
        batch: str | None = None,
        label: str | None = None,
        features: Sequence[str] | None = None,
        metric: str | Metric = "euclidean",
        sqrt: bool | None = None,
        min_shared: int = 10,
        permutations: int = 999,
        seed: int = 0,
        n_jobs: int = 1,
    ) -> Self:
        """PERMANOVA of the profiles between every pair of batches.

        ``data`` has one row per ``(batch, label)``: a `Profiles` (``batch``, ``label``
        and ``features`` then default to its ``batch_col``, ``variant_col`` and profile
        values) or a DataFrame (defaults ``"meta_experiment"``, ``"meta_aa_changes"`` and
        every numeric column not prefixed ``meta_``). Rows with a non-finite feature are
        left out, so chain ``.drop_nonfinite()`` (or read with
        ``features="intersection"``) first if batches have different features.

        For each pair, the distances between the 2n shared profiles come from
        ``metric`` (`Metric.between_rows`; a similarity ``s`` becomes ``1 - s``).
        ``sqrt`` takes their square root first. By default it does for cosine, Pearson and
        Spearman (and ``"cosine_distance"``): ``1 - s`` isn't a Euclidean distance, so the
        sums of squares can come out negative, but ``√(1 - s)`` is (the chord between the
        normalized profiles). The
        p-value is ``(1 + #{F* ≥ F}) / (1 + permutations)`` over ``permutations`` random
        swaps of A and B within variants; each pair draws from its own stream of
        ``seed``, so the result doesn't depend on ``n_jobs`` (pairs run in parallel with
        joblib).
        """
        from joblib import Parallel, delayed

        from .profiles import Profiles

        if isinstance(data, Profiles):
            batch = batch or data.batch_col
            label = label or data.variant_col
            if features is None:
                features = data.values
        batch = batch or "meta_experiment"
        label = label or "meta_aa_changes"
        df = _data.as_frame(data)
        if features is None:
            features = [
                c
                for c, dtype in df.schema.items()
                if c not in (batch, label)
                and not c.startswith("meta_")
                and dtype.is_numeric()
            ]
        features = list(features)
        if not features:
            raise ValueError("No feature columns to compare")
        _data.require_columns(df, batch, label, *features)
        check_unique(df, batch, label, None)
        resolved = Metric.get(metric)
        if sqrt is None:
            sqrt = isinstance(resolved, _Normalized) and not resolved.squared

        x_all = (
            df.select(pl.col(features).cast(pl.Float64)).fill_null(np.nan).to_numpy()
        )
        finite = np.isfinite(x_all).all(axis=1)
        batch_of = df.get_column(batch).cast(pl.String).to_numpy()
        label_of = df.get_column(label).to_list()
        batches = sorted(set(batch_of[finite]), key=_data._natural_key)
        rows: dict[str, dict[Any, int]] = {b: {} for b in batches}
        for i in np.flatnonzero(finite):
            rows[batch_of[i]][label_of[i]] = i

        n_b = len(batches)
        n_shared = np.zeros((n_b, n_b), dtype=int)
        jobs = []
        for i, j in combinations_with_replacement(range(n_b), 2):
            a, b = rows[batches[i]], rows[batches[j]]
            shared = [lab for lab in a if lab in b]
            n_shared[i, j] = n_shared[j, i] = len(shared)
            if i != j and len(shared) >= min_shared:
                idx = [a[lab] for lab in shared] + [b[lab] for lab in shared]
                jobs.append((i, j, x_all[idx]))
        streams = np.random.SeedSequence(seed).spawn(len(jobs))

        def run(x: np.ndarray, stream: np.random.SeedSequence) -> tuple[float, float]:
            d = resolved.between_rows(x).to_distance().values
            if sqrt:
                d = np.sqrt(np.clip(d, 0, None))
            if not np.isfinite(d).all():
                return np.nan, np.nan
            return _permanova_pair(
                d, len(x) // 2, permutations, np.random.default_rng(stream)
            )

        results = Parallel(n_jobs=n_jobs)(
            delayed(run)(x, s) for (_, _, x), s in zip(jobs, streams, strict=True)
        )
        f, p = np.full((n_b, n_b), np.nan), np.full((n_b, n_b), np.nan)
        for (i, j, _), (fv, pv) in zip(jobs, results, strict=True):
            f[i, j] = f[j, i] = fv
            p[i, j] = p[j, i] = pv
        params = {
            "batch": batch,
            "label": label,
            "metric": resolved.name,
            "sqrt": sqrt,
            "n_features": len(features),
            "min_shared": min_shared,
            "permutations": permutations,
            "seed": seed,
        }
        return cls(batches, f, p, n_shared, "permanova", params)

    @classmethod
    def anova(
        cls,
        data: Any,
        *,
        batch: str,
        label: str,
        score: str,
        min_shared: int = 10,
        aggregate: str | None = None,
    ) -> Self:
        """Paired ANOVA of ``score`` between every pair of batches, on long-form data
        (one row per ``(batch, label)``, as `BatchCorrelationHeatmap`). Each pair uses the
        labels where both batches' scores are finite; F is the paired t², and p comes
        from the F distribution on 1 and n - 1 degrees of freedom. Duplicate
        ``(batch, label)`` rows are an error unless ``aggregate`` is given."""
        df = _data.as_frame(data)
        _data.require_columns(df, batch, label, score)
        check_unique(df, batch, label, aggregate)
        wide = pivot_by_batch(df, batch, label, score, aggregate).drop(label)
        batches = sorted(wide.columns, key=_data._natural_key)
        x = wide.select(batches).fill_null(np.nan).to_numpy()
        ok = np.isfinite(x)
        n_shared = ok.T.astype(int) @ ok.astype(int)
        n_b = len(batches)
        f, p = np.full((n_b, n_b), np.nan), np.full((n_b, n_b), np.nan)
        for i, j in combinations(range(n_b), 2):
            if n_shared[i, j] >= min_shared:
                both = ok[:, i] & ok[:, j]
                f[i, j], p[i, j] = _anova_pair(x[both, i], x[both, j])
                f[j, i], p[j, i] = f[i, j], p[i, j]
        params = {
            "batch": batch,
            "label": label,
            "score": score,
            "min_shared": min_shared,
        }
        return cls(batches, f, p, n_shared, "anova", params)

    # ----- views ---------------------------------------------------------------------

    def matrix(self, value: Value = "f") -> MetricMatrix:
        """``value`` (``"f"``, ``"p"`` or ``"-log10p"``) as a symmetric `MetricMatrix`
        over the batches, with ``n_shared``; draw it with `Heatmap.from_matrix`."""
        stat = "pseudo-F" if self.test == "permanova" else "F"
        if value == "f":
            values = self.f
            metric = Precomputed(kind="distance", name="f", label=stat, symbol="F")
        elif value == "p":
            values = self.p
            metric = Precomputed(kind="similarity", name="p", label="p", symbol="p")
        elif value == "-log10p":
            with np.errstate(divide="ignore"):
                values = -np.log10(self.p)
            metric = Precomputed(
                kind="distance", name="neglog10_p", label="−log₁₀ p", symbol="−log₁₀ p"
            )
        else:
            raise ValueError(f"value must be 'f', 'p' or '-log10p', got {value!r}")
        return MetricMatrix(
            values,
            metric,
            list(self.batches),
            list(self.batches),
            n_shared=self.n_shared,
            key=self.params.get("batch"),
            symmetric=True,
        )

    def pairs(self) -> pl.DataFrame:
        """One row per unordered batch pair (including each batch with itself):
        ``batch_a, batch_b, f, p, n_shared``."""
        cells = list(combinations_with_replacement(range(len(self.batches)), 2))
        return pl.DataFrame(
            {
                "batch_a": [self.batches[i] for i, _ in cells],
                "batch_b": [self.batches[j] for _, j in cells],
                "f": [float(self.f[i, j]) for i, j in cells],
                "p": [float(self.p[i, j]) for i, j in cells],
                "n_shared": [int(self.n_shared[i, j]) for i, j in cells],
            },
            schema={
                "batch_a": pl.String,
                "batch_b": pl.String,
                "f": pl.Float64,
                "p": pl.Float64,
                "n_shared": pl.Int64,
            },
        )

    # ----- saving --------------------------------------------------------------------

    def save(self, path: str | PathLike) -> Self:
        """Write `pairs` to parquet, with the test and its parameters in the file's
        metadata so `load` restores the whole result. Returns the result."""
        table = self.pairs().to_arrow()
        meta = {"test": self.test, "batches": self.batches, "params": self.params}
        table = table.replace_schema_metadata(
            {**(table.schema.metadata or {}), _METADATA_KEY: json.dumps(meta)}
        )
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        return self

    @classmethod
    def load(cls, path: str | PathLike) -> Self:
        """Read a result written by `save`."""
        table = pq.read_table(path)
        raw = (table.schema.metadata or {}).get(_METADATA_KEY)
        if raw is None:
            raise ValueError(f"{path} wasn't written by BatchTestMatrix.save")
        meta = json.loads(raw)
        batches: list[str] = meta["batches"]
        pos = {b: i for i, b in enumerate(batches)}
        n_b = len(batches)
        f, p = np.full((n_b, n_b), np.nan), np.full((n_b, n_b), np.nan)
        n_shared = np.zeros((n_b, n_b), dtype=int)
        df = pl.from_arrow(table)
        assert isinstance(df, pl.DataFrame)
        for a, b, fv, pv, ns in df.iter_rows():
            i, j = pos[a], pos[b]
            f[i, j] = f[j, i] = np.nan if fv is None else fv
            p[i, j] = p[j, i] = np.nan if pv is None else pv
            n_shared[i, j] = n_shared[j, i] = ns
        return cls(batches, f, p, n_shared, meta["test"], meta["params"])


# ----- the plots ---------------------------------------------------------------------------

_FMT = {"f": ".2f", "p": ".3f", "-log10p": ".1f"}


class _BatchTestHeatmap(Heatmap):
    """Heatmap of one `BatchTestMatrix` value, styled as `BatchCorrelationHeatmap`."""

    _test: Test

    def _init_result(
        self,
        result: BatchTestMatrix,
        value: Value,
        order: Sequence[Any] | None,
        figsize: tuple[float, float] | None,
        kw: dict[str, Any],
    ) -> None:
        if result.test != self._test:
            raise ValueError(
                f"{type(self).__name__} draws a {self._test} result, got {result.test}"
            )
        #: The `BatchTestMatrix` drawn; `BatchTestMatrix.save` it to reuse it.
        self.result = result
        self.value = value
        #: The drawn value as a `MetricMatrix`.
        self.metric_matrix = result.matrix(value)
        values = self.metric_matrix.values
        finite = values[np.isfinite(values)]
        top = float(finite.max()) if finite.size else 1.0
        n = len(result.batches)
        if figsize is None:
            figsize = (max(6.0, 0.65 * n + 1.0), max(5.0, 0.6 * n + 0.6))
        kw = with_cbar_label(
            {
                "cmap": "Blues" if value == "p" else "Blues_r",
                "vmin": 0,
                "vmax": 1 if value == "p" else top,
                "annot": True,
                "fmt": _FMT[value],
                "nan_color": "black",
                "linewidths": 0.5,
                "linecolor": "black",
                "row_order": order,
                "col_order": order,
                **kw,
            },
            self.metric_matrix.metric.label,
        )
        super().__init__(
            self.metric_matrix.to_frame("batch"),
            index="batch",
            columns=list(result.batches),
            figsize=figsize,
            **kw,
        )

    def pairs(self) -> pl.DataFrame:
        """The numbers behind the plot: `BatchTestMatrix.pairs`."""
        return self.result.pairs()

    def _draw(self, ax: Axes) -> None:
        super()._draw(ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set(xlabel="", ylabel="")


class BatchPermanovaHeatmap(_BatchTestHeatmap):
    """Heatmap of a PERMANOVA between every pair of batches, over the variants they share
    (see `BatchTestMatrix.permanova` and the module docs for the design).

    Cell ``(a, b)`` tests whether batches a and b give different profiles to the same
    variants, beyond the replicate noise. The diagonal and pairs with fewer than
    ``min_shared`` shared variants are missing, drawn in ``nan_color`` (black).

    Parameters
    ----------
    data : Profiles | pl.DataFrame | BatchTestMatrix
        One row per ``(batch, label)`` with the profile features, or an earlier result
        (e.g. `BatchTestMatrix.load`), which is drawn without computing anything.
    batch, label, features, metric, sqrt, min_shared, permutations, seed, n_jobs
        Passed to `BatchTestMatrix.permanova`; ignored for a result.
    value : {"f", "p", "-log10p"}
        What the cells show: the pseudo-F (``Blues_r`` from 0, so batches that agree are
        dark), the permutation p-value (``Blues`` from 0 to 1) or its ``-log10``.
    order : Sequence | None
        Batch order for both rows and columns. Default: natural sort.
    **kw
        Passed to `Heatmap`. Defaults: ``annot=True, nan_color="black", linewidths=0.5,
        linecolor="black"`` and a colorbar labeled with the value.
    """

    _test = "permanova"

    def __init__(
        self,
        data: Any,
        *,
        batch: str | None = None,
        label: str | None = None,
        features: Sequence[str] | None = None,
        metric: str | Metric = "euclidean",
        sqrt: bool | None = None,
        min_shared: int = 10,
        permutations: int = 999,
        seed: int = 0,
        n_jobs: int = 1,
        value: Value = "f",
        order: Sequence[Any] | None = None,
        figsize: tuple[float, float] | None = None,
        **kw: Any,
    ) -> None:
        result = (
            data
            if isinstance(data, BatchTestMatrix)
            else BatchTestMatrix.permanova(
                data,
                batch=batch,
                label=label,
                features=features,
                metric=metric,
                sqrt=sqrt,
                min_shared=min_shared,
                permutations=permutations,
                seed=seed,
                n_jobs=n_jobs,
            )
        )
        self._init_result(result, value, order, figsize, kw)


class BatchAnovaHeatmap(_BatchTestHeatmap):
    """Heatmap of a paired ANOVA of one score between every pair of batches, over the
    labels they share (see `BatchTestMatrix.anova`): cell ``(a, b)`` tests whether batch
    a's ``score`` differs from batch b's for the same labels (F is the paired t²).

    Parameters
    ----------
    data : pl.DataFrame | BatchTestMatrix
        Long form, one row per ``(batch, label)`` (as `BatchCorrelationHeatmap`), or an
        earlier result, which is drawn as it is.
    batch, label, score, min_shared, aggregate
        Passed to `BatchTestMatrix.anova`; required unless ``data`` is a result.
    value, order, **kw
        As `BatchPermanovaHeatmap`; ``value="p"`` is the F-test p-value.
    """

    _test = "anova"

    def __init__(
        self,
        data: Any,
        *,
        batch: str | None = None,
        label: str | None = None,
        score: str | None = None,
        min_shared: int = 10,
        aggregate: str | None = None,
        value: Value = "f",
        order: Sequence[Any] | None = None,
        figsize: tuple[float, float] | None = None,
        **kw: Any,
    ) -> None:
        if isinstance(data, BatchTestMatrix):
            result = data
        else:
            if batch is None or label is None or score is None:
                raise ValueError(
                    "batch, label and score are needed to compute the ANOVA"
                )
            result = BatchTestMatrix.anova(
                data,
                batch=batch,
                label=label,
                score=score,
                min_shared=min_shared,
                aggregate=aggregate,
            )
        self._init_result(result, value, order, figsize, kw)
