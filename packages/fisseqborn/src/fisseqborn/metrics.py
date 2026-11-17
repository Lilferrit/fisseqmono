"""Distances and similarities between vectors, shared by every fisseqborn module that
compares profiles, features or batches.

A `Metric` says how to compare two vectors; a `MetricMatrix` holds the comparisons it
computed (or ones computed elsewhere), so they can be reused:

>>> d = profiles.distances("pearson")           # every pair of variants, once
>>> d.save("vis/pearson.parquet")
>>> profiles.umap(metric=d).cluster(metric=d)   # both reuse d
>>> fb.ClusterMap(profiles.df, metric=d)

Names accepted wherever a metric is: ``"euclidean"``, ``"cityblock"`` (Manhattan),
``"cosine_distance"``, ``"cosine"``,
``"pearson"``, ``"spearman"``, and ``"cosine_squared"``, ``"pearson_squared"``,
``"spearman_squared"``. Where a distance is needed (clustering, UMAP, nearest neighbors) a
similarity ``s`` becomes ``1 - s``, a squared one ``1 - s²``.
"""

import dataclasses
import json
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import combinations_with_replacement, product
from os import PathLike
from typing import Any, ClassVar, Literal, Self

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from scipy import stats
from scipy.spatial.distance import cdist, squareform

Kind = Literal["similarity", "distance"]

_METADATA_KEY = b"fisseqborn.metric"


class Metric(ABC):
    """How to compare two vectors. Get one by name with `Metric.get`, or construct a
    subclass: `Euclidean`, `CosineDistance`, `Cosine`, `Pearson`, `Spearman`."""

    #: ``"similarity"`` (higher is more alike) or ``"distance"`` (lower is).
    kind: ClassVar[Kind]
    #: Whether the value is a squared similarity.
    squared: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """The name `Metric.get` resolves, e.g. ``"spearman_squared"``."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Axis / colorbar label, e.g. ``"Spearman ρ²"``."""

    @property
    @abstractmethod
    def symbol(self) -> str:
        """Short symbol for annotations, e.g. ``"ρ²"``."""

    @property
    def default_scale(self) -> dict[str, Any]:
        """Heatmap color scale: ``cmap``, ``vmin``, ``vmax`` and ``center``."""
        if self.kind == "distance":
            return {"cmap": "Blues_r", "vmin": 0, "vmax": None, "center": None}
        if self.squared:
            return {"cmap": "Blues", "vmin": 0, "vmax": 1, "center": None}
        return {"cmap": "vlag", "vmin": -1, "vmax": 1, "center": 0}

    @property
    def distance_bound(self) -> float | None:
        """Largest value `to_distance` can give, or ``None`` if unbounded."""
        return None

    @staticmethod
    def get(metric: "str | Metric") -> "Metric":
        """The metric named ``metric`` (a `Metric` is returned as is)."""
        if isinstance(metric, Metric):
            return metric
        if isinstance(metric, MetricMatrix):
            raise TypeError("Expected a metric name or Metric here, not a MetricMatrix")
        try:
            return _REGISTRY[metric]
        except (KeyError, TypeError):
            raise ValueError(
                f"Unknown metric {metric!r}; expected one of {names()}"
            ) from None

    # ----- the formula ---------------------------------------------------------------

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        """Each row of ``x`` made ready for the base distance (ranked, centred, ...)."""
        return x

    @abstractmethod
    def _from_prepared(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Values between every row of ``a`` and every row of ``b`` (both prepared)."""

    def to_distance(self, values: np.ndarray) -> np.ndarray:
        """``values`` of this metric as distances: unchanged for a distance, ``1 - s`` for
        a similarity (``1 - s²`` for a squared one, as ``s`` is then already squared)."""
        values = np.asarray(values, dtype=float)
        return values if self.kind == "distance" else 1.0 - values

    def value(self, x: np.ndarray, y: np.ndarray) -> float:
        """The metric between two finite vectors; NaN where it is undefined (a zero
        vector for cosine, zero variance for Pearson / Spearman)."""
        x = np.asarray(x, dtype=float)[None, :]
        y = np.asarray(y, dtype=float)[None, :]
        return float(self._from_prepared(self._prepare(x), self._prepare(y))[0, 0])

    # ----- comparing rows ------------------------------------------------------------

    def between_rows(
        self,
        data: pl.DataFrame | np.ndarray,
        features: Sequence[str] | None = None,
        *,
        key: str | None = None,
    ) -> "MetricMatrix":
        """The metric between every pair of rows of ``data`` (which must be finite).

        From a DataFrame, ``features`` defaults to every numeric column that isn't
        ``key`` or prefixed ``meta_``, and the rows are labelled by ``key`` (so consumers
        can line the matrix up with their own rows).
        """
        x, labels = _rows(data, features, key)
        if not np.isfinite(x).all():
            raise ValueError(
                "between_rows needs finite values; drop or fill null/NaN/inf first"
            )
        p = self._prepare(x)
        values = self._from_prepared(p, p)
        if self.kind == "similarity":
            ok = np.isfinite(np.diag(values))
            values[np.diag_indices_from(values)] = np.where(ok, 1.0, np.nan)
        else:
            np.fill_diagonal(values, 0.0)
        return MetricMatrix(values, self, labels, labels, key=key, symmetric=True)

    def native(self, x: np.ndarray) -> tuple[np.ndarray, str] | None:
        """``(rows, name)`` such that the ``name`` distance (``"euclidean"``,
        ``"manhattan"`` or ``"cosine"``, which scikit-learn and umap-learn both know)
        between ``rows`` is this metric's distance; ``None`` when there is no such form (squared metrics),
        so the full distance matrix is needed."""
        return None

    # ----- comparing columns ---------------------------------------------------------

    def pairwise(
        self,
        data: pl.DataFrame | np.ndarray,
        columns: Sequence[str] | None = None,
        *,
        min_shared: int = 10,
    ) -> "MetricMatrix":
        """The metric between every pair of columns of ``data`` (default: every numeric
        column), each pair over the rows where both are finite.

        Spearman ranks within each pair's shared rows, matching `scipy.stats.spearmanr`
        on that pair. Pairs with fewer than ``min_shared`` shared rows are NaN; the
        counts are the matrix's ``n_shared``.
        """
        mat, labels = _columns(data, columns)
        values, n_shared = self._column_values(mat, mat, min_shared, symmetric=True)
        diag = np.diag(values)
        if self.kind == "similarity":  # a column with itself, exactly
            values[np.diag_indices_from(values)] = np.where(
                np.isfinite(diag), 1.0, diag
            )
        else:
            values[np.diag_indices_from(values)] = np.where(
                np.isfinite(diag), 0.0, diag
            )
        return MetricMatrix(
            values, self, labels, labels, n_shared=n_shared, symmetric=True
        )

    def cross(
        self,
        a: pl.DataFrame | np.ndarray,
        b: pl.DataFrame | np.ndarray,
        *,
        min_shared: int = 10,
    ) -> "MetricMatrix":
        """The metric between each column of ``a`` (the matrix rows) and each column of
        ``b`` (its columns), on the same rows; as `pairwise` otherwise, so
        ``cross(m, m)`` has the values of ``pairwise(m)``."""
        mat_a, labels_a = _columns(a, None)
        mat_b, labels_b = _columns(b, None)
        if mat_a.shape[0] != mat_b.shape[0]:
            raise ValueError(
                f"a and b need the same rows, got {mat_a.shape[0]} and {mat_b.shape[0]}"
            )
        values, n_shared = self._column_values(
            mat_a, mat_b, min_shared, symmetric=False
        )
        return MetricMatrix(
            values, self, labels_a, labels_b, n_shared=n_shared, symmetric=False
        )

    def _column_values(
        self, a: np.ndarray, b: np.ndarray, min_shared: int, *, symmetric: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        finite_a, finite_b = np.isfinite(a), np.isfinite(b)
        n_shared = finite_a.T.astype(int) @ finite_b.astype(int)
        values = np.full((a.shape[1], b.shape[1]), np.nan)
        if finite_a.all() and finite_b.all():
            if a.shape[0] >= min_shared:
                values = self._from_prepared(self._prepare(a.T), self._prepare(b.T))
            return values, n_shared
        for i in range(a.shape[1]):
            for j in range(i if symmetric else 0, b.shape[1]):
                if n_shared[i, j] >= min_shared:
                    ok = finite_a[:, i] & finite_b[:, j]
                    values[i, j] = self.value(a[ok, i], b[ok, j])
                    if symmetric:
                        values[j, i] = values[i, j]
        return values, n_shared


# ----- the metrics ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Euclidean(Metric):
    """Euclidean distance."""

    kind: ClassVar[Kind] = "distance"
    name = "euclidean"  # type: ignore[assignment]
    label = "Euclidean distance"  # type: ignore[assignment]
    symbol = "d"  # type: ignore[assignment]

    def _from_prepared(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return cdist(a, b)

    def value(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(x, float) - np.asarray(y, float)))

    def native(self, x: np.ndarray) -> tuple[np.ndarray, str]:
        return x, "euclidean"


@dataclasses.dataclass(frozen=True)
class Cityblock(Metric):
    """Manhattan (city block, L1) distance."""

    kind: ClassVar[Kind] = "distance"
    name = "cityblock"  # type: ignore[assignment]
    label = "Manhattan distance"  # type: ignore[assignment]
    symbol = "d₁"  # type: ignore[assignment]

    def _from_prepared(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return cdist(a, b, "cityblock")

    def native(self, x: np.ndarray) -> tuple[np.ndarray, str]:
        return x, "manhattan"


class _Normalized(Metric):
    """Metrics computed as the cosine of prepared rows (cosine, Pearson, Spearman)."""

    def _center(self, x: np.ndarray) -> np.ndarray:
        """Rows before normalizing: as they are (cosine), centred (Pearson), ranked and
        centred (Spearman)."""
        return x

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        c = self._center(np.asarray(x, dtype=float))
        norm = np.sqrt((c**2).sum(axis=1, keepdims=True))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(norm > 0, c / norm, np.nan)

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.clip(a @ b.T, -1.0, 1.0)

    def native(self, x: np.ndarray) -> tuple[np.ndarray, str] | None:
        return None if self.squared else (self._center(x), "cosine")


@dataclasses.dataclass(frozen=True)
class _Similarity(_Normalized):
    squared: bool = False
    kind: ClassVar[Kind] = "similarity"
    _base_name: ClassVar[str]
    _base_label: ClassVar[str]
    _base_symbol: ClassVar[str]

    @property
    def name(self) -> str:
        return self._base_name + ("_squared" if self.squared else "")

    @property
    def label(self) -> str:
        return self._base_label + ("²" if self.squared else "")

    @property
    def symbol(self) -> str:
        return self._base_symbol + ("²" if self.squared else "")

    @property
    def distance_bound(self) -> float:
        return 1.0 if self.squared else 2.0

    def _from_prepared(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        s = self._similarity(a, b)
        return s**2 if self.squared else s


@dataclasses.dataclass(frozen=True)
class Cosine(_Similarity):
    """Cosine similarity of the (uncentred) vectors; ``squared=True`` for cos²."""

    _base_name: ClassVar[str] = "cosine"
    _base_label: ClassVar[str] = "Cosine similarity"
    _base_symbol: ClassVar[str] = "cos"


@dataclasses.dataclass(frozen=True)
class Pearson(_Similarity):
    """Pearson correlation; ``squared=True`` for r²."""

    _base_name: ClassVar[str] = "pearson"
    _base_label: ClassVar[str] = "Pearson r"
    _base_symbol: ClassVar[str] = "r"

    def _center(self, x: np.ndarray) -> np.ndarray:
        return x - x.mean(axis=1, keepdims=True)


@dataclasses.dataclass(frozen=True)
class Spearman(_Similarity):
    """Spearman rank correlation (average ranks for ties); ``squared=True`` for ρ²."""

    _base_name: ClassVar[str] = "spearman"
    _base_label: ClassVar[str] = "Spearman ρ"
    _base_symbol: ClassVar[str] = "ρ"

    def _center(self, x: np.ndarray) -> np.ndarray:
        r = stats.rankdata(x, axis=1)
        return r - r.mean(axis=1, keepdims=True)


@dataclasses.dataclass(frozen=True)
class CosineDistance(_Normalized):
    """Cosine distance, ``1 - cosine similarity`` (0 to 2)."""

    kind: ClassVar[Kind] = "distance"
    name = "cosine_distance"  # type: ignore[assignment]
    label = "Cosine distance"  # type: ignore[assignment]
    symbol = "d_cos"  # type: ignore[assignment]

    @property
    def distance_bound(self) -> float:
        return 2.0

    def _from_prepared(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return 1.0 - self._similarity(a, b)


@dataclasses.dataclass(frozen=True)
class Precomputed(Metric):
    """The metric of a `MetricMatrix` computed outside fisseqborn: only its ``kind``
    and labels are known, so it can't compute anything itself. A precomputed similarity
    is taken to be at most 1, so ``1 - s`` is its distance."""

    kind: Kind = "distance"  # type: ignore[misc]
    name: str = "precomputed"  # type: ignore[assignment]
    label: str = "Distance"  # type: ignore[assignment]
    symbol: str = "d"  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.kind not in ("similarity", "distance"):
            raise ValueError(
                f"kind must be 'similarity' or 'distance', got {self.kind!r}"
            )

    @property
    def default_scale(self) -> dict[str, Any]:
        if self.kind == "distance":
            return super().default_scale
        return {"cmap": "Blues", "vmin": None, "vmax": None, "center": None}

    def _from_prepared(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        raise TypeError(
            f"{self.name!r} is precomputed; it has no formula to compute new values"
        )


_REGISTRY: dict[str, Metric] = {
    m.name: m
    for m in (
        Euclidean(),
        Cityblock(),
        CosineDistance(),
        Cosine(),
        Pearson(),
        Spearman(),
        Cosine(squared=True),
        Pearson(squared=True),
        Spearman(squared=True),
    )
}


def names() -> list[str]:
    """Every metric name `Metric.get` accepts."""
    return list(_REGISTRY)


# ----- results ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class MetricMatrix:
    """Values of a metric between labelled items (rows or columns of some data), computed
    by a `Metric` or loaded from elsewhere (`from_array`, `from_frame`, `load`).

    Pass it as ``metric=`` to `Profiles.umap`, `Profiles.cluster` and `ClusterMap` to
    reuse it, or draw it with `Heatmap.from_matrix`.

    Attributes
    ----------
    values : np.ndarray
        ``(n_rows, n_cols)``.
    metric : Metric
        What the values are (`Precomputed` for values from elsewhere).
    row_labels, col_labels : list | None
        What each row / column is, e.g. variants or batch names.
    n_shared : np.ndarray | None
        Per cell, how many finite observations it was computed from (`Metric.pairwise`).
    key : str | None
        The data column the labels are values of (e.g. ``"meta_aa_changes"``): consumers
        use it to line the matrix up with their own rows.
    symmetric : bool
        Rows and columns are the same items (every cell ``(i, j)`` equals ``(j, i)``).
    """

    values: np.ndarray
    metric: Metric
    row_labels: list[Any] | None = None
    col_labels: list[Any] | None = None
    n_shared: np.ndarray | None = None
    key: str | None = None
    symmetric: bool = False

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2:
            raise ValueError(f"values must be 2-D, got shape {values.shape}")
        object.__setattr__(self, "values", values)
        for labels, n, axis in (
            (self.row_labels, values.shape[0], "row"),
            (self.col_labels, values.shape[1], "column"),
        ):
            if labels is not None and len(labels) != n:
                raise ValueError(f"{len(labels)} {axis} labels for {n} {axis}s")
        if self.symmetric and values.shape[0] != values.shape[1]:
            raise ValueError(f"A symmetric matrix must be square, got {values.shape}")

    # ----- constructors --------------------------------------------------------------

    @classmethod
    def from_array(
        cls,
        values: np.ndarray,
        labels: Sequence[Any] | None = None,
        *,
        col_labels: Sequence[Any] | None = None,
        kind: Kind = "distance",
        label: str | None = None,
        key: str | None = None,
    ) -> Self:
        """A matrix computed elsewhere. Without ``col_labels`` it compares ``labels`` with
        themselves, so it must be square and symmetric; with them it is rectangular,
        ``labels`` on the rows."""
        values = np.asarray(values, dtype=float)
        metric = Precomputed(
            kind=kind,
            label=label or kind.capitalize(),
            symbol="d" if kind == "distance" else "s",
        )
        rows = None if labels is None else list(labels)
        if col_labels is not None:
            return cls(values, metric, rows, list(col_labels), key=key)
        if values.ndim != 2 or values.shape[0] != values.shape[1]:
            raise ValueError(
                f"Without col_labels the matrix must be square, got shape {values.shape}"
            )
        if not np.allclose(values, values.T, equal_nan=True):
            raise ValueError(
                "The matrix isn't symmetric; pass col_labels for a non-symmetric one"
            )
        return cls(values, metric, rows, rows, key=key, symmetric=True)

    @classmethod
    def from_frame(
        cls,
        df: pl.DataFrame,
        index: str,
        *,
        kind: Kind = "distance",
        label: str | None = None,
        key: str | None = None,
    ) -> Self:
        """A matrix from a wide frame: labels in ``index``, one column per label. It is
        symmetric when the column names are the ``index`` values (as strings)."""
        cols = [c for c in df.columns if c != index]
        values = df.select(pl.col(cols).cast(pl.Float64)).fill_null(np.nan).to_numpy()
        rows = df.get_column(index).to_list()
        if cols == [str(r) for r in rows]:
            return cls.from_array(values, rows, kind=kind, label=label, key=key)
        return cls.from_array(
            values, rows, col_labels=cols, kind=kind, label=label, key=key
        )

    @classmethod
    def load(cls, path: str | PathLike) -> Self:
        """Read a matrix written by `save`."""
        table = pq.read_table(path)
        meta = json.loads((table.schema.metadata or {})[_METADATA_KEY])
        df = pl.from_arrow(table)
        assert isinstance(df, pl.DataFrame)
        index = meta["index"]
        if meta["name"] in _REGISTRY:
            metric: Metric = _REGISTRY[meta["name"]]
        else:
            metric = Precomputed(
                kind=meta["kind"],
                name=meta["name"],
                label=meta["label"],
                symbol=meta["symbol"],
            )
        rows = df.get_column(index).to_list()
        cols = [c for c in df.columns if c != index]
        values = df.select(cols).fill_null(np.nan).to_numpy()
        return cls(
            values,
            metric,
            rows,
            rows if meta["symmetric"] else cols,
            key=meta["key"],
            symmetric=meta["symmetric"],
        )

    # ----- views ---------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape  # type: ignore[return-value]

    def _replace(self, **changes: Any) -> Self:
        return dataclasses.replace(self, **changes)

    def to_distance(self) -> Self:
        """The values as distances (see `Metric.to_distance`); itself for a distance."""
        if self.metric.kind == "distance":
            return self
        metric = Precomputed(
            kind="distance",
            name=f"{self.metric.name}_distance",
            label=f"1 − {self.metric.label}",
            symbol="d",
        )
        return self._replace(values=self.metric.to_distance(self.values), metric=metric)

    def squared(self) -> Self:
        """The squared similarity (e.g. Pearson r → r²)."""
        if not isinstance(self.metric, _Similarity) or self.metric.squared:
            raise ValueError(f"Can't square a {self.metric.label} matrix")
        return self._replace(
            values=self.values**2,
            metric=dataclasses.replace(self.metric, squared=True),
        )

    def condensed(self) -> np.ndarray:
        """The distances as a condensed vector, for `scipy.cluster.hierarchy.linkage`."""
        if not self.symmetric:
            raise ValueError("Only a symmetric matrix has a condensed form")
        d = self.to_distance().values.copy()
        np.fill_diagonal(d, 0.0)
        if not np.isfinite(d).all():
            raise ValueError(
                "The distances contain NaN (e.g. a zero-variance profile under "
                "Pearson / Spearman, or a zero vector under cosine); drop those rows"
            )
        return squareform(d, checks=False)

    def align(self, labels: Sequence[Any]) -> Self:
        """The matrix reordered (or subset) to ``labels``: its rows, and its columns too
        when symmetric. Raises if a label is missing."""
        if self.row_labels is None:
            raise ValueError("This matrix has no labels to align by")
        pos: dict[Any, int] = {}
        for i, label in enumerate(self.row_labels):
            pos.setdefault(label, i)
        missing = [lab for lab in labels if lab not in pos]
        if missing:
            raise ValueError(
                f"{len(missing)} label(s) aren't in the matrix, e.g. {missing[:5]}"
            )
        idx = np.array([pos[lab] for lab in labels], dtype=int)
        labels = list(labels)
        n_shared = self.n_shared
        if self.symmetric:
            values = self.values[np.ix_(idx, idx)]
            n_shared = None if n_shared is None else n_shared[np.ix_(idx, idx)]
            return self._replace(
                values=values, row_labels=labels, col_labels=labels, n_shared=n_shared
            )
        n_shared = None if n_shared is None else n_shared[idx]
        return self._replace(
            values=self.values[idx], row_labels=labels, n_shared=n_shared
        )

    def for_rows(self, data: pl.DataFrame) -> Self:
        """The matrix lined up with the rows of ``data``: by its ``key`` column when it
        has one, otherwise it must already have one row per row of ``data``."""
        if self.key is not None:
            if self.key not in data.columns:
                raise ValueError(
                    f"The matrix is keyed by {self.key!r}, which the data doesn't have"
                )
            return self.align(data.get_column(self.key).to_list())
        if self.shape[0] != data.height:
            raise ValueError(
                f"The matrix has {self.shape[0]} rows but the data {data.height}; give "
                "it a key (e.g. compute it with Profiles.distances) to align by labels"
            )
        return self

    def to_frame(self, index: str | None = None) -> pl.DataFrame:
        """Wide form: a label column (``index``, default the ``key`` or ``"label"``),
        then one column per column label (as a string)."""
        index = index or self.key or "label"
        rows = self.row_labels if self.row_labels is not None else range(self.shape[0])
        cols = self.col_labels if self.col_labels is not None else range(self.shape[1])
        return pl.DataFrame({index: list(rows)}).with_columns(
            pl.Series(str(c), self.values[:, j]) for j, c in enumerate(cols)
        )

    def pairs(self, a: str = "a", b: str = "b") -> pl.DataFrame:
        """Long form ``a, b, value, n_shared``: one row per unordered pair (including each
        item with itself) when symmetric, else one per cell (``a`` the row)."""
        n_rows, n_cols = self.shape
        rows = self.row_labels if self.row_labels is not None else list(range(n_rows))
        cols = self.col_labels if self.col_labels is not None else list(range(n_cols))
        cells = (
            combinations_with_replacement(range(n_rows), 2)
            if self.symmetric
            else product(range(n_rows), range(n_cols))
        )
        out = [
            (
                rows[i],
                cols[j],
                float(self.values[i, j]),
                None if self.n_shared is None else int(self.n_shared[i, j]),
            )
            for i, j in cells
        ]
        return pl.DataFrame(
            out,
            schema={
                a: _dtype(rows),
                b: _dtype(cols),
                "value": pl.Float64,
                "n_shared": pl.Int64,
            },
            orient="row",
        )

    def save(self, path: str | PathLike) -> Self:
        """Write the matrix to parquet (wide form, see `to_frame`), keeping the metric and
        key in the file's metadata so `load` restores them. Returns the matrix."""
        index = self.key or "label"
        table = self.to_frame(index).to_arrow()
        meta = {
            "name": self.metric.name,
            "kind": self.metric.kind,
            "label": self.metric.label,
            "symbol": self.metric.symbol,
            "key": self.key,
            "index": index,
            "symmetric": self.symmetric,
        }
        table = table.replace_schema_metadata(
            {**(table.schema.metadata or {}), _METADATA_KEY: json.dumps(meta)}
        )
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        return self


# ----- helpers for consumers -------------------------------------------------------------


def row_distances(
    metric: "str | Metric | MetricMatrix", data: pl.DataFrame, x: np.ndarray | None
) -> MetricMatrix:
    """Distances between the rows of ``data``: from a `MetricMatrix` lined up with them
    (`MetricMatrix.for_rows`), or computed from their features ``x``."""
    if isinstance(metric, MetricMatrix):
        return metric.for_rows(data).to_distance()
    if x is None:
        raise ValueError("No features to compute distances from")
    return Metric.get(metric).between_rows(x).to_distance()


def nearest_neighbor_graph(
    metric: "str | Metric | MetricMatrix",
    data: pl.DataFrame,
    x: np.ndarray | None,
    n_neighbors: int,
) -> Any:
    """Sparse connectivity graph of each row's ``n_neighbors`` nearest other rows."""
    from sklearn.neighbors import kneighbors_graph

    if not isinstance(metric, MetricMatrix):
        native = Metric.get(metric).native(x) if x is not None else None
        if native is not None:
            rows, name = native
            return kneighbors_graph(
                rows, n_neighbors=n_neighbors, mode="connectivity", metric=name
            )
    d = row_distances(metric, data, x).values.copy()
    np.fill_diagonal(d, 0.0)
    if not np.isfinite(d).all():
        raise ValueError("The distances contain NaN; drop the rows that cause them")
    return kneighbors_graph(
        d, n_neighbors=n_neighbors, mode="connectivity", metric="precomputed"
    )


def _rows(
    data: pl.DataFrame | np.ndarray, features: Sequence[str] | None, key: str | None
) -> tuple[np.ndarray, list[Any] | None]:
    if isinstance(data, pl.DataFrame):
        if features is None:
            features = [
                c
                for c, dtype in data.schema.items()
                if c != key and not c.startswith("meta_") and dtype.is_numeric()
            ]
        x = data.select(pl.col(features).cast(pl.Float64)).fill_null(np.nan).to_numpy()
        labels = None if key is None else data.get_column(key).to_list()
        return x, labels
    if features is not None or key is not None:
        raise ValueError("features and key need a DataFrame")
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {x.shape}")
    return x, None


def _columns(
    data: pl.DataFrame | np.ndarray, columns: Sequence[str] | None
) -> tuple[np.ndarray, list[Any]]:
    if isinstance(data, pl.DataFrame):
        if columns is None:
            columns = [c for c, dtype in data.schema.items() if dtype.is_numeric()]
        mat = data.select(pl.col(columns).cast(pl.Float64)).fill_null(np.nan).to_numpy()
        return mat, list(columns)
    mat = np.asarray(data, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {mat.shape}")
    if columns is not None:
        raise ValueError("columns needs a DataFrame")
    return mat, list(range(mat.shape[1]))


def _dtype(labels: Sequence[Any]) -> Any:
    if labels and all(isinstance(v, str) for v in labels):
        return pl.String
    if labels and all(isinstance(v, int) for v in labels):
        return pl.Int64
    return None
