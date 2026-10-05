"""Polars / numpy implementations behind the `Profiles` methods."""

import importlib
import logging
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Literal

import numpy as np
import polars as pl
import polars.selectors as cs

logger = logging.getLogger(__name__)

#: Standard deviations below this are treated as zero (matches the pipeline's Normalizer).
EPS = float(np.finfo(np.float32).eps)


def require_extra(module: str, extra: str) -> ModuleType:
    """Import an optional dependency, naming the extra that provides it if missing."""
    try:
        return importlib.import_module(module)
    except ImportError as err:
        raise ImportError(
            f"{module!r} is required for this method; install it with "
            f"`pip install 'fisseqborn[{extra}]'`"
        ) from err


def finite(expr: pl.Expr) -> pl.Expr:
    """``expr`` as Float64 with NaN and +/-inf replaced by null."""
    expr = expr.cast(pl.Float64)
    return pl.when(expr.is_finite()).then(expr)


def normalize_exprs(
    features: Sequence[str], *, control_col: str, by: str | Sequence[str] | None
) -> list[pl.Expr]:
    """Z-score each feature against the mean and std of the control rows.

    With ``by``, statistics are computed within each group. NaN counts as missing, and a
    feature whose control std is below `EPS` becomes null rather than exploding.
    """
    control = pl.col(control_col)

    def zscore(feature: str) -> pl.Expr:
        value = pl.col(feature).cast(pl.Float64).fill_nan(None)
        mean = value.filter(control).mean()
        std = value.filter(control).std()
        if by is not None:
            mean, std = mean.over(by), std.over(by)
        std = pl.when(std.abs() >= EPS).then(std)
        return ((value - mean) / std).fill_nan(None).alias(feature)

    return [zscore(f) for f in features]


def intersect_features(present: Mapping[str, Sequence[str]]) -> list[str]:
    """The features present in every batch of ``present`` (batch -> its features), in the
    order of the first batch.

    Logs one warning per batch that loses features, naming them, and raises `ValueError`
    when no feature is shared by every batch.
    """
    sets = [set(cols) for cols in present.values()]
    common = set.intersection(*sets) if sets else set()
    for batch, cols in present.items():
        dropped = [c for c in cols if c not in common]
        if dropped:
            logger.warning(
                "features='intersection': dropping %d feature(s) of batch %s missing from "
                "another batch: %s",
                len(dropped),
                batch,
                ", ".join(dropped),
            )
    if not common:
        raise ValueError(
            "features='intersection': no feature is present in every batch"
        )
    first = next(iter(present.values()))
    return [c for c in first if c in common]


def features_by_batch(
    lf: pl.LazyFrame, features: Sequence[str], *, batch_col: str
) -> dict[str, list[str]]:
    """For each batch (in order of appearance), the ``features`` with at least one
    non-null, non-NaN value in it."""
    counts = (
        lf.group_by(batch_col, maintain_order=True)
        .agg(pl.col(f).cast(pl.Float64).fill_nan(None).count() for f in features)
        .collect()
    )
    return {
        row[batch_col]: [f for f in features if row[f] > 0]
        for row in counts.iter_rows(named=True)
    }


def median_across_batches(
    lf: pl.LazyFrame,
    features: Sequence[str],
    *,
    variant_col: str,
    batch_col: str,
    paired: Mapping[str, str] | None,
    n_col: str,
    sum_cols: Sequence[str] = (),
) -> pl.LazyFrame:
    """One row per variant: the median of each feature across batches.

    ``paired`` maps a value suffix to a companion suffix (e.g. ``{"_median":
    "_KSnegLogP"}``). For each feature with both columns, the batch holding the (lower)
    middle value is picked and both columns come from that same batch, so a p-value
    always belongs to the median it is shown with. Other ``meta_`` columns keep their
    first value (``sum_cols`` are summed instead) and ``n_col`` counts the batches each
    variant was measured in.
    """
    feature_set = set(features)
    paired_exprs: list[pl.Expr] = []
    handled: set[str] = set()
    for value_suffix, companion_suffix in (paired or {}).items():
        for col in features:
            if not col.endswith(value_suffix):
                continue
            companion = col.removesuffix(value_suffix) + companion_suffix
            if companion not in feature_set or col in handled or companion in handled:
                continue
            # Nulls sort last, so the middle index is taken over the non-null values. Ties
            # on the value are broken by the companion, so the pick doesn't depend on the
            # order of the batches. A variant with no value in any batch gets null for both.
            idx = ((pl.col(col).count().cast(pl.Int64) - 1) // 2).clip(lower_bound=0)
            has_value = pl.col(col).count() > 0
            paired_exprs += [
                pl.when(has_value)
                .then(pl.col(c).sort_by(col, companion, nulls_last=True).get(idx))
                .alias(c)
                for c in (col, companion)
            ]
            handled.update((col, companion))

    plain = [c for c in features if c not in handled]
    meta = cs.starts_with("meta_") - cs.by_name(variant_col, batch_col, *sum_cols)
    return lf.group_by(variant_col, maintain_order=True).agg(
        meta.first(),
        *[pl.col(c).sum() for c in sum_cols],
        pl.col(batch_col).n_unique().cast(pl.UInt32).alias(n_col),
        *[pl.col(c).median() for c in plain],
        *paired_exprs,
    )


def impact_score_expr(features: Sequence[str], *, control_col: str) -> pl.Expr:
    """Cosine distance to the control rows' median profile, halved to lie in [0, 1].

    Non-finite values count as missing, and a feature is skipped for a row whenever it
    is missing in that row or in the control median, as in the pipeline.
    """
    control = pl.col(control_col)
    rows = [finite(pl.col(f)) for f in features]
    refs = [finite(pl.col(f)).filter(control).median() for f in features]
    valid = [a.is_not_null() & b.is_not_null() for a, b in zip(rows, refs)]

    def masked_sum(terms: list[pl.Expr]) -> pl.Expr:
        return pl.sum_horizontal(
            [pl.when(v).then(t).otherwise(0.0) for t, v in zip(terms, valid)]
        )

    norm_a = masked_sum([a.pow(2) for a in rows]).sqrt()
    norm_b = masked_sum([b.pow(2) for b in refs]).sqrt()
    dot = masked_sum([a * b for a, b in zip(rows, refs)])
    norm_a = pl.when(norm_a == 0.0).then(1.0).otherwise(norm_a)
    norm_b = pl.when(norm_b == 0.0).then(1.0).otherwise(norm_b)
    return (1 - dot / (norm_a * norm_b)) / 2


def pca_noise_floor(
    x: np.ndarray, explained_variance_ratio: np.ndarray, seed: int
) -> tuple[int, float]:
    """Number of components that explain more variance than the first PC of ``x`` with
    every column shuffled independently (same marginals, no correlation structure), and
    that noise-floor ratio."""
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    shuffled = x.copy()
    for col in range(shuffled.shape[1]):
        rng.shuffle(shuffled[:, col])
    noise_floor = PCA(n_components=1).fit(shuffled).explained_variance_ratio_[0]
    return int(np.sum(explained_variance_ratio > noise_floor)), float(noise_floor)


def cluster_labels(
    x: np.ndarray,
    *,
    method: Literal["leiden", "kmeans"],
    n_neighbors: int,
    resolution: float,
    n_clusters: int | None,
    seed: int,
) -> np.ndarray:
    """Leiden communities of the k-nearest-neighbor graph, or k-means labels."""
    if method == "kmeans":
        if n_clusters is None:
            raise ValueError("method='kmeans' requires n_clusters")
        from sklearn.cluster import KMeans

        return KMeans(
            n_clusters=n_clusters, random_state=seed, n_init="auto"
        ).fit_predict(x)
    if method != "leiden":
        raise ValueError(f"Unknown method {method!r}; expected 'leiden' or 'kmeans'")
    if n_clusters is not None:
        raise ValueError(
            "n_clusters is not used by method='leiden' (Leiden picks the number of "
            "clusters itself); set resolution instead"
        )
    ig = require_extra("igraph", "cluster")
    leidenalg = require_extra("leidenalg", "cluster")
    from sklearn.neighbors import kneighbors_graph

    knn = kneighbors_graph(x, n_neighbors=n_neighbors, mode="connectivity")
    sources, targets = knn.nonzero()
    graph = ig.Graph(directed=False)
    graph.add_vertices(x.shape[0])
    graph.add_edges(list(zip(sources.tolist(), targets.tolist())))
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )
    return np.asarray(partition.membership)
