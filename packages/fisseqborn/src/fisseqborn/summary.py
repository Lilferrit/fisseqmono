"""Per-cluster summaries of a per-variant frame, ready for `ClusterMap` groups."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import polars as pl

from . import _data, _transforms
from .dataset import Dataset

if TYPE_CHECKING:
    from .clustermap import FeatureGroup

#: A share block: a categorical column (one share per level) or ``{level name: boolean expr}``.
ShareSpec = str | Mapping[str, pl.Expr]


class ClusterSummary(Dataset):
    """One row per cluster, built by `Dataset.cluster_summary`.

    Columns are the cluster id (``by``), the variant count (``count_col``), a
    ``"<id> (n=<count>)"`` label (``label_col``), the per-cluster medians and, for each
    share block, one column per level: the fraction of that level's variants that fall
    in the cluster (each level column sums to 1 over the clusters).

    `group` turns a share block into a `FeatureGroup` whose feature labels carry the
    level totals, e.g. ``"Head (n=412)"``:

    >>> summary = profiles.cluster_summary(shares={"domain": "meta_domain"})
    >>> fb.ClusterMap(summary, row_labels="label",
    ...               groups=[summary.group("domain", "Share of domain", palette="tab10")])

    Attributes
    ----------
    by, count_col, label_col : str
        The cluster id, count and label columns.
    share_columns : dict[str, list[str]]
        Per share block, its level columns in order.
    """

    def __init__(
        self,
        data: "pl.DataFrame | pl.LazyFrame | Dataset",
        *,
        by: str = "meta_cluster_idx",
        count_col: str = "n",
        label_col: str = "label",
        share_columns: Mapping[str, Sequence[str]] | None = None,
        counts: pl.LazyFrame | None = None,
        variant_col: str = "meta_aa_changes",
    ) -> None:
        super().__init__(data, variant_col=variant_col)
        self.by, self.count_col, self.label_col = by, count_col, label_col
        self.share_columns = {k: list(v) for k, v in (share_columns or {}).items()}
        self._counts = counts
        self._totals: dict[str, dict[str, int]] | None = None

    def _block(self, key: str) -> list[str]:
        if key not in self.share_columns:
            raise KeyError(
                f"No share block {key!r}; blocks are {list(self.share_columns)}"
            )
        return self.share_columns[key]

    def shares(self, key: str) -> list[str]:
        """The level columns of share block ``key``, in order."""
        return list(self._block(key))

    @property
    def totals(self) -> dict[str, dict[str, int]]:
        """Per share block, the number of variants of each level (over every cluster).

        Computed from the summary's source when first accessed.
        """
        if self._totals is None:
            cols = [c for block in self.share_columns.values() for c in block]
            if self._counts is None or not cols:
                self._totals = {k: {} for k in self.share_columns}
            else:
                row = (
                    self._counts.select(pl.col(cols).sum()).collect().row(0, named=True)
                )
                self._totals = {
                    key: {c: int(row[c]) for c in block}
                    for key, block in self.share_columns.items()
                }
        return self._totals

    def group(
        self, key: str, name: str | None = None, **group_kw: Any
    ) -> "FeatureGroup":
        """A `FeatureGroup` of share block ``key``, titled ``name`` (default ``key``).

        Features are labelled ``"<level> (n=<total>)"`` and ``vmin`` defaults to 0;
        ``group_kw`` (e.g. ``palette``, ``annot``, ``cluster``) is passed to
        `FeatureGroup` and wins over both.
        """
        from .clustermap import FeatureGroup

        cols = self.shares(key)
        totals = self.totals[key]
        kw: dict[str, Any] = {
            "labels": {c: f"{c} (n={totals[c]})" for c in cols},
            "vmin": 0,
            **group_kw,
        }
        return FeatureGroup(name if name is not None else key, cols, **kw)


def cluster_summary(
    dataset: Dataset,
    by: str = "meta_cluster_idx",
    *,
    medians: Sequence[str] | Mapping[str, str] = (),
    zscore: bool | Sequence[str] = False,
    shares: Mapping[str, ShareSpec] | None = None,
    levels: Mapping[str, Sequence[Any]] | None = None,
    control_col: str = "meta_is_control",
    count_col: str = "n",
    label_col: str = "label",
) -> ClusterSummary:
    """See `Dataset.cluster_summary`."""
    columns = dataset.columns
    schema = pl.DataFrame(schema=dataset.schema)
    median_names = (
        dict(medians) if isinstance(medians, Mapping) else {c: c for c in medians}
    )
    _data.require_columns(schema, by, *median_names)
    shares = dict(shares or {})
    levels = dict(levels or {})
    unknown = set(levels) - set(shares)
    if unknown:
        raise ValueError(f"levels given for unknown share block(s): {sorted(unknown)}")

    if zscore is True:
        zcols = list(median_names)
    elif zscore is False:
        zcols = []
    else:
        zcols = list(zscore)
        extra = [c for c in zcols if c not in median_names]
        if extra:
            raise ValueError(f"zscore columns must also be in medians: {extra}")
    lf = dataset.lazy()
    if zcols:
        if control_col not in columns:
            raise ValueError(
                f"Column {control_col!r} not found; chain .variant_type() before "
                ".cluster_summary(zscore=...)"
            )
        lf = lf.with_columns(
            _transforms.normalize_exprs(zcols, control_col=control_col, by=None)
        )

    share_exprs: dict[str, dict[str, pl.Expr]] = {}
    for key, spec in shares.items():
        if isinstance(spec, str):
            _data.require_columns(schema, spec)
            if key in levels:
                block_levels = list(levels[key])
            else:
                present = lf.select(pl.col(spec).unique()).collect()
                block_levels = _data.resolve_order(present, spec)
            share_exprs[key] = {str(lvl): pl.col(spec) == lvl for lvl in block_levels}
        elif isinstance(spec, Mapping):
            share_exprs[key] = dict(spec)
        else:
            raise TypeError(
                f"Share block {key!r} must be a column name or a mapping of level -> expression"
            )

    outputs = [by, count_col, label_col, *median_names.values()]
    for exprs in share_exprs.values():
        outputs.extend(exprs)
    dupes = sorted({c for c in outputs if outputs.count(c) > 1})
    if dupes:
        raise ValueError(f"cluster_summary would create duplicate columns: {dupes}")

    counts = (
        lf.group_by(by)
        .agg(
            pl.len().alias(count_col),
            *[pl.col(c).median().alias(out) for c, out in median_names.items()],
            *[
                expr.fill_null(False).sum().alias(name)
                for exprs in share_exprs.values()
                for name, expr in exprs.items()
            ],
        )
        .sort(
            pl.col(by).cast(pl.Int64, strict=False),
            pl.col(by).cast(pl.String),
            nulls_last=True,
        )
    )
    share_cols = [name for exprs in share_exprs.values() for name in exprs]
    summary = counts.with_columns(
        *[(pl.col(c) / pl.col(c).sum()).alias(c) for c in share_cols],
        pl.format("{} (n={})", pl.col(by), pl.col(count_col)).alias(label_col),
    )
    return ClusterSummary(
        summary,
        by=by,
        count_col=count_col,
        label_col=label_col,
        share_columns={k: list(v) for k, v in share_exprs.items()},
        counts=counts,
        variant_col=dataset.variant_col,
    )
