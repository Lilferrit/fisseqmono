"""Per-variant feature profiles from the pipeline's feature-selection outputs."""

import fnmatch
import pathlib
import re
from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import polars as pl
import polars.selectors as cs

from . import _pipeline, _transforms
from .dataset import Dataset

if TYPE_CHECKING:
    from .ovwt import OvwtScores

#: Aggregate statistics the pipeline appends to feature names (``<feature>_<statistic>``).
STATISTICS: tuple[str, ...] = (
    "median",
    "mean",
    "KS",
    "signedKS",
    "AUROC",
    "KSnegLogP",
    "AUROCnegLogP",
)

#: Feature categories, matched as substrings of the feature name in this order.
CATEGORIES: tuple[str, ...] = (
    "AreaShape",
    "Correlation",
    "Texture",
    "RadialDistribution",
    "Intensity",
    "Granularity",
    "Neighbors",
    "Zernike",
    "Location",
)

_COMPARTMENT_PREFIXES = ("Mean", "Median", "Children")
_CHANNEL_RE = re.compile(r"CH\d+")


class Profiles(Dataset):
    """Feature profiles: one row per variant, or per (variant, batch) before
    `median_across_batches`.

    Feature columns are named ``<feature>_<statistic>`` (e.g. ``AreaShape_Area_median``)
    as the pipeline writes them. Build them from a pipeline run with `from_pipeline`,
    then chain transforms. Every method returns a new `Profiles`:

    >>> profiles = (
    ...     fb.Profiles.from_pipeline(run_dir, types=["median"], passthrough=["KSnegLogP"])
    ...       .variant_type()
    ...       .normalize(by="meta_experiment")
    ...       .median_across_batches(paired={"_median": "_KSnegLogP"})
    ...       .clinvar("clinvar_converted.parquet")
    ... )
    >>> fb.VolcanoPlot.from_wide(profiles).save("vis/volcano.png")

    Parameters
    ----------
    data : pl.DataFrame | pl.LazyFrame | Dataset
        The profiles.
    variant_col : str, default "meta_aa_changes"
        Variant label column.
    batch_col : str, default "meta_experiment"
        Column naming each row's batch, while there is one row per (variant, batch).
    passthrough : Sequence[str], default ()
        Statistics (column suffixes such as ``"KSnegLogP"``) that are carried along but
        are not profile values: `normalize` leaves them alone and `keep_features` keeps
        them next to their feature. `from_pipeline` sets this.
    """

    def __init__(
        self,
        data: "pl.DataFrame | pl.LazyFrame | Dataset",
        *,
        variant_col: str = "meta_aa_changes",
        batch_col: str = "meta_experiment",
        passthrough: Sequence[str] = (),
    ) -> None:
        super().__init__(data, variant_col=variant_col)
        self.batch_col = batch_col
        self.passthrough = tuple(s.lstrip("_") for s in passthrough)
        #: Loadings from the last `pca` / `pca_reduce` call: a ``component`` and an
        #: ``explained_variance_ratio`` column, then one column per input feature.
        self.pca_loadings: pl.DataFrame | None = None
        #: Explained variance of every fitted component from the last `pca` /
        #: `pca_reduce` / `impact_scores` call: ``component``, ``explained_variance_ratio``,
        #: ``cumulative_explained_variance`` and ``retained`` (kept as a feature). Pass it to
        #: `ExplainedVariancePlot`.
        self.pca_explained_variance: pl.DataFrame | None = None
        #: With ``pca_reduce(noise_floor=True)``: the variance ratio of the first component
        #: of the column-shuffled data.
        self.pca_noise_floor: float | None = None
        #: From the last `feature_select` call: one row per input feature, with
        #: ``feature`` and ``kept``.
        self.feature_selection: pl.DataFrame | None = None

    # ----- constructors ---------------------------------------------------------------

    @classmethod
    def from_pipeline(
        cls,
        pipeline_dir: str | PathLike,
        types: Sequence[str] = ("median",),
        passthrough: Sequence[str] = (),
        *,
        batches: Sequence[str] | None = None,
        variant_col: str = "meta_aa_changes",
        batch_col: str = "meta_experiment",
    ) -> Self:
        """Read ``feature_select_batchwise`` aggregates, one row per (variant, batch).

        For each batch, the ``aggregates/<type>.parquet`` files for ``types`` and the
        ``passthrough_aggregates/<type>.parquet`` files for ``passthrough`` (e.g.
        ``"KSnegLogP"``) are joined on the variant; variants missing from any of them are
        dropped. Each batch is tagged in ``batch_col`` with its directory name, and batches
        are stacked in natural order (``T2_R1`` before ``T10_R1``). ``batches`` selects a
        subset.

        The values are raw; chain `variant_type` and `normalize` to z-score them against
        each batch's synonymous controls.
        """
        if not types and not passthrough:
            raise ValueError("Pass at least one aggregate type or passthrough type")
        frames = []
        for batch_dir in _pipeline.batch_dirs(pipeline_dir, _pipeline.FEATURE_SELECT, batches):
            paths = [batch_dir / "aggregates" / f"{t}.parquet" for t in types]
            paths += [batch_dir / "passthrough_aggregates" / f"{t}.parquet" for t in passthrough]
            parts = [_pipeline.scan(p).select(variant_col, ~cs.starts_with("meta_")) for p in paths]
            batch_lf = parts[0]
            for part in parts[1:]:
                batch_lf = batch_lf.join(part, on=variant_col)
            frames.append(_pipeline.tag(batch_lf, batch_dir.name, batch_col))
        return cls(
            pl.concat(frames, how="diagonal_relaxed"),
            variant_col=variant_col,
            batch_col=batch_col,
            passthrough=passthrough,
        )

    @classmethod
    def from_global(cls, pipeline_dir: str | PathLike, channel: str, **kw: Any) -> Self:
        """Read the global feature-selected profiles,
        ``global/<channel>/feature_select/aggregate.parquet`` (one row per variant,
        already normalized and aggregated across batches by the pipeline)."""
        path = (
            pathlib.Path(pipeline_dir) / "global" / channel / "feature_select" / "aggregate.parquet"
        )
        return cls(_pipeline.scan(path), **kw)

    def _replace(self, data: pl.LazyFrame | pl.DataFrame) -> Self:
        new = super()._replace(data)
        new.pca_loadings = None
        new.pca_explained_variance = None
        new.pca_noise_floor = None
        new.feature_selection = None
        return new

    # ----- helpers --------------------------------------------------------------------

    def _is_passthrough(self, column: str) -> bool:
        return any(column.endswith(f"_{s}") for s in self.passthrough)

    @property
    def values(self) -> list[str]:
        """Feature columns holding profile values (the features minus passthrough ones)."""
        return [c for c in self.features if not self._is_passthrough(c)]

    def _matrix(self) -> tuple[pl.DataFrame, np.ndarray]:
        df = self.df
        if not self.values:
            raise ValueError("No feature columns to embed")
        x = df.select(self.values).cast(pl.Float64).to_numpy()
        if not np.isfinite(x).all():
            raise ValueError(
                "Features contain null, NaN or infinite values; chain .drop_nonfinite() first"
            )
        return df, x

    # ----- transforms -----------------------------------------------------------------

    def normalize(
        self,
        *,
        by: str | Sequence[str] | None = None,
        types: Sequence[str] | None = None,
        control_col: str = "meta_is_control",
    ) -> Self:
        """Z-score each feature against the control rows: ``(x - mean) / std``.

        ``by`` computes the statistics within groups, e.g. ``by="meta_experiment"`` to
        normalize each batch against its own synonymous variants. ``types`` limits it to
        features with those statistic suffixes (default: every non-passthrough feature).
        NaN is treated as missing, and a feature whose control std is ~0 becomes null.
        Run `variant_type` first to create ``control_col``.
        """
        if control_col not in self.columns:
            raise ValueError(
                f"Column {control_col!r} not found; chain .variant_type() before .normalize()"
            )
        if by is not None:
            self._require(*([by] if isinstance(by, str) else by))
        features = self.values
        if types is not None:
            suffixes = tuple(f"_{t.lstrip('_')}" for t in types)
            features = [c for c in features if c.endswith(suffixes)]
        return self.with_columns(
            _transforms.normalize_exprs(features, control_col=control_col, by=by)
        )

    def median_across_batches(
        self,
        *,
        paired: Mapping[str, str] | None = None,
        n_col: str = "meta_n_experiments",
    ) -> Self:
        """Collapse to one row per variant: each feature's median across batches.

        ``paired`` maps a value suffix to a companion suffix, e.g. ``{"_median":
        "_KSnegLogP"}``: for each feature with both columns, both values come from the
        same batch (the one holding the lower-middle value), so every p-value belongs to
        the median it is shown with. Other ``meta_`` columns keep their first value, and
        ``n_col`` counts the batches each variant was measured in.
        """
        self._require(self.variant_col, self.batch_col)
        return self._replace(
            _transforms.median_across_batches(
                self._lf,
                self.features,
                variant_col=self.variant_col,
                batch_col=self.batch_col,
                paired=paired,
                n_col=n_col,
            )
        )

    def keep_features(self, patterns: str | Iterable[str]) -> Self:
        """Keep only the features matching ``patterns`` (names or ``fnmatch`` globs such
        as ``"*_CH1_*"``), plus every ``meta_`` column.

        Names that aren't present are ignored, so a blocklist consensus covering more
        statistics than these profiles can be passed directly. Passthrough columns (e.g.
        ``X_KSnegLogP``) are kept whenever their feature (``X_median``) is.
        """
        return self.select(self._keep_columns(self._match(patterns)))

    def _keep_columns(self, features: Iterable[str]) -> list[str]:
        """Every ``meta_`` column, ``features``, and the passthrough columns of those
        features, in their current order."""
        keep = set(features)
        bases = {c.rsplit("_", 1)[0] for c in keep}
        for col in self.features:
            if self._is_passthrough(col) and any(
                col.removesuffix(f"_{s}") in bases for s in self.passthrough
            ):
                keep.add(col)
        return [c for c in self.columns if c.startswith("meta_") or c in keep]

    def drop_features(self, patterns: str | Iterable[str]) -> Self:
        """Drop the features matching ``patterns`` (names or ``fnmatch`` globs such as
        ``"*CH2*"``)."""
        return self.drop(self._match(patterns))

    def _match(self, patterns: str | Iterable[str]) -> list[str]:
        patterns = [patterns] if isinstance(patterns, str) else list(patterns)
        exact = {p for p in patterns if not any(ch in p for ch in "*?[")}
        globs = [p for p in patterns if p not in exact]
        return [
            c for c in self.features if c in exact or any(fnmatch.fnmatchcase(c, g) for g in globs)
        ]

    def drop_nonfinite(self) -> Self:
        """Drop every feature column that holds a null, NaN or infinite value."""
        df = self.df
        bad = df.select(
            (~_transforms.finite(pl.col(c)).is_not_null().all()).alias(c) for c in self.features
        )
        return self._replace(df.drop([c for c in bad.columns if bad[0, c]]))

    def feature_select(
        self,
        operations: Sequence[str] = ("variance_threshold", "blocklist", "correlation_threshold"),
        *,
        corr_threshold: float = 0.9,
        corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
        freq_cut: float = 0.05,
        unique_cut: float = 0.01,
        na_cutoff: float = 0.05,
        blocklist_file: str | PathLike | None = None,
        outlier_cutoff: float = 500.0,
        min_variance: float = 1e-6,
        **pycytominer_kw: Any,
    ) -> Self:
        """Drop redundant and uninformative features with `pycytominer.feature_select`.

        The ``operations`` run in order over the profile values (passthrough columns are
        not inputs, and are kept whenever their feature is). The defaults are the
        pipeline's: near-zero variance (``freq_cut``, ``unique_cut``, ``min_variance``),
        the blocklist, then one of every pair of features correlated above
        ``corr_threshold``. Other pycytominer operations are ``"drop_na_columns"``
        (``na_cutoff``), ``"drop_outliers"`` (``outlier_cutoff``) and ``"noise_removal"``
        (pass ``noise_removal_perturb_groups`` and ``noise_removal_stdev_cutoff`` through
        ``pycytominer_kw``). ``"blocklist"`` uses pycytominer's built-in Cell Painting
        blocklist unless ``blocklist_file`` is given.

        Which features were kept is stored on `feature_selection` (``feature``, ``kept``).
        Needs ``fisseqborn[select]``.

        >>> selected = (profiles.median_across_batches()
        ...                     .drop_nonfinite()
        ...                     .feature_select(corr_threshold=0.8)
        ...                     .pca_reduce(variance=0.9))
        """
        pycytominer = _transforms.require_extra("pycytominer", "select")
        df = self.df
        values = self.values
        if not values:
            raise ValueError("No feature columns to select from")
        selected = pycytominer.feature_select(
            profiles=df.select(values).to_pandas(),
            features=values,
            image_features=False,
            samples="all",
            operation=list(operations),
            corr_threshold=corr_threshold,
            corr_method=corr_method,
            freq_cut=freq_cut,
            unique_cut=unique_cut,
            na_cutoff=na_cutoff,
            blocklist_file=None if blocklist_file is None else str(blocklist_file),
            outlier_cutoff=outlier_cutoff,
            min_variance=min_variance,
            **pycytominer_kw,
        )
        kept = set(selected.columns)
        new = self._replace(df.select(self._keep_columns(kept)))
        new.feature_selection = pl.DataFrame(
            {"feature": values, "kept": [c in kept for c in values]},
            schema={"feature": pl.String, "kept": pl.Boolean},
        )
        return new

    def impact_score(
        self, *, control_col: str = "meta_is_control", output_col: str = "meta_impact_score"
    ) -> Self:
        """Add the impact score: the cosine distance between each profile and the median
        control profile, halved so it runs from 0 (same direction) to 1 (opposite).

        Missing values are skipped per row and feature. Run `variant_type` first.
        """
        if control_col not in self.columns:
            raise ValueError(
                f"Column {control_col!r} not found; chain .variant_type() before .impact_score()"
            )
        return self.with_columns(
            _transforms.impact_score_expr(self.values, control_col=control_col).alias(output_col)
        )

    # ----- embeddings -----------------------------------------------------------------

    def pca(self, n_components: int = 5, *, prefix: str = "meta_pc_") -> Self:
        """Add the first ``n_components`` principal component scores as
        ``meta_pc_1 .. meta_pc_<n>``; loadings are kept on `pca_loadings`."""
        from sklearn.decomposition import PCA

        df, x = self._matrix()
        model = PCA(n_components=n_components)
        scores = model.fit_transform(x)
        names = [f"{prefix}{i + 1}" for i in range(scores.shape[1])]
        new = self._replace(
            df.drop(names, strict=False).with_columns(
                pl.Series(name, scores[:, i]) for i, name in enumerate(names)
            )
        )
        new.pca_loadings = _loadings(model, names, self.values)
        new.pca_explained_variance = _explained_variance(
            model.explained_variance_ratio_, len(names), prefix, start=1
        )
        return new

    def pca_reduce(
        self,
        variance: float = 0.9,
        *,
        noise_floor: bool = False,
        seed: int = 0,
        prefix: str = "X_",
    ) -> Self:
        """Replace the features with their principal components ``X_0 .. X_<k-1>``.

        ``k`` is the fewest components that together explain ``variance`` of the total
        variance, or with ``noise_floor=True``, the number of components explaining more
        variance than the first component of the data with each column shuffled
        independently (seeded by ``seed``). Loadings of the kept components are stored on
        `pca_loadings`, the explained variance of every component on
        `pca_explained_variance` and, with ``noise_floor=True``, the floor ratio on
        `pca_noise_floor`:

        >>> reduced = profiles.pca_reduce(variance=0.9)
        >>> fb.ExplainedVariancePlot(reduced.pca_explained_variance, thresholds=[0.9])
        """
        df, x = self._matrix()
        model, scores = _fit_pca(x)
        ratios = model.explained_variance_ratio_
        floor = None
        if noise_floor:
            k, floor = _transforms.pca_noise_floor(x, ratios, seed)
            k = max(1, min(k, len(ratios)))
        else:
            k = _n_components(ratios, variance)
        names = [f"{prefix}{i}" for i in range(k)]
        new = self._replace(
            df.drop(self.features).with_columns(
                pl.Series(name, scores[:, i]) for i, name in enumerate(names)
            )
        )
        new.passthrough = ()
        new.pca_loadings = _loadings(model, names, self.values, k)
        new.pca_explained_variance = _explained_variance(ratios, k, prefix)
        new.pca_noise_floor = floor
        return new

    def impact_scores(
        self,
        variances: Sequence[float] = (0.7, 0.8, 0.9, 1.0),
        *,
        control_col: str = "meta_is_control",
        prefix: str = "meta_impact_score_",
    ) -> Self:
        """Add the impact score computed on the principal components, once per
        cumulative-variance threshold.

        PCA is fitted once. For each ratio in ``variances``, the impact score (see
        `impact_score`) is computed on the fewest components explaining that share of the
        variance, as ``pca_reduce(variance=v).impact_score()`` would, and added as
        ``<prefix><v>`` (e.g. ``meta_impact_score_0.9``). The features themselves are kept,
        and `pca_explained_variance` is set (``retained`` marks the largest threshold's
        components).

        >>> scored = profiles.impact_scores([0.7, 0.9])
        >>> fb.RocPlot(scored, label="meta_variant_type", positive="Single Missense",
        ...            score=["meta_impact_score_0.7", "meta_impact_score_0.9"])
        """
        if control_col not in self.columns:
            raise ValueError(
                f"Column {control_col!r} not found; chain .variant_type() before .impact_scores()"
            )
        if not variances:
            raise ValueError("Pass at least one variance ratio")
        df, x = self._matrix()
        model, scores = _fit_pca(x)
        ratios = model.explained_variance_ratio_
        ks = {v: _n_components(ratios, v) for v in variances}
        pcs = [f"__pc_{i}" for i in range(max(ks.values()))]
        tmp = df.select(control_col).with_columns(
            pl.Series(name, scores[:, i]) for i, name in enumerate(pcs)
        )
        outputs = [f"{prefix}{v}" for v in variances]
        impact = tmp.select(
            _transforms.impact_score_expr(pcs[: ks[v]], control_col=control_col).alias(out)
            for v, out in zip(variances, outputs)
        )
        new = self._replace(df.drop(outputs, strict=False).hstack(impact))
        new.pca_explained_variance = _explained_variance(ratios, max(ks.values()), "X_")
        return new

    def umap(
        self,
        *,
        n_neighbors: int = 30,
        min_dist: float = 0.1,
        metric: str = "cosine",
        seed: int = 42,
        output_cols: tuple[str, str] = ("meta_notebook_umap_1", "meta_notebook_umap_2"),
        **umap_kw: Any,
    ) -> Self:
        """Add a 2-D UMAP embedding of the features. Needs ``fisseqborn[umap]``."""
        umap = _transforms.require_extra("umap", "umap")
        df, x = self._matrix()
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
            **umap_kw,
        )
        coords = reducer.fit_transform(x)
        return self._replace(
            df.with_columns(pl.Series(name, coords[:, i]) for i, name in enumerate(output_cols))
        )

    def cluster(
        self,
        method: Literal["leiden", "kmeans"] = "leiden",
        *,
        n_neighbors: int = 15,
        resolution: float = 1.0,
        n_clusters: int | None = None,
        seed: int = 0,
        output_col: str = "meta_cluster_idx",
    ) -> Self:
        """Cluster the profiles and add the cluster ids as strings (``"0"``, ``"1"``, ...).

        ``"leiden"`` runs Leiden community detection on the ``n_neighbors``-nearest
        neighbor graph at the given ``resolution`` and needs ``fisseqborn[cluster]``.
        ``"kmeans"`` needs ``n_clusters``.
        """
        df, x = self._matrix()
        labels = _transforms.cluster_labels(
            x,
            method=method,
            n_neighbors=n_neighbors,
            resolution=resolution,
            n_clusters=n_clusters,
            seed=seed,
        )
        return self._replace(
            df.with_columns(pl.Series(output_col, labels.astype(str), dtype=pl.String))
        )

    def distinguishability(
        self,
        ovwt: "OvwtScores",
        *,
        score: str = "auroc_pooled",
        reference: Literal["synonymous", "all"] | None = "synonymous",
        output_col: str = "meta_distinguishability_score",
    ) -> Self:
        """Join a per-variant distinguishability score from OvWT results.

        Per-batch results (`OvwtScores.from_pipeline`) are batch-corrected against
        ``reference`` (see `OvwtScores.correct`; ``None`` skips the correction) and then
        take the median across batches. Results that are already per variant (e.g.
        `OvwtScores.from_global`, or after `OvwtScores.per_variant`) are joined as they
        are, reading ``score``.
        """
        if score not in ovwt.columns and "test_auroc" in ovwt.columns:
            raise ValueError(
                f"Score column {score!r} not found; these look like legacy OvWT results, "
                "pass score='test_auroc'"
            )
        per_variant = ovwt
        if ovwt.batch_col in ovwt.columns:
            if reference is not None:
                per_variant = per_variant.correct(score, reference=reference)
                score = f"{score}_corrected"
            per_variant = per_variant.per_variant(score)
        per_variant._require(per_variant.variant_col, score)
        right = per_variant.lazy().select(
            pl.col(per_variant.variant_col).alias(self.variant_col),
            pl.col(score).alias(output_col),
        )
        return self._replace(
            self._lf.drop(output_col, strict=False).join(
                right, on=self.variant_col, how="left", maintain_order="left"
            )
        )

    # ----- inspection -----------------------------------------------------------------

    def feature_info(self) -> pl.DataFrame:
        """One row per feature column, parsed from its CellProfiler-style name.

        Columns: ``feature`` (the column name), ``base`` (without the statistic suffix),
        ``statistic`` (e.g. ``"median"``, null if unrecognized), ``compartment`` (e.g.
        ``"Nuclei"``, ``"Cytoplasm"``, null for whole-cell features), ``category`` (the
        first of `CATEGORIES` in the name) and ``channels`` (e.g. ``"CH1+CH3"``).
        """
        return feature_info(self.features)


def feature_info(columns: Iterable[str]) -> pl.DataFrame:
    """Parse feature column names; see `Profiles.feature_info`."""
    rows = []
    for col in columns:
        base, _, last = col.rpartition("_")
        statistic = last if base and last in STATISTICS else None
        base = base if statistic else col
        tokens = base.split("_")
        compartment = tokens[1] if len(tokens) > 2 and tokens[0] in _COMPARTMENT_PREFIXES else None
        category = next((c for c in CATEGORIES if c in base), None)
        channels = sorted(set(_CHANNEL_RE.findall(base)), key=lambda c: int(c[2:]))
        rows.append(
            {
                "feature": col,
                "base": base,
                "statistic": statistic,
                "compartment": compartment,
                "category": category,
                "channels": "+".join(channels) or None,
            }
        )
    schema = {
        k: pl.String
        for k in ("feature", "base", "statistic", "compartment", "category", "channels")
    }
    return pl.DataFrame(rows, schema=schema)


def _fit_pca(x: np.ndarray) -> tuple[Any, np.ndarray]:
    """Every principal component of ``x``: the fitted model and the scores."""
    from sklearn.decomposition import PCA

    model = PCA(n_components=None)
    return model, model.fit_transform(x)


def _n_components(ratios: np.ndarray, variance: float) -> int:
    """The fewest components that together explain ``variance`` of the total."""
    if not 0 < variance <= 1:
        raise ValueError(f"variance must be in (0, 1], got {variance}")
    k = int(np.searchsorted(np.cumsum(ratios), variance - 1e-12)) + 1
    return max(1, min(k, len(ratios)))


def _explained_variance(
    ratios: np.ndarray, k: int, prefix: str, *, start: int = 0
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "component": [f"{prefix}{i + start}" for i in range(len(ratios))],
            "explained_variance_ratio": ratios,
            "cumulative_explained_variance": np.cumsum(ratios),
            "retained": np.arange(len(ratios)) < k,
        }
    )


def _loadings(
    model: Any, names: list[str], features: list[str], k: int | None = None
) -> pl.DataFrame:
    k = len(names) if k is None else k
    return pl.DataFrame(
        {
            "component": names,
            "explained_variance_ratio": model.explained_variance_ratio_[:k],
            **{f: model.components_[:k, j] for j, f in enumerate(features)},
        }
    )
