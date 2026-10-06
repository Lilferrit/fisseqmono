"""Cross-experiment aggregation of a pipeline run: `write_global` and ``fisseqborn-global``.

Neither pipeline aggregates across experiments. `write_global` does it from either pipeline's
per-experiment outputs (read through `fisseq_common.layout`), with the methods of the
embeddings pipeline's former global stages (`fisseq_common.global_aggregation`)::

    <out>/<track>/
      blocklist.parquet              # the vote across experiments: feature, n_batches, n_ok,
                                     #   feature_ok (tracks with blocklists)
      median_aggregate.parquet       # per variant, each reproducible feature's median across
                                     #   experiments
      pca_scores.parquet             # full-rank PCA of median_aggregate: meta_pc_1 .. meta_pc_n
      pca_components.parquet         # meta_component_idx + one loading column per feature
      pca_variance_explained.parquet # meta_component_idx, meta_variance_explained,
                                     #   meta_cumulative_variance_explained
      pca_reduced.parquet            # the leading PCs reaching cumulative_variance_explained,
                                     #   meta_is_control, meta_impact_score
    <out>/<ovwt dir>/global_scores.parquet
                                     # per variant: meta_median_<score>, meta_num_experiments

The directories are named after the old global outputs of each pipeline:

- data pipeline: ``feature_select/`` and ``ovwt_distinguishability/``;
- embeddings pipeline: ``embeddings/`` and ``distinguishability/`` for the Cell-DINO track,
  ``cp_features/`` and ``distinguishability_cp_features/`` for the CellProfiler track (no
  blocklist and no summed metadata: that track has no feature selection).
"""

import argparse
import logging
import pathlib
import posixpath
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from typing import Literal

import numpy as np
import polars as pl

from fisseq_common.global_aggregation import (
    OVWT_SCORES,
    blocklist_vote,
    drop_blocked,
    median_across_batches,
    n_components_for_variance,
    ovwt_global_scores,
    reduced_pca,
)
from fisseq_common.layout import (
    DataPipelineLayout,
    EmbeddingsPipelineLayout,
    PipelineLayout,
)
from fisseq_common.schema import (
    COMPONENT_IDX_COL,
    CUMULATIVE_VARIANCE_EXPLAINED_COL,
    PC_COL_PREFIX,
    VARIANCE_EXPLAINED_COL,
)

from . import _pipeline, _transforms
from .blocklist import Blocklists
from .profiles import Profiles, batch_aggregates

logger = logging.getLogger(__name__)

#: `Profiles.feature_select`'s operations, as the data pipeline runs them per experiment.
#: Pass them as ``operations`` to also select features after pooling (off by default).
OPERATIONS: tuple[str, ...] = (
    "variance_threshold",
    "blocklist",
    "correlation_threshold",
)


@dataclass(frozen=True)
class Track:
    """One set of global outputs: a layout and the directories its outputs go to."""

    layout: PipelineLayout
    features_dir: str
    ovwt_dir: str


def tracks(layout: PipelineLayout) -> list[Track]:
    """The tracks of a run with ``layout``'s pipeline, in the order they are written."""
    if isinstance(layout, DataPipelineLayout):
        return [Track(layout, "feature_select", "ovwt_distinguishability")]
    return [
        Track(
            EmbeddingsPipelineLayout("embeddings"), "embeddings", "distinguishability"
        ),
        Track(
            EmbeddingsPipelineLayout("cp_features"),
            "cp_features",
            "distinguishability_cp_features",
        ),
    ]


def write_global(
    pipeline_dir: "str | PathLike | _pipeline.Source",
    out: str | PathLike,
    *,
    layout: "_pipeline.LayoutSpec" = None,
    exclude: "_pipeline.Patterns | None" = None,
    types: Sequence[str] | None = None,
    passthrough: Sequence[str] = (),
    blocklist: bool = True,
    min_batches: int | None = None,
    min_correlation: float | None = None,
    missing: Literal["ignore", "fail"] = "ignore",
    cumulative_variance_explained: float = 0.9,
    seed: int = 0,
    ovwt: bool = True,
    scores: Sequence[str] | None = None,
    cp_features: bool = True,
    operations: Sequence[str] = (),
    corr_threshold: float = 0.9,
    impact_score: bool = False,
    umap: int = 0,
    metadata: bool = False,
    paired: Mapping[str, str] | None = None,
    pca: int | None = None,
    features: Literal["intersection"] | None = None,
    download_dir: str | PathLike | None = None,
    refresh: bool = False,
    variant_col: str = "meta_aa_changes",
    batch_col: str = "meta_experiment",
) -> dict[str, pathlib.Path]:
    """Aggregate a pipeline run across experiments (see the module docstring for the files).

    For each track (the data pipeline's one; the embeddings pipeline's Cell-DINO track and,
    with ``cp_features`` and when the run has it, its CellProfiler track):

    1. **Blocklist vote** (tracks with blocklists, unless ``blocklist=False``): each
       experiment's combined ``blocklist.parquet`` → `blocklist_vote`: a feature is OK when
       every experiment that reports it marks it OK, or at least ``min_batches`` do.
       ``min_correlation`` first recomputes each experiment's verdict as ``median_r >=
       min_correlation``; ``missing="fail"`` counts an experiment that doesn't report a
       feature against it.
    2. **Pooling**: each experiment's aggregates of ``types`` (default: every method the
       run aggregated) and ``passthrough`` → `drop_blocked` (only features voted not OK are
       dropped) → `median_across_batches` (per variant, the median over the features every
       experiment has) → ``median_aggregate.parquet``.
    3. **PCA** at full rank (every component the data supports), seeded with ``seed`` →
       ``pca_scores``, ``pca_components``, ``pca_variance_explained``; and
       ``pca_reduced``: the fewest leading components reaching
       ``cumulative_variance_explained``, with ``meta_is_control`` and an impact score
       computed on them.
    4. **Distinguishability** (unless ``ovwt=False``): each experiment's OvWT ``scores``
       (default ``auroc_pooled``, ``auroc_median_barcode``, ``auroc_median_fold``)
       z-scored against its own synonymous variants, then the median across experiments
       → ``global_scores.parquet``.

    Off by default, applied to ``median_aggregate`` before the PCA: ``operations``
    (pycytominer `Profiles.feature_select`, e.g. `OPERATIONS`, with ``corr_threshold``),
    ``impact_score`` (a pre-PCA ``meta_impact_score`` column), ``umap`` (an ``N``-D UMAP,
    ``meta_notebook_umap_*``), ``metadata`` (per-variant integer ``meta_`` counts summed
    across experiments) and ``paired`` (take each companion column from the experiment
    holding its value's median, e.g. ``{"_median": "_KSnegLogP"}``).

    ``layout`` (``"data"``, ``"embeddings"`` or a `PipelineLayout`) is detected by
    default. ``exclude`` drops experiments by name (``fnmatch`` globs or compiled
    regexes). ``pipeline_dir`` can be remote (``"user@host:/path"``): only the files read
    are copied with scp, into ``download_dir`` (default: a temporary directory), reusing
    files already there unless ``refresh``. Returns the paths written, keyed
    ``"<dir>/<stem>"``.

    ``pca`` and ``features`` are accepted for compatibility and ignored: the PCA is always
    full rank, and pooling always keeps the features every experiment has.
    """
    if pca:
        warnings.warn(
            "pca= is ignored: write_global always computes every principal component",
            DeprecationWarning,
            stacklevel=2,
        )
    if features not in (None, "intersection"):
        raise ValueError(
            "write_global pools over the features every experiment has; features= can "
            f"only be 'intersection', got {features!r}"
        )
    if missing not in ("ignore", "fail"):
        raise ValueError(f"missing must be 'ignore' or 'fail', got {missing!r}")
    out = pathlib.Path(out)
    src = _pipeline.source(pipeline_dir, download_dir, refresh, layout)
    written: dict[str, pathlib.Path] = {}
    for track in tracks(src.layout):
        track_src = src.with_layout(track.layout)
        if track.features_dir == "cp_features" and (
            not cp_features or not track_src.has_stage("feature_select")
        ):
            continue
        written |= _write_track(
            track_src,
            out,
            track,
            exclude=exclude,
            types=types,
            passthrough=passthrough,
            blocklist=blocklist,
            min_batches=min_batches,
            min_correlation=min_correlation,
            missing=missing,
            cumulative_variance_explained=cumulative_variance_explained,
            seed=seed,
            ovwt=ovwt,
            scores=scores,
            operations=operations,
            corr_threshold=corr_threshold,
            impact_score=impact_score,
            umap=umap,
            metadata=metadata,
            paired=paired,
            variant_col=variant_col,
            batch_col=batch_col,
        )
    return written


def _write_track(
    src: "_pipeline.Source",
    out: pathlib.Path,
    track: Track,
    *,
    exclude: "_pipeline.Patterns | None",
    types: Sequence[str] | None,
    passthrough: Sequence[str],
    blocklist: bool,
    min_batches: int | None,
    min_correlation: float | None,
    missing: str,
    cumulative_variance_explained: float,
    seed: int,
    ovwt: bool,
    scores: Sequence[str] | None,
    operations: Sequence[str],
    corr_threshold: float,
    impact_score: bool,
    umap: int,
    metadata: bool,
    paired: Mapping[str, str] | None,
    variant_col: str,
    batch_col: str,
) -> dict[str, pathlib.Path]:
    lay = src.layout
    fs_dir = out / track.features_dir
    written: dict[str, pathlib.Path] = {}

    def save(df: pl.DataFrame, directory: pathlib.Path, stem: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{stem}.parquet"
        df.write_parquet(path)
        written[f"{directory.name}/{stem}"] = path

    names = src.batches("feature_select", exclude=exclude)
    if metadata and lay.selected(names[0]) is None:
        # The embeddings pipeline's CellProfiler track writes no output.parquet.
        logger.info("%s: no per-variant metadata to sum", track.features_dir)
        metadata = False
    frames, meta_paths = batch_aggregates(
        src, names, types, passthrough, variant_col, metadata=metadata
    )
    batch_lfs = [frames[b] for b in names]

    if blocklist and lay.blocklist(names[0]) is not None:
        vote = _vote(src, names, min_batches, min_correlation, missing, batch_col)
        save(vote, fs_dir, "blocklist")
        batch_lfs = drop_blocked(batch_lfs, vote)

    if paired:
        median_df = _paired_median(batch_lfs, names, paired, variant_col, batch_col)
    else:
        median_df = median_across_batches(batch_lfs, variant_col, names)
    median_df = median_df.sort(variant_col)
    median_df = _extras(
        median_df,
        operations=operations,
        corr_threshold=corr_threshold,
        impact_score=impact_score,
        umap=umap,
        meta_paths=meta_paths,
        variant_col=variant_col,
    )
    save(median_df, fs_dir, "median_aggregate")

    scores_df, components_df = full_rank_pca(median_df, variant_col, seed)
    variance_df = components_df.select(
        COMPONENT_IDX_COL, VARIANCE_EXPLAINED_COL, CUMULATIVE_VARIANCE_EXPLAINED_COL
    )
    n_selected = n_components_for_variance(variance_df, cumulative_variance_explained)
    logger.info(
        "%s: %d/%d component(s) reach %.4f of the variance",
        track.features_dir,
        n_selected,
        variance_df.height,
        cumulative_variance_explained,
    )
    save(scores_df, fs_dir, "pca_scores")
    save(
        components_df.drop(VARIANCE_EXPLAINED_COL, CUMULATIVE_VARIANCE_EXPLAINED_COL),
        fs_dir,
        "pca_components",
    )
    save(variance_df, fs_dir, "pca_variance_explained")
    save(reduced_pca(scores_df, variant_col, n_selected), fs_dir, "pca_reduced")

    if ovwt and src.has_stage("ovwt"):
        ovwt_names = src.batches("ovwt", exclude=exclude)
        paths = src.files([lay.ovwt_results(b) for b in ovwt_names])
        results = [pl.read_parquet(p) for p in paths]
        global_scores = ovwt_global_scores(
            results, variant_col, scores if scores is not None else _present(results)
        )
        save(global_scores.sort(variant_col), out / track.ovwt_dir, "global_scores")
    return written


def _present(results: Sequence[pl.DataFrame]) -> list[str]:
    """The `OVWT_SCORES` columns every experiment's results have."""
    present = [s for s in OVWT_SCORES if all(s in df.columns for df in results)]
    if not present:
        raise ValueError(f"OvWT results have none of the score columns {OVWT_SCORES}")
    return present


def _vote(
    src: "_pipeline.Source",
    names: Sequence[str],
    min_batches: int | None,
    min_correlation: float | None,
    missing: str,
    batch_col: str,
) -> pl.DataFrame:
    """The blocklist vote over each experiment's blocklist."""
    frames = _batch_blocklists(src, names)
    if min_correlation is not None:
        frames = [
            df.with_columns(
                (pl.col("median_r") >= min_correlation)
                .fill_null(False)
                .alias("feature_ok")
            )
            for df in frames
        ]
    if missing == "ignore":
        return blocklist_vote(frames, min_batches)
    stacked = pl.concat(
        [df.with_columns(pl.lit(b).alias(batch_col)) for b, df in zip(names, frames)]
    )
    return (
        Blocklists(stacked, batch_col=batch_col)
        .table(min_batches, missing="fail")
        .sort("feature")
    )


def _batch_blocklists(
    src: "_pipeline.Source", names: Sequence[str]
) -> list[pl.DataFrame]:
    """Each experiment's combined ``blocklist.parquet``; for an experiment without one
    (an older run), its per-method blocklists, concatenated (the same rows)."""
    lay = src.layout
    combined = {b: lay.blocklist(b) for b in names}
    found = set(src.glob(list(combined.values())))
    missing = [b for b in names if combined[b] not in found]
    per_method: dict[str, list[str]] = {}
    if missing:
        patterns = {b: lay.method_blocklist(b, "*") for b in missing}
        listed = src.glob(list(patterns.values()))
        for b, pattern in patterns.items():
            directory = posixpath.dirname(pattern)
            per_method[b] = sorted(
                r for r in listed if posixpath.dirname(r) == directory
            )
            if not per_method[b]:
                raise FileNotFoundError(
                    f"No blocklist in {src}/{posixpath.dirname(combined[b])}"
                )
    rels = [
        r
        for b in names
        for r in ([combined[b]] if b not in per_method else per_method[b])
    ]
    local = dict(zip(rels, src.files(rels)))
    frames = []
    for b in names:
        files = [combined[b]] if b not in per_method else per_method[b]
        frames.append(
            pl.concat(
                [pl.read_parquet(local[r]) for r in files], how="diagonal_relaxed"
            )
        )
    return frames


def _paired_median(
    batch_lfs: Sequence[pl.LazyFrame],
    names: Sequence[str],
    paired: Mapping[str, str],
    variant_col: str,
    batch_col: str,
) -> pl.DataFrame:
    """`median_across_batches` over the common features, taking each ``paired`` companion
    from the experiment that holds its value's median."""
    common = sorted(
        set.intersection(
            *(set(lf.collect_schema().names()) - {variant_col} for lf in batch_lfs)
        )
    )
    common = [c for c in common if not c.startswith("meta_")]
    stacked = pl.concat(
        [
            lf.select(variant_col, *common).with_columns(pl.lit(b).alias(batch_col))
            for b, lf in zip(names, batch_lfs)
        ]
    )
    return (
        _transforms.median_across_batches(
            stacked,
            common,
            variant_col=variant_col,
            batch_col=batch_col,
            paired=paired,
            n_col="meta_n_experiments",
        )
        .select(variant_col, *common)
        .collect()
    )


def _extras(
    median_df: pl.DataFrame,
    *,
    operations: Sequence[str],
    corr_threshold: float,
    impact_score: bool,
    umap: int,
    meta_paths: Mapping[str, pathlib.Path],
    variant_col: str,
) -> pl.DataFrame:
    """The opt-in steps on the pooled table: feature selection, impact score, UMAP and
    summed metadata."""
    if operations:
        selected = (
            Profiles(median_df, variant_col=variant_col)
            .variant_type()
            .feature_select(list(operations), corr_threshold=corr_threshold)
        )
        median_df = median_df.select(variant_col, *selected.features)
    if impact_score:
        scored = (
            Profiles(median_df, variant_col=variant_col).variant_type().impact_score()
        )
        median_df = median_df.with_columns(scored.df.get_column("meta_impact_score"))
    if umap:
        embedded = (
            Profiles(median_df, variant_col=variant_col)
            .drop_nonfinite()
            .umap(n_components=umap)
        )
        median_df = median_df.with_columns(
            embedded.df.select(
                c for c in embedded.columns if c.startswith("meta_notebook_umap_")
            ).get_columns()
        )
    if meta_paths:
        counts = pl.concat(
            [
                pl.scan_parquet(p).select(
                    variant_col,
                    *[
                        c
                        for c, dtype in pl.read_parquet_schema(p).items()
                        if c.startswith("meta_")
                        and c != variant_col
                        and dtype.is_integer()
                    ],
                )
                for p in meta_paths.values()
            ],
            how="diagonal_relaxed",
        )
        sums = counts.group_by(variant_col).agg(pl.all().sum()).collect()
        median_df = median_df.join(
            sums, on=variant_col, how="left", maintain_order="left"
        )
    return median_df


def full_rank_pca(
    df: pl.DataFrame, variant_col: str, seed: int = 0
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Every principal component of ``df``'s feature (non-``meta_``) columns.

    Columns that are null in every row are dropped first; the PCA keeps ``min(n_rows,
    n_features)`` components, with ``random_state=seed``.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        ``(scores, components)``: ``variant_col`` and ``meta_pc_1 .. meta_pc_n`` per row;
        and per component ``meta_component_idx``, one loading column per feature,
        ``meta_variance_explained`` and ``meta_cumulative_variance_explained``.
    """
    from sklearn.decomposition import PCA

    features = [c for c in df.columns if not c.startswith("meta_")]
    null_counts = df.select(features).null_count().row(0, named=True)
    dropped = sorted(c for c in features if null_counts[c] == df.height)
    if dropped:
        logger.warning(
            "Dropping %d all-null feature column(s) before PCA: %s",
            len(dropped),
            dropped,
        )
    features = [c for c in features if c not in dropped]
    if not features:
        raise ValueError("Every feature column is entirely null; nothing to fit")
    n_components = min(df.height, len(features))
    pca = PCA(n_components=n_components, random_state=seed)
    scores = pca.fit_transform(df.select(features).to_numpy())
    pc_cols = [f"{PC_COL_PREFIX}{i}" for i in range(1, n_components + 1)]
    scores_df = df.select(variant_col).with_columns(
        pl.Series(name, scores[:, i]) for i, name in enumerate(pc_cols)
    )
    ratio = pca.explained_variance_ratio_
    components_df = (
        pl.DataFrame({COMPONENT_IDX_COL: np.arange(1, n_components + 1)})
        .with_columns(
            pl.Series(feature, pca.components_[:, j])
            for j, feature in enumerate(features)
        )
        .with_columns(
            pl.Series(VARIANCE_EXPLAINED_COL, ratio),
            pl.Series(CUMULATIVE_VARIANCE_EXPLAINED_COL, np.cumsum(ratio)),
        )
    )
    return scores_df, components_df


def _paired(values: Sequence[str]) -> dict[str, str] | None:
    pairs = {}
    for value in values:
        stat, sep, companion = value.partition(":")
        if not sep or not stat or not companion:
            raise argparse.ArgumentTypeError(
                f"--paired expects VALUE:COMPANION (e.g. median:KSnegLogP), got {value!r}"
            )
        pairs[f"_{stat.lstrip('_')}"] = f"_{companion.lstrip('_')}"
    return pairs or None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fisseqborn-global",
        description=(
            "Aggregate a fisseq-data-pipeline or fisseq-embeddings-pipeline run across "
            "experiments: per track, blocklist.parquet (the vote), median_aggregate.parquet, "
            "pca_{scores,components,variance_explained,reduced}.parquet, and the "
            "distinguishability global_scores.parquet."
        ),
    )
    parser.add_argument(
        "pipeline_dir",
        help="pipeline output directory, local or remote (user@host:/path)",
    )
    parser.add_argument(
        "--out", type=pathlib.Path, required=True, help="output directory"
    )
    parser.add_argument(
        "--layout",
        choices=["data", "embeddings"],
        help="which pipeline wrote the run (default: detected)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="exclude batches matching this fnmatch glob (repeatable), e.g. 'T10_*'",
    )
    parser.add_argument(
        "--exclude-regex",
        action="append",
        default=[],
        metavar="REGEX",
        help="exclude batches whose name this regex matches (repeatable), e.g. '_R3$'",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        help="aggregate types (default: every method the run aggregated)",
    )
    parser.add_argument(
        "--passthrough",
        nargs="*",
        default=[],
        help="passthrough aggregate types to carry along, e.g. KSnegLogP",
    )
    parser.add_argument(
        "--no-blocklist",
        dest="blocklist",
        action="store_false",
        help="pool every feature, without the blocklist vote",
    )
    parser.add_argument(
        "--min-batches",
        type=int,
        help="a feature must be OK in at least this many batches (default: every batch "
        "that reports it)",
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        help="recompute each batch's feature_ok as median_r >= this before voting",
    )
    parser.add_argument(
        "--missing",
        choices=["ignore", "fail"],
        default="ignore",
        help="a feature missing from a batch's blocklist is judged only on the batches "
        "reporting it, or fails there (default: ignore)",
    )
    parser.add_argument(
        "--cumulative-variance-explained",
        type=float,
        default=0.9,
        help="pca_reduced keeps the fewest leading PCs reaching this (default: 0.9)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="PCA random_state (default: 0)"
    )
    parser.add_argument(
        "--scores",
        nargs="+",
        help="OvWT score columns (default: auroc_pooled, auroc_median_barcode, "
        "auroc_median_fold)",
    )
    parser.add_argument(
        "--no-ovwt",
        dest="ovwt",
        action="store_false",
        help="skip the distinguishability global_scores.parquet",
    )
    parser.add_argument(
        "--no-cp-features",
        dest="cp_features",
        action="store_false",
        help="skip the embeddings pipeline's CellProfiler track",
    )
    parser.add_argument(
        "--operations",
        nargs="*",
        default=[],
        help="pycytominer feature_select operations on the pooled table, e.g. "
        + " ".join(OPERATIONS)
        + " (default: none)",
    )
    parser.add_argument("--corr-threshold", type=float, default=0.9)
    parser.add_argument(
        "--impact-score",
        action="store_true",
        help="add a pre-PCA meta_impact_score to median_aggregate",
    )
    parser.add_argument(
        "--umap", type=int, default=0, metavar="N", help="add an N-D UMAP"
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="join each variant's integer meta_ counts, summed across batches",
    )
    parser.add_argument(
        "--paired",
        action="append",
        default=[],
        metavar="VALUE:COMPANION",
        help="take COMPANION from the batch holding VALUE's median, e.g. median:KSnegLogP",
    )
    parser.add_argument("--pca", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--download-dir",
        type=pathlib.Path,
        help="for a remote pipeline_dir (user@host:/path): where to copy its files "
        "(default: a temporary directory)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download remote files already in --download-dir",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of the ``fisseqborn-global`` command."""
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        paired = _paired(args.paired)
    except argparse.ArgumentTypeError as err:
        parser.error(str(err))
    exclude = [*args.exclude, *(re.compile(r) for r in args.exclude_regex)]
    written = write_global(
        args.pipeline_dir,
        args.out,
        layout=args.layout,
        exclude=exclude or None,
        types=args.types,
        passthrough=args.passthrough,
        blocklist=args.blocklist,
        min_batches=args.min_batches,
        min_correlation=args.min_correlation,
        missing=args.missing,
        cumulative_variance_explained=args.cumulative_variance_explained,
        seed=args.seed,
        ovwt=args.ovwt,
        scores=args.scores,
        cp_features=args.cp_features,
        operations=args.operations,
        corr_threshold=args.corr_threshold,
        impact_score=args.impact_score,
        umap=args.umap,
        metadata=args.metadata,
        paired=paired,
        pca=args.pca,
        download_dir=args.download_dir,
        refresh=args.refresh,
    )
    for path in written.values():
        print(path)


if __name__ == "__main__":
    main()
