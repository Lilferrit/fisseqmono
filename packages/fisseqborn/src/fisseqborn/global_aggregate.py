"""Cross-experiment aggregation that the pipeline's global stages used to write.

fisseq-data-pipeline no longer runs GLOBAL_FEATURE_SELECT or GLOBAL_OVWT, so
`write_global` (and the ``fisseqborn-global`` command) rebuilds their artifacts from the
per-batch outputs::

    <out>/
      feature_select/
        aggregate.parquet        # one row per variant, feature-selected, with meta_impact_score
        blocklist.parquet        # Blocklists.table(): feature, n_batches, n_ok, feature_ok
        pca_components.parquet   # Profiles.pca_loadings, with --pca N
      ovwt_distinguishability/
        global_scores.parquet    # meta_aa_changes, meta_median_<score>, meta_num_experiments
"""

import argparse
import logging
import pathlib
import re
from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Literal

import polars as pl

from . import _pipeline
from .blocklist import Blocklists
from .ovwt import OvwtScores
from .profiles import Profiles

#: `Profiles.feature_select`'s default operations (the pipeline's).
OPERATIONS: tuple[str, ...] = ("variance_threshold", "blocklist", "correlation_threshold")


def write_global(
    pipeline_dir: str | PathLike,
    out: str | PathLike,
    *,
    exclude: "_pipeline.Patterns | None" = None,
    types: Sequence[str] = ("median",),
    passthrough: Sequence[str] = (),
    paired: Mapping[str, str] | None = None,
    min_correlation: float | None = None,
    min_batches: int | None = None,
    missing: Literal["fail", "ignore"] = "fail",
    features: Literal["union", "intersection"] = "union",
    metadata: bool = False,
    operations: Sequence[str] = OPERATIONS,
    corr_threshold: float = 0.9,
    pca: int = 0,
    umap: int = 0,
    ovwt: bool = True,
    scores: Sequence[str] | None = None,
    download_dir: str | PathLike | None = None,
    refresh: bool = False,
    variant_col: str = "meta_aa_changes",
    batch_col: str = "meta_experiment",
) -> dict[str, pathlib.Path]:
    """Aggregate a pipeline run across experiments and write the old global artifacts.

    Feature selection (``feature_select/``):

    1. `Blocklists.from_pipeline` for ``types`` (re-thresholded at ``min_correlation``
       when given) → `Blocklists.table` (``min_batches``, ``missing``), written to
       ``blocklist.parquet``.
    2. `Profiles.from_pipeline` (``types``, ``passthrough``, ``features``, and with
       ``metadata`` the ``output.parquet`` counts) → `keep_features` (the consensus) →
       `median_across_batches` (``paired``; the integer metadata columns are summed) →
       `variant_type` → `feature_select` (``operations``, ``corr_threshold``; skipped when
       ``operations`` is empty) → `impact_score`.
    3. With ``pca=N``, the first ``N`` PC scores (``meta_pc_*``) are added and the loadings
       written to ``pca_components.parquet``; with ``umap=N``, an ``N``-D UMAP
       (``meta_notebook_umap_*``). Both are fitted on the features without non-finite
       values (`Profiles.drop_nonfinite`).

    The result is ``aggregate.parquet``.

    Distinguishability (``ovwt_distinguishability/global_scores.parquet``, unless
    ``ovwt=False``): `OvwtScores.from_pipeline` → ``correct(scores, rescale=False)`` (each
    experiment z-scored against its synonymous controls) → `OvwtScores.per_variant` (the
    median), with columns ``meta_median_<score>`` and ``meta_num_experiments``. ``scores``
    defaults to every column of `fisseqborn.ovwt.SCORES` present.

    ``exclude`` drops batches by name from every input (``fnmatch`` globs or compiled
    regexes). ``pipeline_dir`` can be remote (``"user@host:/path"``): only the files above
    are copied with scp, into ``download_dir`` (default: a temporary directory), reusing
    files already there unless ``refresh``. Returns the paths written, keyed by file stem.
    """
    out = pathlib.Path(out)
    pipeline_dir = _pipeline.source(pipeline_dir, download_dir, refresh)
    fs_dir = out / "feature_select"
    written: dict[str, pathlib.Path] = {}

    blocklists = Blocklists.from_pipeline(
        pipeline_dir, types=types, exclude=exclude, batch_col=batch_col
    )
    if min_correlation is not None:
        blocklists = blocklists.rethreshold(min_correlation)
    table = blocklists.table(min_batches, missing=missing)
    ok = table.filter("feature_ok").get_column("feature").to_list()

    profiles = Profiles.from_pipeline(
        pipeline_dir,
        types,
        passthrough,
        exclude=exclude,
        features=features,
        metadata=metadata,
        variant_col=variant_col,
        batch_col=batch_col,
    )
    schema = profiles.schema
    sum_cols = [
        c for c in profiles.meta if c not in (variant_col, batch_col) and schema[c].is_integer()
    ]
    aggregate = (
        profiles.keep_features(ok)
        .median_across_batches(paired=paired, sum_cols=sum_cols)
        .variant_type()
    )
    if operations:
        aggregate = aggregate.feature_select(operations, corr_threshold=corr_threshold)
    aggregate = aggregate.impact_score()
    if pca:
        embedded = aggregate.drop_nonfinite().pca(pca)
        written["pca_components"] = fs_dir / "pca_components.parquet"
        embedded.save_pca_loadings(written["pca_components"])
        aggregate = _add_columns(aggregate, embedded, prefix="meta_pc_")
    if umap:
        embedded = aggregate.drop_nonfinite().umap(n_components=umap)
        aggregate = _add_columns(aggregate, embedded, prefix="meta_notebook_umap_")

    written["aggregate"] = fs_dir / "aggregate.parquet"
    aggregate.save(written["aggregate"])
    written["blocklist"] = fs_dir / "blocklist.parquet"
    table.write_parquet(written["blocklist"])

    if ovwt:
        ovwt_scores = OvwtScores.from_pipeline(
            pipeline_dir, exclude=exclude, variant_col=variant_col, batch_col=batch_col
        )
        scores = ovwt_scores._score_columns(scores)
        corrected = [f"{s}_corrected" for s in scores]
        global_scores = (
            ovwt_scores.correct(scores, rescale=False)
            .per_variant(corrected, n_col="meta_num_experiments")
            .select(
                variant_col,
                *[pl.col(c).alias(f"meta_median_{s}") for s, c in zip(scores, corrected)],
                "meta_num_experiments",
            )
        )
        written["global_scores"] = out / "ovwt_distinguishability" / "global_scores.parquet"
        global_scores.save(written["global_scores"])
    return written


def _add_columns(profiles: Profiles, embedded: Profiles, *, prefix: str) -> Profiles:
    """``profiles`` with the ``prefix`` columns of ``embedded`` (same rows) added."""
    columns = [c for c in embedded.columns if c.startswith(prefix)]
    return profiles.with_columns(embedded.df.select(columns).get_columns())


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
            "Aggregate a fisseq-data-pipeline run across experiments and write the "
            "artifacts the pipeline's global stages used to: feature_select/"
            "{aggregate,blocklist,pca_components}.parquet and "
            "ovwt_distinguishability/global_scores.parquet."
        ),
    )
    parser.add_argument(
        "pipeline_dir", help="pipeline output directory, local or remote (user@host:/path)"
    )
    parser.add_argument("--out", type=pathlib.Path, required=True, help="output directory")
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
        "--types", nargs="+", default=["median"], help="aggregate types (default: median)"
    )
    parser.add_argument(
        "--passthrough",
        nargs="*",
        default=[],
        help="passthrough aggregate types to carry along, e.g. KSnegLogP",
    )
    parser.add_argument(
        "--paired",
        action="append",
        default=[],
        metavar="VALUE:COMPANION",
        help="take COMPANION from the batch holding VALUE's median, e.g. median:KSnegLogP",
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        help="recompute feature_ok as median_r >= this (default: the pipeline's feature_ok)",
    )
    parser.add_argument(
        "--min-batches",
        type=int,
        help="a feature must be OK in at least this many batches (default: all of them)",
    )
    parser.add_argument(
        "--missing",
        choices=["fail", "ignore"],
        default="fail",
        help="a feature missing from a batch's blocklist fails there, or is judged only on "
        "the batches reporting it (default: fail)",
    )
    parser.add_argument(
        "--features",
        choices=["union", "intersection"],
        default="union",
        help="keep features of any batch, or only those in every batch (default: union)",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="join the meta_ counts of each batch's output.parquet (summed across batches)",
    )
    parser.add_argument(
        "--operations",
        nargs="*",
        default=list(OPERATIONS),
        help="pycytominer feature_select operations; pass none to skip feature selection",
    )
    parser.add_argument("--corr-threshold", type=float, default=0.9)
    parser.add_argument(
        "--pca", type=int, default=0, metavar="N", help="add N PCs and write pca_components"
    )
    parser.add_argument("--umap", type=int, default=0, metavar="N", help="add an N-D UMAP")
    parser.add_argument(
        "--scores",
        nargs="+",
        help="OvWT score columns (default: auroc_pooled, auroc_median_barcode, auroc_median_fold)",
    )
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
    parser.add_argument(
        "--no-ovwt",
        dest="ovwt",
        action="store_false",
        help="skip ovwt_distinguishability/global_scores.parquet",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of the ``fisseqborn-global`` command."""
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        paired = _paired(args.paired)
    except argparse.ArgumentTypeError as err:
        parser.error(str(err))
    exclude = [*args.exclude, *(re.compile(r) for r in args.exclude_regex)]
    written = write_global(
        args.pipeline_dir,
        args.out,
        exclude=exclude or None,
        types=args.types,
        passthrough=args.passthrough,
        paired=paired,
        min_correlation=args.min_correlation,
        min_batches=args.min_batches,
        missing=args.missing,
        features=args.features,
        metadata=args.metadata,
        operations=args.operations,
        corr_threshold=args.corr_threshold,
        pca=args.pca,
        umap=args.umap,
        ovwt=args.ovwt,
        scores=args.scores,
        download_dir=args.download_dir,
        refresh=args.refresh,
    )
    for path in written.values():
        print(path)


if __name__ == "__main__":
    main()
