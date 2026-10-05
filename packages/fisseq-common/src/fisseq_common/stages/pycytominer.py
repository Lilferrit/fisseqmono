"""pycytominer feature selection of a per-variant aggregate table.

The data pipeline's FINALIZE_FEATURE_SELECT runs it after the reproducibility blocklist, with
:data:`DEFAULT_OPERATIONS`: low-variance removal, pycytominer's own blocklist of CellProfiler
feature names, and redundancy removal by pairwise correlation. Those filters are written for
named morphological features, so the embeddings pipeline doesn't run it; there the
reproducibility verdict is the only selection.
"""

from typing import Sequence

import polars as pl
import pycytominer

from fisseq_common.schema import FEATURE_SELECTOR

#: The data pipeline's operations.
DEFAULT_OPERATIONS = ("variance_threshold", "blocklist", "correlation_threshold")


def pyc_feature_select(
    agg_df: pl.DataFrame,
    operations: Sequence[str] = DEFAULT_OPERATIONS,
    feature_selector: pl.Expr = FEATURE_SELECTOR,
) -> pl.DataFrame:
    """
    Select informative features of ``agg_df`` with :func:`pycytominer.feature_select`.

    Parameters
    ----------
    agg_df : pl.DataFrame
        Per-variant aggregate table.
    operations : sequence of str
        pycytominer operations, applied in order. Empty returns ``agg_df`` unchanged.
    feature_selector : pl.Expr
        Which columns are features. Defaults to every non-``meta_`` column.

    Returns
    -------
    pl.DataFrame
        ``agg_df`` restricted to the features that pass every operation; the other columns
        are kept unchanged.
    """
    if not operations:
        return agg_df
    selected = pycytominer.feature_select(
        profiles=agg_df.to_pandas(),
        features=agg_df.select(feature_selector).columns,
        image_features=False,
        samples="all",
        operation=list(operations),
    )
    return pl.from_pandas(selected)
