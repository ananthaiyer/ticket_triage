"""Utilities for managing the held-out evaluation set.

The eval set is fixed at creation time and must never be used for training.
Call ``exclude_eval_tickets`` on any DataFrame before passing it to a model.
"""

from pathlib import Path

import polars as pl

_ROOT = Path(__file__).parent.parent.parent
EVAL_PATH = _ROOT / "data" / "eval" / "eval_set.csv"


def load_eval_ids(path: Path = EVAL_PATH) -> frozenset[str]:
    """Return the ticket_ids in the eval set as an immutable set."""
    return frozenset(pl.read_csv(path)["ticket_id"].to_list())


def exclude_eval_tickets(df: pl.DataFrame, eval_path: Path = EVAL_PATH) -> pl.DataFrame:
    """Return *df* with all eval-set ticket_ids removed."""
    eval_ids = load_eval_ids(eval_path)
    if not eval_ids:
        return df
    return df.filter(~pl.col("ticket_id").is_in(list(eval_ids)))
