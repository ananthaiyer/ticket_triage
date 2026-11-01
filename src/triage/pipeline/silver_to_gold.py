"""Silver → Gold: aggregate tables for reporting."""

import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent.parent
GOLD_PATH = _ROOT / "data" / "gold"


def aggregate(df: pl.DataFrame, path: Path = GOLD_PATH) -> None:
    path.mkdir(parents=True, exist_ok=True)

    # Tickets per category per day
    cat_day = (
        df.with_columns(pl.col("created_at").dt.date().alias("date"))
        .group_by(["category", "date"])
        .agg(pl.len().alias("count"))
        .sort(["category", "date"])
    )
    cat_day.write_parquet(path / "tickets_per_category_day.parquet")
    log.info("Wrote %d rows → gold/tickets_per_category_day.parquet", len(cat_day))

    # Tickets per urgency per week (week starts Monday)
    urg_week = (
        df.with_columns(pl.col("created_at").dt.truncate("1w").alias("week_start"))
        .group_by(["urgency", "week_start"])
        .agg(pl.len().alias("count"))
        .sort(["urgency", "week_start"])
    )
    urg_week.write_parquet(path / "tickets_per_urgency_week.parquet")
    log.info("Wrote %d rows → gold/tickets_per_urgency_week.parquet", len(urg_week))
