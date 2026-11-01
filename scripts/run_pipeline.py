"""Orchestrate the bronze → silver → gold pipeline."""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running as `python scripts/run_pipeline.py` without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl

from triage.pipeline import bronze_to_silver, silver_to_gold

REJECTION_THRESHOLD = 0.20


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
        })


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


log = logging.getLogger(__name__)


def main() -> None:
    _configure_logging()
    log.info("Pipeline started")

    bronze_df = bronze_to_silver.read_bronze()
    rows_in = len(bronze_df)
    log.info("Read %d rows from bronze", rows_in)

    clean_df, rejected_df = bronze_to_silver.clean(bronze_df)
    rows_out = len(clean_df)
    rows_rejected = len(rejected_df)
    rejection_rate = rows_rejected / rows_in if rows_in > 0 else 0.0

    if rows_rejected > 0:
        reason_counts = (
            rejected_df.group_by("rejection_reason")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        for row in reason_counts.iter_rows(named=True):
            log.info("  rejected %d rows: %s", row["count"], row["rejection_reason"])

    log.info(
        "Silver summary — in: %d, out: %d, rejected: %d (%.1f%%)",
        rows_in, rows_out, rows_rejected, rejection_rate * 100,
    )

    if rejection_rate > REJECTION_THRESHOLD:
        log.error(
            "Rejection rate %.1f%% exceeds %.0f%% threshold — aborting",
            rejection_rate * 100,
            REJECTION_THRESHOLD * 100,
        )
        sys.exit(1)

    bronze_to_silver.write_silver(clean_df, rejected_df)
    silver_to_gold.aggregate(clean_df)

    log.info("Pipeline complete")


if __name__ == "__main__":
    main()
