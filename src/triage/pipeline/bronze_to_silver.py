"""Bronze → Silver: parse, clean, deduplicate, and quarantine invalid rows."""

import logging
from pathlib import Path

import polars as pl

from triage import quality

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent.parent
BRONZE_PATH = _ROOT / "data" / "bronze" / "tickets.jsonl"
SILVER_PATH = _ROOT / "data" / "silver"

# Ordered list of (rejection_reason_label, quality_function).
# First matching reason wins for rows that fail multiple checks.
_CHECKS: list[tuple[str, str]] = [
    ("null_ticket_id", "null_ticket_id"),
    ("invalid_timestamp", "_invalid_timestamp"),  # handled inline
    ("empty_body", "empty_body"),
    ("invalid_category", "invalid_category"),
    ("invalid_urgency", "invalid_urgency"),
    ("future_timestamp", "future_timestamp"),
    ("duplicate_ticket_id", "duplicate_ticket_id"),
    ("near_duplicate", "near_duplicate"),
]


def read_bronze(path: Path = BRONZE_PATH) -> pl.DataFrame:
    return pl.read_ndjson(path)


def clean(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return ``(clean_df, rejected_df)``.

    ``rejected_df`` has all original columns plus a ``rejection_reason`` column.
    """
    missing = quality.check_required_columns(df)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Trim whitespace on all string columns
    str_cols = [c for c, d in zip(df.columns, df.dtypes) if d == pl.String]
    if str_cols:
        df = df.with_columns([pl.col(c).str.strip_chars() for c in str_cols])

    # Parse created_at string → Datetime (strict=False → unparseable → null).
    # Explicit chrono format handles optional fractional seconds and UTC offset (+00:00).
    df = df.with_columns(
        pl.col("created_at")
        .str.to_datetime("%Y-%m-%dT%H:%M:%S%.f%:z", strict=False)
        .alias("created_at")
    )

    # Build masks; _invalid_timestamp is the parsed-null check
    masks: dict[str, pl.Series] = {
        "null_ticket_id": quality.null_ticket_id(df),
        "invalid_timestamp": df["created_at"].is_null(),
        "empty_body": quality.empty_body(df),
        "invalid_category": quality.invalid_category(df),
        "invalid_urgency": quality.invalid_urgency(df),
        "future_timestamp": quality.future_timestamp(df),
        "duplicate_ticket_id": quality.duplicate_ticket_id(df),
        "near_duplicate": quality.near_duplicate(df),
    }

    # Stack masks as temporary columns, build rejection_reason with when/then chain
    df = df.hstack([s.rename(name) for name, s in masks.items()])

    names = list(masks.keys())
    chain = pl.when(pl.col(names[0])).then(pl.lit(names[0]))
    for name in names[1:]:
        chain = chain.when(pl.col(name)).then(pl.lit(name))
    chain = chain.otherwise(pl.lit(None, dtype=pl.String)).alias("rejection_reason")

    df = df.with_columns(chain).drop(names)

    rejected = df.filter(pl.col("rejection_reason").is_not_null())
    clean_df = df.filter(pl.col("rejection_reason").is_null()).drop("rejection_reason")
    return clean_df, rejected


def write_silver(
    clean_df: pl.DataFrame,
    rejected_df: pl.DataFrame,
    path: Path = SILVER_PATH,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    clean_df.write_parquet(path / "tickets.parquet")
    rejected_df.write_parquet(path / "rejected.parquet")
    log.info("Wrote %d clean rows → %s/tickets.parquet", len(clean_df), path)
    log.info("Wrote %d rejected rows → %s/rejected.parquet", len(rejected_df), path)
