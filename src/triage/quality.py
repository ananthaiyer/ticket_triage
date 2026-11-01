"""Data quality checks for ticket DataFrames.

Each mask function returns a boolean Series (True = row fails the check).
The caller decides what to do with failing rows (quarantine, log, etc.).
"""

from datetime import datetime, timezone

import polars as pl

REQUIRED_COLUMNS = ["ticket_id", "created_at", "subject", "body", "category", "urgency"]
VALID_CATEGORIES = frozenset(["hardware", "software", "access", "network", "email", "other"])
VALID_URGENCIES = frozenset(["low", "medium", "high"])


def check_required_columns(df: pl.DataFrame) -> list[str]:
    """Return names of required columns missing from df."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def null_ticket_id(df: pl.DataFrame) -> pl.Series:
    return df["ticket_id"].is_null()


def empty_body(df: pl.DataFrame) -> pl.Series:
    body = df["body"]
    if body.dtype == pl.Null:
        return pl.Series("empty_body", [True] * len(df), dtype=pl.Boolean)
    return body.is_null() | (body.str.strip_chars() == "")


def invalid_category(df: pl.DataFrame) -> pl.Series:
    return ~df["category"].is_in(list(VALID_CATEGORIES))


def invalid_urgency(df: pl.DataFrame) -> pl.Series:
    return ~df["urgency"].is_in(list(VALID_URGENCIES))


def future_timestamp(df: pl.DataFrame) -> pl.Series:
    """True where created_at (Datetime column) is in the future. Nulls are not flagged."""
    col = df["created_at"]
    if not isinstance(col.dtype, pl.Datetime):
        return pl.Series("future_timestamp", [False] * len(df), dtype=pl.Boolean)
    now = datetime.now(timezone.utc)
    tz = col.dtype.time_zone
    cmp = now if tz else now.replace(tzinfo=None)
    return col.is_not_null() & (col > cmp)


def duplicate_ticket_id(df: pl.DataFrame) -> pl.Series:
    """True for non-first occurrences of a ticket_id. Nulls are excluded."""
    not_null = df["ticket_id"].is_not_null()
    is_dup = df["ticket_id"].is_duplicated() & ~df["ticket_id"].is_first_distinct()
    return not_null & is_dup


def near_duplicate(df: pl.DataFrame) -> pl.Series:
    """True for non-first occurrences of the same (subject, body) pair."""
    key = df["subject"].fill_null("") + "\x00" + df["body"].fill_null("")
    return key.is_duplicated() & ~key.is_first_distinct()
