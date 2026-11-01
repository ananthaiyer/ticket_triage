"""Tests for bronze → silver cleaning and silver → gold aggregation."""

from datetime import datetime, timezone

import polars as pl
import pytest

from triage.pipeline import bronze_to_silver, silver_to_gold

# Schema for bronze-style DataFrames (created_at as String, like JSONL input)
_SCHEMA = {
    "ticket_id": pl.String,
    "created_at": pl.String,
    "subject": pl.String,
    "body": pl.String,
    "category": pl.String,
    "urgency": pl.String,
}


def _ticket(**overrides) -> dict:
    base = {
        "ticket_id": "t1",
        "created_at": "2024-06-01T10:00:00+00:00",
        "subject": "Printer jammed in room 3B",
        "body": "The printer keeps jamming every time I print more than 5 pages.",
        "category": "hardware",
        "urgency": "low",
    }
    base.update(overrides)
    return base


def _make_bronze(*rows: dict) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=_SCHEMA)


class TestBronzeToSilver:
    def test_clean_ticket_passes(self):
        df = _make_bronze(_ticket())
        clean, rejected = bronze_to_silver.clean(df)
        assert len(clean) == 1
        assert len(rejected) == 0

    def test_null_ticket_id_rejected(self):
        df = _make_bronze(_ticket(ticket_id=None))
        _, rejected = bronze_to_silver.clean(df)
        assert rejected["rejection_reason"][0] == "null_ticket_id"

    def test_empty_body_rejected(self):
        df = _make_bronze(_ticket(ticket_id="t2", body=""))
        _, rejected = bronze_to_silver.clean(df)
        assert rejected["rejection_reason"][0] == "empty_body"

    def test_invalid_category_rejected(self):
        df = _make_bronze(_ticket(ticket_id="t3", category="billing"))
        _, rejected = bronze_to_silver.clean(df)
        assert rejected["rejection_reason"][0] == "invalid_category"

    def test_invalid_urgency_rejected(self):
        df = _make_bronze(_ticket(ticket_id="t4", urgency="critical"))
        _, rejected = bronze_to_silver.clean(df)
        assert rejected["rejection_reason"][0] == "invalid_urgency"

    def test_null_timestamp_rejected(self):
        df = _make_bronze(_ticket(ticket_id="t5", created_at=None))
        _, rejected = bronze_to_silver.clean(df)
        assert rejected["rejection_reason"][0] == "invalid_timestamp"

    def test_duplicate_ticket_id_rejected(self):
        df = _make_bronze(
            _ticket(ticket_id="dup"),
            _ticket(ticket_id="dup", subject="Other subject", body="Other body"),
        )
        _, rejected = bronze_to_silver.clean(df)
        assert len(rejected) == 1
        assert rejected["rejection_reason"][0] == "duplicate_ticket_id"

    def test_near_duplicate_rejected(self):
        df = _make_bronze(
            _ticket(ticket_id="a"),
            _ticket(ticket_id="b"),  # same subject+body, different id
        )
        _, rejected = bronze_to_silver.clean(df)
        assert len(rejected) == 1
        assert rejected["rejection_reason"][0] == "near_duplicate"

    def test_whitespace_trimmed(self):
        df = _make_bronze(_ticket(subject="  Broken printer  ", body="  Fix it please  "))
        clean, _ = bronze_to_silver.clean(df)
        assert clean["subject"][0] == "Broken printer"
        assert clean["body"][0] == "Fix it please"

    def test_first_reason_wins(self):
        # null ticket_id AND empty body: null_ticket_id should win
        df = _make_bronze(_ticket(ticket_id=None, body=""))
        _, rejected = bronze_to_silver.clean(df)
        assert rejected["rejection_reason"][0] == "null_ticket_id"

    def test_missing_required_column_raises(self):
        df = pl.DataFrame({"ticket_id": ["t1"], "body": ["some text"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            bronze_to_silver.clean(df)

    def test_rejection_rate_above_threshold(self):
        good = [_ticket(ticket_id=f"g{i}") for i in range(8)]
        bad = [_ticket(ticket_id=None)] * 3  # 3/11 ≈ 27%
        df = _make_bronze(*good, *bad)
        clean, rejected = bronze_to_silver.clean(df)
        assert len(rejected) / len(df) > 0.20


class TestSilverToGold:
    def _silver_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "ticket_id": f"t{i}",
                    "created_at": datetime(2024, 6, 1 + i, tzinfo=timezone.utc),
                    "subject": f"Ticket {i}",
                    "body": f"Body {i}",
                    "category": "hardware" if i % 2 == 0 else "software",
                    "urgency": "low" if i < 3 else "high",
                }
                for i in range(5)
            ]
        )

    def test_category_day_columns(self, tmp_path):
        silver_to_gold.aggregate(self._silver_df(), path=tmp_path)
        result = pl.read_parquet(tmp_path / "tickets_per_category_day.parquet")
        assert set(result.columns) == {"category", "date", "count"}

    def test_urgency_week_columns(self, tmp_path):
        silver_to_gold.aggregate(self._silver_df(), path=tmp_path)
        result = pl.read_parquet(tmp_path / "tickets_per_urgency_week.parquet")
        assert set(result.columns) == {"urgency", "week_start", "count"}

    def test_total_count_preserved(self, tmp_path):
        df = self._silver_df()
        silver_to_gold.aggregate(df, path=tmp_path)
        result = pl.read_parquet(tmp_path / "tickets_per_category_day.parquet")
        assert result["count"].sum() == len(df)
