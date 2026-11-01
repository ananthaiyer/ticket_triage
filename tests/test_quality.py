"""Tests for triage.quality mask functions."""

from datetime import datetime, timezone

import polars as pl

from triage import quality

_DTYPES = {
    "ticket_id": pl.String,
    "created_at": pl.Datetime("us", "UTC"),
    "subject": pl.String,
    "body": pl.String,
    "category": pl.String,
    "urgency": pl.String,
}


def _df(**cols) -> pl.DataFrame:
    """Build a single-row DataFrame with explicit dtypes (avoids Null-inference edge cases)."""
    return pl.DataFrame(
        [pl.Series(k, [v], dtype=_DTYPES[k]) for k, v in cols.items()]
    )


def _base(**overrides):
    base = {
        "ticket_id": "abc-1",
        "created_at": datetime(2024, 6, 1, tzinfo=timezone.utc),
        "subject": "Laptop won't boot",
        "body": "Screen stays black after pressing power.",
        "category": "hardware",
        "urgency": "high",
    }
    base.update(overrides)
    return _df(**base)


class TestNullTicketId:
    def test_flags_null(self):
        df = pl.concat([_base(ticket_id=None), _base(ticket_id="abc-2")])
        assert quality.null_ticket_id(df).to_list() == [True, False]

    def test_clean_passes(self):
        assert not quality.null_ticket_id(_base()).any()


class TestEmptyBody:
    def test_flags_empty_string(self):
        assert quality.empty_body(_base(body="")).all()

    def test_flags_whitespace_only(self):
        assert quality.empty_body(_base(body="   ")).all()

    def test_flags_null(self):
        assert quality.empty_body(_base(body=None)).all()

    def test_clean_passes(self):
        assert not quality.empty_body(_base()).any()


class TestInvalidCategory:
    def test_flags_unknown_value(self):
        assert quality.invalid_category(_base(category="billing")).all()

    def test_all_valid_categories_pass(self):
        for cat in quality.VALID_CATEGORIES:
            assert not quality.invalid_category(_base(category=cat)).any(), cat


class TestInvalidUrgency:
    def test_flags_unknown_value(self):
        assert quality.invalid_urgency(_base(urgency="urgent")).all()

    def test_all_valid_urgencies_pass(self):
        for urg in quality.VALID_URGENCIES:
            assert not quality.invalid_urgency(_base(urgency=urg)).any(), urg


class TestFutureTimestamp:
    def test_flags_far_future(self):
        far_future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        assert quality.future_timestamp(_base(created_at=far_future)).all()

    def test_past_passes(self):
        past = datetime(2020, 1, 1, tzinfo=timezone.utc)
        assert not quality.future_timestamp(_base(created_at=past)).any()

    def test_null_not_flagged(self):
        assert not quality.future_timestamp(_base(created_at=None)).any()


class TestDuplicateTicketId:
    def test_flags_second_occurrence(self):
        df = pl.concat([_base(ticket_id="dup"), _base(ticket_id="dup")])
        assert quality.duplicate_ticket_id(df).to_list() == [False, True]

    def test_unique_ids_pass(self):
        df = pl.concat([_base(ticket_id="a"), _base(ticket_id="b")])
        assert not quality.duplicate_ticket_id(df).any()

    def test_null_ids_not_flagged(self):
        df = pl.concat([_base(ticket_id=None), _base(ticket_id=None)])
        assert not quality.duplicate_ticket_id(df).any()


class TestNearDuplicate:
    def test_flags_same_subject_and_body(self):
        t1 = _base(ticket_id="a", subject="Help", body="Need help please")
        t2 = _base(ticket_id="b", subject="Help", body="Need help please")
        df = pl.concat([t1, t2])
        assert quality.near_duplicate(df).to_list() == [False, True]

    def test_different_body_not_flagged(self):
        df = pl.concat([_base(ticket_id="a", body="Issue A"), _base(ticket_id="b", body="Issue B")])
        assert not quality.near_duplicate(df).any()
