"""Tests for triage.eval exclusion helper."""

from pathlib import Path

import polars as pl

from triage.eval import exclude_eval_tickets, load_eval_ids


def _write_eval_csv(path: Path, ticket_ids: list[str]) -> None:
    pl.DataFrame(
        {"ticket_id": ticket_ids, "checked": [""] * len(ticket_ids)}
    ).write_csv(path)


def _tickets(*ids: str) -> pl.DataFrame:
    return pl.DataFrame({"ticket_id": list(ids), "body": [f"body {i}" for i in ids]})


class TestLoadEvalIds:
    def test_returns_ids_as_frozenset(self, tmp_path):
        csv = tmp_path / "eval.csv"
        _write_eval_csv(csv, ["t1", "t2", "t3"])
        ids = load_eval_ids(csv)
        assert ids == frozenset({"t1", "t2", "t3"})

    def test_empty_file_returns_empty_set(self, tmp_path):
        csv = tmp_path / "eval.csv"
        _write_eval_csv(csv, [])
        assert load_eval_ids(csv) == frozenset()


class TestExcludeEvalTickets:
    def test_removes_eval_ids(self, tmp_path):
        csv = tmp_path / "eval.csv"
        _write_eval_csv(csv, ["t1", "t3"])
        df = _tickets("t1", "t2", "t3", "t4")
        result = exclude_eval_tickets(df, eval_path=csv)
        assert result["ticket_id"].to_list() == ["t2", "t4"]

    def test_empty_eval_returns_full_df(self, tmp_path):
        csv = tmp_path / "eval.csv"
        _write_eval_csv(csv, [])
        df = _tickets("t1", "t2")
        result = exclude_eval_tickets(df, eval_path=csv)
        assert len(result) == 2

    def test_no_overlap_returns_full_df(self, tmp_path):
        csv = tmp_path / "eval.csv"
        _write_eval_csv(csv, ["x1", "x2"])
        df = _tickets("t1", "t2", "t3")
        result = exclude_eval_tickets(df, eval_path=csv)
        assert len(result) == 3

    def test_all_overlap_returns_empty_df(self, tmp_path):
        csv = tmp_path / "eval.csv"
        _write_eval_csv(csv, ["t1", "t2"])
        df = _tickets("t1", "t2")
        result = exclude_eval_tickets(df, eval_path=csv)
        assert len(result) == 0

    def test_row_count_decreases_by_overlap(self, tmp_path):
        csv = tmp_path / "eval.csv"
        _write_eval_csv(csv, ["t2", "t4"])
        df = _tickets("t1", "t2", "t3", "t4", "t5")
        result = exclude_eval_tickets(df, eval_path=csv)
        assert len(result) == 3
