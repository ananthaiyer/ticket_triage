"""Tests for triage.baseline — train / predict without loading real data files."""

import polars as pl

from triage.baseline import load, predict, save, train
from triage.quality import VALID_CATEGORIES, VALID_URGENCIES


def _synthetic_df(n_per_class: int = 4) -> pl.DataFrame:
    """Small training set: n_per_class rows for every (category, urgency) pair."""
    rows = []
    for cat in sorted(VALID_CATEGORIES):
        for urg in sorted(VALID_URGENCIES):
            for i in range(n_per_class):
                rows.append({
                    "ticket_id": f"{cat}-{urg}-{i}",
                    "subject": f"{cat} problem number {i}",
                    "body": (
                        f"I have a {cat} issue and it feels {urg}. "
                        f"Please help with this {cat} situation urgently {i}."
                    ),
                    "category": cat,
                    "urgency": urg,
                })
    return pl.DataFrame(rows)


class TestTrain:
    def test_returns_two_pipelines(self):
        cat_m, urg_m = train(_synthetic_df())
        assert cat_m is not None
        assert urg_m is not None

    def test_models_have_predict(self):
        cat_m, urg_m = train(_synthetic_df())
        assert callable(cat_m.predict)
        assert callable(urg_m.predict)


class TestPredict:
    def setup_method(self):
        df = _synthetic_df()
        self.cat_m, self.urg_m = train(df)

    def test_length_matches_input(self):
        tickets = [{"ticket_id": f"t{i}", "subject": "test", "body": "test"} for i in range(5)]
        results = predict(tickets, self.cat_m, self.urg_m)
        assert len(results) == 5

    def test_output_has_required_keys(self):
        ticket = {"ticket_id": "t1", "subject": "broken laptop", "body": "screen cracked"}
        results = predict([ticket], self.cat_m, self.urg_m)
        assert set(results[0].keys()) >= {"ticket_id", "category", "urgency"}

    def test_category_is_valid(self):
        tickets = [{"ticket_id": "t1", "subject": "network down", "body": "VPN not working at all"}]
        results = predict(tickets, self.cat_m, self.urg_m)
        assert results[0]["category"] in VALID_CATEGORIES

    def test_urgency_is_valid(self):
        tickets = [{"ticket_id": "t1", "subject": "printer jammed", "body": "keeps jamming"}]
        results = predict(tickets, self.cat_m, self.urg_m)
        assert results[0]["urgency"] in VALID_URGENCIES

    def test_ticket_id_preserved(self):
        results = predict([{"ticket_id": "my-special-id", "subject": "x", "body": "y"}],
                          self.cat_m, self.urg_m)
        assert results[0]["ticket_id"] == "my-special-id"

    def test_missing_subject_and_body_does_not_crash(self):
        results = predict([{"ticket_id": "t1"}], self.cat_m, self.urg_m)
        assert results[0]["category"] in VALID_CATEGORIES


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        df = _synthetic_df()
        cat_m, urg_m = train(df)
        save(cat_m, urg_m, path=tmp_path)
        cat_m2, urg_m2 = load(path=tmp_path)
        tickets = [{"ticket_id": "t1", "subject": "printer", "body": "broken printer hardware"}]
        r1 = predict(tickets, cat_m, urg_m)
        r2 = predict(tickets, cat_m2, urg_m2)
        assert r1[0]["category"] == r2[0]["category"]
        assert r1[0]["urgency"] == r2[0]["urgency"]
