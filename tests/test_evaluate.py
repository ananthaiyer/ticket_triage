"""Tests for triage.evaluate pure helper functions — no API calls needed."""


import polars as pl

from triage.evaluate import compute_metrics, find_top_disagreements, write_report


class TestComputeMetrics:
    def test_perfect_agreement(self):
        labels = ["a", "b", "c", "a", "b"]
        m = compute_metrics(labels, labels)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0

    def test_all_wrong_accuracy_zero(self):
        true = ["a", "b", "c"]
        pred = ["b", "c", "a"]
        m = compute_metrics(true, pred)
        assert m["accuracy"] == 0.0

    def test_returns_rounded_floats(self):
        true = ["a", "a", "b"]
        pred = ["a", "b", "b"]
        m = compute_metrics(true, pred)
        assert isinstance(m["accuracy"], float)
        assert isinstance(m["macro_f1"], float)


class TestFindTopDisagreements:
    def _eval_df(self, ids, cats, urgs) -> pl.DataFrame:
        return pl.DataFrame({
            "ticket_id": ids,
            "subject": [f"Subject {i}" for i in ids],
            "category": cats,
            "urgency": urgs,
        })

    def test_no_disagreements_returns_empty(self):
        df = self._eval_df(["t1"], ["hardware"], ["low"])
        preds = [{"ticket_id": "t1", "category": "hardware", "urgency": "low"}]
        assert find_top_disagreements(df, preds, preds) == []

    def test_single_disagreement_returned(self):
        df = self._eval_df(["t1"], ["hardware"], ["low"])
        baseline = [{"ticket_id": "t1", "category": "software", "urgency": "low"}]
        llm_r = [{"ticket_id": "t1", "category": "hardware", "urgency": "low"}]
        result = find_top_disagreements(df, baseline, llm_r)
        assert len(result) == 1
        assert result[0]["ticket_id"] == "t1"

    def test_orders_by_score_descending(self):
        df = self._eval_df(["t1", "t2"], ["hardware", "software"], ["low", "medium"])
        # t1: disagrees on both (score 2); t2: disagrees on category only (score 1)
        baseline = [
            {"ticket_id": "t1", "category": "software", "urgency": "high"},
            {"ticket_id": "t2", "category": "hardware", "urgency": "medium"},
        ]
        llm_r = [
            {"ticket_id": "t1", "category": "hardware", "urgency": "low"},
            {"ticket_id": "t2", "category": "software", "urgency": "medium"},
        ]
        result = find_top_disagreements(df, baseline, llm_r)
        assert result[0]["ticket_id"] == "t1"

    def test_respects_n_limit(self):
        ids = [f"t{i}" for i in range(10)]
        cats = ["hardware"] * 10
        urgs = ["low"] * 10
        df = self._eval_df(ids, cats, urgs)
        baseline = [{"ticket_id": f"t{i}", "category": "software", "urgency": "low"}
                    for i in range(10)]
        llm_r = [{"ticket_id": f"t{i}", "category": "hardware", "urgency": "low"}
                 for i in range(10)]
        result = find_top_disagreements(df, baseline, llm_r, n=3)
        assert len(result) == 3


class TestWriteReport:
    def _metrics(self) -> dict:
        return {
            "baseline": {
                "category": {"accuracy": 0.82, "macro_f1": 0.80},
                "urgency": {"accuracy": 0.74, "macro_f1": 0.71},
            },
            "llm": {
                "category": {"accuracy": 0.88, "macro_f1": 0.86},
                "urgency": {"accuracy": 0.78, "macro_f1": 0.75},
            },
        }

    def test_creates_evaluation_md(self, tmp_path):
        write_report(self._metrics(), avg_latency=1.5, cost_per_1k=0.12,
                     disagreements=[], reports_path=tmp_path)
        assert (tmp_path / "evaluation.md").exists()

    def test_report_contains_metrics(self, tmp_path):
        write_report(self._metrics(), avg_latency=1.5, cost_per_1k=0.12,
                     disagreements=[], reports_path=tmp_path)
        content = (tmp_path / "evaluation.md").read_text()
        assert "0.820" in content
        assert "0.880" in content

    def test_report_contains_latency(self, tmp_path):
        write_report(self._metrics(), avg_latency=2.34, cost_per_1k=0.20,
                     disagreements=[], reports_path=tmp_path)
        content = (tmp_path / "evaluation.md").read_text()
        assert "2.34" in content

    def test_report_contains_cost(self, tmp_path):
        write_report(self._metrics(), avg_latency=1.0, cost_per_1k=0.456,
                     disagreements=[], reports_path=tmp_path)
        content = (tmp_path / "evaluation.md").read_text()
        assert "0.456" in content

    def test_disagreements_appear_in_report(self, tmp_path):
        d = [{
            "ticket_id": "abc-def-123",
            "subject": "VPN keeps dropping",
            "true_category": "network",
            "true_urgency": "high",
            "baseline_category": "software",
            "baseline_urgency": "high",
            "llm_category": "network",
            "llm_urgency": "high",
            "score": 1,
        }]
        write_report(self._metrics(), 1.0, 0.1, d, reports_path=tmp_path)
        content = (tmp_path / "evaluation.md").read_text()
        assert "VPN keeps dropping" in content
