"""Tests for triage.llm — API is mocked so no key is required."""

import json
from unittest.mock import MagicMock, patch

import pytest

from triage.llm import LLMUnavailableError, classify

_TICKET = {"ticket_id": "t1", "subject": "Broken screen", "body": "Screen cracked after drop."}

_VALID_JSON = json.dumps({
    "category": "hardware",
    "urgency": "low",
    "reason": "A cracked screen is a physical hardware failure.",
    "suggested_reply": "Thank you for reporting this. We will arrange a repair shortly.",
})


def _mock_client(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.usage = MagicMock(input_tokens=120, output_tokens=60)
    client = MagicMock()
    client.messages.create.return_value = response
    return client


class TestClassifySuccess:
    def test_returns_expected_category(self):
        result = classify(_TICKET, client=_mock_client(_VALID_JSON))
        assert result["category"] == "hardware"

    def test_returns_expected_urgency(self):
        result = classify(_TICKET, client=_mock_client(_VALID_JSON))
        assert result["urgency"] == "low"

    def test_returns_reason(self):
        result = classify(_TICKET, client=_mock_client(_VALID_JSON))
        assert result["reason"]

    def test_returns_suggested_reply(self):
        result = classify(_TICKET, client=_mock_client(_VALID_JSON))
        assert result["suggested_reply"]

    def test_strips_markdown_fences(self):
        wrapped = f"```json\n{_VALID_JSON}\n```"
        result = classify(_TICKET, client=_mock_client(wrapped))
        assert result["category"] == "hardware"

    def test_usage_tracking(self):
        usage: list = []
        classify(_TICKET, client=_mock_client(_VALID_JSON), _usage=usage)
        assert len(usage) == 1
        assert usage[0]["input_tokens"] == 120
        assert usage[0]["output_tokens"] == 60


class TestClassifyFailure:
    @patch("triage.llm.time.sleep")
    def test_raises_on_invalid_json(self, _sleep):
        with pytest.raises(LLMUnavailableError):
            classify(_TICKET, client=_mock_client("not json at all"))

    @patch("triage.llm.time.sleep")
    def test_raises_on_invalid_category(self, _sleep):
        bad = json.dumps({"category": "billing", "urgency": "low",
                          "reason": "x", "suggested_reply": "y"})
        with pytest.raises(LLMUnavailableError):
            classify(_TICKET, client=_mock_client(bad))

    @patch("triage.llm.time.sleep")
    def test_raises_on_invalid_urgency(self, _sleep):
        bad = json.dumps({"category": "hardware", "urgency": "urgent",
                          "reason": "x", "suggested_reply": "y"})
        with pytest.raises(LLMUnavailableError):
            classify(_TICKET, client=_mock_client(bad))

    @patch("triage.llm.time.sleep")
    def test_raises_on_missing_reason(self, _sleep):
        bad = json.dumps({"category": "hardware", "urgency": "low",
                          "reason": "", "suggested_reply": "y"})
        with pytest.raises(LLMUnavailableError):
            classify(_TICKET, client=_mock_client(bad))

    @patch("triage.llm.time.sleep")
    def test_raises_on_api_exception(self, _sleep):
        client = MagicMock()
        client.messages.create.side_effect = Exception("connection refused")
        with pytest.raises(LLMUnavailableError):
            classify(_TICKET, client=client)


class TestRetry:
    @patch("triage.llm.time.sleep")
    def test_retries_on_bad_json_then_succeeds(self, mock_sleep):
        # First two calls return bad JSON, third returns valid
        bad_block = MagicMock()
        bad_block.type = "text"
        bad_block.text = "not json"
        bad_response = MagicMock()
        bad_response.content = [bad_block]
        bad_response.usage = MagicMock(input_tokens=0, output_tokens=0)

        good_block = MagicMock()
        good_block.type = "text"
        good_block.text = _VALID_JSON
        good_response = MagicMock()
        good_response.content = [good_block]
        good_response.usage = MagicMock(input_tokens=120, output_tokens=60)

        client = MagicMock()
        client.messages.create.side_effect = [bad_response, bad_response, good_response]

        result = classify(_TICKET, client=client)
        assert result["category"] == "hardware"
        assert client.messages.create.call_count == 3

    @patch("triage.llm.time.sleep")
    def test_raises_after_all_retries_exhausted(self, _sleep):
        client = MagicMock()
        client.messages.create.side_effect = Exception("always fails")
        with pytest.raises(LLMUnavailableError):
            classify(_TICKET, client=client)
        # MAX_RETRIES=2 → 3 total attempts
        assert client.messages.create.call_count == 3
