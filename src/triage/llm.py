"""Claude-based IT ticket classifier and reply suggester."""

import json
import logging
import time
from pathlib import Path

import anthropic

log = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    """Raised when all retry attempts to classify a ticket have failed."""


_PROMPT_PATH = Path(__file__).parent / "prompts" / "classify.txt"
_SYSTEM = "You are an IT support triage assistant for Apex GmbH, a mid-sized company."
MODEL = "claude-haiku-5-5"
MAX_RETRIES = 2
_FALLBACK = {
    "category": "other",
    "urgency": "medium",
    "reason": "",
    "suggested_reply": "",
}
_VALID_CATEGORIES = frozenset(["hardware", "software", "access", "network", "email", "other"])
_VALID_URGENCIES = frozenset(["low", "medium", "high"])


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _parse_and_validate(raw: str) -> dict:
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(raw)
    if result.get("category") not in _VALID_CATEGORIES:
        raise ValueError(f"invalid category: {result.get('category')!r}")
    if result.get("urgency") not in _VALID_URGENCIES:
        raise ValueError(f"invalid urgency: {result.get('urgency')!r}")
    if not result.get("reason"):
        raise ValueError("missing reason")
    if not result.get("suggested_reply"):
        raise ValueError("missing suggested_reply")
    return result


def classify(
    ticket: dict,
    client: anthropic.Anthropic | None = None,
    _usage: list | None = None,
) -> dict:
    """Classify a single ticket with Claude.

    Returns dict with category, urgency, reason, suggested_reply.
    Raises LLMUnavailableError after MAX_RETRIES + 1 failed attempts.

    Pass a list as *_usage* to collect per-call token counts (for cost estimation).
    """
    if client is None:
        client = anthropic.Anthropic()

    prompt_template = _load_prompt()
    user_message = prompt_template.format(
        subject=ticket.get("subject", ""),
        body=ticket.get("body", ""),
    )

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=512,
                system=_SYSTEM,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = next(
                (b.text for b in response.content if b.type == "text"), ""
            ).strip()
            result = _parse_and_validate(raw)
            if _usage is not None:
                _usage.append({
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                })
            return result

        except (json.JSONDecodeError, ValueError, anthropic.APIError, Exception) as exc:
            if attempt < MAX_RETRIES:
                delay = 2 ** attempt
                log.warning(
                    "classify attempt %d failed for ticket %s: %s — retrying in %ds",
                    attempt + 1, ticket.get("ticket_id"), exc, delay,
                )
                time.sleep(delay)
            else:
                raise LLMUnavailableError(
                    f"classify failed after {MAX_RETRIES + 1} attempts "
                    f"for ticket {ticket.get('ticket_id')!r}: {exc}"
                ) from exc
