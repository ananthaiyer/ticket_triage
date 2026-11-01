"""Generate synthetic IT support tickets and write to data/bronze/tickets.jsonl."""

import json
import logging
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Logging — one JSON object per line
# ---------------------------------------------------------------------------

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
    logging.basicConfig(level=logging.INFO, handlers=[handler])


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES = ["hardware", "software", "access", "network", "email", "other"]
URGENCIES = ["low", "medium", "high"]
TICKETS_PER_CATEGORY = 50          # 6 × 50 = 300 clean tickets
MESSY_COUNT = 15                   # ~5% of 300
MODEL = "claude-haiku-5-5"
MAX_RETRIES = 2
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "bronze" / "tickets.jsonl"

# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _random_ts(days_back: int = 60) -> str:
    now = datetime.now(tz=timezone.utc)
    offset_seconds = random.uniform(0, days_back * 86400)
    return (now - timedelta(seconds=offset_seconds)).isoformat()


# ---------------------------------------------------------------------------
# Claude API call with retry
# ---------------------------------------------------------------------------

def _call_claude(client: anthropic.Anthropic, category: str) -> list[dict]:
    system = (
        "You are generating synthetic IT support tickets for Apex GmbH, a fictional "
        "mid-sized company with ~600 employees. Tickets come from employees with "
        "varying levels of technical skill. Write in English, but occasionally mix in "
        "a German word or short phrase (e.g. 'Bitte', 'ich kann nicht', 'dringend'). "
        "Vary the writing style: some tickets are one-line, some are three paragraphs; "
        "some have typos or autocorrect errors; some are vague ('it doesn't work'); "
        "some are very specific. Mix urgency levels naturally."
    )

    user = (
        f"Generate exactly {TICKETS_PER_CATEGORY} IT support tickets for the "
        f"'{category}' category.\n\n"
        "Return ONLY a JSON array — no markdown, no explanation — where each element has:\n"
        '  "ticket_id": a UUID string,\n'
        '  "subject": short summary (5–12 words),\n'
        '  "body": ticket text from the employee,\n'
        '  "category": must be exactly "' + category + '",\n'
        '  "urgency": one of "low", "medium", "high".\n\n'
        "Do not include created_at — it will be added later."
    )

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=8192,
                messages=[{"role": "user", "content": user}],
                system=system,
            )
            raw = next(
                (b.text for b in response.content if b.type == "text"), ""
            ).strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            tickets = json.loads(raw)
            if not isinstance(tickets, list):
                raise ValueError("Expected a JSON array")

            validated = []
            for t in tickets:
                if t.get("category") not in CATEGORIES:
                    t["category"] = category
                if t.get("urgency") not in URGENCIES:
                    t["urgency"] = "medium"
                if not t.get("ticket_id"):
                    t["ticket_id"] = str(uuid.uuid4())
                validated.append(t)

            log.info("Generated %d tickets for category '%s'", len(validated), category)
            return validated

        except (json.JSONDecodeError, ValueError, anthropic.APIError) as exc:
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                log.warning(
                    "Attempt %d failed for category '%s': %s — retrying in %ds",
                    attempt + 1, category, exc, wait,
                )
                time.sleep(wait)
            else:
                log.warning(
                    "All retries exhausted for category '%s': %s — skipping batch",
                    category, exc,
                )
                return []

    return []


# ---------------------------------------------------------------------------
# Messy record injection
# ---------------------------------------------------------------------------

def _inject_messy(tickets: list[dict]) -> list[dict]:
    messy: list[dict] = []

    # Duplicates — copy an existing ticket verbatim (same ticket_id)
    for _ in range(4):
        original = random.choice(tickets)
        messy.append(dict(original))

    # Empty body
    for _ in range(4):
        t = dict(random.choice(tickets))
        t["ticket_id"] = str(uuid.uuid4())
        t["body"] = ""
        messy.append(t)

    # Invalid category
    for _ in range(4):
        t = dict(random.choice(tickets))
        t["ticket_id"] = str(uuid.uuid4())
        t["category"] = random.choice(["billing", "unknown", "facilities"])
        messy.append(t)

    # Missing timestamp (set created_at to None)
    for _ in range(3):
        t = dict(random.choice(tickets))
        t["ticket_id"] = str(uuid.uuid4())
        t["created_at"] = None
        messy.append(t)

    return messy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _configure_logging()
    log.info("Starting ticket generation — model=%s", MODEL)

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    all_tickets: list[dict] = []

    for category in CATEGORIES:
        batch = _call_claude(client, category)
        for t in batch:
            t["created_at"] = _random_ts()
        all_tickets.extend(batch)

    log.info("Generated %d clean tickets across all categories", len(all_tickets))

    messy = _inject_messy(all_tickets)
    # Messy records inherit created_at from their source ticket (or None for missing-ts records).
    all_tickets.extend(messy)

    random.shuffle(all_tickets)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        for ticket in all_tickets:
            fh.write(json.dumps(ticket, ensure_ascii=False) + "\n")

    log.info(
        "Wrote %d total tickets (including %d messy) to %s",
        len(all_tickets), len(messy), OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
