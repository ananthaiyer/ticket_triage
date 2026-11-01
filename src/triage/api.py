"""FastAPI application for IT ticket triage.

Set SERVICE_API_KEY in the environment to require an X-Api-Key header on /triage.
Leave it unset to allow unauthenticated access (useful for local dev).
"""

import logging
import os
from contextlib import asynccontextmanager

import anthropic
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from triage import baseline, llm

log = logging.getLogger(__name__)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["cat_model"], _state["urg_model"] = baseline.load_or_train()
    _state["client"] = anthropic.Anthropic()
    log.info("startup complete")
    yield
    _state.clear()


app = FastAPI(title="Ticket Triage API", version="0.1.0", lifespan=lifespan)


class TriageRequest(BaseModel):
    ticket_id: str = ""
    subject: str
    body: str


class TriageResponse(BaseModel):
    ticket_id: str
    category: str
    urgency: str
    reason: str
    suggested_reply: str
    source: str  # "llm" or "baseline"


def _require_api_key(x_api_key: str | None = Header(None, alias="X-Api-Key")) -> None:
    """Reject requests whose X-Api-Key header does not match SERVICE_API_KEY.

    Skipped when SERVICE_API_KEY is not set (local dev / CI).
    """
    expected = os.getenv("SERVICE_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Api-Key header")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest, _: None = Depends(_require_api_key)):
    ticket = {"ticket_id": req.ticket_id, "subject": req.subject, "body": req.body}
    try:
        result = llm.classify(ticket, client=_state.get("client"))
        source = "llm"
    except llm.LLMUnavailableError as exc:
        log.warning(
            "LLM unavailable for ticket %s, falling back to baseline: %s",
            req.ticket_id, exc,
        )
        try:
            b = baseline.predict([ticket], _state.get("cat_model"), _state.get("urg_model"))
            b0 = b[0]
            result = {"category": b0["category"], "urgency": b0["urgency"],
                      "reason": "", "suggested_reply": ""}
            source = "baseline"
        except Exception as exc2:
            log.error(
                "Baseline also unavailable for ticket %s: %s", req.ticket_id, exc2,
            )
            raise HTTPException(
                status_code=503,
                detail="Both classifiers are unavailable; please try again later",
            ) from exc2
    return TriageResponse(
        ticket_id=req.ticket_id,
        category=result["category"],
        urgency=result["urgency"],
        reason=result.get("reason", ""),
        suggested_reply=result.get("suggested_reply", ""),
        source=source,
    )
