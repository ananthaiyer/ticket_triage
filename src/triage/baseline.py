"""TF-IDF + Logistic Regression baseline classifiers for category and urgency."""

import logging
from pathlib import Path

import joblib
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from triage.eval import exclude_eval_tickets

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent
MODELS_PATH = _ROOT / "models"
SILVER_PATH = _ROOT / "data" / "silver" / "tickets.parquet"


def _texts(df: pl.DataFrame) -> list[str]:
    return (df["subject"] + " " + df["body"]).to_list()


def _pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(C=1.0, max_iter=1000)),
    ])


def train(df: pl.DataFrame) -> tuple[Pipeline, Pipeline]:
    """Fit category and urgency pipelines on *df*. Returns (cat_model, urg_model)."""
    texts = _texts(df)
    cat_model = _pipeline()
    cat_model.fit(texts, df["category"].to_list())
    urg_model = _pipeline()
    urg_model.fit(texts, df["urgency"].to_list())
    log.info("Trained on %d tickets", len(df))
    return cat_model, urg_model


def save(cat_model: Pipeline, urg_model: Pipeline, path: Path = MODELS_PATH) -> None:
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(cat_model, path / "category_clf.joblib")
    joblib.dump(urg_model, path / "urgency_clf.joblib")
    log.info("Saved models to %s", path)


def load(path: Path = MODELS_PATH) -> tuple[Pipeline, Pipeline]:
    cat_model = joblib.load(path / "category_clf.joblib")
    urg_model = joblib.load(path / "urgency_clf.joblib")
    return cat_model, urg_model


def load_or_train() -> tuple[Pipeline, Pipeline]:
    """Load saved models; train and save from silver data if they don't exist."""
    cat_path = MODELS_PATH / "category_clf.joblib"
    urg_path = MODELS_PATH / "urgency_clf.joblib"
    if cat_path.exists() and urg_path.exists():
        log.info("Loading models from %s", MODELS_PATH)
        return load()
    log.info("Models not found — training from silver data")
    df = pl.read_parquet(SILVER_PATH)
    df = exclude_eval_tickets(df)
    cat_model, urg_model = train(df)
    save(cat_model, urg_model)
    return cat_model, urg_model


def predict(
    tickets: list[dict],
    cat_model: Pipeline | None = None,
    urg_model: Pipeline | None = None,
) -> list[dict]:
    """Predict category and urgency for each ticket dict. Loads models if not provided."""
    if cat_model is None or urg_model is None:
        cat_model, urg_model = load_or_train()
    texts = [f"{t.get('subject', '')} {t.get('body', '')}" for t in tickets]
    cats = cat_model.predict(texts)
    urgs = urg_model.predict(texts)
    return [
        {"ticket_id": t.get("ticket_id", ""), "category": c, "urgency": u}
        for t, c, u in zip(tickets, cats, urgs)
    ]
