# Deploying to Render

This service runs as a Docker container on Render's free web service tier.

## Prerequisites

- The repo pushed to GitHub (any branch works; `main` is used here)
- An Anthropic API key (`ANTHROPIC_API_KEY`)
- A random secret string for `SERVICE_API_KEY` (authenticates `/triage` requests)
- [Docker](https://docs.docker.com/get-docker/) installed locally (optional, for pre-deploy testing)

---

## Local smoke test (optional)

```bash
# Build
docker build -t ticket-triage .

# Run (reads ANTHROPIC_API_KEY from your shell environment)
docker run --rm -p 8000:8000 \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  -e SERVICE_API_KEY="local-test-key" \
  ticket-triage

# Health check
curl http://localhost:8000/health

# Triage a ticket
curl -s -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: local-test-key" \
  -d '{"ticket_id":"test-1","subject":"VPN not working","body":"I cannot connect to the VPN since this morning."}' \
  | python3 -m json.tool
```

---

## Deploy on Render

### 1. Create a Render account
Sign up at [render.com](https://render.com) if you do not have an account yet.

### 2. Create a new Web Service

From the Render dashboard click **New → Web Service**.

### 3. Connect your GitHub repository

- Select **Build and deploy from a Git repository**
- Connect your GitHub account and choose this repo
- Branch: `main`

### 4. Configure the service

| Setting | Value |
|---|---|
| Name | `ticket-triage` (or your choice) |
| Region | **Frankfurt (EU Central)** |
| Runtime | **Docker** |
| Instance type | **Free** |
| Auto-deploy | On (deploys every push to `main`) |

Render detects the `Dockerfile` automatically. No build command or start command overrides are needed.

### 5. Set environment variables

Under **Environment → Environment Variables** add:

| Key | Value |
|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key |
| `SERVICE_API_KEY` | A strong random secret (e.g. `openssl rand -hex 32`) |

> Render does **not** automatically use a `.env` file — you must add variables through the dashboard or the Render CLI.

### 6. Set the health check path

Under **Settings → Health & Alerts** set the health check path to `/health`.

Render pings this endpoint after each deploy to confirm the service is up before routing traffic to the new instance.

### 7. Deploy

Click **Create Web Service**. Render builds the Docker image and deploys it. The first build takes ~2 minutes. Subsequent deploys (triggered by a push to `main`) are faster because Docker layer caching is reused.

---

## Testing the live service

Replace `<YOUR_SERVICE_URL>` with the URL shown in the Render dashboard (e.g. `https://ticket-triage.onrender.com`).

```bash
# Health check (no auth required)
curl https://<YOUR_SERVICE_URL>/health

# Triage a ticket
curl -s -X POST https://<YOUR_SERVICE_URL>/triage \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: <YOUR_SERVICE_API_KEY>" \
  -d '{
    "ticket_id": "demo-1",
    "subject": "Cannot open Outlook",
    "body": "Since this morning Outlook crashes immediately on launch. I have tried restarting."
  }' | python3 -m json.tool
```

Expected response shape:

```json
{
  "ticket_id": "demo-1",
  "category": "software",
  "urgency": "medium",
  "reason": "...",
  "suggested_reply": "...",
  "baseline_category": "software",
  "baseline_urgency": "medium"
}
```

---

## Notes on model files

The pre-trained baseline models (`models/category_clf.joblib`, `models/urgency_clf.joblib`) are committed to the repository (~1.2 MB total) and baked into the Docker image at build time. This keeps the Dockerfile simple and avoids shipping data files into the build context. The models are deterministic (fixed seed) and can be regenerated locally with `python -m triage.evaluate` if needed.
