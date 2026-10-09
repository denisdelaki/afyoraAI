# Afyora Clinical Decision AI

## Deploy to Render

The repository includes a Render Blueprint in `render.yaml` and a `build.sh` build script. Push these files to your Git repository, then in Render choose **New + → Blueprint**, connect the repository, and apply the Blueprint. It creates a Python web service and a persistent PostgreSQL database. The selected smallest persistent web/database plans are paid Render resources; review Render's current pricing before applying.

During Blueprint setup, enter the rotated Groq credential when prompted for `GROQ_API_KEY` and a comma-separated model pool for `GROQ_MODELS` (for example, `openai/gpt-oss-120b,openai/gpt-oss-20b`). Both are managed in the Render service's Environment settings, not hardcoded in the Blueprint. On existing services, verify these values in Render; `sync: false` variables are requested during initial Blueprint creation and are not updated by subsequent Blueprint syncs. The Blueprint generates `DJANGO_SECRET_KEY`; do not add secrets to the repository.

Render-provided environment variables take precedence over local `.env` values. `DATABASE_URL` comes from the provisioned database, `RENDER_EXTERNAL_HOSTNAME` is added to allowed hosts and trusted HTTPS origins, and Render supplies `PORT` for Gunicorn. Optional custom domains can be set with comma-separated `ALLOWED_HOSTS` and `CSRF_TRUSTED_ORIGINS` variables.

| Environment variable           | Default | Purpose                            |
| ------------------------------ | ------- | ---------------------------------- |
| `GROQ_TIMEOUT_SECONDS`         | `30`    | Provider request timeout           |
| `CLINICAL_MCP_TIMEOUT_SECONDS` | `5`     | Local context preparation deadline |
| `CLINICAL_MAX_REQUEST_BYTES`   | `65536` | Clinical request body limit        |
| `WEB_CONCURRENCY`              | `1`     | Gunicorn worker processes          |
| `GUNICORN_TIMEOUT_SECONDS`     | `90`    | Gunicorn worker timeout            |

Timeouts must be finite positive numbers and the request limit must be a positive integer; invalid Django limits fail at startup. Keep the Gunicorn timeout above the combined Groq and context preparation deadlines, with additional overhead. Increase worker count only within the service's memory budget. Gunicorn variables must be supplied through Render or exported in the launching shell; Django's `.env` loader does not configure its parent shell.

Render build command:

```sh
bash build.sh
```

Render start command:

```sh
python manage.py migrate --noinput && gunicorn afyoraAIAgent.wsgi:application --bind "0.0.0.0:$PORT" --workers "${WEB_CONCURRENCY:-1}" --timeout "${GUNICORN_TIMEOUT_SECONDS:-90}" --access-logfile -
```

The start command applies outstanding database migrations before launching Gunicorn. After the first deploy, create an administrator from the Render Shell with `python manage.py createsuperuser`. The app's self-service signup currently has no email verification; add identity verification and abuse prevention before opening it to the public.

## Backend setup

Install the declared dependencies and apply Django migrations:

```sh
python -m pip install -r requirements.txt
python manage.py migrate
```

Copy `.env.example` to `.env` and set `GROQ_API_KEY` to a key obtained from the Groq Console. `.env` is ignored by Git. Never put the key in frontend code or commit it. Configure `GROQ_MODELS` as a comma-separated list of model IDs; the service randomly chooses one for each analysis request. The legacy `GROQ_MODEL` setting is still accepted when `GROQ_MODELS` is unset. Set `DJANGO_SECRET_KEY` to a stable secret in deployed environments; when it is absent, Django generates a development-only key at startup.

Users can create an account at `/accounts/signup/`, then register and manage connected applications from `/integrations/`. Each integration's bearer token is displayed once; only its SHA-256 hash is stored. Keep the token in the calling application's backend secret store. An account can only view or deactivate its own integrations. Operators can still create integrations with `python manage.py create_integration_app "Scheduling app"`. Never include integration tokens in browser JavaScript.

Signed-in users can try a sample request in the `/sandbox/` page. Sandbox submissions call the configured Groq provider and may consume API quota; use synthetic or approved de-identified information only.

Account self-registration is open and email verification is not configured in this starter. Before exposing account creation publicly, add the identity verification, abuse prevention, and account recovery controls appropriate to your deployment.

## Health check API

`GET /api/health/` checks that the application can serve requests and query its default database. No authentication is required; `HEAD` is also supported. Responses use `Cache-Control: no-store`.

```sh
curl -i http://localhost:8000/api/health/
```

A healthy application returns HTTP `200`:

```json
{ "status": "healthy", "checks": { "database": "ok" } }
```

A database failure returns HTTP `503`:

```json
{ "status": "unhealthy", "checks": { "database": "unavailable" } }
```

This endpoint does not call Groq or verify AI-provider availability, credentials, or database migrations. The Render Blueprint uses it as the service health check.

## Clinical analysis API

For credential setup, complete request/response schemas, error handling, and examples in Python, Node.js, Java, and Go, see the [Application Integration Guide](docs/integration.md).

`POST /api/clinical-ai/analyze/`

Send `Authorization: Bearer <integration-token>` and `Content-Type: application/json` headers. For example, a connected application can send:

```sh
curl -X POST http://localhost:8000/api/clinical-ai/analyze/ \
  -H "Authorization: Bearer $CLINICAL_AI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symptoms":["fever"],"vitals":{"temperature":38.5}}'
```

Example request:

```json
{
  "age": 47,
  "sex": "female",
  "symptoms": ["fever", "headache", "weakness"],
  "medical_history": [],
  "allergies": [],
  "current_medications": [],
  "vitals": {
    "temperature": 38.5,
    "heart_rate": 110,
    "blood_pressure": "150/95"
  },
  "labs": [],
  "radiology": [],
  "previous_diagnoses": [],
  "observations": []
}
```

`symptoms` is required and must be a nonempty list of nonblank strings. `labs` and `radiology` also accept the legacy names `laboratory_results` and `radiology_results`, but sending both names for the same field is rejected. Request bodies are limited to 64 KiB; clinical inputs must be de-identified before submission.

Successful responses contain `supported_diagnosis`, `possible_disease`, `drugs_admissible`, `further_labs_to_be_done`, and backend-generated `triage`. This replaces the previous response contract. Strict Pydantic validation runs before a result is returned; invalid provider output produces a controlled error instead.

Example response:

```json
{
  "supported_diagnosis": "Assessment is uncertain; qualified clinician review is required.",
  "possible_disease": [],
  "drugs_admissible": [],
  "further_labs_to_be_done": [
    "Further investigations depend on clinical assessment."
  ],
  "triage": {
    "level": "not_assessed",
    "urgent_care_recommended": false,
    "reasons": [],
    "unassessed_vitals": ["respiratory_rate", "oxygen_saturation"],
    "recommendation": "Vital triage is incomplete; obtain missing measurements and seek qualified clinician review.",
    "requires_human_review": true,
    "rule_set": "adult-vital-screening-v1"
  }
}
```

The DRF endpoint prepares context through an embedded FastMCP in-memory client in the request thread, then builds the schema-hydrated prompt and calls Groq in native JSON mode. No MCP network listener or subprocess is started. There is no configured RAG source; retrieval must be explicitly implemented against approved, tenant-scoped sources before deployment.

Top-level `patient_id` is excluded and known nested identifying fields are rejected. Free text still requires caller-side de-identification, and clinical information is sent to Groq. Requests fail closed when tools or the provider fail, or output does not validate. Every successful analysis requires qualified clinician review (`X-Clinical-Review-Required: true`). `drugs_admissible` is advisory only, not permission to prescribe or administer medication. API responses use `Cache-Control: no-store`.

Vital-sign triage is computed independently of the AI and survives provider failure. Markedly abnormal adult vitals produce `triage.level: "urgent"` and an urgent-care recommendation. Missing, invalid, conflicting, or incomplete measurements are not treated as normal. Adult screening requires age >=16 and is not applied to known pregnancy (`is_pregnant: true`). Use documented units and obtain clinical approval of the [screening rules and limitations](docs/integration.md#vital-sign-triage) before clinical deployment. This is not a full NEWS2 score or autonomous triage system.

This repository currently contains no Angular application or consultation UI. Other applications should call this Django endpoint from their backend; browser frontends should call their own backend, which can securely hold the integration token. No application should call Groq directly.

## Tests

Run the backend suite with:

```sh
python manage.py test clinicalDecisionAI
```

Groq is mocked in automated tests; tests do not make provider requests.
