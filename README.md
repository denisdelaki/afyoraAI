# Afyora Clinical Decision AI

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
  "laboratory_results": [],
  "radiology_results": [],
  "previous_diagnoses": [],
  "observations": []
}
```

Successful responses contain `clinical_summary`, `risk_level`, `possible_conditions`, `abnormal_findings`, `recommended_investigations`, `management_considerations`, `medication_considerations`, `red_flags`, `referral_recommendation`, `confidence`, and `requires_human_review`. The response is validated before it is returned; invalid provider output produces a controlled error instead.

Example response:

```json
{
  "clinical_summary": "Assessment is limited to the supplied information.",
  "risk_level": "moderate",
  "possible_conditions": [],
  "abnormal_findings": [],
  "recommended_investigations": [],
  "management_considerations": [
    "Review in the context of a clinical assessment."
  ],
  "medication_considerations": [],
  "red_flags": [],
  "referral_recommendation": {
    "required": false,
    "urgency": "routine",
    "reason": "No urgent referral indication was identified from the supplied information."
  },
  "confidence": 0.35,
  "requires_human_review": true
}
```

The backend builds the clinical prompt and calls Groq. Patient identifiers are not sent to the model. Requests fail closed when the provider is unavailable or its response does not validate. Every successful analysis requires human review; the output is decision support only and does not prescribe or make treatment decisions.

This repository currently contains no Angular application or consultation UI. Other applications should call this Django endpoint from their backend; browser frontends should call their own backend, which can securely hold the integration token. No application should call Groq directly.

## Tests

Run the backend suite with:

```sh
python manage.py test clinicalDecisionAI
```

Groq is mocked in automated tests; tests do not make provider requests.
