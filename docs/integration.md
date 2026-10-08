# Clinical AI Application Integration Guide

This guide is for backend teams connecting another application to the Afyora Clinical Decision AI API. The API uses synchronous HTTP request/response: send a JSON `POST`, then read the validated JSON result from that response. There are no callbacks or webhooks.

## 1. Create an Account and Register an Application

1. Open `{BASE_URL}/accounts/signup/` and create an account with a username and password. Email is optional in the current implementation.
2. Sign in at `{BASE_URL}/accounts/login/` if you are not already signed in.
3. Open `{BASE_URL}/integrations/`, enter a recognizable application name, and select **Create integration**.
4. Copy the bearer token shown in the creation response and save it to the calling application's backend secret manager. The dashboard will not show the token again.

Before connecting a client, sign in and open `/sandbox/` to try the clinical analysis flow with sample JSON. The sandbox calls the configured Groq provider and may consume quota. Use synthetic or approved de-identified inputs only; sandbox analysis is not a substitute for a clinician's review.

The integration dashboard lists the applications registered to your account and lets you deactivate one. Deactivation immediately prevents that token from authorizing API calls. Accounts cannot view or deactivate integrations owned by another account.

The CLI remains available to Afyora operators who need to provision an integration directly:

```sh
python manage.py create_integration_app "Scheduling app"
```

The command prints the application ID and a bearer token once. The API stores only a SHA-256 hash, so the token cannot be retrieved later. Store the original token in the calling application's server-side secret manager. Do not put it in browser JavaScript, mobile app bundles, source control, or logs. To rotate a lost or exposed token, deactivate the old integration and create a replacement.

Self-registration is open and email verification is not configured in this starter. Before exposing registration publicly, configure appropriate identity verification, abuse prevention, and account recovery controls.

Use HTTPS for deployed environments. The local development base URL is `http://127.0.0.1:8001`; deployments must provide their own base URL.

## 2. Endpoint

```text
POST {BASE_URL}/api/clinical-ai/analyze/
Authorization: Bearer {INTEGRATION_TOKEN}
Content-Type: application/json
Accept: application/json
```

This is an API endpoint, not a browser page. Opening it directly sends `GET` and returns `405 Method Not Allowed`; use an HTTP client to submit a `POST` request. The endpoint does not require cookie-based CSRF tokens because application access is authorized by the bearer token.

An optional UUID `X-Request-ID` request header may be supplied for correlation. If omitted or not a UUID, the API generates one. The response includes the resulting ID in its `X-Request-ID` header. Do not put patient information in this header. All API responses use `Cache-Control: no-store`. Successful responses include `X-Clinical-Review-Required: true`.

The service selects one model at random from its configured `GROQ_MODELS` pool for each analysis. Configure this as a comma-separated list of model IDs in the backend environment; the legacy `GROQ_MODEL` setting is used when that list is empty and also accepts a comma-separated list. Both settings trim whitespace, discard empty entries, and deduplicate model IDs before selection. Only one model ID is sent to Groq per request. The selected model ID is returned in the `X-Groq-Model` response header and recorded in request metadata logs. Callers do not choose a model in the request body. Selection does not automatically retry another model if the chosen one fails, so configure only model IDs enabled for your Groq account.

## 3. Request Body

`symptoms` is required and must contain at least one nonblank string. Other fields are optional; omitted lists and `vitals` default to empty values. Types are strict: numeric strings are not accepted as ages, while vital measurements may be numbers or strings as documented below. Unknown top-level fields are rejected. `patient_id` may be a string or integer, but it is not sent to MCP tools or the AI provider and is not used to look up patient records in this service.

The JSON body is limited to 65,536 bytes by default; operators can change `CLINICAL_MAX_REQUEST_BYTES` in the server environment. Clinical text is limited to 2,048 characters, each list to 64 entries, and `vitals` to 128 entries. Nested data is limited to six levels and 4,096 nodes. Named identifying fields such as `patient_name`, `mrn`, and `date_of_birth` are rejected in nested clinical data. This is not automatic de-identification: remove identifiers from free text before submission. Configure matching request limits at the reverse proxy.

| Field                 | Type                                     | Description                                            |
| --------------------- | ---------------------------------------- | ------------------------------------------------------ |
| `patient_id`          | string, integer, or null                 | Caller-side reference; not sent to Groq                |
| `age`                 | integer from 0 to 130, or null           | Patient age                                            |
| `sex`                 | string or null                           | Supplied sex information                               |
| `is_pregnant`         | boolean or null                          | Known pregnancy; adult triage is not applied when true |
| `symptoms`            | nonempty array of nonblank strings       | Required reported symptoms                             |
| `medical_history`     | array of strings                         | Relevant medical history                               |
| `allergies`           | array of strings                         | Known allergies                                        |
| `current_medications` | array of strings                         | Current medications                                    |
| `vitals`              | object of number, string, or null values | Vital signs, such as temperature or blood pressure     |
| `labs`                | array of strings or JSON objects         | Supplied lab results                                   |
| `radiology`           | array of strings or JSON objects         | Supplied imaging findings                              |
| `previous_diagnoses`  | array of strings                         | Previous diagnoses supplied by the caller              |
| `observations`        | array of strings                         | Other relevant clinical observations                   |

Example:

```json
{
  "patient_id": "encounter-84921",
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

The legacy names `laboratory_results` and `radiology_results` remain accepted. Do not supply a canonical name and its legacy alias together; this returns `400` even if the values match.

Send only the minimum clinical information needed. The integration service does not persist the request as a patient record or log submitted clinical data. Clinical data is still transmitted to Groq; in-memory MCP does not prevent provider transmission or define provider retention policy.

## 4. Successful Response

The API returns HTTP `200` with the four validated AI fields plus a backend-generated `triage` object. Clients must migrate their response parsing from the previous response contract. Qualified clinician review is mandatory, as indicated by `X-Clinical-Review-Required: true`.

```json
{
  "supported_diagnosis": "Assessment is uncertain; qualified clinician review is required.",
  "possible_disease": ["Example differential"],
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

`supported_diagnosis` is a provisional assessment, not a definitive diagnosis. `possible_disease` contains differential considerations; `further_labs_to_be_done` contains suggested investigations. Despite its name, `drugs_admissible` contains advisory medication considerations only, not prescriptions or permission to administer medication. Empty lists are valid. The response does not provide a validated aggregate risk score or confidence estimate; absence of a threshold alert is not evidence of low risk.

### Vital-Sign Triage

`triage` is computed deterministically by the backend, not supplied by Groq. It is also returned alongside controlled AI/provider errors after valid input is assessed, including unexpected application failures. Callers must inspect triage even on non-200 responses. A model cannot overwrite or downgrade this assessment.

The current rules are conservative adult screening alerts, not a complete NEWS2 implementation or a validated autonomous triage system. They require known age >=16 and are not applied to known pregnancy. Missing age, childhood, or known pregnancy produces `not_assessed` with an age-appropriate/obstetric review recommendation. All rules require clinical-owner approval and local validation before clinical deployment, particularly for chronic respiratory disease, oxygen treatment, and individual baseline differences.

| Vital and accepted names                   | Required unit            | Urgent screening trigger |
| ------------------------------------------ | ------------------------ | ------------------------ |
| `heart_rate`, `pulse`                      | beats/min                | <=40 or >=131            |
| `respiratory_rate`                         | breaths/min              | <=8 or >=25              |
| `oxygen_saturation`, `spo2`                | percentage, not fraction | <=91                     |
| `systolic_blood_pressure`, `systolic_bp`   | mmHg                     | <=90 or >=180            |
| `diastolic_blood_pressure`, `diastolic_bp` | mmHg                     | >=120                    |
| `temperature`, `temperature_celsius`       | Celsius, not Fahrenheit  | <=35 or >=39.1           |

`blood_pressure` also accepts a string such as `"120/80"` in mmHg. Numeric strings are accepted by the triage parser for vital measurements. Unit-suffixed strings are unassessed rather than guessed. Conflicting aliases are marked unassessed, but an extreme reading still triggers urgent care. Missing, invalid, nonfinite, negative, or out-of-range saturation measurements are not treated as normal. Unknown vital names do not substitute for the named measurements.

`level` is `urgent` when a trigger is met, `not_assessed` when eligible measurements are incomplete or invalid without a trigger, or `no_threshold_triggered` when all named measurements were assessed without a trigger. Neither non-urgent level certifies safety. `urgent_care_recommended: false` on an incomplete assessment does not mean urgent care is unnecessary. Urgent results recommend immediate in-person clinical assessment and local emergency services for severe breathlessness, chest pain, collapse, or new confusion. Do not wait for an AI result to seek care.

Clinical references: [Royal College of Physicians NEWS2 resources](https://www.rcp.ac.uk/improving-care/resources/national-early-warning-score-news-2/) and [American Heart Association severe-hypertension guidance](https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings/hypertensive-crisis-when-you-should-call-911-for-high-blood-pressure). These custom screening rules are not endorsed by either organization and do not compute the full NEWS2 score, consciousness assessment, supplemental-oxygen contribution, or combined moderate-abnormality escalation. High-temperature and blood-pressure cutoffs are conservative alerts, not diagnostic labels.

## 5. Errors

Errors use this shape:

```json
{
  "error": {
    "code": "authentication_required",
    "message": "A registered application bearer token is required."
  }
}
```

| HTTP status | Meaning                                         | Suggested caller behavior                                                 |
| ----------- | ----------------------------------------------- | ------------------------------------------------------------------------- |
| `400`       | Invalid JSON or request fields                  | Correct the request; do not retry unchanged                               |
| `401`       | Missing, unknown, or inactive integration token | Check token configuration; ask the operator to provision/re-enable access |
| `405`       | Method is not `POST`                            | Send a `POST` to the full `/analyze/` path                                |
| `406`       | Requested response format is not JSON           | Set `Accept: application/json`                                            |
| `413`       | Request body exceeds the size limit             | Reduce the clinical payload                                               |
| `415`       | Content type is not JSON                        | Set `Content-Type: application/json`                                      |
| `429`       | Groq rate limit                                 | Retry with bounded exponential backoff and jitter                         |
| `500`       | Unexpected application failure                  | Preserve the request ID and contact the service operator                  |
| `502`       | Provider failure or invalid model response      | Retry cautiously; preserve the request ID for support                     |
| `503`       | AI provider is not configured on the service    | Contact the service operator                                              |
| `504`       | Provider timed out                              | Retry cautiously with a new request ID if needed                          |

The API does not currently provide idempotency keys. Avoid automatic retries that could create unwanted duplicate provider requests. Never display a failed or unvalidated model result as a successful clinical assessment.

## 6. Python Example

Backend orchestration creates a separate embedded FastMCP server and in-memory client per request, with a five-second context preparation budget by default (`CLINICAL_MCP_TIMEOUT_SECONDS`). Tool execution stays in the view thread; no server lifespan or patient context is shared across request event loops. It creates no SSE/HTTP MCP listener, subprocess, or externally callable tool route. The only current tool prepares supplied clinical context; no RAG source or patient-record lookup is configured. Future retrieval sources must be operator-controlled, tenant-scoped, and independently approved before use. Groq defaults to a 30-second request timeout (`GROQ_TIMEOUT_SECONDS`) with automatic SDK retries disabled. Operators configure these positive limits in Render's Environment settings or the local process environment.

Requires Python 3 and `requests` (`python -m pip install requests`). Store the integration token in the calling service's `CLINICAL_AI_TOKEN` environment variable.

```python
import os

import requests

base_url = os.environ.get("CLINICAL_AI_BASE_URL", "http://127.0.0.1:8001")
token = os.environ["CLINICAL_AI_TOKEN"]
payload = {
    "symptoms": ["fever", "headache"],
    "vitals": {"temperature": 38.5},
}

response = requests.post(
    f"{base_url}/api/clinical-ai/analyze/",
    headers={"Authorization": f"Bearer {token}"},
    json=payload,
    timeout=45,
)
response.raise_for_status()
analysis = response.json()
print(analysis["supported_diagnosis"])
```

## 7. Node.js Example

Requires Node.js 18 or later for built-in `fetch`. Store the token in the server process environment, not a client-side bundle.

```javascript
const baseUrl = process.env.CLINICAL_AI_BASE_URL ?? "http://127.0.0.1:8001";
const token = process.env.CLINICAL_AI_TOKEN;

if (!token) {
  throw new Error("CLINICAL_AI_TOKEN is not configured");
}

const response = await fetch(`${baseUrl}/api/clinical-ai/analyze/`, {
  method: "POST",
  headers: {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
    Accept: "application/json",
  },
  body: JSON.stringify({
    symptoms: ["fever", "headache"],
    vitals: { temperature: 38.5 },
  }),
  signal: AbortSignal.timeout(45_000),
});

const result = await response.json();
if (!response.ok) {
  throw new Error(
    `Clinical AI API ${response.status}: ${JSON.stringify(result)}`,
  );
}

console.log(result.supported_diagnosis);
```

## 8. Java Example

Requires Java 17 or later. This example uses the built-in `HttpClient`; store the token in the backend process environment.

```java
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

public class ClinicalAiExample {
    public static void main(String[] args) throws Exception {
        String baseUrl = System.getenv().getOrDefault(
            "CLINICAL_AI_BASE_URL", "http://127.0.0.1:8001");
        String token = System.getenv("CLINICAL_AI_TOKEN");
        if (token == null || token.isBlank()) {
            throw new IllegalStateException("CLINICAL_AI_TOKEN is not configured");
        }

        String payload = """
            {"symptoms":["fever","headache"],"vitals":{"temperature":38.5}}
            """;
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(baseUrl + "/api/clinical-ai/analyze/"))
            .timeout(Duration.ofSeconds(45))
            .header("Authorization", "Bearer " + token)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(payload))
            .build();

        HttpClient client = HttpClient.newHttpClient();
        HttpResponse<String> response = client.send(
            request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            throw new IllegalStateException(
                "Clinical AI API " + response.statusCode() + ": " + response.body());
        }
        System.out.println(response.body());
    }
}
```

## 9. Go Example

Uses Go's standard library. Store the token in the backend process environment.

```go
package main

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

func main() {
	baseURL := os.Getenv("CLINICAL_AI_BASE_URL")
	if baseURL == "" {
		baseURL = "http://127.0.0.1:8001"
	}
	token := os.Getenv("CLINICAL_AI_TOKEN")
	if token == "" {
		panic("CLINICAL_AI_TOKEN is not configured")
	}

	payload := []byte(`{"symptoms":["fever","headache"],"vitals":{"temperature":38.5}}`)
	request, err := http.NewRequest(
		http.MethodPost,
		baseURL+"/api/clinical-ai/analyze/",
		bytes.NewReader(payload),
	)
	if err != nil {
		panic(err)
	}
	request.Header.Set("Authorization", "Bearer "+token)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")

	client := &http.Client{Timeout: 45 * time.Second}
	response, err := client.Do(request)
	if err != nil {
		panic(err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		panic(err)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		panic(fmt.Sprintf("Clinical AI API %d: %s", response.StatusCode, body))
	}
	fmt.Println(string(body))
}
```

## 10. Clinical Safety and Privacy

- The integration token authorizes an application, not an individual clinician. The calling system remains responsible for user identity, permissions, consent, and audit controls.
- Keep tokens and patient data in trusted backend services and transmit them only over HTTPS in deployed environments.
- Do not log bearer tokens, complete requests, or unnecessary patient data.
- A successful response is advisory only. A qualified clinician must review it before any diagnosis, referral, or treatment decision.
- Do not automatically prescribe or execute treatment based on this response.
- The current API does not persist patient records or associate `patient_id` with a database record.
