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

An optional `X-Request-ID` request header may be supplied for correlation. If omitted, the API generates one. The response includes the resulting ID in its `X-Request-ID` header. Do not put patient information in this header.

The service selects one model at random from its configured `GROQ_MODELS` pool for each analysis. Configure this as a comma-separated list of model IDs in the backend environment; the legacy `GROQ_MODEL` setting is used as a one-model fallback when the list is not set. The selected model ID is returned in the `X-Groq-Model` response header and recorded in request metadata logs. Callers do not choose a model in the request body. Selection does not automatically retry another model if the chosen one fails, so configure only model IDs enabled for your Groq account.

## 3. Request Body

All fields are optional. Omitted lists and `vitals` default to empty values. Unknown fields are rejected. `patient_id` may be a string or integer, but it is not sent to the AI provider and is not used to look up patient records in this service.

| Field                 | Type                                     | Description                                        |
| --------------------- | ---------------------------------------- | -------------------------------------------------- |
| `patient_id`          | string, integer, or null                 | Caller-side reference; not sent to Groq            |
| `age`                 | integer from 0 to 130, or null           | Patient age                                        |
| `sex`                 | string or null                           | Supplied sex information                           |
| `symptoms`            | array of strings                         | Reported symptoms                                  |
| `medical_history`     | array of strings                         | Relevant medical history                           |
| `allergies`           | array of strings                         | Known allergies                                    |
| `current_medications` | array of strings                         | Current medications                                |
| `vitals`              | object of number, string, or null values | Vital signs, such as temperature or blood pressure |
| `laboratory_results`  | array of strings or objects              | Supplied lab results                               |
| `radiology_results`   | array of strings or objects              | Supplied imaging findings                          |
| `previous_diagnoses`  | array of strings                         | Previous diagnoses supplied by the caller          |
| `observations`        | array of strings                         | Other relevant clinical observations               |

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
  "laboratory_results": [],
  "radiology_results": [],
  "previous_diagnoses": [],
  "observations": []
}
```

Send only the minimum clinical information needed. The integration service does not persist the request as a patient record. It excludes `patient_id` from the model prompt and does not log the submitted clinical data.

## 4. Successful Response

The API returns HTTP `200` with a validated JSON object. `requires_human_review` is always `true`.

```json
{
  "clinical_summary": "Assessment is limited to the supplied information.",
  "risk_level": "moderate",
  "possible_conditions": [
    {
      "condition": "Example differential",
      "likelihood": "Uncertain",
      "supporting_evidence": ["Fever was reported."]
    }
  ],
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

`risk_level` is `low`, `moderate`, `high`, or `critical`. Referral `urgency` is `routine`, `urgent`, or `emergency`. `confidence` is between `0.0` and `1.0`. Other response fields are documented by their names in the example and should be treated as decision support, not as an autonomous diagnosis or treatment plan.

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
| `415`       | Content type is not JSON                        | Set `Content-Type: application/json`                                      |
| `429`       | Groq rate limit                                 | Retry with bounded exponential backoff and jitter                         |
| `502`       | Provider failure or invalid model response      | Retry cautiously; preserve the request ID for support                     |
| `503`       | AI provider is not configured on the service    | Contact the service operator                                              |
| `504`       | Provider timed out                              | Retry cautiously with a new request ID if needed                          |

The API does not currently provide idempotency keys. Avoid automatic retries that could create unwanted duplicate provider requests. Never display a failed or unvalidated model result as a successful clinical assessment.

## 6. Python Example

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
print(analysis["clinical_summary"])
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

console.log(result.clinical_summary);
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
