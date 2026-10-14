"""
Ollama client for local open-weights model inference.

Allows the complete triage workflow to run without a proprietary cloud model,
satisfying the challenge requirement that patient data not leave the country.

Setup:
  1. Install Ollama: https://ollama.com/download
  2. Pull a model: ollama pull llama3.2:3b  (runs on ~4 GB RAM)
  3. Set environment:
     USE_LOCAL_MODEL=true
     LOCAL_MODEL=llama3.2:3b
     OLLAMA_BASE_URL=http://localhost:11434  (default)

The workflow is identical regardless of model backend. The clinical rules
(SATS) are deterministic Python — they do not change with the model choice.

Model requirements (minimum for llama3.2:3b):
  - RAM: 4 GB
  - Storage: 2 GB
  - CPU: any modern x86-64 or arm64
  - GPU: optional (inference is slower without GPU)
"""
from __future__ import annotations

import json
import logging
import re

import httpx
from django.conf import settings
from pydantic import ValidationError

from clinicalDecisionAI.schemas import ClinicalAnalysisRequest, ClinicalDecisionSchema
from clinicalDecisionAI.prompts import SYSTEM_INSTRUCTION, build_clinical_prompt
from clinicalDecisionAI.services import (
    ClinicalResponseError,
    _log_rejected_response,
    _normalize_clinical_output,
)

logger = logging.getLogger(__name__)


class OllamaConfigurationError(Exception):
    pass


class OllamaProviderError(Exception):
    def __init__(self, category: str):
        self.category = category
        super().__init__(category)


class OllamaClient:
    """
    Client for locally running Ollama models.

    Patient data never leaves the local machine when this client is used.
    """

    def __init__(self, model: str | None = None):
        base_url = getattr(settings, "OLLAMA_BASE_URL", "").strip()
        if not base_url:
            raise OllamaConfigurationError("OLLAMA_BASE_URL is not configured")
        self.base_url = base_url.rstrip("/")
        self.model = model or getattr(settings, "LOCAL_MODEL", "llama3.2:3b")
        self.timeout = getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 120.0)

    def ask(self, prompt: str) -> str:
        """Send a prompt to the local Ollama server and return the response content."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1},
        }
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise OllamaProviderError("timeout") from exc
        except httpx.ConnectError as exc:
            raise OllamaProviderError("connection") from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaProviderError("provider") from exc
        except Exception as exc:
            logger.warning("Ollama request failed (%s)", type(exc).__name__)
            raise OllamaProviderError("provider") from exc

        try:
            data = response.json()
            content = data["message"]["content"]
        except (KeyError, TypeError, ValueError):
            raise OllamaProviderError("invalid_response")

        if not isinstance(content, str) or not content.strip():
            raise ClinicalResponseError("empty_response")

        return content

    def is_available(self) -> bool:
        """Check if the local Ollama server is reachable."""
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False


def run_clinical_analysis_local(
    patient: ClinicalAnalysisRequest,
    model: str | None = None,
) -> ClinicalDecisionSchema:
    """
    Run clinical analysis using a local Ollama model.

    Identical validation pipeline to run_clinical_analysis() in services.py.
    Patient data does not leave the local machine.
    """
    client = OllamaClient(model=model)
    content = client.ask(build_clinical_prompt(patient))

    json_content = content.strip()
    fenced = re.fullmatch(
        r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", json_content, re.DOTALL | re.IGNORECASE
    )
    if fenced:
        json_content = fenced.group(1)

    try:
        output = json.loads(json_content)
    except json.JSONDecodeError as exc:
        _log_rejected_response(client.model, "invalid_json", content, [
            {"message": exc.msg, "line": exc.lineno, "column": exc.colno},
        ])
        raise ClinicalResponseError("invalid_json") from exc

    if not isinstance(output, dict):
        _log_rejected_response(client.model, "schema_validation", content, [
            {"message": "Expected a JSON object", "received_type": type(output).__name__},
        ])
        raise ClinicalResponseError("schema_validation")

    try:
        output = _normalize_clinical_output(output)
    except ValueError as exc:
        _log_rejected_response(client.model, "schema_validation", content, [{"message": str(exc)}])
        raise ClinicalResponseError("schema_validation") from exc

    try:
        return ClinicalDecisionSchema(**output)
    except ValidationError as exc:
        _log_rejected_response(client.model, "schema_validation", content)
        raise ClinicalResponseError("schema_validation") from exc


def get_clinical_analysis_runner():
    """
    Return the appropriate analysis runner based on AI_BACKEND setting.

    AI_BACKEND=local  → OllamaClient (patient data stays on-premises)
    AI_BACKEND=groq   → GroqClient (patient data sent to US cloud)

    Default is 'local' when USE_LOCAL_MODEL=true.
    """
    use_local = getattr(settings, "USE_LOCAL_MODEL", False)
    backend = getattr(settings, "AI_BACKEND", "groq")

    if use_local or backend == "local":
        return run_clinical_analysis_local
    else:
        from clinicalDecisionAI.services import run_clinical_analysis
        return run_clinical_analysis
