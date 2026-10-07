import logging
import random

from django.conf import settings
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    RateLimitError,
)
from pydantic import ValidationError

from clinicalDecisionAI.prompts import SYSTEM_INSTRUCTION, build_clinical_prompt
from clinicalDecisionAI.schemas import ClinicalAnalysisRequest, ClinicalDecisionResponse

logger = logging.getLogger(__name__)


class GroqConfigurationError(Exception):
    pass


class GroqProviderError(Exception):
    def __init__(self, category: str, status_code: int | None = None):
        self.category = category
        self.status_code = status_code
        super().__init__(category)


class ClinicalResponseError(Exception):
    def __init__(self, category: str):
        self.category = category
        super().__init__(category)


def select_groq_model() -> str:
    models = getattr(settings, 'GROQ_MODELS', ()) or (settings.GROQ_MODEL,)
    return random.choice(models)


class GroqClient:
    def __init__(self, model: str | None = None):
        api_key = (settings.GROQ_API_KEY or '').strip()
        if not api_key:
            raise GroqConfigurationError

        self.model = model or select_groq_model()
        self.client = Groq(
            api_key=api_key,
            timeout=settings.GROQ_TIMEOUT_SECONDS,
        )

    def ask(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_INSTRUCTION},
                    {'role': 'user', 'content': prompt},
                ],
                response_format={'type': 'json_object'},
            )
        except AuthenticationError as exc:
            raise GroqProviderError('authentication') from exc
        except RateLimitError as exc:
            raise GroqProviderError('rate_limit') from exc
        except (APITimeoutError, TimeoutError) as exc:
            raise GroqProviderError('timeout') from exc
        except APIConnectionError as exc:
            raise GroqProviderError('network') from exc
        except APIStatusError as exc:
            logger.warning(
                'Groq request failed with HTTP status %s (%s)',
                exc.status_code,
                type(exc).__name__,
            )
            raise GroqProviderError('provider', status_code=exc.status_code) from exc
        except Exception as exc:
            logger.warning(
                'Groq request failed (%s)',
                type(exc).__name__,
            )
            raise GroqProviderError('provider') from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            content = None
        if not isinstance(content, str) or not content.strip():
            raise ClinicalResponseError('empty_response')
        return content


def run_clinical_analysis(
    patient: ClinicalAnalysisRequest,
    model: str | None = None,
) -> ClinicalDecisionResponse:
    client = GroqClient(model=model)
    content = client.ask(build_clinical_prompt(patient))
    try:
        return ClinicalDecisionResponse.model_validate_json(content)
    except ValidationError as exc:
        category = (
            'invalid_json'
            if any(error['type'] == 'json_invalid' for error in exc.errors())
            else 'schema_validation'
        )
        logger.warning('Clinical AI response rejected: %s', category)
        raise ClinicalResponseError(category) from exc