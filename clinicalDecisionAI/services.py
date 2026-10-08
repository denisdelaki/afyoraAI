import json
import logging
import math
import random
import re

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
from clinicalDecisionAI.schemas import ClinicalAnalysisRequest, ClinicalDecisionSchema, VitalTriageSchema

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


def assess_vital_triage(patient: ClinicalAnalysisRequest) -> VitalTriageSchema:
    if patient.age is None or patient.age < 16 or patient.is_pregnant is True:
        return VitalTriageSchema(
            level='not_assessed', urgent_care_recommended=False, reasons=[],
            unassessed_vitals=['adult eligibility'],
            recommendation='Adult vital screening cannot be applied without age >=16, or during pregnancy. '
            'Obtain age-appropriate or obstetric assessment. Do not delay urgent care for concerning symptoms or abnormal vitals.',
        )

    readings = dict(patient.vitals)
    blood_pressure_values = {}
    reasons = []
    unassessed = []
    if isinstance(readings.get('blood_pressure'), str):
        match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*', readings['blood_pressure'])
        if match:
            blood_pressure_values['systolic_blood_pressure'] = float(match.group(1))
            blood_pressure_values['diastolic_blood_pressure'] = float(match.group(2))
        else:
            unassessed.append('blood_pressure')
    elif 'blood_pressure' in readings:
        unassessed.append('blood_pressure')

    rules = (
        ('heart_rate', ('heart_rate', 'pulse'), 40, 131, 'beats/min'),
        ('respiratory_rate', ('respiratory_rate',), 8, 25, 'breaths/min'),
        ('oxygen_saturation', ('oxygen_saturation', 'spo2'), 91, None, '%'),
        ('systolic_blood_pressure', ('systolic_blood_pressure', 'systolic_bp'), 90, 180, 'mmHg'),
        ('diastolic_blood_pressure', ('diastolic_blood_pressure', 'diastolic_bp'), None, 120, 'mmHg'),
        ('temperature', ('temperature', 'temperature_celsius'), 35, 39.1, 'C'),
    )
    for name, aliases, lower, upper, unit in rules:
        supplied = [readings[key] for key in aliases if key in readings]
        if name in blood_pressure_values:
            supplied.append(blood_pressure_values[name])
        valid = []
        invalid = not supplied
        for value in supplied:
            try:
                number = float(value) if not isinstance(value, bool) else float('nan')
            except (TypeError, ValueError, OverflowError):
                invalid = True
                continue
            if not math.isfinite(number) or number < 0 or (name == 'oxygen_saturation' and number > 100):
                invalid = True
                continue
            valid.append(number)
            if (lower is not None and number <= lower) or (upper is not None and number >= upper):
                reasons.append(f'{name}: {number:g} {unit} meets an urgent screening threshold.')
        if invalid or len(set(valid)) > 1:
            unassessed.append(name)

    if reasons:
        level = 'urgent'
        recommendation = (
            'Urgent in-person clinical assessment is recommended now. Recheck measurements if feasible, '
            'but do not delay care or wait for AI advice. Call local emergency services for severe breathlessness, '
            'chest pain, collapse, or new confusion. These thresholds are screening alerts, not diagnoses.'
        )
    elif unassessed:
        level = 'not_assessed'
        recommendation = (
            'Vital triage is incomplete: obtain or verify missing, invalid, or conflicting measurements and seek clinician review. '
            'Do not delay urgent care for concerning symptoms. Unassessed vitals are not normal findings.'
        )
    else:
        level = 'no_threshold_triggered'
        recommendation = (
            'No configured adult vital threshold was triggered. This does not establish low risk or rule out serious illness. '
            'Qualified clinician review remains required; seek urgent care for concerning symptoms.'
        )
    return VitalTriageSchema(
        level=level, urgent_care_recommended=bool(reasons), reasons=reasons,
        unassessed_vitals=unassessed, recommendation=recommendation,
    )


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
            max_retries=0,
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
) -> ClinicalDecisionSchema:
    client = GroqClient(model=model)
    content = client.ask(build_clinical_prompt(patient))
    try:
        output = json.loads(content)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ClinicalResponseError('invalid_json') from exc
    if not isinstance(output, dict):
        raise ClinicalResponseError('schema_validation')
    try:
        return ClinicalDecisionSchema(**output)
    except ValidationError as exc:
        category = 'schema_validation'
        logger.warning('Clinical AI response rejected: %s', category)
        raise ClinicalResponseError(category) from exc