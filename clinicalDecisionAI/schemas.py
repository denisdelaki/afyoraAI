from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator


ClinicalText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2048)]


class ClinicalDecisionSchema(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')

    supported_diagnosis: ClinicalText
    possible_disease: list[ClinicalText] = Field(max_length=64)
    drugs_admissible: list[ClinicalText] = Field(max_length=64)
    further_labs_to_be_done: list[ClinicalText] = Field(max_length=64)


class VitalTriageSchema(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')

    level: Literal['urgent', 'no_threshold_triggered', 'not_assessed']
    urgent_care_recommended: bool
    reasons: list[ClinicalText]
    unassessed_vitals: list[ClinicalText]
    recommendation: ClinicalText
    requires_human_review: Literal[True] = True
    rule_set: Literal['adult-vital-screening-v1'] = 'adult-vital-screening-v1'


class ClinicalAnalysisResponse(ClinicalDecisionSchema):
    triage: VitalTriageSchema


class ClinicalAnalysisRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', allow_inf_nan=False)

    patient_id: int | str | None = None
    age: int | None = Field(default=None, ge=0, le=130)
    sex: ClinicalText | None = None
    is_pregnant: bool | None = None
    symptoms: list[ClinicalText] = Field(min_length=1, max_length=64)
    medical_history: list[ClinicalText] = Field(default_factory=list, max_length=64)
    allergies: list[ClinicalText] = Field(default_factory=list, max_length=64)
    current_medications: list[ClinicalText] = Field(default_factory=list, max_length=64)
    vitals: dict[str, int | float | str | None] = Field(default_factory=dict, max_length=128)
    laboratory_results: list[ClinicalText | dict[str, JsonValue]] = Field(
        default_factory=list, max_length=64,
        validation_alias=AliasChoices('labs', 'laboratory_results'), serialization_alias='labs',
    )
    radiology_results: list[ClinicalText | dict[str, JsonValue]] = Field(
        default_factory=list, max_length=64,
        validation_alias=AliasChoices('radiology', 'radiology_results'), serialization_alias='radiology',
    )
    previous_diagnoses: list[ClinicalText] = Field(default_factory=list, max_length=64)
    observations: list[ClinicalText] = Field(default_factory=list, max_length=64)

    @model_validator(mode='before')
    @classmethod
    def validate_payload(cls, value):
        if not isinstance(value, dict):
            raise ValueError('Request body must be a JSON object.')
        for canonical, legacy in (('labs', 'laboratory_results'), ('radiology', 'radiology_results')):
            if canonical in value and legacy in value:
                raise ValueError(f'Supply only one of {canonical} and {legacy}.')
        identifier_keys = {
            'patient_id', 'patient_name', 'full_name', 'first_name', 'last_name',
            'mrn', 'medical_record_number', 'national_id', 'date_of_birth',
            'address', 'email', 'phone',
        }
        remaining_nodes = 4096

        def check(item, depth=0):
            nonlocal remaining_nodes
            remaining_nodes -= 1
            if depth > 6 or remaining_nodes < 0:
                raise ValueError('Clinical data exceeds the nesting or complexity limit.')
            if isinstance(item, str) and len(item) > 2048:
                raise ValueError('Clinical text must not exceed 2048 characters.')
            if isinstance(item, dict):
                for key, nested in item.items():
                    if not isinstance(key, str) or len(key) > 128:
                        raise ValueError('Clinical field names must be strings of at most 128 characters.')
                    if key.lower() in identifier_keys and not (depth == 0 and key == 'patient_id'):
                        raise ValueError('Remove identifying fields from clinical data before submission.')
                    check(nested, depth + 1)
            elif isinstance(item, list):
                for nested in item:
                    check(nested, depth + 1)

        check(value)
        return value


class PossibleCondition(BaseModel):
    model_config = ConfigDict(extra='forbid')

    condition: str
    likelihood: str
    supporting_evidence: list[str]


class ReferralRecommendation(BaseModel):
    model_config = ConfigDict(extra='forbid')

    required: bool
    urgency: Literal['routine', 'urgent', 'emergency']
    reason: str


class ClinicalDecisionResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    clinical_summary: str
    risk_level: Literal['low', 'moderate', 'high', 'critical']
    possible_conditions: list[PossibleCondition]
    abnormal_findings: list[str]
    recommended_investigations: list[str]
    management_considerations: list[str]
    medication_considerations: list[str]
    red_flags: list[str]
    referral_recommendation: ReferralRecommendation
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human_review: Literal[True]