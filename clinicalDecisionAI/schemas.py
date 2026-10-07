from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ClinicalAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    patient_id: int | str | None = None
    age: int | None = Field(default=None, ge=0, le=130)
    sex: str | None = None
    symptoms: list[str] = Field(default_factory=list)
    medical_history: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    current_medications: list[str] = Field(default_factory=list)
    vitals: dict[str, int | float | str | None] = Field(default_factory=dict)
    laboratory_results: list[str | dict[str, Any]] = Field(default_factory=list)
    radiology_results: list[str | dict[str, Any]] = Field(default_factory=list)
    previous_diagnoses: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)


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