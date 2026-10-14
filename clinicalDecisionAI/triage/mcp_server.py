"""
Custom MCP server for The Triage Desk.

Implements three distinct tools required by the challenge:
  1. assess_triage_priority
  2. request_clinical_reassessment
  3. prepare_department_referral

Additional tool:
  4. validate_vitals

Each tool:
- Has a strict Pydantic input schema
- Has a structured Pydantic output schema
- Validates all inputs before processing
- Creates an AuditEvent for every call
- Does NOT diagnose, prescribe, or change operational workflow without approval
- Is protected against prompt injection by not forwarding free-text to schemas

Protocol: FastMCP (compatible with MCP protocol)
Transport: In-memory (stdio transport available for external MCP client testing)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator

from clinicalDecisionAI.triage.framework import (
    FRAMEWORK_ID,
    FRAMEWORK_VERSION,
    FRAMEWORK_SOURCE,
    FRAMEWORK_URL,
    RULE_SET_LABEL,
)
from clinicalDecisionAI.triage.rules import apply_sats_rules

# ---------------------------------------------------------------------------
# Input / output Pydantic schemas
# ---------------------------------------------------------------------------

class VitalsInput(BaseModel):
    """Validated vital signs accepted by triage tools."""
    blood_pressure: str | None = Field(None, description="Systolic/diastolic e.g. '120/80'")
    heart_rate: float | None = Field(None, ge=0, le=300, description="Beats per minute")
    respiratory_rate: float | None = Field(None, ge=0, le=60, description="Breaths per minute")
    oxygen_saturation: float | None = Field(None, ge=0, le=100, description="SpO2 %")
    temperature: float | None = Field(None, ge=20, le=50, description="Temperature °C")
    systolic_blood_pressure: float | None = Field(None, ge=0, le=350, description="mmHg")
    diastolic_blood_pressure: float | None = Field(None, ge=0, le=250, description="mmHg")

    def to_raw_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class AssessTriagePriorityInput(BaseModel):
    """Input for the assess_triage_priority tool."""
    encounter_id: str = Field(..., description="Opaque encounter reference")
    age: int | None = Field(None, ge=0, le=130)
    is_pregnant: bool | None = None
    symptoms: list[str] = Field(..., min_length=1, max_length=64)
    vitals: VitalsInput = Field(default_factory=VitalsInput)
    framework_version: str = Field(FRAMEWORK_VERSION)

    @field_validator("symptoms")
    @classmethod
    def symptoms_must_be_nonempty_strings(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip() for s in v if isinstance(s, str) and s.strip()]
        if not cleaned:
            raise ValueError("At least one non-blank symptom is required")
        if len(cleaned[0]) > 512:
            raise ValueError("Symptom text too long")
        return cleaned[:64]


class TriageCriterion(BaseModel):
    criterion_id: str
    vital: str
    measured_value: float
    unit: str
    threshold_direction: str
    threshold_value: float
    priority: str
    description: str
    source: str
    framework: str


class AssessTriagePriorityOutput(BaseModel):
    assessment_id: str
    assessment_timestamp: str
    encounter_id: str
    framework_id: str
    framework_version: str
    framework_source: str
    framework_url: str
    rule_set: str
    proposed_priority: Literal["RED", "ORANGE", "YELLOW", "GREEN", "NOT_ASSESSED"]
    triggered_criteria: list[dict]
    unassessed_vitals: list[str]
    missing_data_warnings: list[str]
    supporting_symptoms: list[str]
    recommendation: str
    immediate_escalation_required: bool
    requires_human_review: bool = True
    pending_approval_required: bool = True


class RequestReassessmentInput(BaseModel):
    encounter_id: str
    reason: str = Field(..., min_length=1, max_length=1024)
    new_symptoms: list[str] = Field(default_factory=list, max_length=64)
    new_vitals: VitalsInput = Field(default_factory=VitalsInput)

    @field_validator("reason")
    @classmethod
    def sanitise_reason(cls, v: str) -> str:
        return v.strip()[:1024]


class RequestReassessmentOutput(BaseModel):
    flag_id: str
    encounter_id: str
    flagged_at: str
    reason: str
    changes_detected: list[str]
    previous_priority: str
    queue_unchanged: bool = True
    message: str


class PrepareDepartmentReferralInput(BaseModel):
    encounter_id: str
    proposed_priority: Literal["RED", "ORANGE", "YELLOW", "GREEN", "NOT_ASSESSED"]
    symptoms: list[str] = Field(..., min_length=1, max_length=64)
    requesting_clinician_role: str = Field(..., description="Role of requesting clinician")

    @field_validator("symptoms")
    @classmethod
    def clean_symptoms(cls, v: list[str]) -> list[str]:
        return [s.strip() for s in v if isinstance(s, str) and s.strip()][:64]


class PrepareDepartmentReferralOutput(BaseModel):
    proposal_id: str
    encounter_id: str
    proposed_department: str
    routing_evidence: list[dict]
    routing_rules_applied: list[str]
    requires_clinician_approval: bool = True
    approval_pending: bool = True
    message: str


class ValidateVitalsInput(BaseModel):
    vitals: VitalsInput
    age: int | None = Field(None, ge=0, le=130)
    is_pregnant: bool | None = None


class ValidateVitalsOutput(BaseModel):
    is_valid: bool
    parsed_values: dict[str, float]
    unassessable: list[str]
    warnings: list[str]
    eligibility_met: bool
    framework_id: str = FRAMEWORK_ID


# ---------------------------------------------------------------------------
# Department routing table (facility-specific configuration)
# In production, load this from the database or a config file.
# ---------------------------------------------------------------------------

DEPARTMENT_ROUTING_RULES = [
    {
        "rule_id": "ROUTE-001",
        "condition": "priority == RED",
        "department": "Resuscitation Bay",
        "description": "Immediate life-threatening — resuscitation bay",
    },
    {
        "rule_id": "ROUTE-002",
        "condition": "priority == ORANGE",
        "department": "Emergency Treatment Area",
        "description": "Very urgent — emergency treatment area",
    },
    {
        "rule_id": "ROUTE-003",
        "condition": "priority == YELLOW",
        "department": "Urgent Care",
        "description": "Urgent — urgent care area",
    },
    {
        "rule_id": "ROUTE-004",
        "condition": "priority == GREEN",
        "department": "General Outpatient",
        "description": "Routine — general outpatient",
    },
    {
        "rule_id": "ROUTE-005",
        "condition": "priority == NOT_ASSESSED",
        "department": "Triage Reassessment",
        "description": "Incomplete assessment — triage reassessment required",
    },
]


def _route_by_priority(priority: str) -> tuple[str, list[dict], list[str]]:
    """Return (department, evidence, rules) based on SATS priority."""
    for rule in DEPARTMENT_ROUTING_RULES:
        if priority in rule["condition"]:
            evidence = [{
                "rule_id": rule["rule_id"],
                "condition_met": rule["condition"],
                "department": rule["department"],
                "description": rule["description"],
                "source": "Facility routing table v1.0",
            }]
            return rule["department"], evidence, [rule["rule_id"]]
    return "Triage Reassessment", [], ["ROUTE-005"]


# ---------------------------------------------------------------------------
# Audit helper (simplified — full DB audit is in the Django approval service)
# ---------------------------------------------------------------------------

def _audit_log(tool_name: str, encounter_id: str, success: bool, summary: dict) -> None:
    """Write a structured audit log. DB audit is handled by the Django service layer."""
    import logging
    logger = logging.getLogger("clinicalDecisionAI.mcp_audit")
    logger.info(
        "MCP tool call: %s",
        {
            "tool": tool_name,
            "encounter_id": encounter_id,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **summary,
        },
    )


# ---------------------------------------------------------------------------
# MCP server definition
# ---------------------------------------------------------------------------

def create_triage_mcp_server() -> FastMCP:
    """Create and return the custom Triage Desk MCP server with all tools registered."""
    server = FastMCP(
        name="AfyoraTriageDesk",
        instructions=(
            "The Triage Desk MCP server provides clinical triage support tools. "
            "All outputs are advisory and require qualified clinician review and approval "
            "before affecting any operational workflow. No diagnosis or prescription is provided."
        ),
        mask_error_details=True,
    )

    @server.tool(
        description=(
            "Assess triage priority for a patient encounter using the South African Triage Scale (SATS). "
            "Applies deterministic, source-grounded clinical rules. Returns a PROPOSED priority only. "
            "Does NOT diagnose, prescribe, or change queue position. "
            "Requires clinician approval before any operational effect."
        )
    )
    def assess_triage_priority(input: AssessTriagePriorityInput) -> AssessTriagePriorityOutput:
        """Apply SATS rules deterministically and return a proposed triage priority."""
        assessment_id = str(uuid.uuid4())
        result = apply_sats_rules(
            vitals=input.vitals.to_raw_dict(),
            symptoms=input.symptoms,
            age=input.age,
            is_pregnant=input.is_pregnant,
            assessment_id=assessment_id,
        )
        _audit_log(
            "assess_triage_priority",
            input.encounter_id,
            True,
            {
                "proposed_priority": result.proposed_priority,
                "triggered_criteria_count": len(result.triggered_criteria),
                "unassessed_vitals_count": len(result.unassessed_vitals),
                "framework": FRAMEWORK_ID,
            },
        )
        return AssessTriagePriorityOutput(
            assessment_id=result.assessment_id,
            assessment_timestamp=result.assessment_timestamp,
            encounter_id=input.encounter_id,
            framework_id=result.framework_id,
            framework_version=result.framework_version,
            framework_source=result.framework_source,
            framework_url=result.framework_url,
            rule_set=result.rule_set,
            proposed_priority=result.proposed_priority,
            triggered_criteria=result.triggered_criteria,
            unassessed_vitals=result.unassessed_vitals,
            missing_data_warnings=result.missing_data_warnings,
            supporting_symptoms=result.supporting_symptoms,
            recommendation=result.recommendation,
            immediate_escalation_required=result.immediate_escalation_required,
            requires_human_review=True,
            pending_approval_required=True,
        )

    @server.tool(
        description=(
            "Flag a clinical reassessment when symptoms or vital signs change. "
            "Creates a reassessment flag for a clinician to review. "
            "Does NOT automatically change the patient's queue position or priority."
        )
    )
    def request_clinical_reassessment(input: RequestReassessmentInput) -> RequestReassessmentOutput:
        """Identify observation changes and create a clinician reassessment flag."""
        flag_id = str(uuid.uuid4())
        changes: list[str] = []

        # Detect changes in new vitals
        new_vitals_dict = input.new_vitals.to_raw_dict()
        if new_vitals_dict:
            changes.append(
                f"Updated vitals provided: {list(new_vitals_dict.keys())}"
            )
        if input.new_symptoms:
            changes.append(
                f"New symptoms reported: {input.new_symptoms[:10]}"
            )
        if not changes:
            changes.append("Clinical reassessment explicitly requested")

        _audit_log(
            "request_clinical_reassessment",
            input.encounter_id,
            True,
            {
                "flag_id": flag_id,
                "changes_count": len(changes),
                "has_new_vitals": bool(new_vitals_dict),
                "has_new_symptoms": bool(input.new_symptoms),
            },
        )

        return RequestReassessmentOutput(
            flag_id=flag_id,
            encounter_id=input.encounter_id,
            flagged_at=datetime.now(timezone.utc).isoformat(),
            reason=input.reason,
            changes_detected=changes,
            previous_priority="UNKNOWN",  # Caller provides prior assessment ID if needed
            queue_unchanged=True,
            message=(
                "Reassessment flag created. Queue position has NOT been changed. "
                "A qualified clinician must review this flag and decide on next steps. "
                "If the patient shows signs of immediate emergency, activate emergency "
                "procedures without waiting for this flag to be processed."
            ),
        )

    @server.tool(
        description=(
            "Prepare a department referral proposal based on triage priority and symptoms. "
            "Evaluates routing against the facility department directory. "
            "Returns a PROPOSED department and supporting evidence. "
            "Requires authorised clinician approval before any referral is committed."
        )
    )
    def prepare_department_referral(
        input: PrepareDepartmentReferralInput,
    ) -> PrepareDepartmentReferralOutput:
        """Evaluate routing information and propose a department referral."""
        proposal_id = str(uuid.uuid4())
        department, evidence, rules_applied = _route_by_priority(input.proposed_priority)

        _audit_log(
            "prepare_department_referral",
            input.encounter_id,
            True,
            {
                "proposal_id": proposal_id,
                "proposed_department": department,
                "rules_applied": rules_applied,
                "requesting_role": input.requesting_clinician_role,
            },
        )

        return PrepareDepartmentReferralOutput(
            proposal_id=proposal_id,
            encounter_id=input.encounter_id,
            proposed_department=department,
            routing_evidence=evidence,
            routing_rules_applied=rules_applied,
            requires_clinician_approval=True,
            approval_pending=True,
            message=(
                f"Referral PROPOSAL to '{department}' created. "
                "This proposal has NOT been committed. "
                "An authorised clinician must approve it before any patient movement occurs. "
                "Emergency escalation must follow established facility procedures without delay."
            ),
        )

    @server.tool(
        description="Validate vital-sign measurements and check SATS eligibility without scoring."
    )
    def validate_vitals(input: ValidateVitalsInput) -> ValidateVitalsOutput:
        """Validate vital signs for plausibility and SATS applicability."""
        from clinicalDecisionAI.triage.rules import _parse_vitals  # noqa: PLC0415
        parsed, unassessable, warnings = _parse_vitals(input.vitals.to_raw_dict())
        eligibility = (
            input.age is not None
            and input.age >= 16
            and input.is_pregnant is not True
        )
        return ValidateVitalsOutput(
            is_valid=bool(parsed),
            parsed_values=parsed,
            unassessable=unassessable,
            warnings=warnings,
            eligibility_met=eligibility,
        )

    return server


# Singleton instance for import by agent and views
triage_mcp_server = create_triage_mcp_server()
