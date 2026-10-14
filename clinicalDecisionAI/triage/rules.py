"""
Deterministic SATS rule engine.

Applies SATS vital-sign thresholds (see triage/framework.py) to validated
patient input. Does NOT call an LLM. Does NOT make autonomous clinical decisions.
Returns a TriageRuleResult with full citation trail.
"""
from __future__ import annotations

import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from clinicalDecisionAI.triage.framework import (
    ALL_THRESHOLDS,
    FRAMEWORK_ID,
    PRIORITY_ORDER,
    TriageRuleResult,
    SATSPriority,
    higher_priority,
)

# Vital aliases (maps canonical name → list of accepted field names)
_VITAL_ALIASES: dict[str, tuple[str, ...]] = {
    "heart_rate":               ("heart_rate", "pulse", "hr"),
    "respiratory_rate":         ("respiratory_rate", "rr"),
    "oxygen_saturation":        ("oxygen_saturation", "spo2", "o2_sat"),
    "systolic_blood_pressure":  ("systolic_blood_pressure", "systolic_bp", "sbp"),
    "diastolic_blood_pressure": ("diastolic_blood_pressure", "diastolic_bp", "dbp"),
    "temperature":              ("temperature", "temp", "temperature_celsius"),
}

# Escalation symptoms that flag immediate emergency messaging
_IMMEDIATE_ESCALATION_SYMPTOMS = frozenset({
    "unconscious", "unresponsive", "not breathing", "no pulse",
    "cardiac arrest", "seizure", "fitting", "major trauma", "penetrating injury",
    "airway obstruction", "choking", "severe bleeding", "haemorrhage",
})


def _parse_vitals(raw_vitals: dict[str, Any]) -> tuple[dict[str, float], list[str], list[str]]:
    """
    Parse raw vitals dict into canonical float values.

    Returns:
        parsed: canonical_name → float value
        unassessed: list of vital names that could not be parsed
        warnings: human-readable warnings for missing/invalid data
    """
    parsed: dict[str, float] = {}
    unassessed: list[str] = []
    warnings: list[str] = []

    # Handle blood_pressure string "systolic/diastolic"
    bp_raw = raw_vitals.get("blood_pressure")
    if isinstance(bp_raw, str):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*", bp_raw)
        if match:
            parsed["systolic_blood_pressure"] = float(match.group(1))
            parsed["diastolic_blood_pressure"] = float(match.group(2))
        else:
            unassessed.append("blood_pressure")
            warnings.append("blood_pressure: provided but could not be parsed (expected 'systolic/diastolic')")
    elif bp_raw is not None:
        unassessed.append("blood_pressure")
        warnings.append("blood_pressure: invalid type; expected 'systolic/diastolic' string")

    for canonical, aliases in _VITAL_ALIASES.items():
        # Skip if already parsed from blood_pressure string
        if canonical in parsed:
            continue
        values_found: list[float] = []
        conflicting = False
        for alias in aliases:
            raw = raw_vitals.get(alias)
            if raw is None:
                continue
            try:
                num = float(raw) if not isinstance(raw, bool) else float("nan")
            except (TypeError, ValueError, OverflowError):
                conflicting = True
                warnings.append(
                    f"{canonical}: value '{raw}' is not a valid number and was not assessed"
                )
                continue
            if not math.isfinite(num) or num < 0:
                conflicting = True
                warnings.append(
                    f"{canonical}: value {num} is implausible (must be finite and non-negative)"
                )
                continue
            if canonical == "oxygen_saturation" and num > 100:
                conflicting = True
                warnings.append(f"{canonical}: value {num}% exceeds 100 — implausible")
                continue
            values_found.append(num)

        if conflicting or len(set(values_found)) > 1:
            unassessed.append(canonical)
            if len(set(values_found)) > 1:
                warnings.append(
                    f"{canonical}: conflicting values {values_found} — not assessed"
                )
        elif values_found:
            parsed[canonical] = values_found[0]
        # else: vital genuinely absent — not warned here (reported as unassessed below)

    # Report vitals that are simply missing
    for canonical in _VITAL_ALIASES:
        if canonical not in parsed and canonical not in unassessed:
            # Check if any alias key was present (already handled above)
            aliases = _VITAL_ALIASES[canonical]
            if not any(a in raw_vitals for a in aliases) and canonical not in (
                "systolic_blood_pressure", "diastolic_blood_pressure"
            ):
                unassessed.append(canonical)
                warnings.append(
                    f"{canonical}: not provided — unassessed vitals are NOT treated as normal"
                )

    return parsed, unassessed, warnings


def apply_sats_rules(
    vitals: dict[str, Any],
    symptoms: list[str],
    age: int | None,
    is_pregnant: bool | None,
    assessment_id: str | None = None,
) -> TriageRuleResult:
    """
    Apply SATS adult vital-sign thresholds deterministically.

    This function is pure-deterministic: same inputs → same outputs.
    It does NOT call an LLM. All citations reference SATS source material.

    Args:
        vitals: Raw vitals dict from patient record.
        symptoms: List of reported symptoms (used for escalation check only).
        age: Patient age in years, or None if unknown.
        is_pregnant: Known pregnancy, or None if unknown.
        assessment_id: Optional caller-supplied ID; generated if absent.

    Returns:
        TriageRuleResult with full citation trail.
    """
    assessment_id = assessment_id or str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # Eligibility check
    if age is None or age < 16 or is_pregnant is True:
        reason = "unknown age" if age is None else (
            f"age {age} < 16 (paediatric)" if age < 16 else "known pregnancy"
        )
        return TriageRuleResult(
            assessment_id=assessment_id,
            assessment_timestamp=timestamp,
            proposed_priority="NOT_ASSESSED",
            unassessed_vitals=["adult_eligibility"],
            missing_data_warnings=[
                f"SATS adult rules not applied: {reason}. "
                "Obtain age-appropriate or obstetric triage assessment."
            ],
            recommendation=(
                "SATS adult vital-sign screening cannot be applied: patient does not meet "
                f"eligibility criteria ({reason}). Obtain appropriate specialist triage. "
                "Do not delay urgent care pending AI assessment."
            ),
            immediate_escalation_required=False,
            requires_human_review=True,
        )

    parsed_vitals, unassessed, warnings = _parse_vitals(vitals)

    triggered_criteria: list[dict] = []
    proposed_priority: SATSPriority = "GREEN"  # default if vitals present
    if not parsed_vitals:
        proposed_priority = "NOT_ASSESSED"

    # Apply each threshold deterministically
    for threshold in ALL_THRESHOLDS:
        value = parsed_vitals.get(threshold.vital)
        if value is None:
            continue
        triggered = False
        if threshold.lower is not None and value <= threshold.lower:
            triggered = True
        if threshold.upper is not None and value >= threshold.upper:
            triggered = True
        if triggered:
            triggered_criteria.append({
                "criterion_id": threshold.criterion_id,
                "vital": threshold.vital,
                "measured_value": value,
                "unit": threshold.unit,
                "threshold_direction": "low" if (threshold.lower and value <= threshold.lower) else "high",
                "threshold_value": threshold.lower if (threshold.lower and value <= threshold.lower) else threshold.upper,
                "priority": threshold.priority,
                "description": threshold.description,
                "source": threshold.source_table,
                "framework": FRAMEWORK_ID,
            })
            proposed_priority = higher_priority(proposed_priority, threshold.priority)

    # If all vitals are unassessed, the priority cannot be determined
    if unassessed and not triggered_criteria and proposed_priority == "GREEN":
        proposed_priority = "NOT_ASSESSED"

    # Symptom-based escalation check (messaging only — does not change priority)
    lower_symptoms = {s.lower() for s in symptoms}
    escalation_triggers = lower_symptoms & _IMMEDIATE_ESCALATION_SYMPTOMS
    immediate_escalation_required = bool(escalation_triggers) or proposed_priority == "RED"

    # Build recommendation
    recommendation = _build_recommendation(
        proposed_priority, triggered_criteria, unassessed, immediate_escalation_required
    )

    return TriageRuleResult(
        assessment_id=assessment_id,
        assessment_timestamp=timestamp,
        proposed_priority=proposed_priority,
        triggered_criteria=triggered_criteria,
        unassessed_vitals=unassessed,
        missing_data_warnings=warnings,
        supporting_symptoms=list(symptoms),
        recommendation=recommendation,
        immediate_escalation_required=immediate_escalation_required,
        requires_human_review=True,
    )


def _build_recommendation(
    priority: SATSPriority,
    triggered: list[dict],
    unassessed: list[str],
    immediate_escalation: bool,
) -> str:
    if immediate_escalation or priority == "RED":
        return (
            "IMMEDIATE EMERGENCY: Activate emergency procedures NOW. Do not delay for AI output "
            "or software steps. Call emergency services or activate the facility emergency "
            "response immediately. This message is a screening alert — clinical assessment "
            "by a qualified clinician is mandatory and must not be delayed."
        )
    if priority == "ORANGE":
        return (
            "VERY URGENT (SATS Orange): Abnormal vital signs detected. The patient should be seen "
            "by a clinician within 10 minutes. Recheck measurements if feasible, but do not delay "
            "care. This is a screening alert; a qualified clinician must confirm the priority."
        )
    if priority == "YELLOW":
        return (
            "URGENT (SATS Yellow): Some vital signs are outside normal ranges. Clinical review "
            "within 60 minutes. A qualified clinician must confirm the priority."
        )
    if priority == "GREEN":
        if unassessed:
            return (
                "Routine priority (SATS Green) based on available vitals, but some vital signs "
                "are missing or could not be assessed. Obtain and verify missing measurements. "
                "Missing vitals are NOT treated as normal. Clinician review required."
            )
        return (
            "No SATS vital-sign threshold triggered on available measurements. This does not "
            "establish low risk or exclude serious illness. Qualified clinician review is required. "
            "Seek urgent care for concerning symptoms."
        )
    # NOT_ASSESSED
    return (
        "Vital-sign triage could not be completed: insufficient or unassessable data. "
        "Obtain or verify missing measurements and seek clinician review. "
        "Do not delay urgent care for concerning symptoms. "
        "Missing vitals are NOT treated as normal."
    )
