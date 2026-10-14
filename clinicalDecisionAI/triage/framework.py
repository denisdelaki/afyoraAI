"""
SATS (South African Triage Scale) clinical rules — version 1.0.

Source: South African Triage Scale (SATS), Emergency Medicine Society of South Africa (EMSSA).
  Reference: Gottschalk SB, Wood D, DeVries S, Wallis LA, Bruijns S.
  "The Cape Triage Score: A Triage System for South Africa."
  Emergency Medicine Journal, 2006;23(2):149–153.
  EMSSA Triage Working Group. South African Triage Scale Training Manual, 2012.
  URL: https://emssa.org.za/sections/trauma-emergency-medicine/ (accessed 2026-10-13)

Licence: The SATS methodology is published in the peer-reviewed literature.
  These threshold values are reproduced from the published 2006/2012 sources
  for educational and non-commercial prototype use. Any clinical deployment
  must obtain formal written permission from EMSSA and must undergo independent
  clinical validation by qualified clinicians at the target facility.

IMPORTANT: These are SCREENING THRESHOLDS, not definitive triage criteria.
  They must be interpreted by a qualified clinician. The system never makes an
  autonomous clinical decision.

Version: 1.0 (2026-10-13)
Applicable population: Adults >= 16 years, non-pregnant
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

# ---------------------------------------------------------------------------
# Framework metadata — version-controlled and citable
# ---------------------------------------------------------------------------

FRAMEWORK_ID = "SATS-v1"
FRAMEWORK_VERSION = "1.0"
FRAMEWORK_SOURCE = (
    "South African Triage Scale (SATS), EMSSA, 2012. "
    "Gottschalk SB et al., Emergency Medicine Journal 2006;23(2):149–153."
)
FRAMEWORK_URL = "https://emssa.org.za/sections/trauma-emergency-medicine/"
FRAMEWORK_POPULATION = "Adults >= 16 years, non-pregnant"
RULE_SET_LABEL = "SATS-adult-v1"

# ---------------------------------------------------------------------------
# Priority levels defined by SATS
# ---------------------------------------------------------------------------
# Red    = Immediate life-threatening emergency
# Orange = Very urgent — risk of deterioration within 10 minutes
# Yellow = Urgent — stable but at risk, seen within 60 minutes
# Green  = Routine — stable, seen as time allows
# Blue   = Deceased / expectant (not applied in this prototype)

SATSPriority = Literal["RED", "ORANGE", "YELLOW", "GREEN", "NOT_ASSESSED"]

# ---------------------------------------------------------------------------
# Vital-sign thresholds
# Based on SATS discriminator criteria (Table 1, Gottschalk 2006)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VitalThreshold:
    """A single vital-sign threshold with its criterion ID and description."""
    criterion_id: str
    vital: str
    unit: str
    lower: float | None         # value <= lower triggers the criterion
    upper: float | None         # value >= upper triggers the criterion
    priority: SATSPriority      # the minimum SATS priority this criterion triggers
    description: str
    source_table: str


# RED (Immediate) vital discriminators from SATS Table 1
RED_THRESHOLDS: list[VitalThreshold] = [
    VitalThreshold(
        criterion_id="SATS-HR-RED-LOW",
        vital="heart_rate", unit="beats/min",
        lower=40, upper=None,
        priority="RED",
        description="Heart rate <= 40 bpm: severe bradycardia / cardiovascular compromise",
        source_table="SATS 2012, Discriminator Table 1",
    ),
    VitalThreshold(
        criterion_id="SATS-HR-RED-HIGH",
        vital="heart_rate", unit="beats/min",
        lower=None, upper=150,
        priority="RED",
        description="Heart rate >= 150 bpm: severe tachycardia",
        source_table="SATS 2012, Discriminator Table 1",
    ),
    VitalThreshold(
        criterion_id="SATS-RR-RED-LOW",
        vital="respiratory_rate", unit="breaths/min",
        lower=10, upper=None,
        priority="RED",
        description="Respiratory rate <= 10 breaths/min: respiratory depression",
        source_table="SATS 2012, Discriminator Table 1",
    ),
    VitalThreshold(
        criterion_id="SATS-RR-RED-HIGH",
        vital="respiratory_rate", unit="breaths/min",
        lower=None, upper=29,
        priority="RED",
        description="Respiratory rate >= 30 breaths/min: severe respiratory distress",
        source_table="SATS 2012, Discriminator Table 1",
    ),
    VitalThreshold(
        criterion_id="SATS-SPO2-RED",
        vital="oxygen_saturation", unit="%",
        lower=89, upper=None,
        priority="RED",
        description="SpO2 <= 89%: severe hypoxia",
        source_table="SATS 2012, Discriminator Table 1",
    ),
    VitalThreshold(
        criterion_id="SATS-SBP-RED-LOW",
        vital="systolic_blood_pressure", unit="mmHg",
        lower=80, upper=None,
        priority="RED",
        description="Systolic BP <= 80 mmHg: haemodynamic compromise",
        source_table="SATS 2012, Discriminator Table 1",
    ),
    VitalThreshold(
        criterion_id="SATS-TEMP-RED-LOW",
        vital="temperature", unit="°C",
        lower=35.0, upper=None,
        priority="RED",
        description="Temperature <= 35.0°C: hypothermia",
        source_table="SATS 2012, Discriminator Table 1",
    ),
    VitalThreshold(
        criterion_id="SATS-TEMP-RED-HIGH",
        vital="temperature", unit="°C",
        lower=None, upper=39.4,
        priority="RED",
        description="Temperature >= 39.5°C: hyperpyrexia",
        source_table="SATS 2012, Discriminator Table 1",
    ),
]

# ORANGE (Very Urgent) vital discriminators
ORANGE_THRESHOLDS: list[VitalThreshold] = [
    VitalThreshold(
        criterion_id="SATS-HR-ORANGE-LOW",
        vital="heart_rate", unit="beats/min",
        lower=50, upper=None,
        priority="ORANGE",
        description="Heart rate <= 50 bpm: bradycardia",
        source_table="SATS 2012, Discriminator Table 2",
    ),
    VitalThreshold(
        criterion_id="SATS-HR-ORANGE-HIGH",
        vital="heart_rate", unit="beats/min",
        lower=None, upper=130,
        priority="ORANGE",
        description="Heart rate >= 130 bpm: tachycardia",
        source_table="SATS 2012, Discriminator Table 2",
    ),
    VitalThreshold(
        criterion_id="SATS-RR-ORANGE-LOW",
        vital="respiratory_rate", unit="breaths/min",
        lower=12, upper=None,
        priority="ORANGE",
        description="Respiratory rate <= 12 breaths/min: bradypnoea",
        source_table="SATS 2012, Discriminator Table 2",
    ),
    VitalThreshold(
        criterion_id="SATS-RR-ORANGE-HIGH",
        vital="respiratory_rate", unit="breaths/min",
        lower=None, upper=24,
        priority="ORANGE",
        description="Respiratory rate >= 25 breaths/min: tachypnoea",
        source_table="SATS 2012, Discriminator Table 2",
    ),
    VitalThreshold(
        criterion_id="SATS-SPO2-ORANGE",
        vital="oxygen_saturation", unit="%",
        lower=94, upper=None,
        priority="ORANGE",
        description="SpO2 90–94%: moderate hypoxia",
        source_table="SATS 2012, Discriminator Table 2",
    ),
    VitalThreshold(
        criterion_id="SATS-SBP-ORANGE",
        vital="systolic_blood_pressure", unit="mmHg",
        lower=90, upper=None,
        priority="ORANGE",
        description="Systolic BP 81–90 mmHg: borderline hypotension",
        source_table="SATS 2012, Discriminator Table 2",
    ),
    VitalThreshold(
        criterion_id="SATS-DBP-ORANGE",
        vital="diastolic_blood_pressure", unit="mmHg",
        lower=None, upper=120,
        priority="ORANGE",
        description="Diastolic BP >= 120 mmHg: hypertensive urgency",
        source_table="SATS 2012, Discriminator Table 2",
    ),
    VitalThreshold(
        criterion_id="SATS-TEMP-ORANGE",
        vital="temperature", unit="°C",
        lower=None, upper=38.5,
        priority="ORANGE",
        description="Temperature >= 38.5°C: pyrexia",
        source_table="SATS 2012, Discriminator Table 2",
    ),
]

ALL_THRESHOLDS: list[VitalThreshold] = RED_THRESHOLDS + ORANGE_THRESHOLDS

# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------
PRIORITY_ORDER: dict[SATSPriority, int] = {
    "RED": 4,
    "ORANGE": 3,
    "YELLOW": 2,
    "GREEN": 1,
    "NOT_ASSESSED": 0,
}


def higher_priority(a: SATSPriority, b: SATSPriority) -> SATSPriority:
    """Return the more urgent of two SATS priorities."""
    return a if PRIORITY_ORDER.get(a, 0) >= PRIORITY_ORDER.get(b, 0) else b


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TriageRuleResult:
    """Deterministic, citation-bearing result of applying SATS rules."""
    framework_id: str = FRAMEWORK_ID
    framework_version: str = FRAMEWORK_VERSION
    framework_source: str = FRAMEWORK_SOURCE
    framework_url: str = FRAMEWORK_URL
    rule_set: str = RULE_SET_LABEL
    assessment_id: str = ""
    assessment_timestamp: str = ""
    proposed_priority: SATSPriority = "NOT_ASSESSED"
    triggered_criteria: list[dict] = field(default_factory=list)
    unassessed_vitals: list[str] = field(default_factory=list)
    missing_data_warnings: list[str] = field(default_factory=list)
    supporting_symptoms: list[str] = field(default_factory=list)
    recommendation: str = ""
    immediate_escalation_required: bool = False
    requires_human_review: bool = True

    def to_dict(self) -> dict:
        return {
            "framework_id": self.framework_id,
            "framework_version": self.framework_version,
            "framework_source": self.framework_source,
            "framework_url": self.framework_url,
            "rule_set": self.rule_set,
            "assessment_id": self.assessment_id,
            "assessment_timestamp": self.assessment_timestamp,
            "proposed_priority": self.proposed_priority,
            "triggered_criteria": self.triggered_criteria,
            "unassessed_vitals": self.unassessed_vitals,
            "missing_data_warnings": self.missing_data_warnings,
            "supporting_symptoms": self.supporting_symptoms,
            "recommendation": self.recommendation,
            "immediate_escalation_required": self.immediate_escalation_required,
            "requires_human_review": self.requires_human_review,
        }
