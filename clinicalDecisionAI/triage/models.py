"""
Triage database models.

Implements:
- TriageEncounter: a single patient triage session
- TriageAssessment: AI-proposed triage assessment with full citation trail
- TriageApproval: clinician approval or rejection of a proposed assessment
- ReassessmentFlag: flag for clinical reassessment when observations change
- ReferralProposal: AI-proposed department referral (requires approval)
- AuditEvent: immutable structured audit log for every tool call / action
- ClinicalRole: role-based access control for clinicians
"""

from __future__ import annotations

import hashlib
import json
import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone


class ClinicalRole(models.Model):
    """Role assignment for a user in the clinical workflow."""

    NURSE = "NURSE"
    DOCTOR = "DOCTOR"
    ADMIN = "ADMIN"

    ROLE_CHOICES = [
        (NURSE, "Nurse / Triage Clinician"),
        (DOCTOR, "Doctor / Physician"),
        (ADMIN, "Administrator"),
    ]

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="clinical_role",
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    assigned_at = models.DateTimeField(auto_now_add=True)
    assigned_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="clinical_role_assignments",
    )

    class Meta:
        verbose_name = "Clinical Role"

    def __str__(self) -> str:
        return f"{self.user.username} ({self.role})"


class TriageEncounter(models.Model):
    """
    A single patient triage encounter.

    Stores de-identified clinical data. Never stores patient name, MRN,
    national ID, date of birth, or contact details. The caller-side
    patient_id is an opaque reference only.
    """

    STATUS_WAITING = "WAITING"
    STATUS_ASSESSED = "ASSESSED"
    STATUS_APPROVED = "APPROVED"
    STATUS_REFERRED = "REFERRED"
    STATUS_DISCHARGED = "DISCHARGED"

    STATUS_CHOICES = [
        (STATUS_WAITING, "Waiting — assessment pending"),
        (STATUS_ASSESSED, "Assessed — AI priority proposed"),
        (STATUS_APPROVED, "Approved — clinician confirmed priority"),
        (STATUS_REFERRED, "Referred — department referral proposed"),
        (STATUS_DISCHARGED, "Discharged"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    # Caller-supplied opaque reference; not used for clinical decisions
    caller_encounter_ref = models.CharField(max_length=128, blank=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    registered_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="triage_encounters",
    )
    # De-identified clinical data (JSON)
    age = models.PositiveSmallIntegerField(null=True, blank=True)
    sex = models.CharField(max_length=32, blank=True)
    is_pregnant = models.BooleanField(null=True, blank=True)
    symptoms_json = models.JSONField(default=list)
    vitals_json = models.JSONField(default=dict)
    medical_history_json = models.JSONField(default=list)

    workflow_status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default=STATUS_WAITING, db_index=True
    )
    # AI-proposed priority (NOT operational until approved)
    proposed_priority = models.CharField(max_length=20, blank=True)
    # Clinician-confirmed priority (operational)
    confirmed_priority = models.CharField(max_length=20, blank=True)

    # Hash of clinical input used to detect stale approvals
    clinical_input_hash = models.CharField(max_length=64, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Triage Encounter"

    def __str__(self) -> str:
        return f"Encounter {self.id} [{self.workflow_status}]"

    def compute_input_hash(self) -> str:
        """SHA-256 of the clinical input fields; used to detect stale approvals."""
        payload = {
            "age": self.age,
            "sex": self.sex,
            "is_pregnant": self.is_pregnant,
            "symptoms": sorted(self.symptoms_json),
            "vitals": dict(sorted(self.vitals_json.items())),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()
        ).hexdigest()

    def save(self, *args, **kwargs):
        self.clinical_input_hash = self.compute_input_hash()
        super().save(*args, **kwargs)


class TriageAssessment(models.Model):
    """
    An AI-proposed triage assessment for a TriageEncounter.

    Stores the full citation trail from the SATS rule engine.
    This is a PROPOSAL only; it has no operational effect until a clinician approves it.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    encounter = models.ForeignKey(
        TriageEncounter,
        on_delete=models.CASCADE,
        related_name="assessments",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    framework_id = models.CharField(max_length=64)
    framework_version = models.CharField(max_length=32)
    framework_source = models.TextField()
    framework_url = models.URLField(blank=True)
    rule_set = models.CharField(max_length=64)
    proposed_priority = models.CharField(max_length=20)
    triggered_criteria_json = models.JSONField(default=list)
    unassessed_vitals_json = models.JSONField(default=list)
    missing_data_warnings_json = models.JSONField(default=list)
    supporting_symptoms_json = models.JSONField(default=list)
    recommendation = models.TextField()
    immediate_escalation_required = models.BooleanField(default=False)
    # LLM-generated advisory (not used for priority)
    llm_summary = models.TextField(blank=True)
    llm_model_used = models.CharField(max_length=128, blank=True)
    # Clinical input hash at time of assessment (for stale-approval detection)
    clinical_input_hash = models.CharField(max_length=64)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Triage Assessment"

    def __str__(self) -> str:
        return f"Assessment {self.id} → {self.proposed_priority}"


class TriageApproval(models.Model):
    """
    A clinician approval or rejection of a TriageAssessment.

    Only a user with ClinicalRole can create an approval record.
    The AI cannot write to this model.
    """

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"

    STATUS_CHOICES = [
        (PENDING, "Pending clinician decision"),
        (APPROVED, "Approved — priority confirmed"),
        (REJECTED, "Rejected — priority not confirmed"),
        (EXPIRED, "Expired — assessment too old"),
        (INVALIDATED, "Invalidated — clinical input changed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    assessment = models.ForeignKey(
        TriageAssessment,
        on_delete=models.CASCADE,
        related_name="approvals",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=PENDING, db_index=True)

    # Clinician identity (populated only on decision)
    decided_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="triage_approvals",
    )
    decided_at = models.DateTimeField(null=True, blank=True)
    clinician_role = models.CharField(max_length=20, blank=True)
    decision = models.CharField(max_length=20, blank=True)  # APPROVED / REJECTED
    clinician_comments = models.TextField(blank=True)

    # Assessment snapshot hash at approval time
    assessment_snapshot_hash = models.CharField(max_length=64)

    # Idempotency key to prevent duplicate submissions
    idempotency_key = models.CharField(max_length=128, unique=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Triage Approval"

    def __str__(self) -> str:
        return f"Approval {self.id} [{self.status}]"

    def is_stale(self, current_input_hash: str) -> bool:
        """Return True if the clinical input has changed since this approval was created."""
        return self.assessment_snapshot_hash != current_input_hash

    def is_expired(self) -> bool:
        return timezone.now() > self.expires_at


class ReassessmentFlag(models.Model):
    """
    A flag indicating that clinical reassessment is needed.

    Created when a change in symptoms or vitals is detected.
    Does NOT autonomously change the patient's queue position.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    encounter = models.ForeignKey(
        TriageEncounter,
        on_delete=models.CASCADE,
        related_name="reassessment_flags",
    )
    flagged_at = models.DateTimeField(auto_now_add=True)
    flagged_by_tool = models.CharField(max_length=64)  # MCP tool name
    reason = models.TextField()
    changes_detected_json = models.JSONField(default=list)
    previous_assessment = models.ForeignKey(
        TriageAssessment,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    is_acknowledged = models.BooleanField(default=False)
    acknowledged_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="acknowledged_flags",
    )
    acknowledged_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-flagged_at"]
        verbose_name = "Reassessment Flag"

    def __str__(self) -> str:
        return f"ReassessmentFlag {self.id} — encounter {self.encounter_id}"


class ReferralProposal(models.Model):
    """
    AI-proposed department referral. Requires clinician approval before any
    operational action is taken. The AI cannot commit a referral.
    """

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"

    STATUS_CHOICES = [
        (PENDING, "Proposed — awaiting clinician approval"),
        (APPROVED, "Approved — referral confirmed"),
        (REJECTED, "Rejected — referral not confirmed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    encounter = models.ForeignKey(
        TriageEncounter,
        on_delete=models.CASCADE,
        related_name="referral_proposals",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    proposed_department = models.CharField(max_length=128)
    routing_evidence_json = models.JSONField(default=list)
    routing_rules_applied_json = models.JSONField(default=list)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=PENDING)
    approved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="approved_referrals",
    )
    approved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Referral Proposal"

    def __str__(self) -> str:
        return f"Referral → {self.proposed_department} [{self.status}]"


class AuditEvent(models.Model):
    """
    Immutable structured audit log entry.

    Every tool call, approval decision, and consequential action creates one record.
    Ordinary users and the AI cannot modify or delete audit events.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    event_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    correlation_id = models.CharField(max_length=128, blank=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    # Actor (service or clinician)
    actor_type = models.CharField(max_length=32)  # 'system', 'clinician', 'mcp_tool'
    actor_id = models.CharField(max_length=128, blank=True)
    actor_role = models.CharField(max_length=32, blank=True)

    # Tool / action
    tool_name = models.CharField(max_length=128, db_index=True)
    action_type = models.CharField(max_length=64)  # 'tool_call', 'approval', 'reassessment'

    # Input/output (privacy-safe representation — no free-text clinical content)
    input_summary_json = models.JSONField(default=dict)
    output_summary_json = models.JSONField(default=dict)

    # Validation and result
    validation_outcome = models.CharField(max_length=32)  # 'accepted', 'rejected', 'error'
    success = models.BooleanField()
    error_code = models.CharField(max_length=64, blank=True)
    error_message = models.CharField(max_length=512, blank=True)

    # Source references / framework versions
    source_references_json = models.JSONField(default=list)
    framework_version = models.CharField(max_length=64, blank=True)
    model_version = models.CharField(max_length=128, blank=True)

    # State change
    previous_state = models.CharField(max_length=64, blank=True)
    resulting_state = models.CharField(max_length=64, blank=True)

    # Related objects
    encounter_id = models.UUIDField(null=True, blank=True, db_index=True)
    assessment_id = models.UUIDField(null=True, blank=True)
    approval_id = models.UUIDField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Audit Event"
        # Prevent ordinary users from having update/delete permissions
        default_permissions = ("view",)

    def __str__(self) -> str:
        return f"AuditEvent {self.event_id} — {self.tool_name} [{self.validation_outcome}]"
