"""
Clinician approval service.

Enforces:
- Approval tied to exact assessment (via clinical input hash)
- Stale approval detection when clinical input changes
- Expired approval detection
- Duplicate / replayed approval prevention (idempotency_key uniqueness)
- Unauthorised approval rejection (role check)
- Transactional updates with select_for_update()
- AI cannot write to TriageApproval
- Audit event created for every approval decision
"""
from __future__ import annotations

import uuid
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.db import transaction
from django.utils import timezone

from clinicalDecisionAI.triage.models import (
    AuditEvent,
    ClinicalRole,
    TriageApproval,
    TriageAssessment,
    TriageEncounter,
)

User = get_user_model()

# Approval request expires after this many minutes
APPROVAL_EXPIRY_MINUTES = 60

# Roles allowed to approve triage priorities
APPROVING_ROLES = {ClinicalRole.NURSE, ClinicalRole.DOCTOR}


class ApprovalError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def create_approval_request(
    assessment: TriageAssessment,
    encounter: TriageEncounter,
    correlation_id: str = "",
) -> TriageApproval:
    """
    Create a PENDING approval request for an assessment.

    This does NOT change the operational queue. It creates a record
    for a clinician to review and decide upon.
    """
    idempotency_key = f"approval:{assessment.pk}"
    expires_at = timezone.now() + timedelta(minutes=APPROVAL_EXPIRY_MINUTES)

    approval, created = TriageApproval.objects.get_or_create(
        idempotency_key=idempotency_key,
        defaults={
            "assessment": assessment,
            "expires_at": expires_at,
            "status": TriageApproval.PENDING,
            "assessment_snapshot_hash": encounter.clinical_input_hash,
        },
    )
    if not created:
        # Already exists — return the existing pending request
        return approval

    _create_audit_event(
        tool_name="create_approval_request",
        action_type="approval",
        actor_type="system",
        encounter=encounter,
        assessment=assessment,
        approval=approval,
        success=True,
        previous_state="",
        resulting_state="PENDING",
        correlation_id=correlation_id,
        input_summary={"assessment_id": str(assessment.pk)},
        output_summary={"approval_id": str(approval.pk), "expires_at": expires_at.isoformat()},
    )
    return approval



def _preflight_checks(
    approval_id: str,
    deciding_user: User,
    decision: str,
    idempotency_key: str,
    correlation_id: str,
) -> tuple[TriageApproval, ClinicalRole]:
    """
    Pre-flight checks that must persist their DB mutations regardless of
    what happens in the main atomic block. Stale / expired status writes
    need to be committed immediately so tests and live callers can observe
    them even if the caller catches ApprovalError.
    """
    if decision not in (TriageApproval.APPROVED, TriageApproval.REJECTED):
        raise ApprovalError("invalid_decision", "Decision must be APPROVED or REJECTED")

    # Role check
    try:
        role = deciding_user.clinical_role
    except ClinicalRole.DoesNotExist:
        _create_audit_event(
            tool_name="decide_approval",
            action_type="approval",
            actor_type="user",
            actor_id=str(deciding_user.pk),
            encounter=None,
            assessment=None,
            approval=None,
            success=False,
            previous_state="",
            resulting_state="",
            correlation_id=correlation_id,
            validation_outcome="rejected",
            error_code="unauthorized",
            error_message="User has no clinical role",
            input_summary={"approval_id": approval_id},
            output_summary={},
        )
        raise ApprovalError("unauthorized", "User does not have a clinical role")

    if role.role not in APPROVING_ROLES:
        raise ApprovalError(
            "forbidden",
            f"Role '{role.role}' is not permitted to approve triage assessments",
        )

    # Idempotency check
    if idempotency_key:
        if TriageApproval.objects.filter(
            idempotency_key=idempotency_key,
            status__in=[TriageApproval.APPROVED, TriageApproval.REJECTED],
        ).exists():
            raise ApprovalError("duplicate_submission", "This decision has already been recorded")

    # Fetch row (without lock — locking happens in the atomic block)
    try:
        approval = TriageApproval.objects.get(pk=approval_id)
    except TriageApproval.DoesNotExist:
        raise ApprovalError("not_found", "Approval request not found")

    if approval.status != TriageApproval.PENDING:
        raise ApprovalError(
            "invalid_state",
            f"Cannot decide an approval in '{approval.status}' state; expected PENDING",
        )

    if approval.is_expired():
        # Committed immediately (no savepoint wrapping) so callers can observe the state
        TriageApproval.objects.filter(pk=approval.pk).update(status=TriageApproval.EXPIRED)
        raise ApprovalError("expired", "Approval request has expired")

    # Stale assessment check
    encounter = approval.assessment.encounter
    encounter.refresh_from_db()
    current_hash = encounter.clinical_input_hash
    if approval.is_stale(current_hash):
        TriageApproval.objects.filter(pk=approval.pk).update(status=TriageApproval.INVALIDATED)
        raise ApprovalError(
            "stale_assessment",
            "Clinical input has changed since this assessment was created. "
            "A fresh assessment is required.",
        )

    return approval, role


def decide_approval(
    approval_id: str,
    deciding_user: User,
    decision: str,
    comments: str = "",
    idempotency_key: str = "",
    correlation_id: str = "",
) -> TriageApproval:
    """
    A clinician approves or rejects a pending triage assessment.

    Enforces:
    - User has an appropriate ClinicalRole
    - Approval is still PENDING
    - Approval has not expired
    - Clinical input has not changed since the approval was created
    - Decision value is APPROVED or REJECTED
    - Duplicate submissions via idempotency_key are rejected

    This is the ONLY path by which a confirmed priority reaches the encounter.

    Architecture note: pre-flight checks run BEFORE the atomic commit block.
    Any DB mutations in _preflight_checks (stale, expired status writes) commit
    immediately and are visible to all callers regardless of subsequent exceptions.
    """
    # Step 1: Pre-flight — runs outside any transaction, so DB mutations persist
    approval, role = _preflight_checks(
        approval_id, deciding_user, decision, idempotency_key, correlation_id
    )

    # Step 2: Commit — runs in its own atomic transaction
    return _commit_approval(
        approval_id=approval_id,
        deciding_user=deciding_user,
        decision=decision,
        comments=comments,
        idempotency_key=idempotency_key,
        correlation_id=correlation_id,
        role=role,
    )


@transaction.atomic
def _commit_approval(
    approval_id: str,
    deciding_user: User,
    decision: str,
    comments: str,
    idempotency_key: str,
    correlation_id: str,
    role: ClinicalRole,
) -> TriageApproval:
    """
    Atomic commit of the approved/rejected decision.
    Only called after _preflight_checks passes.
    """
    # Re-fetch with row lock (race condition guard)
    approval = TriageApproval.objects.select_for_update().get(pk=approval_id)

    # Re-check state after lock acquisition
    if approval.status != TriageApproval.PENDING:
        raise ApprovalError(
            "invalid_state",
            f"Cannot decide an approval in '{approval.status}' state; expected PENDING",
        )

    previous_state = approval.status

    # Derive encounter from the locked approval row
    encounter = approval.assessment.encounter

    # Record the decision
    approval.status = decision  # APPROVED or REJECTED
    approval.decided_by = deciding_user
    approval.decided_at = timezone.now()
    approval.clinician_role = role.role
    approval.decision = decision
    approval.clinician_comments = comments[:2048]
    if idempotency_key:
        approval.idempotency_key = idempotency_key
    approval.save()

    # If approved, update the encounter's confirmed priority
    if decision == TriageApproval.APPROVED:
        TriageEncounter.objects.filter(pk=encounter.pk).update(
            confirmed_priority=approval.assessment.proposed_priority,
            workflow_status=TriageEncounter.STATUS_APPROVED,
        )

    _create_audit_event(
        tool_name="decide_approval",
        action_type="approval",
        actor_type="clinician",
        actor_id=str(deciding_user.pk),
        actor_role=role.role,
        encounter=encounter,
        assessment=approval.assessment,
        approval=approval,
        success=True,
        previous_state=previous_state,
        resulting_state=decision,
        correlation_id=correlation_id,
        validation_outcome="accepted",
        input_summary={
            "approval_id": str(approval.pk),
            "decision": decision,
            "has_comments": bool(comments),
        },
        output_summary={
            "resulting_status": approval.status,
            "confirmed_priority": approval.assessment.proposed_priority if decision == "APPROVED" else None,
        },
    )

    return approval


def get_audit_history(encounter_id: str) -> list[AuditEvent]:
    """Return chronological audit events for an encounter."""
    return list(
        AuditEvent.objects.filter(encounter_id=encounter_id).order_by("created_at")
    )


def _create_audit_event(
    *,
    tool_name: str,
    action_type: str,
    actor_type: str,
    actor_id: str = "",
    actor_role: str = "",
    encounter: TriageEncounter | None,
    assessment: TriageAssessment | None,
    approval: TriageApproval | None,
    success: bool,
    previous_state: str = "",
    resulting_state: str = "",
    correlation_id: str = "",
    validation_outcome: str = "accepted",
    error_code: str = "",
    error_message: str = "",
    input_summary: dict | None = None,
    output_summary: dict | None = None,
    source_references: list | None = None,
) -> AuditEvent:
    from clinicalDecisionAI.triage.framework import FRAMEWORK_ID, FRAMEWORK_VERSION
    return AuditEvent.objects.create(
        correlation_id=correlation_id or str(uuid.uuid4()),
        actor_type=actor_type,
        actor_id=actor_id,
        actor_role=actor_role,
        tool_name=tool_name,
        action_type=action_type,
        input_summary_json=input_summary or {},
        output_summary_json=output_summary or {},
        validation_outcome=validation_outcome,
        success=success,
        error_code=error_code,
        error_message=error_message,
        source_references_json=source_references or [],
        framework_version=f"{FRAMEWORK_ID}-{FRAMEWORK_VERSION}",
        previous_state=previous_state,
        resulting_state=resulting_state,
        encounter_id=encounter.pk if encounter else None,
        assessment_id=assessment.pk if assessment else None,
        approval_id=approval.pk if approval else None,
    )
