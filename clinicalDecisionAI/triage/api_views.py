"""
Triage workflow API views.

Endpoints:
  POST /api/triage/encounters/             — register a triage encounter
  GET  /api/triage/encounters/{id}/        — retrieve encounter + workflow status
  POST /api/triage/encounters/{id}/assess/ — run the LangGraph agent, create assessment
  GET  /api/triage/encounters/{id}/assessments/ — list assessments
  POST /api/triage/approvals/             — create approval request for an assessment
  POST /api/triage/approvals/{id}/decide/ — clinician approves or rejects
  POST /api/triage/encounters/{id}/reassess/ — flag for reassessment
  GET  /api/triage/encounters/{id}/audit/ — audit history
  GET  /api/triage/encounters/{id}/referrals/ — referral proposals

All queue-changing operations require:
  1. An authenticated user (session or bearer token)
  2. A ClinicalRole of NURSE or DOCTOR
  3. A valid, non-stale, non-expired approval
"""
from __future__ import annotations

import asyncio
import logging
import uuid

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from rest_framework import status
from rest_framework.authentication import SessionAuthentication
from rest_framework.parsers import JSONParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView

from clinicalDecisionAI.authentication import (
    IntegrationApplicationPermission,
    IntegrationTokenAuthentication,
)
from clinicalDecisionAI.schemas import ClinicalAnalysisRequest
from clinicalDecisionAI.triage.approval_service import (
    ApprovalError,
    create_approval_request,
    decide_approval,
    get_audit_history,
    APPROVAL_EXPIRY_MINUTES,
)
from clinicalDecisionAI.triage.models import (
    AuditEvent,
    ClinicalRole,
    ReassessmentFlag,
    TriageApproval,
    TriageAssessment,
    TriageEncounter,
)

logger = logging.getLogger(__name__)


class ClinicalRolePermission:
    """Mixin that enforces clinician role on the requesting user."""

    REQUIRED_ROLES = {ClinicalRole.NURSE, ClinicalRole.DOCTOR}

    def _check_clinical_role(self, request) -> ClinicalRole | None:
        try:
            role = request.user.clinical_role
            if role.role in self.REQUIRED_ROLES:
                return role
        except ClinicalRole.DoesNotExist:
            pass
        return None


def _request_correlation_id(request) -> str:
    try:
        return str(uuid.UUID(request.headers.get("X-Request-ID", "")))
    except (ValueError, AttributeError):
        return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Encounter management
# ---------------------------------------------------------------------------

class TriageEncounterCreateView(APIView):
    """POST /api/triage/encounters/ — register a new triage encounter."""
    authentication_classes = [SessionAuthentication, IntegrationTokenAuthentication]
    permission_classes = [IsAuthenticated]
    parser_classes = [JSONParser]
    renderer_classes = [JSONRenderer]

    def post(self, request):
        data = request.data
        if not isinstance(data, dict):
            return Response(
                {"error": {"code": "invalid_request", "message": "Body must be a JSON object"}},
                status=status.HTTP_400_BAD_REQUEST,
            )
        symptoms = data.get("symptoms", [])
        if not isinstance(symptoms, list) or not symptoms:
            return Response(
                {"error": {"code": "invalid_request", "message": "symptoms must be a nonempty list"}},
                status=status.HTTP_400_BAD_REQUEST,
            )
        age = data.get("age")
        if age is not None and (not isinstance(age, int) or age < 0 or age > 130):
            return Response(
                {"error": {"code": "invalid_request", "message": "age must be an integer 0–130"}},
                status=status.HTTP_400_BAD_REQUEST,
            )
        encounter = TriageEncounter.objects.create(
            caller_encounter_ref=str(data.get("encounter_ref", ""))[:128],
            age=age,
            sex=str(data.get("sex", ""))[:32],
            is_pregnant=data.get("is_pregnant"),
            symptoms_json=[str(s)[:512] for s in symptoms[:64]],
            vitals_json=data.get("vitals", {}),
            medical_history_json=data.get("medical_history", []),
            registered_by=request.user if hasattr(request.user, "pk") else None,
            workflow_status=TriageEncounter.STATUS_WAITING,
        )
        return Response(
            {
                "encounter_id": str(encounter.pk),
                "workflow_status": encounter.workflow_status,
                "created_at": encounter.created_at.isoformat(),
                "message": "Triage encounter registered. Use POST /assess to run the triage agent.",
            },
            status=status.HTTP_201_CREATED,
        )


class TriageEncounterDetailView(APIView):
    """GET /api/triage/encounters/{id}/ — retrieve encounter status."""
    authentication_classes = [SessionAuthentication, IntegrationTokenAuthentication]
    permission_classes = [IsAuthenticated]
    renderer_classes = [JSONRenderer]

    def get(self, request, encounter_id):
        try:
            encounter = TriageEncounter.objects.get(pk=encounter_id)
        except (TriageEncounter.DoesNotExist, ValueError):
            return Response({"error": {"code": "not_found", "message": "Encounter not found"}}, status=404)

        latest = encounter.assessments.first()
        return Response({
            "encounter_id": str(encounter.pk),
            "workflow_status": encounter.workflow_status,
            "proposed_priority": encounter.proposed_priority,
            "confirmed_priority": encounter.confirmed_priority,
            "created_at": encounter.created_at.isoformat(),
            "latest_assessment_id": str(latest.pk) if latest else None,
            "immediate_escalation_required": latest.immediate_escalation_required if latest else False,
        })


class TriageAssessmentView(APIView):
    """POST /api/triage/encounters/{id}/assess/ — run the LangGraph triage agent."""
    authentication_classes = [SessionAuthentication, IntegrationTokenAuthentication]
    permission_classes = [IsAuthenticated]
    parser_classes = [JSONParser]
    renderer_classes = [JSONRenderer]

    def post(self, request, encounter_id):
        try:
            encounter = TriageEncounter.objects.get(pk=encounter_id)
        except (TriageEncounter.DoesNotExist, ValueError):
            return Response({"error": {"code": "not_found", "message": "Encounter not found"}}, status=404)

        correlation_id = _request_correlation_id(request)

        # Run the LangGraph agent
        try:
            from clinicalDecisionAI.triage.agent import run_triage_agent
            agent_state = asyncio.run(run_triage_agent(
                encounter_id=str(encounter.pk),
                age=encounter.age,
                is_pregnant=encounter.is_pregnant,
                symptoms=encounter.symptoms_json,
                vitals_raw=encounter.vitals_json,
                timeout_seconds=getattr(settings, "TRIAGE_AGENT_TIMEOUT_SECONDS", 60.0),
            ))
        except asyncio.TimeoutError:
            return Response(
                {"error": {"code": "agent_timeout", "message": "Triage agent timed out"}},
                status=status.HTTP_504_GATEWAY_TIMEOUT,
            )
        except Exception as exc:
            logger.error("Triage agent error: %s", type(exc).__name__, extra={"correlation_id": correlation_id})
            return Response(
                {"error": {"code": "agent_error", "message": "Triage agent failed"}},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        # Persist assessment
        assessment = TriageAssessment.objects.create(
            encounter=encounter,
            framework_id=agent_state.get("triage_result", {}).get("framework_id", "SATS-v1") if agent_state.get("triage_result") else "SATS-v1",
            framework_version=agent_state.get("triage_result", {}).get("framework_version", "1.0") if agent_state.get("triage_result") else "1.0",
            framework_source=agent_state.get("framework_source", ""),
            framework_url=agent_state.get("triage_result", {}).get("framework_url", "") if agent_state.get("triage_result") else "",
            rule_set="SATS-adult-v1",
            proposed_priority=agent_state.get("proposed_priority", "NOT_ASSESSED"),
            triggered_criteria_json=agent_state.get("triggered_criteria", []),
            unassessed_vitals_json=agent_state.get("unassessed_vitals", []),
            missing_data_warnings_json=agent_state.get("missing_data_warnings", []),
            supporting_symptoms_json=agent_state.get("symptoms", []),
            recommendation=agent_state.get("recommendation", ""),
            immediate_escalation_required=agent_state.get("immediate_escalation", False),
            llm_summary=agent_state.get("llm_summary", ""),
            llm_model_used=agent_state.get("llm_model_used", ""),
            clinical_input_hash=encounter.clinical_input_hash,
        )

        # Update encounter with proposed priority (NOT confirmed)
        TriageEncounter.objects.filter(pk=encounter.pk).update(
            proposed_priority=agent_state.get("proposed_priority", "NOT_ASSESSED"),
            workflow_status=TriageEncounter.STATUS_ASSESSED,
        )

        # Create pending approval request
        approval = create_approval_request(assessment, encounter, correlation_id=correlation_id)

        # Audit the agent run
        AuditEvent.objects.create(
            correlation_id=correlation_id,
            actor_type="system",
            tool_name="triage_agent",
            action_type="assessment",
            input_summary_json={"encounter_id": str(encounter.pk), "symptoms_count": len(encounter.symptoms_json)},
            output_summary_json={
                "proposed_priority": agent_state.get("proposed_priority"),
                "triggered_criteria_count": len(agent_state.get("triggered_criteria", [])),
                "immediate_escalation": agent_state.get("immediate_escalation"),
            },
            validation_outcome="accepted",
            success=True,
            framework_version="SATS-v1-1.0",
            previous_state=TriageEncounter.STATUS_WAITING,
            resulting_state=TriageEncounter.STATUS_ASSESSED,
            encounter_id=encounter.pk,
            assessment_id=assessment.pk,
        )

        immediate = assessment.immediate_escalation_required
        response_data = {
            "assessment_id": str(assessment.pk),
            "encounter_id": str(encounter.pk),
            "proposed_priority": assessment.proposed_priority,
            "framework_source": assessment.framework_source,
            "triggered_criteria": assessment.triggered_criteria_json,
            "unassessed_vitals": assessment.unassessed_vitals_json,
            "missing_data_warnings": assessment.missing_data_warnings_json,
            "recommendation": assessment.recommendation,
            "immediate_escalation_required": immediate,
            "requires_clinician_approval": True,
            "approval_id": str(approval.pk),
            "llm_summary": assessment.llm_summary,
            "workflow_step": "await_approval",
            "agent_audit_trail": agent_state.get("audit_trail", []),
        }
        if immediate:
            response_data["EMERGENCY_NOTICE"] = (
                "Activate emergency procedures immediately. "
                "Do NOT wait for software approval workflow."
            )

        resp = Response(response_data, status=status.HTTP_201_CREATED)
        resp["X-Clinical-Review-Required"] = "true"
        resp["X-Proposed-Priority"] = assessment.proposed_priority
        resp["X-Immediate-Escalation"] = "true" if immediate else "false"
        return resp


class TriageApprovalDecideView(APIView, ClinicalRolePermission):
    """POST /api/triage/approvals/{id}/decide/ — clinician approves or rejects."""
    authentication_classes = [SessionAuthentication]
    permission_classes = [IsAuthenticated]
    parser_classes = [JSONParser]
    renderer_classes = [JSONRenderer]

    def post(self, request, approval_id):
        role = self._check_clinical_role(request)
        if not role:
            return Response(
                {"error": {"code": "forbidden", "message": "A clinical role (NURSE or DOCTOR) is required"}},
                status=status.HTTP_403_FORBIDDEN,
            )

        data = request.data
        supplied_role = data.get("clinical_role")
        if supplied_role not in self.REQUIRED_ROLES:
            return Response(
                {"error": {"code": "invalid_clinical_role", "message": "Pass clinical_role as NURSE or DOCTOR"}},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if supplied_role != role.role:
            return Response(
                {"error": {"code": "forbidden", "message": "clinical_role must match the authenticated clinician's assigned role"}},
                status=status.HTTP_403_FORBIDDEN,
            )
        decision = data.get("decision", "").upper()
        comments = str(data.get("comments", ""))[:2048]
        idempotency_key = str(data.get("idempotency_key", ""))[:128]
        correlation_id = _request_correlation_id(request)

        try:
            approval = decide_approval(
                approval_id=str(approval_id),
                deciding_user=request.user,
                decision=decision,
                comments=comments,
                idempotency_key=idempotency_key,
                correlation_id=correlation_id,
            )
        except ApprovalError as exc:
            status_map = {
                "invalid_decision": 400,
                "unauthorized": 403,
                "forbidden": 403,
                "not_found": 404,
                "invalid_state": 409,
                "expired": 410,
                "stale_assessment": 409,
                "duplicate_submission": 409,
            }
            http_status = status_map.get(exc.code, 400)
            return Response(
                {"error": {"code": exc.code, "message": exc.message}},
                status=http_status,
            )

        return Response({
            "approval_id": str(approval.pk),
            "status": approval.status,
            "decided_at": approval.decided_at.isoformat() if approval.decided_at else None,
            "clinician_role": approval.clinician_role,
            "confirmed_priority": (
                approval.assessment.proposed_priority
                if approval.decision == TriageApproval.APPROVED else None
            ),
            "message": (
                "Priority confirmed and applied to the encounter."
                if approval.decision == TriageApproval.APPROVED
                else "Priority proposal rejected. No change applied."
            ),
        })


class TriageAuditHistoryView(APIView):
    """GET /api/triage/encounters/{id}/audit/ — chronological audit history."""
    authentication_classes = [SessionAuthentication]
    permission_classes = [IsAuthenticated]
    renderer_classes = [JSONRenderer]

    def get(self, request, encounter_id):
        role = getattr(getattr(request.user, "clinical_role", None), "role", None)
        if role not in {ClinicalRole.NURSE, ClinicalRole.DOCTOR, ClinicalRole.ADMIN}:
            return Response(
                {"error": {"code": "forbidden", "message": "Clinical role required to view audit history"}},
                status=403,
            )
        events = get_audit_history(encounter_id)
        return Response({
            "encounter_id": encounter_id,
            "event_count": len(events),
            "events": [
                {
                    "event_id": str(e.event_id),
                    "created_at": e.created_at.isoformat(),
                    "actor_type": e.actor_type,
                    "actor_role": e.actor_role,
                    "tool_name": e.tool_name,
                    "action_type": e.action_type,
                    "validation_outcome": e.validation_outcome,
                    "success": e.success,
                    "previous_state": e.previous_state,
                    "resulting_state": e.resulting_state,
                    "error_code": e.error_code,
                }
                for e in events
            ],
        })


class ReassessmentFlagView(APIView, ClinicalRolePermission):
    """POST /api/triage/encounters/{id}/reassess/ — flag for clinical reassessment."""
    authentication_classes = [SessionAuthentication, IntegrationTokenAuthentication]
    permission_classes = [IsAuthenticated]
    parser_classes = [JSONParser]
    renderer_classes = [JSONRenderer]

    def post(self, request, encounter_id):
        try:
            encounter = TriageEncounter.objects.get(pk=encounter_id)
        except (TriageEncounter.DoesNotExist, ValueError):
            return Response({"error": {"code": "not_found", "message": "Encounter not found"}}, status=404)

        data = request.data
        reason = str(data.get("reason", "Reassessment requested"))[:1024]
        new_vitals = data.get("new_vitals", {})
        new_symptoms = data.get("new_symptoms", [])[:64]

        # Invalidate stale pending approvals
        stale_count = TriageApproval.objects.filter(
            assessment__encounter=encounter,
            status=TriageApproval.PENDING,
        ).update(status=TriageApproval.INVALIDATED)

        flag = ReassessmentFlag.objects.create(
            encounter=encounter,
            flagged_by_tool="reassessment_api",
            reason=reason,
            changes_detected_json=(
                [f"New vitals: {list(new_vitals.keys())}"] if new_vitals else []
            ) + ([f"New symptoms: {new_symptoms[:5]}"] if new_symptoms else []),
            previous_assessment=encounter.assessments.first(),
        )

        # Audit
        AuditEvent.objects.create(
            actor_type="system",
            tool_name="reassessment_flag",
            action_type="reassessment",
            input_summary_json={"reason": reason[:256], "has_new_vitals": bool(new_vitals)},
            output_summary_json={"flag_id": str(flag.pk), "stale_approvals_invalidated": stale_count},
            validation_outcome="accepted",
            success=True,
            previous_state=encounter.workflow_status,
            resulting_state="REASSESSMENT_FLAGGED",
            encounter_id=encounter.pk,
        )

        return Response({
            "flag_id": str(flag.pk),
            "encounter_id": str(encounter.pk),
            "flagged_at": flag.flagged_at.isoformat(),
            "queue_unchanged": True,
            "stale_approvals_invalidated": stale_count,
            "message": (
                "Reassessment flag created. Queue position has NOT changed. "
                "Clinician review required. If emergency signs are present, "
                "activate emergency procedures immediately."
            ),
        }, status=status.HTTP_201_CREATED)
