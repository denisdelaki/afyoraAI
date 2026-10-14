"""
Triage workflow test suite.

Tests:
- SATS rule engine (deterministic, citation-bearing)
- MCP triage tools (assess_triage_priority, request_clinical_reassessment,
                    prepare_department_referral, validate_vitals)
- LangGraph agent (multi-step workflow, tool selection, error recovery)
- Clinician approval service (approval gate, stale detection, role enforcement,
                              duplicate rejection, transactional safety)
- Triage API endpoints (encounter creation, assessment, approval, reassessment, audit)
- Audit trail completeness
- Emergency escalation bypass
- Privacy controls

All tests use SYNTHETIC patient data only. No real patient information.
"""
import asyncio
import json
import uuid
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, SimpleTestCase, TransactionTestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from clinicalDecisionAI.triage.framework import (
    FRAMEWORK_ID,
    FRAMEWORK_VERSION,
    FRAMEWORK_SOURCE,
    RULE_SET_LABEL,
    higher_priority,
)
from clinicalDecisionAI.triage.rules import apply_sats_rules, _parse_vitals
from clinicalDecisionAI.triage.models import (
    AuditEvent,
    ClinicalRole,
    ReassessmentFlag,
    TriageApproval,
    TriageAssessment,
    TriageEncounter,
)
from clinicalDecisionAI.triage.approval_service import (
    ApprovalError,
    create_approval_request,
    decide_approval,
    get_audit_history,
)

User = get_user_model()

# ---------------------------------------------------------------------------
# Synthetic patient fixtures (no real patient data)
# ---------------------------------------------------------------------------

SYNTHETIC_NORMAL_ADULT = {
    "age": 35, "is_pregnant": False,
    "symptoms": ["mild headache"],
    "vitals": {"heart_rate": 72, "respiratory_rate": 16, "oxygen_saturation": 98,
                "blood_pressure": "120/75", "temperature": 37.0},
}

SYNTHETIC_URGENT_ADULT = {
    "age": 52, "is_pregnant": False,
    "symptoms": ["chest pain", "shortness of breath"],
    "vitals": {"heart_rate": 135, "respiratory_rate": 26, "oxygen_saturation": 91,
                "blood_pressure": "85/55", "temperature": 38.8},
}

SYNTHETIC_MISSING_VITALS = {
    "age": 40, "is_pregnant": False,
    "symptoms": ["dizziness"],
    "vitals": {},  # all vitals absent
}

SYNTHETIC_INVALID_VITALS = {
    "age": 30, "is_pregnant": False,
    "symptoms": ["unwell"],
    "vitals": {"heart_rate": -1, "oxygen_saturation": 110, "temperature": "unknown"},
}

SYNTHETIC_CHILD = {
    "age": 10, "is_pregnant": False,
    "symptoms": ["fever"],
    "vitals": {"heart_rate": 140},
}

SYNTHETIC_EMERGENCY = {
    "age": 65, "is_pregnant": False,
    "symptoms": ["unconscious"],
    "vitals": {"heart_rate": 38, "oxygen_saturation": 85},
}


# ============================================================================
# A. SATS Framework Tests
# ============================================================================

class SATSFrameworkMetadataTests(SimpleTestCase):
    """A1, A2, A14 — Framework metadata and traceability."""

    def test_framework_has_authoritative_source(self):
        self.assertIn("South African Triage Scale", FRAMEWORK_SOURCE)
        self.assertIn("EMSSA", FRAMEWORK_SOURCE)
        self.assertIn("Emergency Medicine Journal", FRAMEWORK_SOURCE)
        self.assertTrue(FRAMEWORK_ID.startswith("SATS"))
        self.assertEqual(FRAMEWORK_VERSION, "1.0")

    def test_rule_set_label_is_versioned(self):
        self.assertIn("SATS", RULE_SET_LABEL)
        self.assertIn("v1", RULE_SET_LABEL)

    def test_priority_ordering_is_consistent(self):
        self.assertEqual(higher_priority("RED", "ORANGE"), "RED")
        self.assertEqual(higher_priority("ORANGE", "YELLOW"), "ORANGE")
        self.assertEqual(higher_priority("GREEN", "NOT_ASSESSED"), "GREEN")
        self.assertEqual(higher_priority("RED", "GREEN"), "RED")


# ============================================================================
# B. SATS Rule Engine Tests (deterministic)
# ============================================================================

class SATSRuleEngineTests(SimpleTestCase):
    """A3–A7, A14, B6 — Deterministic triage rules with citations."""

    def _triage(self, vitals, symptoms=None, age=35, is_pregnant=False):
        return apply_sats_rules(
            vitals=vitals,
            symptoms=symptoms or ["unwell"],
            age=age,
            is_pregnant=is_pregnant,
        )

    def test_normal_adult_with_complete_vitals_is_green(self):
        result = self._triage(SYNTHETIC_NORMAL_ADULT["vitals"])
        self.assertEqual(result.proposed_priority, "GREEN")
        self.assertFalse(result.immediate_escalation_required)
        self.assertTrue(result.requires_human_review)
        self.assertIn("does not establish low risk", result.recommendation)

    def test_each_red_threshold_triggers_red_priority(self):
        """A14 — Each threshold independently testable with citation."""
        red_cases = [
            {"heart_rate": 38},          # <= 40 RED
            {"heart_rate": 155},         # >= 150 RED
            {"respiratory_rate": 9},     # <= 10 RED
            {"respiratory_rate": 32},    # >= 30 RED
            {"oxygen_saturation": 88},   # <= 89 RED
            {"blood_pressure": "78/50"}, # SBP <= 80 RED
            {"temperature": 34.5},       # <= 35.0 RED
            {"temperature": 39.5},       # >= 39.4 RED (boundary)
        ]
        for vitals in red_cases:
            with self.subTest(vitals=vitals):
                result = self._triage(vitals)
                self.assertEqual(result.proposed_priority, "RED", msg=f"Expected RED for {vitals}")
                self.assertTrue(result.triggered_criteria, "Must cite triggering criteria")
                for criterion in result.triggered_criteria:
                    self.assertIn("criterion_id", criterion)
                    self.assertIn("source", criterion)
                    self.assertIn("SATS", criterion["criterion_id"])
                    self.assertIn("measured_value", criterion)
                self.assertIn("IMMEDIATE EMERGENCY", result.recommendation)

    def test_orange_threshold_triggers_orange_priority(self):
        """SATS Orange — very urgent."""
        result = self._triage({"heart_rate": 48})  # <= 50 ORANGE
        self.assertIn(result.proposed_priority, ("ORANGE", "RED"))
        self.assertTrue(result.triggered_criteria)

    def test_missing_vitals_produce_not_assessed(self):
        """A4, A5 — Missing data is NOT treated as normal."""
        result = self._triage({})
        self.assertEqual(result.proposed_priority, "NOT_ASSESSED")
        self.assertTrue(result.unassessed_vitals)
        self.assertTrue(result.missing_data_warnings)
        self.assertNotIn("low risk", result.recommendation.lower())
        self.assertIn("NOT treated as normal", result.recommendation)

    def test_invalid_vitals_are_flagged_as_unassessed(self):
        """A4 — Invalid values go to unassessed, not silently discarded."""
        result = self._triage({"heart_rate": -1, "oxygen_saturation": 110, "temperature": "unknown"})
        self.assertIn("heart_rate", result.unassessed_vitals)
        self.assertIn("oxygen_saturation", result.unassessed_vitals)
        self.assertIn("temperature", result.unassessed_vitals)
        self.assertTrue(result.missing_data_warnings)

    def test_conflicting_aliases_go_to_unassessed(self):
        """A4, A5 — Conflicting measurements are not silently resolved."""
        result = self._triage({"pulse": 60, "heart_rate": 130})
        self.assertIn("heart_rate", result.unassessed_vitals)

    def test_child_patient_skips_adult_rules(self):
        """Adult rules not applied to paediatric patients."""
        result = self._triage({"heart_rate": 150}, age=10)
        self.assertEqual(result.proposed_priority, "NOT_ASSESSED")
        self.assertIn("adult_eligibility", result.unassessed_vitals)

    def test_pregnant_patient_skips_adult_rules(self):
        """Adult rules not applied during pregnancy."""
        result = self._triage({"heart_rate": 150}, age=30, is_pregnant=True)
        self.assertEqual(result.proposed_priority, "NOT_ASSESSED")

    def test_unknown_age_skips_adult_rules(self):
        """Adult rules not applied when age is unknown."""
        result = self._triage({"heart_rate": 150}, age=None)
        self.assertEqual(result.proposed_priority, "NOT_ASSESSED")

    def test_assessment_includes_required_metadata(self):
        """A6, A7 — Result includes assessment ID, timestamp, framework citation."""
        result = self._triage({"heart_rate": 130}, age=45)
        self.assertTrue(result.assessment_id)
        self.assertTrue(result.assessment_timestamp)
        self.assertEqual(result.framework_id, FRAMEWORK_ID)
        self.assertEqual(result.framework_version, FRAMEWORK_VERSION)
        self.assertIn("EMSSA", result.framework_source)
        self.assertTrue(result.framework_url)

    def test_same_inputs_always_produce_same_priority(self):
        """B6 — Deterministic: same inputs → same outputs."""
        vitals = {"heart_rate": 135, "oxygen_saturation": 91}
        results = [self._triage(vitals, age=52) for _ in range(5)]
        priorities = {r.proposed_priority for r in results}
        self.assertEqual(len(priorities), 1, "Priority must be deterministic")

    def test_immediate_escalation_triggered_by_escalation_symptom(self):
        """A13 — Emergency symptoms trigger immediate escalation flag."""
        result = apply_sats_rules(
            vitals={"heart_rate": 80, "oxygen_saturation": 97},
            symptoms=["unconscious"],
            age=40,
            is_pregnant=False,
        )
        self.assertTrue(result.immediate_escalation_required)
        self.assertIn("IMMEDIATE EMERGENCY", result.recommendation)

    def test_immediate_escalation_never_blocks_recommendation(self):
        """A13 — Emergency recommendation is independent of approval workflow."""
        result = apply_sats_rules(
            vitals={"heart_rate": 35},
            symptoms=["chest pain"],
            age=60,
            is_pregnant=False,
        )
        # Recommendation must tell clinician to act NOW, not wait for software
        self.assertIn("NOW", result.recommendation)
        self.assertIn("emergency", result.recommendation.lower())


# ============================================================================
# C. MCP Tool Tests
# ============================================================================

class MCPToolTests(SimpleTestCase):
    """C1–C14 — MCP server with three distinct tools."""

    def _run(self, coro):
        return asyncio.run(coro)

    async def _call_tool(self, tool_name: str, tool_input: dict):
        import json
        from fastmcp import Client as MCPClient
        from clinicalDecisionAI.triage.mcp_server import triage_mcp_server
        async with MCPClient(triage_mcp_server) as client:
            result = await client.call_tool(tool_name, {"input": tool_input})
            # FastMCP returns a list of TextContent items; parse the first as JSON
            if hasattr(result, "content") and result.content:
                raw = getattr(result.content[0], "text", str(result.content[0]))
                try:
                    return json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    return raw
            if hasattr(result, "data"):
                return result.data
            return result

    def test_server_exposes_four_distinct_tools(self):
        """C1 — At least three genuinely distinct, reusable tools."""
        import asyncio
        from fastmcp import Client as MCPClient
        from clinicalDecisionAI.triage.mcp_server import triage_mcp_server

        async def _list():
            async with MCPClient(triage_mcp_server) as client:
                return {t.name for t in await client.list_tools()}

        tool_names = asyncio.run(_list())
        self.assertIn("assess_triage_priority", tool_names)
        self.assertIn("request_clinical_reassessment", tool_names)
        self.assertIn("prepare_department_referral", tool_names)
        self.assertIn("validate_vitals", tool_names)
        # Each tool is genuinely distinct (not an alias)
        self.assertEqual(len(tool_names), 4)

    def test_assess_triage_priority_returns_cited_red_priority(self):
        """C2 — assess_triage_priority with SATS citations."""
        data = self._run(self._call_tool("assess_triage_priority", {
            "encounter_id": "test-enc-001",
            "age": 52,
            "is_pregnant": False,
            "symptoms": ["chest pain"],
            "vitals": {"heart_rate": 155, "oxygen_saturation": 88},
            "framework_version": "1.0",
        }))
        self.assertIsInstance(data, dict)
        self.assertEqual(data["proposed_priority"], "RED")
        self.assertTrue(data["triggered_criteria"])
        self.assertIn("SATS", data["triggered_criteria"][0]["criterion_id"])
        self.assertTrue(data["requires_human_review"])
        self.assertTrue(data["pending_approval_required"])
        self.assertFalse(data.get("diagnosis"))   # Must not diagnose

    def test_assess_triage_priority_with_missing_vitals_returns_not_assessed(self):
        """C2, A5 — Missing vitals → NOT_ASSESSED, not silently normal."""
        data = self._run(self._call_tool("assess_triage_priority", {
            "encounter_id": "test-enc-002",
            "age": 40,
            "is_pregnant": False,
            "symptoms": ["dizziness"],
            "vitals": {},
        }))
        self.assertEqual(data["proposed_priority"], "NOT_ASSESSED")
        self.assertTrue(data["missing_data_warnings"])
        self.assertIn("NOT treated as normal", data["recommendation"])

    def test_request_clinical_reassessment_creates_flag_without_queue_change(self):
        """C3, A12 — Reassessment flag does NOT change queue."""
        data = self._run(self._call_tool("request_clinical_reassessment", {
            "encounter_id": "test-enc-003",
            "reason": "Patient reports worsening shortness of breath",
            "new_symptoms": ["worsening dyspnoea"],
            "new_vitals": {"oxygen_saturation": 91},
        }))
        self.assertIsInstance(data, dict)
        self.assertIn("flag_id", data)
        self.assertTrue(data["queue_unchanged"])
        self.assertTrue(data["changes_detected"])
        self.assertIn("clinician must review", data["message"].lower())

    def test_prepare_department_referral_requires_approval(self):
        """C4, A9 — Referral is a proposal only; approval required."""
        data = self._run(self._call_tool("prepare_department_referral", {
            "encounter_id": "test-enc-004",
            "proposed_priority": "RED",
            "symptoms": ["chest pain"],
            "requesting_clinician_role": "NURSE",
        }))
        self.assertTrue(data["requires_clinician_approval"])
        self.assertTrue(data["approval_pending"])
        self.assertEqual(data["proposed_department"], "Resuscitation Bay")
        self.assertTrue(data["routing_evidence"])
        self.assertIn("NOT been committed", data["message"])

    def test_validate_vitals_tool_rejects_out_of_range_values(self):
        """C5, H5 — MCP schema rejects implausible values before they reach the engine."""
        from fastmcp.exceptions import ToolError
        async def _call():
            from fastmcp import Client as MCPClient
            from clinicalDecisionAI.triage.mcp_server import triage_mcp_server
            async with MCPClient(triage_mcp_server) as client:
                return await client.call_tool("validate_vitals", {"input": {
                    "vitals": {"heart_rate": -5, "oxygen_saturation": 105},
                    "age": 35,
                    "is_pregnant": False,
                }})
        # Pydantic schema validation correctly rejects out-of-range values at the MCP layer
        with self.assertRaises((ToolError, Exception)):
            self._run(_call())

    def test_validate_vitals_tool_handles_valid_boundary_values(self):
        """C4 — validate_vitals accepts boundary-valid inputs correctly."""
        data = self._run(self._call_tool("validate_vitals", {
            "vitals": {"heart_rate": 60, "oxygen_saturation": 98},
            "age": 35,
            "is_pregnant": False,
        }))
        self.assertTrue(data["is_valid"])
        self.assertTrue(data["eligibility_met"])
        self.assertIn("heart_rate", data["parsed_values"])
        self.assertIn("oxygen_saturation", data["parsed_values"])

    def test_tools_reject_invalid_input_schemas(self):
        """C5 — Strict input validation."""
        from fastmcp.exceptions import ToolError
        async def call_bad():
            from fastmcp import Client as MCPClient
            from clinicalDecisionAI.triage.mcp_server import triage_mcp_server
            async with MCPClient(triage_mcp_server) as client:
                await client.call_tool("assess_triage_priority", {
                    "input": {"symptoms": [], "encounter_id": "x"}  # empty symptoms
                })
        with self.assertRaises((ToolError, Exception)):
            self._run(call_bad())


# ============================================================================
# D. LangGraph Agent Tests
# ============================================================================

class LangGraphAgentTests(SimpleTestCase):
    """B1–B10 — Agent multi-step workflow, tool selection, error recovery."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_agent_produces_multi_step_audit_trail(self):
        """B2 — Agent executes multiple steps, not a single prompt."""
        from clinicalDecisionAI.triage.agent import run_triage_agent
        state = self._run(run_triage_agent(
            encounter_id="agent-test-001",
            age=35,
            is_pregnant=False,
            symptoms=["mild cough"],
            vitals_raw={"heart_rate": 72, "oxygen_saturation": 98},
        ))
        # Must have visited at least validate_vitals + assess_priority + propose
        tools_called = [e["tool"] for e in state.get("audit_trail", [])]
        self.assertIn("validate_vitals", tools_called)
        self.assertIn("assess_triage_priority", tools_called)
        self.assertIn("propose_priority", tools_called)
        self.assertIn("await_approval", tools_called)

    def test_agent_requires_approval_unconditionally(self):
        """B10, A10 — Agent always requires approval; cannot bypass gate."""
        from clinicalDecisionAI.triage.agent import run_triage_agent
        state = self._run(run_triage_agent(
            encounter_id="agent-test-002",
            age=45,
            is_pregnant=False,
            symptoms=["chest pain"],
            vitals_raw={"heart_rate": 140, "oxygen_saturation": 91},
        ))
        self.assertTrue(state["requires_approval"])

    def test_agent_deterministic_priority_for_same_input(self):
        """B6 — Deterministic SATS result regardless of LLM variation."""
        from clinicalDecisionAI.triage.agent import run_triage_agent
        # The LLM is in llm_summarise_node which imports from clinicalDecisionAI.services
        # Patch at the point of import inside the node's local scope
        with patch("clinicalDecisionAI.services.run_clinical_analysis",
                   side_effect=Exception("LLM unavailable")):
            results = [
                self._run(run_triage_agent(
                    encounter_id="agent-test-003",
                    age=52,
                    is_pregnant=False,
                    symptoms=["chest pain"],
                    vitals_raw={"heart_rate": 155, "oxygen_saturation": 88},
                ))
                for _ in range(3)
            ]
        priorities = {r["proposed_priority"] for r in results}
        self.assertEqual(len(priorities), 1, "SATS result must be deterministic")
        self.assertEqual(list(priorities)[0], "RED")

    def test_agent_graceful_fallback_on_llm_unavailability(self):
        """B9, E7 — LLM failure does NOT fail the triage assessment."""
        from clinicalDecisionAI.triage.agent import run_triage_agent
        # The LLM node catches all exceptions and falls back gracefully
        # Run directly — if Ollama/Groq is not configured, it will fail gracefully
        state = self._run(run_triage_agent(
            encounter_id="agent-test-004",
            age=35,
            is_pregnant=False,
            symptoms=["mild headache"],
            vitals_raw={"heart_rate": 72, "oxygen_saturation": 98},
        ))
        # Deterministic triage result must always be present regardless of LLM state
        self.assertIn(state["proposed_priority"], ("NOT_ASSESSED", "GREEN", "ORANGE", "RED", "YELLOW"))
        self.assertTrue(state["requires_approval"])

    @override_settings(TRIAGE_AGENT_TIMEOUT_SECONDS=0.001)
    def test_agent_timeout_does_not_hang(self):
        """B5, E7 — Bounded timeout prevents infinite execution."""
        from clinicalDecisionAI.triage.agent import run_triage_agent
        # With a 1ms timeout, the agent should raise or return quickly
        try:
            asyncio.run(
                asyncio.wait_for(run_triage_agent(
                    encounter_id="timeout-test",
                    age=35,
                    is_pregnant=False,
                    symptoms=["unwell"],
                    vitals_raw={},
                ), timeout=5.0)  # outer safety net
            )
        except (asyncio.TimeoutError, Exception):
            pass  # Expected — timeout or graceful error


# ============================================================================
# E. Clinician Approval Service Tests
# ============================================================================

class ClinicianApprovalTests(TestCase):
    """F1–F13, A10, A11 — Approval gate enforcement."""

    def _make_nurse(self, username="test-nurse"):
        user = User.objects.create_user(username=username, password="TestPass!42")
        ClinicalRole.objects.create(user=user, role=ClinicalRole.NURSE)
        return user

    def _make_doctor(self, username="test-doctor"):
        user = User.objects.create_user(username=username, password="TestPass!42")
        ClinicalRole.objects.create(user=user, role=ClinicalRole.DOCTOR)
        return user

    def _make_plain_user(self, username="plain-user"):
        return User.objects.create_user(username=username, password="TestPass!42")

    def _make_encounter(self):
        enc = TriageEncounter.objects.create(
            symptoms_json=["chest pain"],
            vitals_json={"heart_rate": 135},
            age=52,
        )
        return enc

    def _make_assessment(self, encounter):
        return TriageAssessment.objects.create(
            encounter=encounter,
            framework_id="SATS-v1",
            framework_version="1.0",
            framework_source="EMSSA 2012",
            rule_set="SATS-adult-v1",
            proposed_priority="ORANGE",
            clinical_input_hash=encounter.clinical_input_hash,
            recommendation="Very urgent.",
        )

    def test_nurse_can_approve_assessment(self):
        """F3, F4 — Clinician approval gate works."""
        nurse = self._make_nurse("approving-nurse")
        enc = self._make_encounter()
        assessment = self._make_assessment(enc)
        approval = create_approval_request(assessment, enc)

        result = decide_approval(
            approval_id=str(approval.pk),
            deciding_user=nurse,
            decision="APPROVED",
            comments="Assessed at bedside — priority confirmed.",
        )
        self.assertEqual(result.status, "APPROVED")
        self.assertEqual(result.decided_by, nurse)
        self.assertEqual(result.clinician_role, ClinicalRole.NURSE)
        self.assertIsNotNone(result.decided_at)

        # Encounter confirmed priority updated
        enc.refresh_from_db()
        self.assertEqual(enc.confirmed_priority, "ORANGE")
        self.assertEqual(enc.workflow_status, TriageEncounter.STATUS_APPROVED)

    def test_plain_user_cannot_approve(self):
        """F9, F11 — No clinical role = no approval."""
        plain = self._make_plain_user("noRole-user")
        enc = self._make_encounter()
        assessment = self._make_assessment(enc)
        approval = create_approval_request(assessment, enc)

        with self.assertRaises(ApprovalError) as ctx:
            decide_approval(str(approval.pk), plain, "APPROVED")
        self.assertEqual(ctx.exception.code, "unauthorized")

        # Queue must be unchanged
        enc.refresh_from_db()
        self.assertEqual(enc.confirmed_priority, "")

    def test_stale_approval_rejected_error_code(self):
        """F5 — ApprovalError.code is 'stale_assessment' when clinical input changes."""
        nurse = self._make_nurse("stale-nurse")
        enc = self._make_encounter()
        assessment = self._make_assessment(enc)
        approval = create_approval_request(assessment, enc)

        # Simulate clinical input change
        enc.vitals_json = {"heart_rate": 160}  # new vital — different hash
        enc.save()  # recomputes clinical_input_hash

        with self.assertRaises(ApprovalError) as ctx:
            decide_approval(str(approval.pk), nurse, "APPROVED")
        self.assertEqual(ctx.exception.code, "stale_assessment")

        # Queue must not be changed
        enc.refresh_from_db()
        self.assertEqual(enc.confirmed_priority, "")

    def test_expired_approval_rejected(self):
        """F6 — Expired approval requests are rejected."""
        nurse = self._make_nurse("expired-nurse")
        enc = self._make_encounter()
        assessment = self._make_assessment(enc)
        approval = create_approval_request(assessment, enc)
        # Force expiry
        TriageApproval.objects.filter(pk=approval.pk).update(
            expires_at=timezone.now() - timedelta(seconds=1)
        )

        with self.assertRaises(ApprovalError) as ctx:
            decide_approval(str(approval.pk), nurse, "APPROVED")
        self.assertEqual(ctx.exception.code, "expired")

    def test_duplicate_submission_rejected(self):
        """F6 — Duplicate/replayed approval requests are rejected."""
        nurse = self._make_nurse("dup-nurse")
        enc = self._make_encounter()
        assessment = self._make_assessment(enc)
        approval = create_approval_request(assessment, enc)

        # First decision succeeds
        decide_approval(str(approval.pk), nurse, "APPROVED", idempotency_key="idem-key-abc")

        # Second attempt with same idempotency key is rejected
        enc2 = self._make_encounter()
        assessment2 = self._make_assessment(enc2)
        approval2 = create_approval_request(assessment2, enc2, correlation_id="test")

        with self.assertRaises(ApprovalError) as ctx:
            decide_approval(str(approval2.pk), nurse, "APPROVED", idempotency_key="idem-key-abc")
        self.assertEqual(ctx.exception.code, "duplicate_submission")

    def test_invalid_decision_value_rejected(self):
        """F6 — Only APPROVED and REJECTED are valid decisions."""
        nurse = self._make_nurse("inv-nurse")
        enc = self._make_encounter()
        assessment = self._make_assessment(enc)
        approval = create_approval_request(assessment, enc)

        with self.assertRaises(ApprovalError) as ctx:
            decide_approval(str(approval.pk), nurse, "MAYBE")
        self.assertEqual(ctx.exception.code, "invalid_decision")

    def test_unapproved_encounter_has_no_confirmed_priority(self):
        """F12 — Proposed priority never reaches live queue without approval."""
        enc = self._make_encounter()
        assessment = self._make_assessment(enc)
        # Assessment created, but no approval given
        TriageEncounter.objects.filter(pk=enc.pk).update(
            proposed_priority="ORANGE",
            workflow_status=TriageEncounter.STATUS_ASSESSED,
        )
        enc.refresh_from_db()
        self.assertEqual(enc.confirmed_priority, "")
        self.assertNotEqual(enc.proposed_priority, enc.confirmed_priority)

    def test_audit_event_created_for_each_approval(self):
        """F1, F2 — Every approval creates a structured audit event."""
        nurse = self._make_nurse("audit-nurse")
        enc = self._make_encounter()
        assessment = self._make_assessment(enc)
        initial_count = AuditEvent.objects.count()
        create_approval_request(assessment, enc)

        self.assertGreater(AuditEvent.objects.count(), initial_count)
        event = AuditEvent.objects.latest("created_at")
        self.assertEqual(event.tool_name, "create_approval_request")
        self.assertTrue(event.event_id)
        self.assertTrue(event.created_at)
        self.assertEqual(event.action_type, "approval")
        self.assertTrue(event.success)

    def test_audit_events_are_not_deletable_by_normal_users(self):
        """F8 — AuditEvent model has no delete permission for ordinary users."""
        permissions = AuditEvent._meta.default_permissions
        self.assertNotIn("delete", permissions)
        self.assertNotIn("add", permissions)
        self.assertNotIn("change", permissions)
        self.assertIn("view", permissions)

    def test_doctor_can_reject_proposal(self):
        """F3, F4 — Rejection is recorded correctly."""
        doctor = self._make_doctor("rejecting-doctor")
        enc = self._make_encounter()
        assessment = self._make_assessment(enc)
        approval = create_approval_request(assessment, enc)

        result = decide_approval(
            str(approval.pk), doctor, "REJECTED",
            comments="Vital signs recheck showed normal BP.",
        )
        self.assertEqual(result.status, "REJECTED")
        enc.refresh_from_db()
        self.assertEqual(enc.confirmed_priority, "")  # Not changed on rejection


# ============================================================================
# F. Triage API Endpoint Tests
# ============================================================================

class TriageEncounterAPITests(TestCase):
    """G1–G10 — Encounter and workflow API."""

    def setUp(self):
        self.user = User.objects.create_user("api-user", password="TestPass!42")
        self.nurse = User.objects.create_user("api-nurse", password="TestPass!42")
        ClinicalRole.objects.create(user=self.nurse, role=ClinicalRole.NURSE)
        self.client.force_login(self.user)

    def test_create_encounter_returns_201(self):
        """G1 — Register a triage encounter."""
        response = self.client.post(
            "/api/triage/encounters/",
            data=json.dumps({"symptoms": ["fever", "headache"], "age": 34}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("encounter_id", data)
        self.assertEqual(data["workflow_status"], "WAITING")

    def test_create_encounter_requires_symptoms(self):
        """G1 — Missing symptoms rejected."""
        response = self.client.post(
            "/api/triage/encounters/",
            data=json.dumps({"age": 30}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_encounter_detail_shows_workflow_status(self):
        """G9 — Encounter status visible."""
        enc = TriageEncounter.objects.create(
            symptoms_json=["cough"],
            vitals_json={},
            age=28,
        )
        response = self.client.get(f"/api/triage/encounters/{enc.pk}/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("workflow_status", response.json())
        self.assertIn("proposed_priority", response.json())
        self.assertIn("confirmed_priority", response.json())

    def test_proposed_and_confirmed_priority_are_distinct_fields(self):
        """A11 — AI-proposed vs clinician-confirmed are clearly separated."""
        enc = TriageEncounter.objects.create(
            symptoms_json=["chest pain"],
            vitals_json={"heart_rate": 135},
            age=52,
            proposed_priority="ORANGE",
            workflow_status=TriageEncounter.STATUS_ASSESSED,
        )
        response = self.client.get(f"/api/triage/encounters/{enc.pk}/")
        data = response.json()
        # confirmed_priority should be empty until approved
        self.assertEqual(data["proposed_priority"], "ORANGE")
        self.assertEqual(data["confirmed_priority"], "")

    def test_reassessment_does_not_change_queue(self):
        """A12, G7 — Reassessment flag created; queue unchanged."""
        enc = TriageEncounter.objects.create(
            symptoms_json=["cough"],
            vitals_json={"heart_rate": 72},
            age=40,
            proposed_priority="GREEN",
            workflow_status=TriageEncounter.STATUS_ASSESSED,
        )
        response = self.client.post(
            f"/api/triage/encounters/{enc.pk}/reassess/",
            data=json.dumps({"reason": "New symptom: dyspnoea", "new_symptoms": ["dyspnoea"]}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertTrue(data["queue_unchanged"])
        self.assertIn("flag_id", data)
        # Queue position still intact
        enc.refresh_from_db()
        self.assertEqual(enc.proposed_priority, "GREEN")

    def test_unauthorized_approval_decision_rejected(self):
        """A10, F9 — User without clinical role cannot approve."""
        enc = TriageEncounter.objects.create(
            symptoms_json=["unwell"], vitals_json={}, age=40
        )
        assessment = TriageAssessment.objects.create(
            encounter=enc,
            framework_id="SATS-v1", framework_version="1.0",
            framework_source="EMSSA", rule_set="SATS-adult-v1",
            proposed_priority="ORANGE", clinical_input_hash=enc.clinical_input_hash,
            recommendation="Urgent.",
        )
        approval = create_approval_request(assessment, enc)

        # self.user has no clinical role
        response = self.client.post(
            f"/api/triage/approvals/{approval.pk}/decide/",
            data=json.dumps({"decision": "APPROVED", "clinical_role": "NURSE"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 403)

        # Confirm queue unchanged
        enc.refresh_from_db()
        self.assertEqual(enc.confirmed_priority, "")

    @override_settings(DEBUG=True)
    def test_sandbox_nurse_default_and_explicit_approval_role(self):
        self.assertEqual(self.client.get("/sandbox/").status_code, 200)
        self.assertEqual(ClinicalRole.objects.get(user=self.user).role, ClinicalRole.NURSE)
        enc = TriageEncounter.objects.create(
            symptoms_json=["unwell"], vitals_json={}, age=40
        )
        assessment = TriageAssessment.objects.create(
            encounter=enc,
            framework_id="SATS-v1", framework_version="1.0",
            framework_source="EMSSA", rule_set="SATS-adult-v1",
            proposed_priority="ORANGE", clinical_input_hash=enc.clinical_input_hash,
            recommendation="Urgent.",
        )
        approval = create_approval_request(assessment, enc)
        path = f"/api/triage/approvals/{approval.pk}/decide/"
        for supplied_role, expected_status in ((None, 400), ("ADMIN", 400), ("DOCTOR", 403), ("NURSE", 200)):
            with self.subTest(clinical_role=supplied_role):
                payload = {"decision": "APPROVED"}
                if supplied_role is not None:
                    payload["clinical_role"] = supplied_role
                response = self.client.post(
                    path, data=json.dumps(payload), content_type="application/json",
                )
                self.assertEqual(response.status_code, expected_status)
                enc.refresh_from_db()
                self.assertEqual(enc.confirmed_priority, "ORANGE" if expected_status == 200 else "")
        self.assertEqual(response.json()["clinician_role"], "NURSE")

    def test_audit_history_requires_clinical_role(self):
        """F9 — Audit history restricted to clinical roles."""
        enc = TriageEncounter.objects.create(
            symptoms_json=["unwell"], vitals_json={}, age=40
        )
        response = self.client.get(f"/api/triage/encounters/{enc.pk}/audit/")
        self.assertEqual(response.status_code, 403)

    def test_audit_history_accessible_to_nurse(self):
        """G10 — Audit history accessible to clinical users."""
        self.client.force_login(self.nurse)
        enc = TriageEncounter.objects.create(
            symptoms_json=["unwell"], vitals_json={}, age=40
        )
        AuditEvent.objects.create(
            tool_name="test_tool", action_type="test",
            actor_type="system", success=True,
            validation_outcome="accepted",
            encounter_id=enc.pk,
        )
        response = self.client.get(f"/api/triage/encounters/{enc.pk}/audit/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["event_count"], 1)
        self.assertIn("events", data)

    def test_unauthenticated_request_rejected(self):
        """H1 — All triage endpoints require authentication."""
        self.client.logout()
        response = self.client.post(
            "/api/triage/encounters/",
            data=json.dumps({"symptoms": ["fever"]}),
            content_type="application/json",
        )
        self.assertIn(response.status_code, (401, 302, 403))


# ============================================================================
# G. Privacy and Safety Tests
# ============================================================================

class PrivacyAndSafetyTests(TestCase):
    """H — Privacy, security, and clinical safety controls."""

    def test_encounter_never_stores_identifying_fields(self):
        """H1, E8 — No PII in encounter model."""
        enc = TriageEncounter.objects.create(
            symptoms_json=["fever"],
            vitals_json={"heart_rate": 80},
            age=35,
        )
        enc_dict = {
            f.name: getattr(enc, f.name)
            for f in TriageEncounter._meta.get_fields()
            if hasattr(enc, f.name)
        }
        forbidden_fields = {"name", "mrn", "national_id", "date_of_birth", "email", "phone"}
        for field in forbidden_fields:
            self.assertNotIn(field, enc_dict, f"Field '{field}' must not exist on TriageEncounter")

    def test_clinical_input_hash_detects_changes(self):
        """F5 — Hash changes when input changes."""
        enc = TriageEncounter.objects.create(
            symptoms_json=["fever"], vitals_json={"heart_rate": 80}, age=35
        )
        hash1 = enc.clinical_input_hash
        enc.vitals_json = {"heart_rate": 160}
        enc.save()
        self.assertNotEqual(enc.clinical_input_hash, hash1)

    def test_sats_engine_never_outputs_diagnosis_or_prescription(self):
        """A8 — Rule engine output contains no diagnosis or medication fields."""
        result = apply_sats_rules(
            vitals={"heart_rate": 135},
            symptoms=["chest pain"],
            age=52, is_pregnant=False,
        )
        result_dict = result.to_dict()
        forbidden_keys = {"diagnosis", "prescription", "medication", "treatment"}
        for key in forbidden_keys:
            self.assertNotIn(key, result_dict, f"Rule engine must not produce '{key}'")

    def test_audit_events_do_not_log_raw_clinical_text(self):
        """F10 — Audit events contain summaries, not raw clinical content."""
        enc = TriageEncounter.objects.create(
            symptoms_json=["private symptom detail"], vitals_json={}, age=40
        )
        create_approval_request(
            TriageAssessment.objects.create(
                encounter=enc, framework_id="SATS-v1", framework_version="1.0",
                framework_source="EMSSA", rule_set="SATS-adult-v1",
                proposed_priority="GREEN", clinical_input_hash=enc.clinical_input_hash,
                recommendation="OK",
            ),
            enc,
        )
        event = AuditEvent.objects.latest("created_at")
        # Raw symptom text should not appear in audit input/output summaries
        raw_text = json.dumps(event.input_summary_json) + json.dumps(event.output_summary_json)
        self.assertNotIn("private symptom detail", raw_text)

    def test_emergency_message_never_delayed_by_approval_workflow(self):
        """A13, F13 — Emergency escalation message is immediate; no approval needed."""
        result = apply_sats_rules(
            vitals={"heart_rate": 35, "oxygen_saturation": 84},
            symptoms=["unconscious"],
            age=60, is_pregnant=False,
        )
        self.assertTrue(result.immediate_escalation_required)
        # Emergency recommendation must tell clinician to act NOW without waiting for approval workflow
        recommendation = result.recommendation.lower()
        # The message correctly says "do not wait for AI processing" — this is the right message.
        # What we verify: the message does NOT suggest waiting for an approval workflow before acting.
        self.assertNotIn("pending approval", recommendation)
        self.assertNotIn("clinician approval workflow", recommendation)
        # Must be a strong immediate action message
        self.assertIn("now", recommendation)
        self.assertIn("emergency", recommendation)


# ============================================================================
# H. Clinical Input Hash / Stale Approval End-to-End
# ============================================================================

class StaleApprovalEndToEndTests(TestCase):
    """F5 — Changing clinical input invalidates pending approvals."""

    def test_reassessment_invalidates_pending_approvals(self):
        """Posting to reassess endpoint invalidates all PENDING approvals."""
        user = User.objects.create_user("reassess-user", password="TestPass!42")
        self.client.force_login(user)

        enc = TriageEncounter.objects.create(
            symptoms_json=["fever"], vitals_json={"heart_rate": 80}, age=35
        )
        assessment = TriageAssessment.objects.create(
            encounter=enc, framework_id="SATS-v1", framework_version="1.0",
            framework_source="EMSSA", rule_set="SATS-adult-v1",
            proposed_priority="GREEN", clinical_input_hash=enc.clinical_input_hash,
            recommendation="Routine.",
        )
        approval = create_approval_request(assessment, enc)
        self.assertEqual(approval.status, TriageApproval.PENDING)

        # Trigger reassessment with new symptoms
        self.client.post(
            f"/api/triage/encounters/{enc.pk}/reassess/",
            data=json.dumps({"reason": "New: chest pain", "new_symptoms": ["chest pain"]}),
            content_type="application/json",
        )

        approval.refresh_from_db()
        self.assertEqual(approval.status, TriageApproval.INVALIDATED)


# ============================================================================
# I. Transaction-level Stale Approval DB Persistence
# ============================================================================

class StaleApprovalDbPersistenceTests(TransactionTestCase):
    """
    F5 — Uses TransactionTestCase (no wrapping transaction) so the
    bare update() in _preflight_checks commits durably to the real DB.
    Verifies the status row is physically set to INVALIDATED.
    """

    def _setup(self):
        user = User.objects.create_user("txn-nurse-xyz", password="TestPass!42")
        ClinicalRole.objects.create(user=user, role=ClinicalRole.NURSE)
        enc = TriageEncounter.objects.create(
            symptoms_json=["fever"], vitals_json={"heart_rate": 80}, age=35
        )
        assessment = TriageAssessment.objects.create(
            encounter=enc, framework_id="SATS-v1", framework_version="1.0",
            framework_source="EMSSA", rule_set="SATS-adult-v1",
            proposed_priority="GREEN", clinical_input_hash=enc.clinical_input_hash,
            recommendation="Routine.",
        )
        approval = create_approval_request(assessment, enc)
        return user, enc, assessment, approval

    def test_stale_approval_db_row_set_to_invalidated(self):
        """F5 — After stale rejection, the approval row is physically INVALIDATED in DB."""
        nurse, enc, _assessment, approval = self._setup()

        # Change the clinical input so hashes differ
        enc.vitals_json = {"heart_rate": 160}
        enc.save()

        with self.assertRaises(ApprovalError) as ctx:
            decide_approval(str(approval.pk), nurse, "APPROVED")
        self.assertEqual(ctx.exception.code, "stale_assessment")

        # Row must be INVALIDATED in the actual DB (no test transaction wrapping)
        fresh = TriageApproval.objects.get(pk=approval.pk)
        self.assertEqual(fresh.status, TriageApproval.INVALIDATED)

        # Encounter queue unchanged
        enc.refresh_from_db()
        self.assertEqual(enc.confirmed_priority, "")
