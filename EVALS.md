# EVALS.md — The Triage Desk Evaluation Protocol

**System:** The Triage Desk  
**Track:** Health & Wellbeing — Agentic AI for Africa Innovation Challenge  
**Version:** 1.0 (2026-10-13)  
**Framework:** South African Triage Scale (SATS), EMSSA 2012

---

## Overview

This document defines the evaluation methodology for The Triage Desk — how each dimension of system quality is measured, what constitutes a pass, and how to reproduce the evaluation.

All evaluations use **synthetic patient data only**. No real patient information appears in any test fixture or evaluation scenario.

---

## 1. Automated Test Suite

**Run:**
```bash
python manage.py test clinicalDecisionAI.triage --verbosity=2
```

**Coverage:**
| Test Class | Count | Requirement IDs |
|-----------|-------|-----------------|
| `SATSFrameworkMetadataTests` | 3 | A1, A2, A14 |
| `SATSRuleEngineTests` | 12 | A3–A8, A13, A14, B6 |
| `MCPToolTests` | 7 | C1–C5 |
| `LangGraphAgentTests` | 5 | B1–B10, E7 |
| `ClinicianApprovalTests` | 10 | F1–F13, A10, A11 |
| `TriageEncounterAPITests` | 8 | G1–G10, A12 |
| `PrivacyAndSafetyTests` | 5 | H1, E8, A8, A13, F10 |
| `StaleApprovalEndToEndTests` | 1 | F5 |
| `StaleApprovalDbPersistenceTests` | 1 | F5 (TransactionTestCase) |
| **Total** | **52+** | All major requirements |

**Pass criteria:** 0 failures, 0 errors.

---

## 2. Clinical Rule Correctness (A3, A14)

### 2.1 Methodology

Each SATS threshold is evaluated independently. For each vital sign, a set of boundary test cases is constructed:

| Test case | Input | Expected priority | Pass condition |
|-----------|-------|-------------------|----------------|
| Normal adult | HR 72, SpO₂ 98, RR 16, Temp 37.0 | GREEN | `proposed_priority == "GREEN"` |
| HR ≤ 40 (RED) | HR 38 | RED | `proposed_priority == "RED"` |
| HR ≥ 150 (RED) | HR 155 | RED | `proposed_priority == "RED"` |
| SpO₂ ≤ 89 (RED) | SpO₂ 88 | RED | `proposed_priority == "RED"` |
| SBP ≤ 80 (RED) | BP 78/50 | RED | `proposed_priority == "RED"` |
| Temp ≤ 35 (RED) | Temp 34.5 | RED | `proposed_priority == "RED"` |
| Temp ≥ 39.4 (RED) | Temp 39.5 | RED | `proposed_priority == "RED"` |
| Missing vitals | `{}` | NOT_ASSESSED | `proposed_priority == "NOT_ASSESSED"` |
| Invalid HR | HR -1 | Unassessed | `"heart_rate" in result.unassessed_vitals` |

### 2.2 Citation requirement

Every triggered criterion must include:
- `criterion_id` prefixed with `SATS`
- `source` field citing "EMSSA" and "Gottschalk"
- `measured_value` and `threshold_value`

**Verified by:** `test_each_red_threshold_triggers_red_priority` (12 subtests)

### 2.3 Determinism check

```python
# Run the same inputs 5 times — output must be identical
results = [apply_sats_rules(vitals=v, symptoms=s, age=a, is_pregnant=p) for _ in range(5)]
assert len({r.proposed_priority for r in results}) == 1
```

**Verified by:** `test_same_inputs_always_produce_same_priority`

---

## 3. Agentic Workflow Evaluation (B1–B10)

### 3.1 Multi-step verification

The agent must execute a minimum of 4 distinct steps (tool calls) per encounter, not a single LLM prompt.

```python
tools_called = [e["tool"] for e in state["audit_trail"]]
assert "validate_vitals" in tools_called
assert "assess_triage_priority" in tools_called
assert "propose_priority" in tools_called
assert "await_approval" in tools_called
```

**Verified by:** `test_agent_produces_multi_step_audit_trail`

### 3.2 Approval gate — cannot be bypassed

```python
# For every run, regardless of priority level, requires_approval must be True
state = await run_triage_agent(...)
assert state["requires_approval"] is True
```

**Verified by:** `test_agent_requires_approval_unconditionally`

### 3.3 LLM failure does not break triage

```python
# With no LLM backend configured, agent must still return a valid triage result
state = await run_triage_agent(age=35, vitals={"heart_rate": 72, "oxygen_saturation": 98}, ...)
assert state["proposed_priority"] in ("GREEN", "YELLOW", "ORANGE", "RED", "NOT_ASSESSED")
assert state["requires_approval"] is True
```

**Verified by:** `test_agent_graceful_fallback_on_llm_unavailability`

---

## 4. MCP Server Evaluation (C1–C5)

### 4.1 Distinct tool count

```python
async with Client(triage_mcp_server) as c:
    tools = await c.list_tools()
    names = {t.name for t in tools}
assert "assess_triage_priority" in names
assert "request_clinical_reassessment" in names
assert "prepare_department_referral" in names
assert "validate_vitals" in names
assert len(names) == 4  # 4 genuinely distinct tools
```

### 4.2 Tool output schema verification

Each tool must return a structured dict with required fields:

| Tool | Required output fields |
|------|----------------------|
| `assess_triage_priority` | `proposed_priority`, `triggered_criteria`, `requires_human_review`, `pending_approval_required`, `recommendation`, `assessment_id` |
| `request_clinical_reassessment` | `flag_id`, `queue_unchanged`, `changes_detected`, `message` |
| `prepare_department_referral` | `requires_clinician_approval`, `approval_pending`, `proposed_department`, `routing_evidence`, `message` |
| `validate_vitals` | `is_valid`, `eligibility_met`, `parsed_values` |

### 4.3 Input schema enforcement

Tools with invalid inputs must raise `ToolError` (not silently produce wrong results):

```python
# Out-of-range heart_rate (-5) must be rejected by Pydantic schema
with pytest.raises(ToolError):
    await client.call_tool("validate_vitals", {"input": {"vitals": {"heart_rate": -5}}})
```

---

## 5. Clinician Approval Gate (F1–F13)

### 5.1 Role enforcement matrix

| Role | Can approve? | Expected result |
|------|-------------|-----------------|
| `NURSE` | ✅ Yes | `approval.status == "APPROVED"` |
| `DOCTOR` | ✅ Yes | `approval.status == "APPROVED"` |
| Plain user (no role) | ❌ No | `ApprovalError(code="unauthorized")` |
| AI system | ❌ No | No code path exists |

### 5.2 Stale detection

```python
# Create approval, change clinical input, attempt approval
enc.vitals_json = {"heart_rate": 160}
enc.save()
with raises(ApprovalError) as exc:
    decide_approval(approval_id, nurse, "APPROVED")
assert exc.value.code == "stale_assessment"
assert TriageApproval.objects.get(pk=approval_id).status == "INVALIDATED"
```

### 5.3 Duplicate prevention

```python
# Same idempotency_key used twice must raise
decide_approval(..., idempotency_key="abc")
with raises(ApprovalError) as exc:
    decide_approval(..., idempotency_key="abc")
assert exc.value.code == "duplicate_submission"
```

### 5.4 Queue separation

```python
# Proposed priority must never equal confirmed priority until clinician approves
enc = TriageEncounter.objects.get(pk=pk)
assert enc.proposed_priority == "ORANGE"    # set by agent
assert enc.confirmed_priority == ""          # not yet approved
```

---

## 6. Privacy Evaluation (H1, E8)

### 6.1 No PII in encounter model

Verify that `TriageEncounter` has none of: `name`, `mrn`, `national_id`, `date_of_birth`, `email`, `phone`.

```python
field_names = {f.name for f in TriageEncounter._meta.get_fields()}
assert not field_names & {"name", "mrn", "national_id", "date_of_birth", "email", "phone"}
```

### 6.2 No PII in audit events

Audit events must not contain raw clinical text (only summary statistics):
- `input_summary_json`: counts and IDs, not free-text symptoms
- `output_summary_json`: priority codes and assessment IDs, not clinical narratives

### 6.3 Local inference option

To verify zero data egress:
```bash
USE_LOCAL_MODEL=true LOCAL_MODEL=llama3.2:3b python manage.py test clinicalDecisionAI.triage
```
With `USE_LOCAL_MODEL=true`, no outbound HTTPS requests to Groq or any cloud AI provider should be made.

---

## 7. Clinical Safety Evaluation (A8, A13)

### 7.1 No diagnosis or prescription in rule output

```python
result = apply_sats_rules(vitals={"heart_rate": 135}, symptoms=["chest pain"], age=52)
output = result.to_dict()
assert "diagnosis" not in output
assert "prescription" not in output
assert "medication" not in output
```

### 7.2 Emergency escalation is immediate

```python
result = apply_sats_rules(
    vitals={"heart_rate": 35, "oxygen_saturation": 84},
    symptoms=["unconscious"],
    age=60,
)
assert result.immediate_escalation_required is True
assert "NOW" in result.recommendation
assert "pending approval" not in result.recommendation.lower()
assert "clinician approval workflow" not in result.recommendation.lower()
```

### 7.3 Missing data is never treated as normal

```python
result = apply_sats_rules(vitals={}, symptoms=["dizziness"], age=40)
assert result.proposed_priority == "NOT_ASSESSED"
assert "NOT treated as normal" in result.recommendation
assert result.missing_data_warnings  # must be non-empty
```

---

## 8. End-to-End Workflow Evaluation

### 8.1 Synthetic Normal Adult Workflow

```
POST /api/triage/encounters/
  → encounter_id returned (201)

POST /api/triage/encounters/{id}/assess/
  → workflow runs; proposed_priority=GREEN; requires_approval=True (202)

GET /api/triage/encounters/{id}/
  → proposed_priority=GREEN; confirmed_priority="" (200)

POST /api/triage/approvals/{id}/decide/  (as NURSE)
  → decision=APPROVED (200)

GET /api/triage/encounters/{id}/
  → confirmed_priority=GREEN; workflow_status=APPROVED (200)
```

### 8.2 Synthetic RED Emergency Workflow

```
POST /api/triage/encounters/
  body: {"age": 65, "symptoms": ["unconscious"], "vitals": {"heart_rate": 35, "oxygen_saturation": 85}}

POST /api/triage/encounters/{id}/assess/
  → proposed_priority=RED; immediate_escalation=True
  → recommendation: "IMMEDIATE EMERGENCY: ... NOW"
  → requires_approval=True (agent halts; recommendation is surfaced immediately)

[Clinician acts on emergency alert NOW — does not wait for software approval]

POST /api/triage/approvals/{id}/decide/  (as DOCTOR)
  → decision=APPROVED
  → confirmed_priority=RED
```

### 8.3 Synthetic Reassessment Workflow

```
[After GREEN assessment is proposed]

POST /api/triage/encounters/{id}/reassess/
  body: {"reason": "New symptom: chest pain", "new_symptoms": ["chest pain"]}
  → queue_unchanged=True; flag_id returned (201)
  → pending approval is INVALIDATED

GET /api/triage/encounters/{id}/
  → proposed_priority=GREEN still (queue position unchanged)
  → pending_reassessment=True
```

---

## 9. Open-Source Model Verification

### 9.1 Ollama availability check

```bash
ollama list | grep llama3.2:3b
# Expected: llama3.2:3b    <hash>    2.0 GB    <timestamp>
```

### 9.2 On-premises inference test

```bash
USE_LOCAL_MODEL=true LOCAL_MODEL=llama3.2:3b \
  python manage.py test clinicalDecisionAI.triage.tests.LangGraphAgentTests --verbosity=2
```

All agent tests must pass with `USE_LOCAL_MODEL=true` and no `GROQ_API_KEY` set.

---

## 10. Evaluation Checklist

| # | Requirement | Evaluation method | Pass criterion |
|---|-------------|------------------|----------------|
| A1 | SATS framework with source citation | `test_framework_has_authoritative_source` | Green |
| A2 | Versioned rule set | `test_rule_set_label_is_versioned` | Green |
| A3 | Deterministic thresholds | `test_each_red_threshold_triggers_red_priority` | Green |
| A4 | Missing data → NOT_ASSESSED | `test_missing_vitals_produce_not_assessed` | Green |
| A5 | Invalid data → unassessed | `test_invalid_vitals_are_flagged_as_unassessed` | Green |
| A6 | Result includes assessment ID | `test_assessment_includes_required_metadata` | Green |
| A7 | Result includes timestamp | `test_assessment_includes_required_metadata` | Green |
| A8 | No diagnosis/prescription | `test_sats_engine_never_outputs_diagnosis_or_prescription` | Green |
| A9 | Referral = proposal only | `test_prepare_department_referral_requires_approval` | Green |
| A10 | AI cannot approve | `test_plain_user_cannot_approve`, `test_unauthorized_approval_decision_rejected` | Green |
| A11 | Proposed ≠ confirmed priority | `test_proposed_and_confirmed_priority_are_distinct_fields` | Green |
| A12 | Reassessment does not change queue | `test_reassessment_does_not_change_queue` | Green |
| A13 | Emergency recommendation is immediate | `test_immediate_escalation_never_blocks_recommendation` | Green |
| A14 | Each threshold independently testable | 8 subtests in `test_each_red_threshold_triggers_red_priority` | Green |
| B1 | Genuine agentic workflow | `test_agent_produces_multi_step_audit_trail` | Green |
| B2 | Multi-step (not single prompt) | `test_agent_produces_multi_step_audit_trail` | Green |
| B5 | Bounded execution time | `test_agent_timeout_does_not_hang` | Green |
| B6 | Deterministic SATS (LLM variation) | `test_agent_deterministic_priority_for_same_input` | Green |
| B9 | LLM failure fallback | `test_agent_graceful_fallback_on_llm_unavailability` | Green |
| B10 | Agent requires approval unconditionally | `test_agent_requires_approval_unconditionally` | Green |
| C1 | 4 distinct MCP tools | `test_server_exposes_four_distinct_tools` | Green |
| C2 | assess_triage_priority with citations | `test_assess_triage_priority_returns_cited_red_priority` | Green |
| C3 | Reassessment flag + queue unchanged | `test_request_clinical_reassessment_creates_flag_without_queue_change` | Green |
| C4 | Referral = proposal, approval required | `test_prepare_department_referral_requires_approval` | Green |
| C5 | Schema rejects invalid input | `test_validate_vitals_tool_rejects_out_of_range_values` | Green |
| F1 | Audit event on every approval | `test_audit_event_created_for_each_approval` | Green |
| F3 | Nurse can approve | `test_nurse_can_approve_assessment` | Green |
| F4 | Rejection recorded | `test_doctor_can_reject_proposal` | Green |
| F5 | Stale detection | `test_stale_approval_rejected_error_code`, `StaleApprovalDbPersistenceTests` | Green |
| F6 | Expiry + duplicate prevention | `test_expired_approval_rejected`, `test_duplicate_submission_rejected` | Green |
| F8 | Audit events not deletable | `test_audit_events_are_not_deletable_by_normal_users` | Green |
| F9 | Role enforcement on API | `test_audit_history_requires_clinical_role`, `test_unauthorized_approval_decision_rejected` | Green |
| F12 | Unapproved encounter has no confirmed priority | `test_unapproved_encounter_has_no_confirmed_priority` | Green |
| G1 | Create encounter API | `test_create_encounter_returns_201` | Green |
| G9 | Encounter status visible | `test_encounter_detail_shows_workflow_status` | Green |
| G10 | Audit history accessible to clinicians | `test_audit_history_accessible_to_nurse` | Green |
| H1 | No PII on encounter model | `test_encounter_never_stores_identifying_fields` | Green |
| E8 | No PII in audit events | `test_audit_events_do_not_log_raw_clinical_text` | Green |
