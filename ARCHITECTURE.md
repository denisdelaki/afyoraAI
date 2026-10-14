# Afyora Triage Desk — Architecture

**System name:** The Triage Desk  
**Track:** Health & Wellbeing  
**Challenge:** Agentic AI for Africa Innovation Challenge  
**Version:** 1.0 (2026-10-13)  
**Repository:** [github.com/denisdelaki/afyoraAI](https://github.com/denisdelaki/afyoraAI)

---

## 1. Overview

The Triage Desk is a clinical decision *support* system for under-resourced healthcare facilities in Africa. It assists qualified nurses and doctors by:

1. Applying the **South African Triage Scale (SATS)** deterministically to structured vital signs
2. Running a **multi-step LangGraph agent** to collect evidence, generate a natural-language summary, and propose a triage priority
3. Surfacing the proposed priority to a **qualified clinician** for review and explicit approval
4. Recording every action in an **immutable structured audit trail**

> [!IMPORTANT]
> The system is a **decision support tool only**. It has no operational authority. A qualified clinician must approve every proposed priority before it affects the patient queue. The AI cannot approve its own proposals.

---

## 2. Clinical Framework

### 2.1 South African Triage Scale (SATS)

| Field | Value |
|-------|-------|
| **Framework** | South African Triage Scale (SATS) |
| **Version in use** | 1.0 (2026-10-13) |
| **Source authority** | Emergency Medicine Society of South Africa (EMSSA) |
| **Primary citation** | Gottschalk SB, Wood D, DeVries S, Wallis LA, Bruijns S. "The Cape Triage Score: A Triage System for South Africa." *Emergency Medicine Journal* 2006;23(2):149–153. |
| **Training manual** | EMSSA Triage Working Group. *South African Triage Scale Training Manual*, 2012. |
| **URL** | https://emssa.org.za/sections/trauma-emergency-medicine/ |
| **Applicable population** | Adults ≥ 16 years, non-pregnant |
| **Licence note** | Threshold values reproduced from peer-reviewed literature for educational/prototype use. Clinical deployment requires EMSSA written permission and independent clinical validation. |

### 2.2 Priority Levels

| Priority | Colour | Description | Target response |
|----------|--------|-------------|-----------------|
| RED | Immediate | Life-threatening emergency | Immediate |
| ORANGE | Very Urgent | Risk of deterioration within 10 min | < 10 minutes |
| YELLOW | Urgent | Stable but at risk | < 60 minutes |
| GREEN | Routine | Non-urgent | As capacity allows |
| NOT_ASSESSED | — | Insufficient data | Reassess immediately |

### 2.3 What the AI Does and Does Not Do

| ✅ AI is permitted to | ❌ AI must not |
|----------------------|----------------|
| Apply deterministic SATS thresholds | Diagnose a condition |
| Generate a readable triage summary | Prescribe or recommend specific medications |
| Propose a triage priority | Approve its own proposal |
| Flag reassessment when observations change | Change the patient queue autonomously |
| Propose a department referral | Commit a referral without clinician approval |
| Return evidence citations for each finding | Delay emergency procedures for software approval |

---

## 3. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                         The Triage Desk                                │
│                                                                        │
│  ┌──────────────┐    REST API     ┌─────────────────────────────────┐  │
│  │  EMR / HIS   │◄──────────────►│   Django + DRF                  │  │
│  │  (caller)    │                │   /api/triage/*                 │  │
│  └──────────────┘                └────────────┬────────────────────┘  │
│                                               │                        │
│                                               ▼                        │
│                          ┌────────────────────────────────────────┐    │
│                          │         LangGraph Agent                │    │
│                          │  ┌──────────────────────────────────┐  │    │
│                          │  │ 1. validate_vitals               │  │    │
│                          │  │ 2. assess_triage_priority (SATS) │  │    │
│                          │  │ 3. llm_summarise (advisory only) │  │    │
│                          │  │ 4. propose_priority              │  │    │
│                          │  │ 5. await_approval ──── STOPS ──► │  │    │
│                          │  └──────────────────────────────────┘  │    │
│                          └────────────┬────────────────────────────┘    │
│                                       │ in-memory                      │
│                          ┌────────────▼────────────────────────────┐    │
│                          │  Custom MCP Server (FastMCP)            │    │
│                          │  • assess_triage_priority (SATS rules)  │    │
│                          │  • request_clinical_reassessment         │    │
│                          │  • prepare_department_referral           │    │
│                          │  • validate_vitals                       │    │
│                          └────────────┬────────────────────────────┘    │
│                                       │                                 │
│              ┌────────────────────────┼────────────────────┐            │
│              │                        │                    │            │
│              ▼                        ▼                    ▼            │
│   ┌─────────────────┐   ┌───────────────────┐  ┌──────────────────┐    │
│   │  Ollama (local) │   │  Groq Cloud API   │  │  External MCP    │    │
│   │  llama3.2:3b    │   │  (optional cloud) │  │  mcp-server-     │    │
│   │  (default)      │   │                   │  │  fetch (public   │    │
│   │  no data egress │   │  USE_LOCAL_MODEL  │  │  references)     │    │
│   └─────────────────┘   │  = false          │  └──────────────────┘    │
│                          └───────────────────┘                          │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                   PostgreSQL (Render / local)                    │  │
│  │  TriageEncounter · TriageAssessment · TriageApproval            │  │
│  │  ReassessmentFlag · ReferralProposal · AuditEvent               │  │
│  │  ClinicalRole · IntegrationApplication                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                   Clinician Approval Gate                        │  │
│  │  POST /api/triage/approvals/{id}/decide/                        │  │
│  │  Requires: authenticated session + ClinicalRole (NURSE/DOCTOR)  │  │
│  │  Enforces: non-stale, non-expired, non-duplicate, transactional │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Breakdown

### 4.1 Custom MCP Server (`clinicalDecisionAI/triage/mcp_server.py`)

Protocol: FastMCP (in-memory; stdio transport available for external testing)  
Licence: FastMCP — MIT

| Tool | Purpose | Output |
|------|---------|--------|
| `assess_triage_priority` | Apply SATS deterministic rules | `AssessTriagePriorityOutput` with citations |
| `request_clinical_reassessment` | Flag observation change | `RequestReassessmentOutput`; queue unchanged |
| `prepare_department_referral` | Evaluate routing options | `PrepareDepartmentReferralOutput`; proposal only |
| `validate_vitals` | Parse and validate vitals | `ValidateVitalsOutput`; no scoring |

All tools: strict Pydantic input/output schemas, per-call audit logging, `asyncio.timeout` execution bounds.

### 4.2 External MCP Server

**mcp-server-fetch** (official MCP reference implementation)  
- Licence: MIT  
- Maintainer: Anthropic / MCP Community  
- Use: Retrieves public clinical reference URLs (WHO, CDC, EMSSA) to surface evidence links  
- Privacy guarantee: No patient data is transmitted; only public URLs are fetched  
- Integration boundary: `clinicalDecisionAI/triage/external_mcp.py`

### 4.3 LangGraph Orchestration Agent (`clinicalDecisionAI/triage/agent.py`)

Framework: **LangGraph 0.4.x** — MIT licence  
Graph type: `StateGraph` with typed state (`TriageAgentState`)

| Node | Type | Purpose |
|------|------|---------|
| `validate_vitals` | MCP tool call | Parse and validate inputs |
| `assess_priority` | MCP tool call | Apply SATS rules deterministically |
| `llm_summarise` | LLM inference | Advisory readable summary (skipped on RED emergency) |
| `propose_priority` | Pure logic | Compose final proposal |
| `await_approval` | Terminal | Halt — clinician decides next |

**Bounded execution:** `MAX_RETRIES = 2`, `NODE_TIMEOUT = 15s`, `agent_timeout = 60s`  
**Conditional edges:** Emergency RED → skip LLM node (do not delay); tool error → `error_terminal`

### 4.4 Open-Weights Local Model (Ollama)

| Parameter | Value |
|-----------|-------|
| Default model | `llama3.2:3b` |
| Minimum RAM | 4 GB |
| Minimum storage | 2 GB |
| Hardware | CPU-only (GPU optional, improves speed) |
| Data locality | Patient data never leaves the machine |
| Install | `brew install ollama` / `ollama.com/download` |
| Pull model | `ollama pull llama3.2:3b` |
| Enable | `USE_LOCAL_MODEL=true` or `AI_BACKEND=local` |

Reproducible test command:
```bash
USE_LOCAL_MODEL=true LOCAL_MODEL=llama3.2:3b python manage.py test clinicalDecisionAI --verbosity=2
```

### 4.5 Clinician Approval Service (`clinicalDecisionAI/triage/approval_service.py`)

| Check | Enforcement |
|-------|-------------|
| Clinician identity | `ClinicalRole` FK on authenticated user |
| Role restriction | `NURSE` or `DOCTOR` only |
| Stale assessment | SHA-256 hash of clinical input fields |
| Expiry | `expires_at` timestamp (default 60 min) |
| Duplicate prevention | `idempotency_key` unique constraint |
| Concurrency safety | `select_for_update()` on approval row |
| AI isolation | No code path allows LLM to write `TriageApproval` |

### 4.6 Audit Trail (`clinicalDecisionAI/triage/models.py::AuditEvent`)

Every tool call, approval, rejection, reassessment flag, and referral proposal creates an immutable `AuditEvent` record.

Fields: `event_id`, `correlation_id`, `created_at`, `actor_type`, `actor_id`, `actor_role`, `tool_name`, `action_type`, `input_summary_json`, `output_summary_json`, `validation_outcome`, `success`, `error_code`, `source_references_json`, `framework_version`, `model_version`, `previous_state`, `resulting_state`, `encounter_id`, `assessment_id`, `approval_id`

Restrictions: `default_permissions = ("view",)` — no add, change, or delete for ordinary users.

---

## 5. Data Model

```
TriageEncounter ──< TriageAssessment ──< TriageApproval
      │                    │
      ├──< ReassessmentFlag │
      │                    │
      └──< ReferralProposal │

AuditEvent (encounter_id, assessment_id, approval_id FKs — denormalised for immutability)

ClinicalRole ──< User (Django auth)
IntegrationApplication (bearer token auth for external EMR integrations)
```

### Privacy by design

- `TriageEncounter` never stores: name, MRN, national ID, date of birth, email, phone
- `patient_id` in incoming requests is excluded before any AI/LLM call
- `CLINICAL_LOG_INVALID_OUTPUT=false` by default (no clinical text in logs)
- All free-text symptoms pass through `sanitise_clinical_text()` before LLM

---

## 6. API Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/api/health/` | Public | Health check |
| POST | `/api/clinical-ai/analyze/` | Bearer token | Legacy single-call analysis |
| POST | `/api/triage/encounters/` | Session / Bearer | Register encounter |
| GET | `/api/triage/encounters/{id}/` | Session / Bearer | Encounter status |
| POST | `/api/triage/encounters/{id}/assess/` | Session / Bearer | Run agent |
| POST | `/api/triage/encounters/{id}/reassess/` | Session / Bearer | Flag reassessment |
| GET | `/api/triage/encounters/{id}/audit/` | Session + Role | Audit history |
| POST | `/api/triage/approvals/{id}/decide/` | Session + Role | Clinician decision |

---

## 7. Deployment

### Local (development)

```bash
git clone https://github.com/denisdelaki/afyoraAI.git
cd afyoraAI
cp .env.example .env          # fill in GROQ_API_KEY or set USE_LOCAL_MODEL=true
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### Local with open-weights model (no cloud API required)

```bash
brew install ollama
ollama pull llama3.2:3b      # one-time download
export USE_LOCAL_MODEL=true
python manage.py runserver
```

### Docker

```bash
docker-compose up
```

### Production (Render)

Configured in `render.yaml`. Requires `GROQ_API_KEY` set in Render environment variables. For in-country inference, deploy on local infrastructure with `USE_LOCAL_MODEL=true`.

---

## 8. Security Controls

| Control | Implementation |
|---------|---------------|
| Authentication | Bearer token (EMR integrations) + Django session (clinicians) |
| Authorisation | `ClinicalRole` model with NURSE/DOCTOR/ADMIN roles |
| CSRF | Django `CsrfViewMiddleware` |
| Input validation | Pydantic v2 strict schemas, `extra='forbid'` |
| Prompt injection | `sanitise_clinical_text()`; LLM instructed to treat input as untrusted |
| Rate limiting | `django-ratelimit` on API views |
| Secret management | All secrets via environment variables; `.env` in `.gitignore` |
| Audit immutability | `AuditEvent.default_permissions = ("view",)` |
| Concurrent approval | `select_for_update()` |
| Data locality | `USE_LOCAL_MODEL=true` keeps all inference on-premises |

---

## 9. Licence

MIT — see `LICENSE` file.
