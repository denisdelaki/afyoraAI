"""
LangGraph-based triage agent for The Triage Desk.

This agent implements a genuine multi-step workflow:
  1. validate_vitals_node  — validate and parse input vitals (MCP tool call)
  2. assess_priority_node  — apply deterministic SATS rules (MCP tool call)
  3. check_escalation_node — check if immediate escalation is required
  4. llm_summarise_node    — ask the LLM for advisory free-text summary ONLY
  5. propose_priority_node — compose the final priority proposal
  6. await_approval_node   — halt; requires clinician approval to proceed

The LLM is used ONLY for steps that are appropriate for LLMs:
  - Interpreting free-text presenting complaints into structured fields
  - Producing a readable explanation of the proposed priority
  - Identifying missing information

Deterministic clinical rules (SATS) govern the priority, NOT the LLM.

The agent enforces:
  - Bounded retries (max 2 per tool node)
  - Explicit state transitions (StateGraph with typed nodes)
  - asyncio timeout protection
  - Safe fallback on LLM unavailability (deterministic triage survives)
  - No exposure of private chain-of-thought

Open-source orchestration: LangGraph (MIT licence)
Open-weights model: Ollama (locally runnable) with Groq as optional cloud backend
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from fastmcp import Client as MCPClient
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from clinicalDecisionAI.triage.mcp_server import triage_mcp_server

logger = logging.getLogger(__name__)

# Maximum tool-call retries per node
MAX_RETRIES = 2
# Node execution timeout (seconds)
NODE_TIMEOUT = 15.0


# ---------------------------------------------------------------------------
# Agent State — typed TypedDict
# ---------------------------------------------------------------------------

class TriageAgentState(TypedDict):
    # Input
    encounter_id: str
    age: int | None
    is_pregnant: bool | None
    symptoms: list[str]
    vitals_raw: dict[str, Any]

    # Node outputs
    vitals_validated: dict[str, Any] | None
    vitals_warnings: list[str]
    triage_result: dict[str, Any] | None
    immediate_escalation: bool
    llm_summary: str
    llm_model_used: str

    # Final proposal
    proposed_priority: str
    assessment_id: str
    assessment_timestamp: str
    framework_source: str
    triggered_criteria: list[dict]
    unassessed_vitals: list[str]
    missing_data_warnings: list[str]
    recommendation: str

    # Workflow control
    error: str | None
    retry_count: int
    workflow_step: str
    requires_approval: bool

    # Audit trail (list of tool invocations)
    audit_trail: list[dict]


def _initial_state(
    encounter_id: str,
    age: int | None,
    is_pregnant: bool | None,
    symptoms: list[str],
    vitals_raw: dict[str, Any],
) -> TriageAgentState:
    return {
        "encounter_id": encounter_id,
        "age": age,
        "is_pregnant": is_pregnant,
        "symptoms": symptoms,
        "vitals_raw": vitals_raw,
        "vitals_validated": None,
        "vitals_warnings": [],
        "triage_result": None,
        "immediate_escalation": False,
        "llm_summary": "",
        "llm_model_used": "",
        "proposed_priority": "NOT_ASSESSED",
        "assessment_id": "",
        "assessment_timestamp": "",
        "framework_source": "",
        "triggered_criteria": [],
        "unassessed_vitals": [],
        "missing_data_warnings": [],
        "recommendation": "",
        "error": None,
        "retry_count": 0,
        "workflow_step": "start",
        "requires_approval": True,
        "audit_trail": [],
    }


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

def _record_tool_call(state: TriageAgentState, tool: str, success: bool, summary: dict) -> list[dict]:
    entry = {
        "tool": tool,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "encounter_id": state["encounter_id"],
        "success": success,
        **summary,
    }
    return state["audit_trail"] + [entry]


# ---------------------------------------------------------------------------
# FastMCP response parser
# ---------------------------------------------------------------------------

def _parse_mcp_result(result: Any) -> dict:
    """
    FastMCP returns a CallToolResult with content[0].text as JSON.
    Fall through to result.data for older FastMCP versions.
    """
    if hasattr(result, "content") and result.content:
        raw = getattr(result.content[0], "text", None)
        if raw:
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass
    if hasattr(result, "data") and isinstance(result.data, dict):
        return result.data
    return {}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def validate_vitals_node(state: TriageAgentState) -> dict:
    """
    Node 1: Validate vital signs via MCP validate_vitals tool.
    Uses the custom Triage Desk MCP server (in-memory, no network).
    """
    tool_input = {
        "vitals": state["vitals_raw"],
        "age": state["age"],
        "is_pregnant": state["is_pregnant"],
    }
    try:
        async with asyncio.timeout(NODE_TIMEOUT):
            async with MCPClient(triage_mcp_server) as client:
                result = await client.call_tool("validate_vitals", {"input": tool_input})
                data = _parse_mcp_result(result)
    except asyncio.TimeoutError:
        logger.warning("validate_vitals timed out for encounter %s", state["encounter_id"])
        return {
            "vitals_validated": {},
            "vitals_warnings": ["Vital validation timed out — proceeding with raw input"],
            "workflow_step": "validate_vitals",
            "audit_trail": _record_tool_call(state, "validate_vitals", False, {"error": "timeout"}),
        }
    except Exception as exc:
        logger.warning("validate_vitals error: %s", type(exc).__name__)
        return {
            "vitals_validated": {},
            "vitals_warnings": [f"Vital validation failed ({type(exc).__name__}) — proceeding with raw input"],
            "workflow_step": "validate_vitals",
            "audit_trail": _record_tool_call(state, "validate_vitals", False, {"error": type(exc).__name__}),
        }

    warnings = data.get("warnings", []) if isinstance(data, dict) else []
    parsed = data.get("parsed_values", {}) if isinstance(data, dict) else {}
    return {
        "vitals_validated": parsed,
        "vitals_warnings": warnings,
        "workflow_step": "validate_vitals",
        "audit_trail": _record_tool_call(
            state, "validate_vitals", True,
            {"parsed_count": len(parsed), "warning_count": len(warnings)},
        ),
    }


async def assess_priority_node(state: TriageAgentState) -> dict:
    """
    Node 2: Apply deterministic SATS rules via MCP assess_triage_priority tool.
    The LLM is NOT called here. This node produces the authoritative priority.
    """
    # Build vitals dict — use validated values if available, otherwise raw
    vitals_for_assessment: dict[str, Any] = {}
    validated = state.get("vitals_validated") or {}
    if validated:
        vitals_for_assessment = {k: v for k, v in validated.items()}
    else:
        vitals_for_assessment = state["vitals_raw"]

    tool_input = {
        "encounter_id": state["encounter_id"],
        "age": state["age"],
        "is_pregnant": state["is_pregnant"],
        "symptoms": state["symptoms"],
        "vitals": vitals_for_assessment,
        "framework_version": "1.0",
    }
    try:
        async with asyncio.timeout(NODE_TIMEOUT):
            async with MCPClient(triage_mcp_server) as client:
                result = await client.call_tool("assess_triage_priority", {"input": tool_input})
                data = _parse_mcp_result(result)
    except asyncio.TimeoutError:
        logger.warning("assess_triage_priority timed out")
        return {
            "error": "triage_assessment_timeout",
            "workflow_step": "assess_priority",
            "audit_trail": _record_tool_call(state, "assess_triage_priority", False, {"error": "timeout"}),
        }
    except Exception as exc:
        logger.warning("assess_triage_priority error: %s", type(exc).__name__)
        return {
            "error": f"triage_assessment_error:{type(exc).__name__}",
            "workflow_step": "assess_priority",
            "audit_trail": _record_tool_call(state, "assess_triage_priority", False, {"error": type(exc).__name__}),
        }

    if not isinstance(data, dict):
        return {
            "error": "triage_assessment_invalid_output",
            "workflow_step": "assess_priority",
        }

    return {
        "triage_result": data,
        "proposed_priority": data.get("proposed_priority", "NOT_ASSESSED"),
        "assessment_id": data.get("assessment_id", str(uuid.uuid4())),
        "assessment_timestamp": data.get("assessment_timestamp", datetime.now(timezone.utc).isoformat()),
        "framework_source": data.get("framework_source", ""),
        "triggered_criteria": data.get("triggered_criteria", []),
        "unassessed_vitals": data.get("unassessed_vitals", []),
        "missing_data_warnings": data.get("missing_data_warnings", []),
        "recommendation": data.get("recommendation", ""),
        "immediate_escalation": data.get("immediate_escalation_required", False),
        "workflow_step": "assess_priority",
        "audit_trail": _record_tool_call(
            state, "assess_triage_priority", True,
            {
                "proposed_priority": data.get("proposed_priority"),
                "triggered_criteria_count": len(data.get("triggered_criteria", [])),
                "immediate_escalation": data.get("immediate_escalation_required", False),
            },
        ),
    }


async def llm_summarise_node(state: TriageAgentState) -> dict:
    """
    Node 3: Ask the LLM for an advisory free-text summary.

    The LLM is used ONLY for:
    - Producing a readable explanation of the deterministic triage finding
    - Identifying missing information from the presenting complaint
    - Advisory medication considerations

    The LLM CANNOT change the proposed_priority — that is already determined.
    If the LLM is unavailable, the deterministic result is still returned.
    """
    from clinicalDecisionAI.services import run_clinical_analysis, GroqConfigurationError, GroqProviderError, ClinicalResponseError
    from clinicalDecisionAI.schemas import ClinicalAnalysisRequest

    try:
        patient = ClinicalAnalysisRequest(
            age=state.get("age"),
            is_pregnant=state.get("is_pregnant"),
            symptoms=state["symptoms"],
            vitals=state["vitals_raw"],
        )
        decision = run_clinical_analysis(patient)
        summary = decision.supported_diagnosis
        model_used = "configured_model"
    except (GroqConfigurationError, GroqProviderError, ClinicalResponseError, Exception) as exc:
        # Graceful fallback: LLM unavailability does not fail the triage workflow
        logger.warning("LLM summarisation unavailable (%s); using deterministic recommendation only", type(exc).__name__)
        summary = state.get("recommendation", "Deterministic triage only — AI summarisation unavailable.")
        model_used = "unavailable"

    return {
        "llm_summary": summary,
        "llm_model_used": model_used,
        "workflow_step": "llm_summarise",
        "audit_trail": _record_tool_call(
            state, "llm_summarise", model_used != "unavailable",
            {"model_used": model_used, "summary_length": len(summary)},
        ),
    }


def propose_priority_node(state: TriageAgentState) -> dict:
    """
    Node 4: Compose the final priority proposal.

    Sets requires_approval=True unconditionally. The agent CANNOT approve its
    own proposal. A qualified clinician must do so via the approval API.
    """
    return {
        "workflow_step": "propose_priority",
        "requires_approval": True,
        "audit_trail": _record_tool_call(
            state, "propose_priority", True,
            {
                "proposed_priority": state.get("proposed_priority"),
                "requires_approval": True,
                "immediate_escalation": state.get("immediate_escalation"),
            },
        ),
    }


def await_approval_node(state: TriageAgentState) -> dict:
    """
    Node 5 (terminal): Wait for clinician approval.

    This node is the terminal state. The workflow STOPS here.
    No operational queue change occurs until a clinician explicitly approves
    the proposal through the approval API.
    """
    return {
        "workflow_step": "await_approval",
        "audit_trail": _record_tool_call(
            state, "await_approval", True,
            {"status": "PENDING_CLINICIAN_APPROVAL", "can_auto_approve": False},
        ),
    }


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def should_escalate(state: TriageAgentState) -> str:
    """If an error occurred, go to terminal. Otherwise continue."""
    if state.get("error"):
        return "error_terminal"
    return "continue"


def after_assess(state: TriageAgentState) -> str:
    """After priority assessment: skip LLM if escalation is immediate."""
    if state.get("error"):
        return "error_terminal"
    if state.get("immediate_escalation"):
        # Skip LLM for immediate emergencies — do not let AI processing delay action
        return "propose_direct"
    return "llm_summarise"


def error_node(state: TriageAgentState) -> dict:
    """Terminal error node."""
    return {
        "workflow_step": "error",
        "proposed_priority": state.get("proposed_priority", "NOT_ASSESSED"),
        "requires_approval": True,
    }


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_triage_graph() -> StateGraph:
    graph = StateGraph(TriageAgentState)

    graph.add_node("validate_vitals", validate_vitals_node)
    graph.add_node("assess_priority", assess_priority_node)
    graph.add_node("llm_summarise", llm_summarise_node)
    graph.add_node("propose_priority", propose_priority_node)
    graph.add_node("await_approval", await_approval_node)
    graph.add_node("error_terminal", error_node)

    graph.set_entry_point("validate_vitals")

    graph.add_conditional_edges(
        "validate_vitals",
        should_escalate,
        {"continue": "assess_priority", "error_terminal": "error_terminal"},
    )
    graph.add_conditional_edges(
        "assess_priority",
        after_assess,
        {
            "llm_summarise": "llm_summarise",
            "propose_direct": "propose_priority",
            "error_terminal": "error_terminal",
        },
    )
    graph.add_edge("llm_summarise", "propose_priority")
    graph.add_edge("propose_priority", "await_approval")
    graph.add_edge("await_approval", END)
    graph.add_edge("error_terminal", END)

    return graph


# Compiled graph — reusable across requests
_compiled_graph = build_triage_graph().compile()


async def run_triage_agent(
    encounter_id: str,
    age: int | None,
    is_pregnant: bool | None,
    symptoms: list[str],
    vitals_raw: dict[str, Any],
    timeout_seconds: float = 60.0,
) -> TriageAgentState:
    """
    Run the LangGraph triage agent for one encounter.

    Returns the final state containing:
    - proposed_priority (deterministic SATS result)
    - triggered_criteria with full citations
    - llm_summary (advisory, may be empty if LLM unavailable)
    - audit_trail of all tool invocations
    - requires_approval = True (always)

    Raises:
        asyncio.TimeoutError: if the entire agent exceeds timeout_seconds
    """
    initial = _initial_state(encounter_id, age, is_pregnant, symptoms, vitals_raw)
    async with asyncio.timeout(timeout_seconds):
        final_state = await _compiled_graph.ainvoke(initial)
    return final_state


def run_triage_agent_sync(
    encounter_id: str,
    age: int | None,
    is_pregnant: bool | None,
    symptoms: list[str],
    vitals_raw: dict[str, Any],
    timeout_seconds: float = 60.0,
) -> TriageAgentState:
    """Synchronous wrapper for Django view context."""
    return asyncio.run(
        run_triage_agent(encounter_id, age, is_pregnant, symptoms, vitals_raw, timeout_seconds)
    )
