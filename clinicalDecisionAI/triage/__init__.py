# Expose triage models at the app level for Django admin and migrations
from clinicalDecisionAI.triage.models import (  # noqa: F401
    AuditEvent,
    ClinicalRole,
    ReassessmentFlag,
    ReferralProposal,
    TriageApproval,
    TriageAssessment,
    TriageEncounter,
)
