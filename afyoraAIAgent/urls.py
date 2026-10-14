"""
URL configuration for afyoraAIAgent project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.contrib.auth.views import LoginView, LogoutView
from django.urls import path, reverse_lazy

from clinicalDecisionAI.views import (
    account_signup,
    ClinicalAnalysisAPIView,
    application_home,
    clinical_sandbox,
    deactivate_integration,
    health_check,
    integration_guide,
    integration_dashboard,
)
from clinicalDecisionAI.triage.api_views import (
    TriageEncounterCreateView,
    TriageEncounterDetailView,
    TriageAssessmentView,
    TriageApprovalDecideView,
    TriageAuditHistoryView,
    ReassessmentFlagView,
)

urlpatterns = [
    path('', application_home, name='home'),
    path('api/health/', health_check, name='health-check'),
    path('admin/', admin.site.urls),
    path('accounts/signup/', account_signup, name='account-signup'),
    path(
        'accounts/login/',
        LoginView.as_view(template_name='registration/login.html', redirect_authenticated_user=True),
        name='account-login',
    ),
    path(
        'accounts/logout/',
        LogoutView.as_view(next_page=reverse_lazy('account-login')),
        name='account-logout',
    ),
    path('integrations/', integration_dashboard, name='integration-dashboard'),
    path('integrations/guide/', integration_guide, name='integration-guide'),
    path('sandbox/', clinical_sandbox, name='clinical-sandbox'),
    path(
        'integrations/<int:integration_id>/deactivate/',
        deactivate_integration,
        name='integration-deactivate',
    ),
    path('api/clinical-ai/analyze/', ClinicalAnalysisAPIView.as_view(), name='clinical-ai-analyze'),

    # Triage workflow API
    path('api/triage/encounters/', TriageEncounterCreateView.as_view(), name='triage-encounter-create'),
    path('api/triage/encounters/<uuid:encounter_id>/', TriageEncounterDetailView.as_view(), name='triage-encounter-detail'),
    path('api/triage/encounters/<uuid:encounter_id>/assess/', TriageAssessmentView.as_view(), name='triage-assess'),
    path('api/triage/encounters/<uuid:encounter_id>/reassess/', ReassessmentFlagView.as_view(), name='triage-reassess'),
    path('api/triage/encounters/<uuid:encounter_id>/audit/', TriageAuditHistoryView.as_view(), name='triage-audit'),
    path('api/triage/approvals/<uuid:approval_id>/decide/', TriageApprovalDecideView.as_view(), name='triage-approval-decide'),
]
