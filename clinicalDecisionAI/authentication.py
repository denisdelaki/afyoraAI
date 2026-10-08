import hashlib

from django.utils import timezone
from django.contrib.auth.models import AnonymousUser
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.permissions import BasePermission

from clinicalDecisionAI.models import IntegrationApplication


def hash_integration_token(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def authenticate_integration_request(request) -> IntegrationApplication | None:
    authorization = request.headers.get('Authorization', '').split()
    if (
        len(authorization) != 2
        or authorization[0].lower() != 'bearer'
        or not authorization[1].startswith('cdai_')
    ):
        return None

    token_hash = hash_integration_token(authorization[1])
    application = IntegrationApplication.objects.filter(
        token_hash=token_hash,
        is_active=True,
    ).first()
    if application is not None:
        IntegrationApplication.objects.filter(pk=application.pk).update(
            last_used_at=timezone.now()
        )
    return application


class IntegrationTokenAuthentication(BaseAuthentication):
    def authenticate(self, request):
        application = authenticate_integration_request(request)
        if application is None:
            raise AuthenticationFailed('A registered application bearer token is required.')
        return AnonymousUser(), application

    def authenticate_header(self, request):
        return 'Bearer'


class IntegrationApplicationPermission(BasePermission):
    def has_permission(self, request, view):
        return isinstance(request.auth, IntegrationApplication) and request.auth.is_active