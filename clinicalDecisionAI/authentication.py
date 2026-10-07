import hashlib

from django.utils import timezone

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