import secrets

from django.core.management.base import BaseCommand

from clinicalDecisionAI.authentication import hash_integration_token
from clinicalDecisionAI.models import IntegrationApplication


class Command(BaseCommand):
    help = 'Register an application and issue its bearer token once.'

    def add_arguments(self, parser):
        parser.add_argument('name', help='Name of the application being connected.')

    def handle(self, *args, **options):
        token = f'cdai_{secrets.token_urlsafe(32)}'
        application = IntegrationApplication.objects.create(
            name=options['name'],
            token_hash=hash_integration_token(token),
        )
        self.stdout.write(f'Application ID: {application.pk}')
        self.stdout.write(f'Integration token: {token}')
        self.stdout.write('Store this token securely; it cannot be retrieved again.')