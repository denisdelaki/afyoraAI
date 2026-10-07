from django.conf import settings
from django.db import models


class IntegrationApplication(models.Model):
	name = models.CharField(max_length=120)
	owner = models.ForeignKey(
		settings.AUTH_USER_MODEL,
		on_delete=models.CASCADE,
		related_name='integration_applications',
		null=True,
		blank=True,
	)
	token_hash = models.CharField(max_length=64, unique=True, editable=False)
	is_active = models.BooleanField(default=True)
	created_at = models.DateTimeField(auto_now_add=True)
	last_used_at = models.DateTimeField(null=True, blank=True)

	def __str__(self):
		return self.name
