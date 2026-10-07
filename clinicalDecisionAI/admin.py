from django.contrib import admin

from clinicalDecisionAI.models import IntegrationApplication


@admin.register(IntegrationApplication)
class IntegrationApplicationAdmin(admin.ModelAdmin):
	list_display = ('name', 'owner', 'is_active', 'created_at', 'last_used_at')
	list_filter = ('is_active',)
	readonly_fields = ('token_hash', 'created_at', 'last_used_at')
