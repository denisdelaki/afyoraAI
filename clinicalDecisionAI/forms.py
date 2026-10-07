from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import UserCreationForm

from clinicalDecisionAI.models import IntegrationApplication


class AccountCreationForm(UserCreationForm):
	email = forms.EmailField(required=False)

	class Meta(UserCreationForm.Meta):
		model = get_user_model()
		fields = ('username', 'email')


class IntegrationRegistrationForm(forms.ModelForm):
	class Meta:
		model = IntegrationApplication
		fields = ('name',)
		widgets = {
			'name': forms.TextInput(attrs={
				'autocomplete': 'organization',
				'placeholder': 'For example, Scheduling app',
			}),
		}