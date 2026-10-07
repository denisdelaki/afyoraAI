import json
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from django.core.management import call_command
from django.contrib.auth import get_user_model
from django.test import Client, TestCase, override_settings
from groq import APITimeoutError

from clinicalDecisionAI.authentication import hash_integration_token
from clinicalDecisionAI.models import IntegrationApplication
from clinicalDecisionAI.schemas import ClinicalAnalysisRequest, ClinicalDecisionResponse
from clinicalDecisionAI.services import (
	ClinicalResponseError,
	GroqConfigurationError,
	GroqClient,
	GroqProviderError,
	run_clinical_analysis,
	select_groq_model,
)

MOCK_API_KEY = 'mock'


VALID_RESPONSE = {
	'clinical_summary': 'Assessment is limited to the supplied information.',
	'risk_level': 'moderate',
	'possible_conditions': [
		{
			'condition': 'Example differential',
			'likelihood': 'Uncertain',
			'supporting_evidence': ['Fever was reported.'],
		}
	],
	'abnormal_findings': [],
	'recommended_investigations': [],
	'management_considerations': ['Review the patient in clinical context.'],
	'medication_considerations': [],
	'red_flags': [],
	'referral_recommendation': {
		'required': False,
		'urgency': 'routine',
		'reason': 'No referral information supplied.',
	},
	'confidence': 0.35,
	'requires_human_review': True,
}


def provider_response(content):
	return SimpleNamespace(
		choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
	)


class GroqServiceTests(TestCase):
	def test_model_is_selected_from_configured_pool(self):
		models = ('openai/gpt-oss-120b', 'openai/gpt-oss-20b')
		with override_settings(GROQ_MODELS=models):
			with patch(
				'clinicalDecisionAI.services.random.choice',
				return_value='openai/gpt-oss-20b',
			) as choose_model:
				selected = select_groq_model()

		self.assertEqual(selected, 'openai/gpt-oss-20b')
		choose_model.assert_called_once_with(models)

	@override_settings(GROQ_API_KEY=MOCK_API_KEY)
	@patch('clinicalDecisionAI.services.Groq')
	def test_selected_model_is_sent_to_groq(self, groq_factory):
		groq = groq_factory.return_value
		groq.chat.completions.create.return_value = provider_response('{}')
		client = GroqClient(model='openai/gpt-oss-20b')

		client.ask('prompt')

		self.assertEqual(
			groq.chat.completions.create.call_args.kwargs['model'],
			'openai/gpt-oss-20b',
		)

	@override_settings(GROQ_API_KEY=MOCK_API_KEY)
	@patch('clinicalDecisionAI.services.Groq')
	def test_successful_groq_request_returns_validated_response(self, groq_factory):
		groq = groq_factory.return_value
		groq.chat.completions.create.return_value = provider_response(
			json.dumps(VALID_RESPONSE)
		)
		patient = ClinicalAnalysisRequest(
			patient_id=318,
			age=47,
			symptoms=['fever'],
			vitals={'temperature': 38.5},
		)

		result = run_clinical_analysis(patient)

		self.assertEqual(result.clinical_summary, VALID_RESPONSE['clinical_summary'])
		groq_factory.assert_called_once_with(
			api_key=MOCK_API_KEY, timeout=30.0
		)
		call = groq.chat.completions.create.call_args.kwargs
		self.assertIn(call['model'], ('openai/gpt-oss-120b', 'openai/gpt-oss-20b'))
		self.assertEqual(call['response_format'], {'type': 'json_object'})
		self.assertNotIn('318', call['messages'][1]['content'])
		self.assertTrue(call['messages'][0]['content'].find('not a doctor') >= 0)

	@override_settings(GROQ_API_KEY='')
	@patch('clinicalDecisionAI.services.Groq')
	def test_missing_api_key_fails_before_client_creation(self, groq_factory):
		with self.assertRaises(GroqConfigurationError):
			GroqClient()
		groq_factory.assert_not_called()

	@override_settings(GROQ_API_KEY=MOCK_API_KEY)
	@patch('clinicalDecisionAI.services.Groq')
	def test_provider_failure_is_categorized(self, groq_factory):
		groq_factory.return_value.chat.completions.create.side_effect = RuntimeError(
			'provider detail must not be returned'
		)

		with self.assertRaises(GroqProviderError) as context:
			GroqClient().ask('prompt')

		self.assertEqual(context.exception.category, 'provider')

	@override_settings(GROQ_API_KEY=MOCK_API_KEY)
	@patch('clinicalDecisionAI.services.Groq')
	def test_timeout_is_categorized(self, groq_factory):
		request = httpx.Request('POST', 'https://api.groq.com')
		groq_factory.return_value.chat.completions.create.side_effect = (
			APITimeoutError(request=request)
		)

		with self.assertRaises(GroqProviderError) as context:
			GroqClient().ask('prompt')

		self.assertEqual(context.exception.category, 'timeout')

	@override_settings(GROQ_API_KEY=MOCK_API_KEY)
	@patch('clinicalDecisionAI.services.Groq')
	def test_empty_provider_response_is_rejected(self, groq_factory):
		groq_factory.return_value.chat.completions.create.return_value = provider_response(
			'  '
		)

		with self.assertRaises(ClinicalResponseError) as context:
			GroqClient().ask('prompt')

		self.assertEqual(context.exception.category, 'empty_response')

	@override_settings(GROQ_API_KEY=MOCK_API_KEY)
	@patch('clinicalDecisionAI.services.Groq')
	def test_provider_response_without_choices_is_rejected(self, groq_factory):
		groq_factory.return_value.chat.completions.create.return_value = SimpleNamespace(
			choices=[]
		)

		with self.assertRaises(ClinicalResponseError) as context:
			GroqClient().ask('prompt')

		self.assertEqual(context.exception.category, 'empty_response')

	@override_settings(GROQ_API_KEY=MOCK_API_KEY)
	@patch('clinicalDecisionAI.services.Groq')
	def test_invalid_json_is_rejected(self, groq_factory):
		groq_factory.return_value.chat.completions.create.return_value = provider_response(
			'{not json'
		)

		with self.assertRaises(ClinicalResponseError) as context:
			run_clinical_analysis(ClinicalAnalysisRequest(symptoms=['fever']))

		self.assertEqual(context.exception.category, 'invalid_json')

	@override_settings(GROQ_API_KEY=MOCK_API_KEY)
	@patch('clinicalDecisionAI.services.Groq')
	def test_schema_invalid_json_object_is_rejected(self, groq_factory):
		invalid_response = {**VALID_RESPONSE, 'risk_level': 'certain'}
		groq_factory.return_value.chat.completions.create.return_value = provider_response(
			json.dumps(invalid_response)
		)

		with self.assertRaises(ClinicalResponseError) as context:
			run_clinical_analysis(ClinicalAnalysisRequest())

		self.assertEqual(context.exception.category, 'schema_validation')


class ClinicalAnalysisEndpointTests(TestCase):
	def setUp(self):
		self.client = Client()
		self.token = 'cdai_test-application-token'
		self.application = IntegrationApplication.objects.create(
			name='Test integration',
			token_hash=hash_integration_token(self.token),
		)
		self.auth_headers = {'HTTP_AUTHORIZATION': f'Bearer {self.token}'}

	def test_browser_get_explains_post_api_usage(self):
		response = self.client.get('/api/clinical-ai/analyze/')

		self.assertEqual(response.status_code, 405)
		self.assertEqual(response['Allow'], 'POST')
		self.assertEqual(
			response.json()['error']['code'], 'method_not_allowed'
		)

	def test_endpoint_returns_validated_analysis(self):
		result = ClinicalDecisionResponse.model_validate(VALID_RESPONSE)
		with patch('clinicalDecisionAI.views.run_clinical_analysis', return_value=result):
			response = self.client.post(
				'/api/clinical-ai/analyze/',
				data=json.dumps({'symptoms': ['fever']}),
				content_type='application/json',
				HTTP_X_REQUEST_ID='test-request-id',
				**self.auth_headers,
			)

		self.assertEqual(response.status_code, 200)
		self.assertEqual(response.json(), VALID_RESPONSE)
		self.assertEqual(response['X-Request-ID'], 'test-request-id')
		self.assertIn(
			response['X-Groq-Model'],
			('openai/gpt-oss-120b', 'openai/gpt-oss-20b'),
		)

	def test_bearer_authenticated_request_does_not_require_cookie_csrf(self):
		client = Client(enforce_csrf_checks=True)
		result = ClinicalDecisionResponse.model_validate(VALID_RESPONSE)
		with patch('clinicalDecisionAI.views.run_clinical_analysis', return_value=result):
			response = client.post(
				'/api/clinical-ai/analyze/',
				data='{}',
				content_type='application/json',
				**self.auth_headers,
			)

		self.assertEqual(response.status_code, 200)

	def test_endpoint_rejects_invalid_patient_data(self):
		with patch('clinicalDecisionAI.views.run_clinical_analysis') as analysis:
			response = self.client.post(
				'/api/clinical-ai/analyze/',
				data=json.dumps({'age': 150}),
				content_type='application/json',
				**self.auth_headers,
			)

		self.assertEqual(response.status_code, 400)
		self.assertEqual(response.json()['error']['code'], 'invalid_request')
		analysis.assert_not_called()

	def test_endpoint_requires_registered_application(self):
		with patch('clinicalDecisionAI.views.run_clinical_analysis') as analysis:
			response = self.client.post(
				'/api/clinical-ai/analyze/',
				data='{}',
				content_type='application/json',
			)

		self.assertEqual(response.status_code, 401)
		self.assertEqual(
			response.json()['error']['code'], 'authentication_required'
		)
		self.assertEqual(response['WWW-Authenticate'], 'Bearer')
		analysis.assert_not_called()

	def test_endpoint_rejects_unknown_application_token(self):
		with patch('clinicalDecisionAI.views.run_clinical_analysis') as analysis:
			response = self.client.post(
				'/api/clinical-ai/analyze/',
				data='{}',
				content_type='application/json',
				HTTP_AUTHORIZATION='Bearer cdai_unknown-token',
			)

		self.assertEqual(response.status_code, 401)
		analysis.assert_not_called()

	def test_endpoint_rejects_deactivated_application(self):
		self.application.is_active = False
		self.application.save(update_fields=['is_active'])
		with patch('clinicalDecisionAI.views.run_clinical_analysis') as analysis:
			response = self.client.post(
				'/api/clinical-ai/analyze/',
				data='{}',
				content_type='application/json',
				**self.auth_headers,
			)

		self.assertEqual(response.status_code, 401)
		analysis.assert_not_called()

	def test_provider_failure_returns_controlled_error(self):
		with patch(
			'clinicalDecisionAI.views.run_clinical_analysis',
			side_effect=GroqProviderError('timeout'),
		):
			response = self.client.post(
				'/api/clinical-ai/analyze/',
				data='{}',
				content_type='application/json',
				**self.auth_headers,
			)

		self.assertEqual(response.status_code, 504)
		self.assertEqual(response.json()['error']['code'], 'ai_timeout')
		self.assertNotIn('GROQ_API_KEY', response.content.decode())

	def test_unconfigured_provider_returns_service_unavailable(self):
		with patch(
			'clinicalDecisionAI.views.run_clinical_analysis',
			side_effect=GroqConfigurationError,
		):
			response = self.client.post(
				'/api/clinical-ai/analyze/',
				data='{}',
				content_type='application/json',
				**self.auth_headers,
			)

		self.assertEqual(response.status_code, 503)
		self.assertEqual(
			response.json()['error']['code'], 'ai_provider_unavailable'
		)


class IntegrationApplicationProvisioningTests(TestCase):
	def test_management_command_outputs_one_time_token_and_stores_only_hash(self):
		output = StringIO()
		call_command('create_integration_app', 'Scheduling app', stdout=output)

		token_line = next(
			line for line in output.getvalue().splitlines()
			if line.startswith('Integration token: ')
		)
		token = token_line.removeprefix('Integration token: ')
		application = IntegrationApplication.objects.get(name='Scheduling app')
		self.assertTrue(token.startswith('cdai_'))
		self.assertEqual(application.token_hash, hash_integration_token(token))
		self.assertNotEqual(application.token_hash, token)


class AccountIntegrationFlowTests(TestCase):
	def test_signup_creates_account_and_signs_user_in(self):
		response = self.client.post(
			'/accounts/signup/',
			{
				'username': 'new-partner',
				'email': 'partner@example.com',
				'password1': 'UniqueAccountPass!582',
				'password2': 'UniqueAccountPass!582',
			},
		)

		self.assertRedirects(response, '/integrations/')
		self.assertTrue(
			get_user_model().objects.filter(username='new-partner').exists()
		)
		self.assertEqual(self.client.get('/integrations/').status_code, 200)

	def test_user_can_register_and_only_see_token_once(self):
		user = get_user_model().objects.create_user(
			username='integration-owner', password='UniqueAccountPass!582'
		)
		self.client.force_login(user)

		response = self.client.post('/integrations/', {'name': 'Scheduling app'})

		self.assertEqual(response.status_code, 201)
		self.assertEqual(response['Cache-Control'], 'no-store')
		integration = IntegrationApplication.objects.get(owner=user)
		self.assertContains(response, 'Copy this token now', status_code=201)
		self.assertNotContains(response, integration.token_hash, status_code=201)
		self.assertContains(response, 'cdai_', status_code=201)
		self.assertNotContains(
			self.client.get('/integrations/'), 'cdai_'
		)

	def test_users_cannot_view_or_deactivate_another_users_integration(self):
		owner = get_user_model().objects.create_user(
			username='first-owner', password='UniqueAccountPass!582'
		)
		other_user = get_user_model().objects.create_user(
			username='second-owner', password='UniqueAccountPass!582'
		)
		integration = IntegrationApplication.objects.create(
			name='Private integration',
			owner=other_user,
			token_hash=hash_integration_token('cdai_private-token'),
		)
		self.client.force_login(owner)

		response = self.client.post(
			f'/integrations/{integration.pk}/deactivate/'
		)

		self.assertEqual(response.status_code, 404)
		integration.refresh_from_db()
		self.assertTrue(integration.is_active)

	def test_owner_can_deactivate_their_integration(self):
		user = get_user_model().objects.create_user(
			username='deactivation-owner', password='UniqueAccountPass!582'
		)
		integration = IntegrationApplication.objects.create(
			name='Retiring integration',
			owner=user,
			token_hash=hash_integration_token('cdai_retiring-token'),
		)
		self.client.force_login(user)

		response = self.client.post(
			f'/integrations/{integration.pk}/deactivate/'
		)

		self.assertRedirects(response, '/integrations/')
		integration.refresh_from_db()
		self.assertFalse(integration.is_active)

	def test_integration_guide_requires_login(self):
		response = self.client.get('/integrations/guide/')

		self.assertRedirects(
			response, '/accounts/login/?next=/integrations/guide/'
		)

	def test_integration_guide_shows_language_tabs_and_examples(self):
		user = get_user_model().objects.create_user(
			username='guide-user', password='UniqueAccountPass!582'
		)
		self.client.force_login(user)

		response = self.client.get('/integrations/guide/')

		self.assertEqual(response.status_code, 200)
		for language in ('Python', 'Node.js', 'Java', 'Go'):
			self.assertContains(response, language)
		self.assertContains(response, '/api/clinical-ai/analyze/')
		self.assertContains(response, 'CLINICAL_AI_TOKEN')

	def test_sandbox_requires_login(self):
		response = self.client.get('/sandbox/')

		self.assertRedirects(response, '/accounts/login/?next=/sandbox/')

	@patch('clinicalDecisionAI.views.run_clinical_analysis')
	def test_sandbox_runs_validated_analysis_without_application_token(self, analyze):
		user = get_user_model().objects.create_user(
			username='sandbox-user', password='UniqueAccountPass!582'
		)
		self.client.force_login(user)
		analyze.return_value = ClinicalDecisionResponse.model_validate(VALID_RESPONSE)

		response = self.client.post(
			'/sandbox/',
			{'patient_json': json.dumps({'symptoms': ['fever']})},
		)

		self.assertEqual(response.status_code, 200)
		self.assertEqual(response['Cache-Control'], 'no-store')
		self.assertContains(response, 'AI Clinical Decision Support')
		self.assertContains(response, 'Human clinician review required: Yes')
		self.assertContains(response, 'Selected Groq model:')
		analyze.assert_called_once()

	@patch('clinicalDecisionAI.views.run_clinical_analysis')
	def test_sandbox_rejects_invalid_json_without_calling_provider(self, analyze):
		user = get_user_model().objects.create_user(
			username='sandbox-invalid-user', password='UniqueAccountPass!582'
		)
		self.client.force_login(user)

		response = self.client.post('/sandbox/', {'patient_json': '{bad json'})

		self.assertEqual(response.status_code, 400)
		self.assertContains(response, 'correct the highlighted validation issues', status_code=400)
		analyze.assert_not_called()

	@patch(
		'clinicalDecisionAI.views.run_clinical_analysis',
		side_effect=GroqConfigurationError,
	)
	def test_sandbox_handles_missing_provider_configuration(self, analyze):
		user = get_user_model().objects.create_user(
			username='sandbox-unconfigured-user', password='UniqueAccountPass!582'
		)
		self.client.force_login(user)

		response = self.client.post(
			'/sandbox/',
			{'patient_json': json.dumps({'symptoms': ['fever']})},
		)

		self.assertEqual(response.status_code, 503)
		self.assertContains(response, 'provider is not configured', status_code=503)

	@patch(
		'clinicalDecisionAI.views.run_clinical_analysis',
		side_effect=GroqProviderError('provider', status_code=400),
	)
	def test_sandbox_shows_safe_provider_status_guidance(self, analyze):
		user = get_user_model().objects.create_user(
			username='sandbox-provider-user', password='UniqueAccountPass!582'
		)
		self.client.force_login(user)

		response = self.client.post(
			'/sandbox/',
			{'patient_json': json.dumps({'symptoms': ['fever']})},
		)

		self.assertEqual(response.status_code, 502)
		self.assertContains(response, 'HTTP 400', status_code=502)
		self.assertContains(response, 'configured model and request options', status_code=502)
