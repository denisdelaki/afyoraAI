import asyncio
import json
import os
import runpy
import threading
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from django.core.management import call_command
from django.core.exceptions import ImproperlyConfigured
from django.core.management.utils import get_random_secret_key
from django.contrib.auth import get_user_model
from django.test import Client, SimpleTestCase, TestCase, override_settings
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from groq import APITimeoutError
from pydantic import ValidationError

from afyoraAIAgent import settings as project_settings
from clinicalDecisionAI.authentication import hash_integration_token
from clinicalDecisionAI.models import IntegrationApplication
from clinicalDecisionAI.schemas import ClinicalAnalysisRequest, ClinicalDecisionSchema
from clinicalDecisionAI.views import _analyze_patient, _local_context
from clinicalDecisionAI.services import (
	ClinicalResponseError,
	GroqConfigurationError,
	GroqClient,
	GroqProviderError,
	run_clinical_analysis,
	select_groq_model,
	assess_vital_triage,
)

MOCK_API_KEY = 'mock'


VALID_RESPONSE = {
	'supported_diagnosis': 'Uncertain assessment; qualified clinician review required.',
	'possible_disease': ['Example differential'],
	'drugs_admissible': [],
	'further_labs_to_be_done': ['Review in the context of a clinical assessment.'],
}


def provider_response(content):
	return SimpleNamespace(
		choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
	)


class DeploymentEnvironmentTests(SimpleTestCase):
	def test_render_environment_settings_take_precedence(self):
		environment = {
			'DEBUG': 'false', 'DJANGO_SECRET_KEY': get_random_secret_key(),
			'GROQ_API_KEY': 'mock-render-key', 'GROQ_MODELS': 'model-one,model-two',
			'GROQ_TIMEOUT_SECONDS': '20.5', 'CLINICAL_MCP_TIMEOUT_SECONDS': '3',
			'CLINICAL_MAX_REQUEST_BYTES': '32768',
			'RENDER_EXTERNAL_HOSTNAME': 'clinical-example.onrender.com',
			'DATABASE_URL': 'postgresql://test@localhost:5432/test',
		}
		with patch.dict(os.environ, environment):
			configuration = runpy.run_path(project_settings.__file__)
		self.assertFalse(configuration['DEBUG'])
		self.assertEqual(configuration['SECRET_KEY'], environment['DJANGO_SECRET_KEY'])
		self.assertEqual(configuration['GROQ_API_KEY'], environment['GROQ_API_KEY'])
		self.assertEqual(configuration['GROQ_MODELS'], ('model-one', 'model-two'))
		self.assertEqual(configuration['GROQ_TIMEOUT_SECONDS'], 20.5)
		self.assertEqual(configuration['CLINICAL_MCP_TIMEOUT_SECONDS'], 3.0)
		self.assertEqual(configuration['CLINICAL_MAX_REQUEST_BYTES'], 32768)
		self.assertEqual(configuration['DATA_UPLOAD_MAX_MEMORY_SIZE'], 32768)
		self.assertIn(environment['RENDER_EXTERNAL_HOSTNAME'], configuration['ALLOWED_HOSTS'])
		self.assertIn('https://clinical-example.onrender.com', configuration['CSRF_TRUSTED_ORIGINS'])
		self.assertEqual(configuration['DATABASES']['default']['ENGINE'], 'django.db.backends.postgresql')

	def test_missing_optional_environment_settings_keep_defaults(self):
		with patch.dict(os.environ), patch('dotenv.load_dotenv') as dotenv:
			for name in ('GROQ_TIMEOUT_SECONDS', 'CLINICAL_MCP_TIMEOUT_SECONDS', 'CLINICAL_MAX_REQUEST_BYTES'):
				os.environ.pop(name, None)
			configuration = runpy.run_path(project_settings.__file__)
		self.assertEqual(configuration['GROQ_TIMEOUT_SECONDS'], 30.0)
		self.assertEqual(configuration['CLINICAL_MCP_TIMEOUT_SECONDS'], 5.0)
		self.assertEqual(configuration['CLINICAL_MAX_REQUEST_BYTES'], 65536)
		self.assertFalse(dotenv.call_args.kwargs['override'])

	def test_invalid_environment_limits_fail_at_startup(self):
		for name, invalid in (
			('GROQ_TIMEOUT_SECONDS', 'nan'), ('GROQ_TIMEOUT_SECONDS', 'inf'),
			('GROQ_TIMEOUT_SECONDS', '-1'), ('CLINICAL_MCP_TIMEOUT_SECONDS', '0'),
			('CLINICAL_MCP_TIMEOUT_SECONDS', 'invalid'), ('CLINICAL_MAX_REQUEST_BYTES', '1.5'),
			('CLINICAL_MAX_REQUEST_BYTES', '0'),
		):
			with self.subTest(name=name, value=invalid), patch.dict(os.environ, {name: invalid}):
				with self.assertRaisesMessage(ImproperlyConfigured, name):
					runpy.run_path(project_settings.__file__)


class VitalTriageTests(SimpleTestCase):
	def test_each_extreme_vital_triggers_urgent_care(self):
		for vitals in (
			{'heart_rate': 40}, {'heart_rate': 131}, {'respiratory_rate': 8},
			{'respiratory_rate': 25}, {'spo2': 91}, {'systolic_bp': 90},
			{'blood_pressure': '180/80'}, {'blood_pressure': '130/120'},
			{'temperature': 35}, {'temperature': 39.1},
		):
			with self.subTest(vitals=vitals):
				triage = assess_vital_triage(ClinicalAnalysisRequest(age=47, symptoms=['unwell'], vitals=vitals))
				self.assertEqual(triage.level, 'urgent')
				self.assertTrue(triage.urgent_care_recommended)
				self.assertTrue(triage.reasons)

	def test_missing_invalid_and_conflicting_vitals_are_not_normalized_away(self):
		for vitals in ({}, {'spo2': 101}, {'temperature': 'unknown'}, {'pulse': 60, 'heart_rate': 75}):
			with self.subTest(vitals=vitals):
				triage = assess_vital_triage(ClinicalAnalysisRequest(age=47, symptoms=['unwell'], vitals=vitals))
				self.assertEqual(triage.level, 'not_assessed')
				self.assertTrue(triage.unassessed_vitals)
		triage = assess_vital_triage(ClinicalAnalysisRequest(age=47, symptoms=['unwell'], vitals={'spo2': 80, 'oxygen_saturation': 98}))
		self.assertEqual(triage.level, 'urgent')
		self.assertIn('oxygen_saturation', triage.unassessed_vitals)
		triage = assess_vital_triage(ClinicalAnalysisRequest(
			age=47, symptoms=['unwell'], vitals={'systolic_bp': 120, 'blood_pressure': '70/40'},
		))
		self.assertEqual(triage.level, 'urgent')
		self.assertIn('systolic_blood_pressure', triage.unassessed_vitals)

	def test_complete_non_triggering_vitals_do_not_claim_low_risk(self):
		patient = ClinicalAnalysisRequest(age=47, symptoms=['unwell'], vitals={
			'heart_rate': 75, 'respiratory_rate': 16, 'spo2': 98,
			'blood_pressure': '120/80', 'temperature': 37,
		})
		triage = assess_vital_triage(patient)
		self.assertEqual(triage.level, 'no_threshold_triggered')
		self.assertFalse(triage.urgent_care_recommended)
		self.assertIn('does not establish low risk', triage.recommendation)

	def test_adult_rules_are_not_applied_to_children_unknown_age_or_pregnancy(self):
		for eligibility in ({}, {'age': 10}, {'age': 30, 'is_pregnant': True}):
			with self.subTest(eligibility=eligibility):
				triage = assess_vital_triage(ClinicalAnalysisRequest(**eligibility, symptoms=['unwell'], vitals={'heart_rate': 140}))
				self.assertEqual(triage.level, 'not_assessed')
				self.assertIn('adult eligibility', triage.unassessed_vitals)


class ClinicalSchemaTests(SimpleTestCase):
	def test_symptoms_are_required_strict_and_nonblank(self):
		for symptoms in (None, [], '', 'fever', [1], [True], [''], ['   '], ['fever'] * 65):
			with self.subTest(symptoms=symptoms), self.assertRaises(ValidationError):
				ClinicalAnalysisRequest.model_validate({'symptoms': symptoms})
		with self.assertRaises(ValidationError):
			ClinicalAnalysisRequest.model_validate({})

	def test_new_and_legacy_aliases_normalize_to_same_data(self):
		new = ClinicalAnalysisRequest(symptoms=['fever'], labs=['CBC'], radiology=['X-ray'])
		legacy = ClinicalAnalysisRequest(
			symptoms=['fever'], laboratory_results=['CBC'], radiology_results=['X-ray'],
		)
		self.assertEqual(new, legacy)
		self.assertEqual(new.model_dump(by_alias=True)['labs'], ['CBC'])
		for canonical, old in (('labs', 'laboratory_results'), ('radiology', 'radiology_results')):
			with self.subTest(field=canonical), self.assertRaises(ValidationError):
				ClinicalAnalysisRequest.model_validate({'symptoms': ['fever'], canonical: [], old: []})

	def test_inputs_are_bounded_and_identifying_fields_rejected(self):
		nested = {'value': 'finding'}
		for depth in range(8):
			nested = {'nested': nested}
		for extra in (
			{'labs': [{'patient_name': 'private'}]},
			{'radiology': [{'details': {'mrn': 'private'}}]},
			{'vitals': {'notes': 'x' * 2049}},
			{'labs': [nested]},
			{'vitals': {'temperature': float('inf')}},
			{'age': '47'},
			{'unexpected': 'value'},
		):
			with self.subTest(extra=extra), self.assertRaises(ValidationError):
				ClinicalAnalysisRequest.model_validate({'symptoms': ['fever'], **extra})

	def test_response_contract_rejects_missing_extra_or_wrong_types(self):
		for payload in (
			{}, {**VALID_RESPONSE, 'extra': 'value'},
			{**VALID_RESPONSE, 'supported_diagnosis': 1},
			{**VALID_RESPONSE, 'drugs_admissible': 'drug'},
			{**VALID_RESPONSE, 'possible_disease': [True]},
		):
			with self.subTest(payload=payload), self.assertRaises(ValidationError):
				ClinicalDecisionSchema(**payload)


class LocalOrchestrationTests(SimpleTestCase):
	def test_real_mcp_dispatch_uses_request_thread_without_network_or_subprocess(self):
		patient = ClinicalAnalysisRequest(symptoms=['fever'], patient_id='private')
		thread_id = threading.get_ident()
		thread_ids = []
		original_dump = ClinicalAnalysisRequest.model_dump

		def record_dump(instance, *args, **kwargs):
			thread_ids.append(threading.get_ident())
			return original_dump(instance, *args, **kwargs)

		with (
			patch('socket.socket.connect', side_effect=AssertionError('MCP networking is prohibited')),
			patch('socket.socket.connect_ex', side_effect=AssertionError('MCP networking is prohibited')),
			patch('subprocess.Popen', side_effect=AssertionError('MCP subprocesses are prohibited')),
			patch.object(ClinicalAnalysisRequest, 'model_dump', autospec=True, side_effect=record_dump),
			patch('clinicalDecisionAI.views.run_clinical_analysis', return_value=ClinicalDecisionSchema(**VALID_RESPONSE)) as provider,
		):
			result = _analyze_patient(patient, 'test-model')
		self.assertEqual(result.model_dump(exclude={'triage'}), VALID_RESPONSE)
		self.assertEqual(result.triage.level, 'not_assessed')
		self.assertIsNone(provider.call_args.args[0].patient_id)
		self.assertTrue(thread_ids)
		self.assertEqual(set(thread_ids), {thread_id})

	def test_concurrent_requests_and_repeated_event_loops_remain_isolated(self):
		async def concurrent_contexts():
			return await asyncio.gather(
				_local_context(ClinicalAnalysisRequest(symptoms=['first'], labs=['first lab'])),
				_local_context(ClinicalAnalysisRequest(symptoms=['second'], labs=['second lab'])),
			)

		for attempt in range(2):
			first, second = asyncio.run(concurrent_contexts())
			self.assertEqual(first.symptoms, ['first'])
			self.assertEqual(second.symptoms, ['second'])
			self.assertEqual(first.laboratory_results, ['first lab'])
			self.assertEqual(second.laboratory_results, ['second lab'])

	def test_tool_failure_and_timeout_do_not_reach_provider(self):
		for failure in (ToolError('private tool detail'), TimeoutError('private timeout detail')):
			with (
				self.subTest(failure=type(failure).__name__),
				patch('clinicalDecisionAI.views._local_context', side_effect=failure),
				patch('clinicalDecisionAI.views.run_clinical_analysis') as provider,
				self.assertRaises(ClinicalResponseError),
			):
				_analyze_patient(ClinicalAnalysisRequest(symptoms=['fever']), 'test-model')
			provider.assert_not_called()

	@override_settings(CLINICAL_MCP_TIMEOUT_SECONDS=0.2)
	def test_real_tool_deadline_is_enforced(self):
		server = FastMCP('BlockedContext', mask_error_details=True)

		@server.tool
		async def prepare_clinical_context(patient: ClinicalAnalysisRequest) -> dict:
			await asyncio.Event().wait()
			return {}

		with (
			patch('clinicalDecisionAI.views._create_clinical_mcp', return_value=server),
			patch('clinicalDecisionAI.views.run_clinical_analysis') as provider,
			self.assertRaises(ClinicalResponseError) as error,
		):
			_analyze_patient(ClinicalAnalysisRequest(symptoms=['fever']), 'test-model')
		self.assertEqual(error.exception.category, 'tool_timeout')
		provider.assert_not_called()

	def test_simultaneous_view_threads_do_not_share_patient_data(self):
		def provider(patient, model):
			return ClinicalDecisionSchema(**{**VALID_RESPONSE, 'supported_diagnosis': patient.symptoms[0]})

		with patch('clinicalDecisionAI.views.run_clinical_analysis', side_effect=provider):
			with ThreadPoolExecutor(max_workers=2) as executor:
				results = list(executor.map(
					lambda symptom: _analyze_patient(ClinicalAnalysisRequest(symptoms=[symptom]), 'test-model'),
					['first patient', 'second patient'],
				))
		self.assertEqual([result.supported_diagnosis for result in results], ['first patient', 'second patient'])


@override_settings(GROQ_MODELS=('openai/gpt-oss-120b', 'openai/gpt-oss-20b'))
class GroqServiceTests(TestCase):
	def test_environment_model_lists_are_split_before_selection(self):
		models = ('openai/gpt-oss-120b', 'openai/gpt-oss-20b')
		for primary, legacy, expected in (
			('', ', openai/gpt-oss-120b, openai/gpt-oss-20b, openai/gpt-oss-120b, ', models),
			('openai/gpt-oss-120b,openai/gpt-oss-20b', 'ignored-model', models),
			(' , ', ' , ', ('openai/gpt-oss-120b',)),
			('', 'openai/gpt-oss-20b', ('openai/gpt-oss-20b',)),
		):
			with self.subTest(primary=primary, legacy=legacy):
				with patch.dict(os.environ, {'GROQ_MODELS': primary, 'GROQ_MODEL': legacy}):
					configuration = runpy.run_path(project_settings.__file__)
				self.assertEqual(configuration['GROQ_MODELS'], expected)
				self.assertEqual(configuration['GROQ_MODEL'], expected[0])
				with override_settings(GROQ_MODELS=configuration['GROQ_MODELS']):
					for attempt in range(5):
						selected = select_groq_model()
						self.assertIn(selected, expected)
						self.assertNotIn(',', selected)

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

		self.assertEqual(result.supported_diagnosis, VALID_RESPONSE['supported_diagnosis'])
		groq_factory.assert_called_once_with(
			api_key=MOCK_API_KEY, timeout=30.0, max_retries=0
		)
		call = groq.chat.completions.create.call_args.kwargs
		self.assertIn(call['model'], ('openai/gpt-oss-120b', 'openai/gpt-oss-20b'))
		self.assertEqual(call['response_format'], {'type': 'json_object'})
		self.assertNotIn('318', call['messages'][1]['content'])
		self.assertTrue(call['messages'][0]['content'].find('not a doctor') >= 0)
		self.assertIn('supported_diagnosis', call['messages'][0]['content'])
		self.assertIn('untrusted data', call['messages'][0]['content'])

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
		invalid_response = {**VALID_RESPONSE, 'possible_disease': [1]}
		groq_factory.return_value.chat.completions.create.return_value = provider_response(
			json.dumps(invalid_response)
		)

		with self.assertRaises(ClinicalResponseError) as context:
			run_clinical_analysis(ClinicalAnalysisRequest(symptoms=['fever']))

		self.assertEqual(context.exception.category, 'schema_validation')

	@override_settings(GROQ_API_KEY=MOCK_API_KEY)
	@patch('clinicalDecisionAI.services.Groq')
	def test_non_object_json_and_extra_output_fields_are_rejected(self, groq_factory):
		for output in ([], None, 'string', {**VALID_RESPONSE, 'extra': 'private'}):
			groq_factory.return_value.chat.completions.create.return_value = provider_response(json.dumps(output))
			with self.subTest(output=output), self.assertRaises(ClinicalResponseError) as error:
				run_clinical_analysis(ClinicalAnalysisRequest(symptoms=['fever']))
			self.assertEqual(error.exception.category, 'schema_validation')


@override_settings(GROQ_MODELS=('openai/gpt-oss-120b', 'openai/gpt-oss-20b'))
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
		result = ClinicalDecisionSchema.model_validate(VALID_RESPONSE)
		with patch('clinicalDecisionAI.views.run_clinical_analysis', return_value=result):
			response = self.client.post(
				'/api/clinical-ai/analyze/',
				data=json.dumps({'symptoms': ['fever']}),
				content_type='application/json',
				HTTP_X_REQUEST_ID='12345678-1234-4234-8234-123456789abc',
				**self.auth_headers,
			)

		self.assertEqual(response.status_code, 200)
		self.assertEqual({key: value for key, value in response.json().items() if key != 'triage'}, VALID_RESPONSE)
		self.assertEqual(response.json()['triage']['level'], 'not_assessed')
		self.assertEqual(response['X-Request-ID'], '12345678-1234-4234-8234-123456789abc')
		self.assertEqual(response['Cache-Control'], 'no-store')
		self.assertEqual(response['X-Clinical-Review-Required'], 'true')
		self.assertIn(
			response['X-Groq-Model'],
			('openai/gpt-oss-120b', 'openai/gpt-oss-20b'),
		)

	def test_bearer_authenticated_request_does_not_require_cookie_csrf(self):
		client = Client(enforce_csrf_checks=True)
		result = ClinicalDecisionSchema.model_validate(VALID_RESPONSE)
		with patch('clinicalDecisionAI.views.run_clinical_analysis', return_value=result):
			response = client.post(
				'/api/clinical-ai/analyze/',
				data=json.dumps({'symptoms': ['fever']}),
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
				data=json.dumps({'symptoms': ['fever']}),
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
				data=json.dumps({'symptoms': ['fever']}),
				content_type='application/json',
				**self.auth_headers,
			)

		self.assertEqual(response.status_code, 503)
		self.assertEqual(
			response.json()['error']['code'], 'ai_provider_unavailable'
		)

	def test_invalid_inputs_fail_before_local_tools_and_provider(self):
		for body in ('{}', '[]', 'null', '{invalid', '{"symptoms":[]}', '{"symptoms":[" "]}'):
			with self.subTest(body=body), patch('clinicalDecisionAI.views._analyze_patient') as analyze:
				response = self.client.post(
					'/api/clinical-ai/analyze/', data=body,
					content_type='application/json', **self.auth_headers,
				)
			self.assertEqual(response.status_code, 400)
			self.assertEqual(response.json()['error']['code'], 'invalid_request')
			self.assertEqual(response['Cache-Control'], 'no-store')
			analyze.assert_not_called()

	def test_json_charset_and_new_fields_are_supported(self):
		with patch('clinicalDecisionAI.views.run_clinical_analysis', return_value=ClinicalDecisionSchema(**VALID_RESPONSE)) as analyze:
			response = self.client.post(
				'/api/clinical-ai/analyze/',
				data=json.dumps({'symptoms': ['fever'], 'labs': ['CBC'], 'radiology': ['X-ray']}),
				content_type='application/json; charset=utf-8', **self.auth_headers,
			)
		self.assertEqual(response.status_code, 200)
		self.assertEqual(analyze.call_args.args[0].laboratory_results, ['CBC'])
		self.assertEqual(analyze.call_args.args[0].radiology_results, ['X-ray'])

	def test_non_json_and_oversized_requests_are_rejected(self):
		with patch('clinicalDecisionAI.views._analyze_patient') as analyze:
			response = self.client.post('/api/clinical-ai/analyze/', data='text', content_type='text/plain', **self.auth_headers)
			self.assertEqual(response.status_code, 415)
			with override_settings(CLINICAL_MAX_REQUEST_BYTES=16):
				response = self.client.post(
					'/api/clinical-ai/analyze/', data=json.dumps({'symptoms': ['fever']}),
					content_type='application/json', **self.auth_headers,
				)
			self.assertEqual(response.status_code, 413)
			analyze.assert_not_called()

	def test_cookie_login_does_not_authorize_api(self):
		user = get_user_model().objects.create_user(username='cookie-only')
		self.client.force_login(user)
		response = self.client.post('/api/clinical-ai/analyze/', data='{}', content_type='application/json')
		self.assertEqual(response.status_code, 401)
		self.assertEqual(response['WWW-Authenticate'], 'Bearer')

	def test_tool_and_output_errors_are_safe_bad_gateway_responses(self):
		for failure in (ClinicalResponseError('tool_error'), ClinicalResponseError('schema_validation')):
			with self.subTest(category=failure.category), patch('clinicalDecisionAI.views._analyze_patient', side_effect=failure):
				response = self.client.post(
					'/api/clinical-ai/analyze/', data=json.dumps({'symptoms': ['fever']}),
					content_type='application/json', **self.auth_headers,
				)
			self.assertEqual(response.status_code, 502)
			self.assertEqual(response.json()['error']['code'], 'ai_invalid_response')

	def test_internal_errors_do_not_expose_exception_or_header_data(self):
		with patch('clinicalDecisionAI.views._analyze_patient', side_effect=RuntimeError('private patient detail')):
			response = self.client.post(
				'/api/clinical-ai/analyze/', data=json.dumps({'symptoms': ['fever']}),
				content_type='application/json', HTTP_X_REQUEST_ID='private patient detail', **self.auth_headers,
			)
		self.assertEqual(response.status_code, 500)
		self.assertNotIn('private patient detail', response.content.decode())
		self.assertNotEqual(response['X-Request-ID'], 'private patient detail')

	def test_urgent_triage_is_computed_independently_of_the_ai_response(self):
		with patch('clinicalDecisionAI.views.run_clinical_analysis', return_value=ClinicalDecisionSchema(**VALID_RESPONSE)):
			response = self.client.post(
				'/api/clinical-ai/analyze/',
				data=json.dumps({'age': 47, 'symptoms': ['unwell'], 'vitals': {'spo2': 80}}),
				content_type='application/json', **self.auth_headers,
			)
		self.assertEqual(response.status_code, 200)
		self.assertEqual(response.json()['triage']['level'], 'urgent')
		self.assertTrue(response.json()['triage']['urgent_care_recommended'])
		self.assertIn('oxygen_saturation', response.json()['triage']['reasons'][0])

	def test_urgent_triage_survives_provider_or_application_failure(self):
		for failure, expected_status in (
			(GroqConfigurationError(), 503), (GroqProviderError('timeout'), 504),
			(ClinicalResponseError('schema_validation'), 502), (RuntimeError('private'), 500),
		):
			with self.subTest(status=expected_status), patch('clinicalDecisionAI.views._analyze_patient', side_effect=failure):
				response = self.client.post(
					'/api/clinical-ai/analyze/',
					data=json.dumps({'age': 47, 'symptoms': ['unwell'], 'vitals': {'heart_rate': 180}}),
					content_type='application/json', **self.auth_headers,
				)
			self.assertEqual(response.status_code, expected_status)
			self.assertEqual(response.json()['triage']['level'], 'urgent')
			self.assertTrue(response.json()['triage']['requires_human_review'])

	def test_model_cannot_inject_its_own_triage(self):
		with override_settings(GROQ_API_KEY=MOCK_API_KEY), patch('clinicalDecisionAI.services.Groq') as factory:
			factory.return_value.chat.completions.create.return_value = provider_response(json.dumps({
				**VALID_RESPONSE, 'triage': {'level': 'routine'},
			}))
			response = self.client.post(
				'/api/clinical-ai/analyze/',
				data=json.dumps({'age': 47, 'symptoms': ['unwell'], 'vitals': {'blood_pressure': '200/130'}}),
				content_type='application/json', **self.auth_headers,
			)
		self.assertEqual(response.status_code, 502)
		self.assertEqual(response.json()['triage']['level'], 'urgent')


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
		analyze.return_value = ClinicalDecisionSchema.model_validate(VALID_RESPONSE)

		response = self.client.post(
			'/sandbox/',
			{'patient_json': json.dumps({'symptoms': ['fever']})},
		)

		self.assertEqual(response.status_code, 200)
		self.assertEqual(response['Cache-Control'], 'no-store')
		self.assertContains(response, 'AI Clinical Decision Support')
		self.assertContains(response, 'Human clinician review required: Yes')
		self.assertContains(response, 'Selected Groq model:')
		self.assertContains(response, VALID_RESPONSE['supported_diagnosis'])
		self.assertContains(response, VALID_RESPONSE['possible_disease'][0])
		self.assertContains(response, VALID_RESPONSE['further_labs_to_be_done'][0])
		self.assertContains(response, 'Medication considerations')
		self.assertNotContains(response, 'Confidence:')
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

	@patch('clinicalDecisionAI.views.run_clinical_analysis', side_effect=GroqProviderError('timeout'))
	def test_sandbox_shows_urgent_triage_even_when_ai_fails(self, analyze):
		user = get_user_model().objects.create_user(username='urgent-sandbox-user')
		self.client.force_login(user)
		response = self.client.post('/sandbox/', {'patient_json': json.dumps({
			'age': 47, 'symptoms': ['unwell'], 'vitals': {'spo2': 80},
		})})
		self.assertEqual(response.status_code, 504)
		self.assertContains(response, 'Urgent clinical assessment recommended now', status_code=504)
		self.assertContains(response, 'oxygen_saturation: 80 %', status_code=504)
		self.assertContains(response, 'do not delay care', status_code=504)

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
