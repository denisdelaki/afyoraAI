import asyncio
import json
import logging
import secrets
import time
from datetime import datetime, timezone
from uuid import UUID, uuid4

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.core.exceptions import RequestDataTooBig
from django.db import connection, DatabaseError
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods, require_POST, require_safe
from fastmcp import Client as MCPClient, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import ValidationError
from rest_framework import exceptions, status
from rest_framework.parsers import JSONParser
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView

from clinicalDecisionAI.authentication import (
	IntegrationApplicationPermission,
	IntegrationTokenAuthentication,
	hash_integration_token,
)
from clinicalDecisionAI.forms import AccountCreationForm, IntegrationRegistrationForm
from clinicalDecisionAI.models import IntegrationApplication
from clinicalDecisionAI.schemas import ClinicalAnalysisRequest, ClinicalAnalysisResponse
from clinicalDecisionAI.services import (
	ClinicalResponseError,
	GroqConfigurationError,
	GroqProviderError,
	run_clinical_analysis,
	select_groq_model,
	assess_vital_triage,
)

logger = logging.getLogger(__name__)


@require_safe
def health_check(request):
	try:
		with connection.cursor() as cursor:
			cursor.execute('SELECT 1')
			cursor.fetchone()
	except DatabaseError:
		payload = {'status': 'unhealthy', 'checks': {'database': 'unavailable'}}
		response_status = 503
	else:
		payload = {'status': 'healthy', 'checks': {'database': 'ok'}}
		response_status = 200
	response = JsonResponse(payload, status=response_status)
	response['Cache-Control'] = 'no-store'
	return response


def _request_id(request):
	try:
		return str(UUID(request.headers.get('X-Request-ID', '')))
	except (ValueError, AttributeError):
		return str(uuid4())


async def prepare_clinical_context(patient: ClinicalAnalysisRequest) -> dict:
	return patient.model_dump(mode='json', by_alias=True, exclude={'patient_id'}, exclude_none=True)


def _create_clinical_mcp():
	server = FastMCP('ClinicalContext', mask_error_details=True)
	server.tool(prepare_clinical_context)
	return server


async def _local_context(patient):
	async with asyncio.timeout(settings.CLINICAL_MCP_TIMEOUT_SECONDS):
		async with MCPClient(_create_clinical_mcp()) as client:
			result = await client.call_tool(
				'prepare_clinical_context',
				{'patient': patient.model_dump(mode='json', by_alias=True, exclude={'patient_id'})},
				timeout=settings.CLINICAL_MCP_TIMEOUT_SECONDS,
			)
			return ClinicalAnalysisRequest.model_validate(result.data)


def _analyze_patient(patient, model):
	try:
		context = asyncio.run(_local_context(patient))
	except TimeoutError as exc:
		raise ClinicalResponseError('tool_timeout') from exc
	except (ToolError, ValidationError) as exc:
		raise ClinicalResponseError('tool_error') from exc
	decision = run_clinical_analysis(context, model=model)
	return ClinicalAnalysisResponse(
		**decision.model_dump(mode='json'), triage=assess_vital_triage(patient),
	)


def application_home(request):
	return redirect('integration-dashboard' if request.user.is_authenticated else 'account-signup')


@require_http_methods(['GET', 'POST'])
def account_signup(request):
	if request.user.is_authenticated:
		return redirect('integration-dashboard')

	form = AccountCreationForm(request.POST or None)
	if request.method == 'POST' and form.is_valid():
		user = form.save()
		login(request, user)
		return redirect('integration-dashboard')
	return render(request, 'registration/signup.html', {'form': form})


@login_required
@require_http_methods(['GET', 'POST'])
def integration_dashboard(request):
	form = IntegrationRegistrationForm(request.POST or None)
	issued_token = None

	if request.method == 'POST' and form.is_valid():
		issued_token = f'cdai_{secrets.token_urlsafe(32)}'
		integration = form.save(commit=False)
		integration.owner = request.user
		integration.token_hash = hash_integration_token(issued_token)
		integration.save()
		form = IntegrationRegistrationForm()

	response = render(
		request,
		'integrations/dashboard.html',
		{
			'form': form,
			'integrations': IntegrationApplication.objects.filter(
				owner=request.user
			).order_by('-created_at'),
			'issued_token': issued_token,
		},
		status=201 if issued_token else 200,
	)
	if issued_token:
		response['Cache-Control'] = 'no-store'
	return response


@login_required
def integration_guide(request):
	return render(request, 'integrations/guide.html')


@login_required
@require_http_methods(['GET', 'POST'])
def clinical_sandbox(request):
	demo_request = {
		'age': 47,
		'sex': 'female',
		'symptoms': ['fever', 'headache'],
		'medical_history': [],
		'allergies': [],
		'current_medications': [],
		'vitals': {'temperature': 38.5, 'heart_rate': 104},
		'labs': [],
		'radiology': [],
		'previous_diagnoses': [],
		'observations': [],
	}
	patient_json = json.dumps(demo_request, indent=2)
	result = None
	triage = None
	selected_model = None
	error = None
	error_details = []
	status = 200
	request_id = _request_id(request)

	if request.method == 'POST':
		patient_json = request.POST.get('patient_json', '')
		try:
			if len(patient_json.encode('utf-8')) > settings.CLINICAL_MAX_REQUEST_BYTES:
				raise ValueError('Clinical input exceeds the request size limit.')
			patient = ClinicalAnalysisRequest.model_validate_json(patient_json)
		except ValidationError as exc:
			error = 'Check the JSON input and correct the highlighted validation issues.'
			error_details = [
				f"{'.'.join(str(part) for part in item['loc']) or 'request'}: {item['msg']}"
				for item in exc.errors(include_input=False)
			]
			status = 400
			logger.warning(
				'Clinical AI sandbox input rejected',
				extra={
					'request_id': request_id,
					'user_id': request.user.pk,
					'error_category': 'request_validation',
				},
			)
		except (ValueError, RecursionError):
			error = 'Clinical input is too large or too deeply nested.'
			status = 400
		else:
			triage = assess_vital_triage(patient)
			started = time.monotonic()
			request_timestamp = datetime.now(timezone.utc).isoformat()
			selected_model = select_groq_model()
			try:
				result = _analyze_patient(patient, model=selected_model)
			except GroqConfigurationError:
				error = 'The AI provider is not configured. Contact the service administrator.'
				status, category = 503, 'missing_configuration'
			except GroqProviderError as exc:
				category = exc.category
				status, error = {
					'authentication': (502, 'The AI provider rejected its configured credentials.'),
					'rate_limit': (429, 'The AI provider is busy. Wait briefly before retrying.'),
					'timeout': (504, 'The AI provider did not respond in time.'),
					'network': (502, 'The AI provider is temporarily unreachable.'),
					'provider': (502, 'The AI provider request failed.'),
				}.get(category, (502, 'The AI provider request failed.'))
				if category == 'provider' and exc.status_code:
					error = f'The AI provider rejected the request (HTTP {exc.status_code}).'
					if exc.status_code == 400:
						error += ' Check the configured model and request options.'
					elif exc.status_code == 401:
						error += ' Check the server-side Groq API key.'
					elif exc.status_code == 403:
						error += ' Check account permissions and model access.'
					elif exc.status_code == 404:
						error += ' Check that the configured model is available.'
					elif exc.status_code >= 500:
						error += ' The provider may be temporarily unavailable.'
			except (ClinicalResponseError, ValidationError) as exc:
				category = getattr(exc, 'category', 'schema_validation')
				status = 502
				error = 'The AI response could not be validated. No result was returned.'
			except Exception:
				category = 'internal_error'
				status = 502
				error = 'The sandbox request could not be completed.'
			else:
				logger.info(
					'Clinical AI sandbox request completed',
					extra={
						'request_id': request_id,
						'user_id': request.user.pk,
						'model': selected_model,
						'request_timestamp': request_timestamp,
						'response_timestamp': datetime.now(timezone.utc).isoformat(),
						'latency_ms': round((time.monotonic() - started) * 1000),
						'success': True,
					},
				)
			if error:
				logger.warning(
					'Clinical AI sandbox request failed',
					extra={
						'request_id': request_id,
						'user_id': request.user.pk,
						'model': selected_model,
						'request_timestamp': request_timestamp,
						'response_timestamp': datetime.now(timezone.utc).isoformat(),
						'latency_ms': round((time.monotonic() - started) * 1000),
						'success': False,
						'error_category': category,
					},
				)

	response = render(
		request,
		'integrations/sandbox.html',
		{
			'patient_json': patient_json,
			'result': result,
			'triage': triage,
			'error': error,
			'error_details': error_details,
			'request_id': request_id,
			'selected_model': selected_model,
		},
		status=status,
	)
	response['Cache-Control'] = 'no-store'
	response['X-Request-ID'] = request_id
	return response


@login_required
@require_POST
def deactivate_integration(request, integration_id):
	integration = get_object_or_404(
		IntegrationApplication,
		pk=integration_id,
		owner=request.user,
	)
	integration.is_active = False
	integration.save(update_fields=['is_active'])
	messages.success(request, f'{integration.name} has been deactivated.')
	return redirect('integration-dashboard')


def _validation_details(error: ValidationError) -> list[dict[str, str]]:
	return [
		{
			'field': '.'.join(str(part) for part in item['loc']),
			'message': item['msg'],
		}
		for item in error.errors(include_input=False, include_context=False, include_url=False)
	]


class ClinicalAnalysisAPIView(APIView):
	authentication_classes = [IntegrationTokenAuthentication]
	permission_classes = [IntegrationApplicationPermission]
	parser_classes = [JSONParser]
	renderer_classes = [JSONRenderer]
	http_method_names = ['post']

	def initial(self, request, *args, **kwargs):
		self.request_id = _request_id(request)
		if request.method != 'POST':
			raise exceptions.MethodNotAllowed(request.method)
		super().initial(request, *args, **kwargs)

	def finalize_response(self, request, response, *args, **kwargs):
		response = super().finalize_response(request, response, *args, **kwargs)
		response['X-Request-ID'] = self.request_id
		response['Cache-Control'] = 'no-store'
		if response.status_code >= 500 and hasattr(self, 'triage'):
			response.data['triage'] = self.triage.model_dump(mode='json')
		if hasattr(self, 'selected_model'):
			response['X-Groq-Model'] = self.selected_model
		if response.status_code == 200:
			response['X-Clinical-Review-Required'] = 'true'
		return response

	def handle_exception(self, exc):
		if isinstance(exc, exceptions.APIException):
			response = super().handle_exception(exc)
			code, message = {
				401: ('authentication_required', 'A registered application bearer token is required.'),
				403: ('permission_denied', 'Application access is not permitted.'),
				405: ('method_not_allowed', 'This API accepts POST requests with a registered application bearer token.'),
				406: ('not_acceptable', 'This API returns application/json.'),
				415: ('invalid_content_type', 'Content-Type must be application/json.'),
			}.get(response.status_code, ('invalid_request', 'Request body must be a valid JSON object.'))
			response.data = {'error': {'code': code, 'message': message}}
			return response
		if isinstance(exc, RequestDataTooBig):
			return self.error('request_too_large', 'Request body exceeds the size limit.', 413)
		if isinstance(exc, RecursionError):
			return self.error('invalid_request', 'Request body is too deeply nested.', 400)
		logger.error('Clinical AI application error', extra={'request_id': self.request_id, 'error_category': 'internal_error'})
		return self.error('internal_error', 'Clinical AI analysis could not be completed.', 500)

	@staticmethod
	def error(code, message, http_status):
		return Response({'error': {'code': code, 'message': message}}, status=http_status)

	def post(self, request):
		if request.content_type.split(';', 1)[0].strip().lower() != 'application/json':
			raise exceptions.UnsupportedMediaType(request.content_type)
		if len(request.body) > settings.CLINICAL_MAX_REQUEST_BYTES:
			raise RequestDataTooBig
		try:
			patient = ClinicalAnalysisRequest.model_validate(request.data)
		except ValidationError as exc:
			return Response({'error': {
				'code': 'invalid_request',
				'message': 'Patient information did not match the expected schema.',
				'details': _validation_details(exc),
			}}, status=status.HTTP_400_BAD_REQUEST)

		started = time.monotonic()
		triage = assess_vital_triage(patient)
		self.triage = triage
		request_timestamp = datetime.now(timezone.utc).isoformat()
		self.selected_model = select_groq_model()
		category = None
		try:
			result = _analyze_patient(patient, model=self.selected_model)
			result = ClinicalAnalysisResponse(**result.model_dump(mode='json'))
		except GroqConfigurationError:
			category = 'missing_configuration'
			response = self.error('ai_provider_unavailable', 'Clinical AI is not configured.', 503)
		except GroqProviderError as exc:
			category = exc.category
			http_status, code, message = {
				'authentication': (502, 'ai_provider_error', 'Clinical AI provider authentication failed.'),
				'rate_limit': (429, 'ai_rate_limited', 'Clinical AI is temporarily rate limited.'),
				'timeout': (504, 'ai_timeout', 'Clinical AI did not respond in time.'),
				'network': (502, 'ai_provider_unavailable', 'Clinical AI provider is unavailable.'),
			}.get(category, (502, 'ai_provider_error', 'Clinical AI provider request failed.'))
			response = self.error(code, message, http_status)
		except (ClinicalResponseError, ValidationError) as exc:
			category = getattr(exc, 'category', 'schema_validation')
			response = self.error('ai_invalid_response', 'Clinical AI returned a response that could not be validated.', 502)
		else:
			response = Response(result.model_dump(mode='json'))
		if category:
			response.data['triage'] = triage.model_dump(mode='json')
		logger.log(
			logging.WARNING if category else logging.INFO,
			'Clinical AI request failed' if category else 'Clinical AI request completed',
			extra={
				'request_id': self.request_id,
				'integration_application_id': request.auth.pk,
				'model': self.selected_model,
				'request_timestamp': request_timestamp,
				'response_timestamp': datetime.now(timezone.utc).isoformat(),
				'latency_ms': round((time.monotonic() - started) * 1000),
				'success': category is None,
				'error_category': category,
			},
		)
		return response
