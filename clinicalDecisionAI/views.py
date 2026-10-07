import json
import logging
import secrets
import time
from datetime import datetime, timezone
from uuid import uuid4

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST
from pydantic import ValidationError

from clinicalDecisionAI.authentication import (
	authenticate_integration_request,
	hash_integration_token,
)
from clinicalDecisionAI.forms import AccountCreationForm, IntegrationRegistrationForm
from clinicalDecisionAI.models import IntegrationApplication
from clinicalDecisionAI.schemas import ClinicalAnalysisRequest
from clinicalDecisionAI.services import (
	ClinicalResponseError,
	GroqConfigurationError,
	GroqProviderError,
	run_clinical_analysis,
	select_groq_model,
)

logger = logging.getLogger(__name__)


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
		'laboratory_results': [],
		'radiology_results': [],
		'previous_diagnoses': [],
		'observations': [],
	}
	patient_json = json.dumps(demo_request, indent=2)
	result = None
	selected_model = None
	error = None
	error_details = []
	status = 200
	request_id = request.headers.get('X-Request-ID') or str(uuid4())

	if request.method == 'POST':
		patient_json = request.POST.get('patient_json', '')
		try:
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
		else:
			started = time.monotonic()
			request_timestamp = datetime.now(timezone.utc).isoformat()
			selected_model = select_groq_model()
			try:
				result = run_clinical_analysis(patient, model=selected_model)
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
			except ClinicalResponseError as exc:
				category = exc.category
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


def _error_response(code: str, message: str, status: int) -> JsonResponse:
	return JsonResponse({'error': {'code': code, 'message': message}}, status=status)


def _validation_details(error: ValidationError) -> list[dict[str, str]]:
	return [
		{
			'field': '.'.join(str(part) for part in item['loc']),
			'message': item['msg'],
		}
		for item in error.errors(include_input=False)
	]


@csrf_exempt
@require_http_methods(['GET', 'POST'])
def analyze_clinical_case(request):
	request_id = request.headers.get('X-Request-ID') or str(uuid4())
	if request.method == 'GET':
		response = _error_response(
			'method_not_allowed',
			'This API accepts POST requests. Send JSON with a registered application bearer token.',
			405,
		)
		response['Allow'] = 'POST'
		response['X-Request-ID'] = request_id
		return response

	application = authenticate_integration_request(request)
	if application is None:
		response = _error_response(
			'authentication_required', 'A registered application bearer token is required.', 401
		)
		response['WWW-Authenticate'] = 'Bearer'
		response['X-Request-ID'] = request_id
		return response

	if request.content_type != 'application/json':
		response = _error_response(
			'invalid_content_type', 'Content-Type must be application/json.', 415
		)
		response['X-Request-ID'] = request_id
		return response

	try:
		body = json.loads(request.body)
		if not isinstance(body, dict):
			raise ValueError
		patient = ClinicalAnalysisRequest.model_validate(body)
	except ValidationError as exc:
		response = JsonResponse(
			{
				'error': {
					'code': 'invalid_request',
					'message': 'Patient information did not match the expected schema.',
					'details': _validation_details(exc),
				}
			},
			status=400,
		)
		response['X-Request-ID'] = request_id
		return response
	except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
		response = _error_response(
			'invalid_request', 'Request body must be a valid JSON object.', 400
		)
		response['X-Request-ID'] = request_id
		return response

	started = time.monotonic()
	request_timestamp = datetime.now(timezone.utc).isoformat()
	selected_model = select_groq_model()
	try:
		result = run_clinical_analysis(patient, model=selected_model)
	except GroqConfigurationError:
		status, code, message, category = (
			503,
			'ai_provider_unavailable',
			'Clinical AI is not configured.',
			'missing_configuration',
		)
	except GroqProviderError as exc:
		category = exc.category
		status, code, message = {
			'authentication': (502, 'ai_provider_error', 'Clinical AI provider authentication failed.'),
			'rate_limit': (429, 'ai_rate_limited', 'Clinical AI is temporarily rate limited.'),
			'timeout': (504, 'ai_timeout', 'Clinical AI did not respond in time.'),
			'network': (502, 'ai_provider_unavailable', 'Clinical AI provider is unavailable.'),
			'provider': (502, 'ai_provider_error', 'Clinical AI provider request failed.'),
		}.get(category, (502, 'ai_provider_error', 'Clinical AI provider request failed.'))
	except ClinicalResponseError as exc:
		category = exc.category
		status, code, message = (
			502,
			'ai_invalid_response',
			'Clinical AI returned a response that could not be validated.',
		)
	except Exception:
		category = 'internal_error'
		status, code, message = (
			502,
			'ai_analysis_failed',
			'Clinical AI analysis could not be completed.',
		)
	else:
		logger.info(
			'Clinical AI request completed',
			extra={
				'request_id': request_id,
				'integration_application_id': application.pk,
				'model': selected_model,
				'request_timestamp': request_timestamp,
				'response_timestamp': datetime.now(timezone.utc).isoformat(),
				'latency_ms': round((time.monotonic() - started) * 1000),
				'success': True,
			},
		)
		response = JsonResponse(result.model_dump(mode='json'))
		response['X-Request-ID'] = request_id
		response['X-Groq-Model'] = selected_model
		return response

	logger.warning(
		'Clinical AI request failed',
		extra={
			'request_id': request_id,
			'integration_application_id': application.pk,
			'model': selected_model,
			'request_timestamp': request_timestamp,
			'response_timestamp': datetime.now(timezone.utc).isoformat(),
			'latency_ms': round((time.monotonic() - started) * 1000),
			'success': False,
			'error_category': category,
		},
	)
	response = _error_response(code, message, status)
	response['X-Request-ID'] = request_id
	response['X-Groq-Model'] = selected_model
	return response
