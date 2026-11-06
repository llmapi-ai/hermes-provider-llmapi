"""Gateway error codes -> Hermes FailoverReasons."""

from __future__ import annotations

import pytest

BILLING = {"reason": "billing", "retryable": False, "should_fallback": True}
NOT_ENTITLED = {"reason": "model_entitlement", "retryable": False, "should_fallback": True}
BLOCKED = {"reason": "auth_permanent", "retryable": False, "should_fallback": False}


@pytest.mark.parametrize(
    ("status", "code", "expected"),
    [
        (402, "insufficient_credits", BILLING),
        (402, "bonus_credits_not_applicable", BILLING),
        (429, "usage_limit_exceeded", BILLING),
        (429, "org_monthly_cap_exceeded", BILLING),
        (429, "rejected_subscription_exhausted", BILLING),
        (429, "rejected_model_budget_exhausted_monthly", BILLING),
        (429, "rejected_model_budget_exhausted_five_h", BILLING),
        (403, "provider_blocked", NOT_ENTITLED),
        (403, "model_access_denied", NOT_ENTITLED),
        (403, "org_model_not_allowed", NOT_ENTITLED),
        (403, "rejected_not_in_plan", NOT_ENTITLED),
        (403, "organization_blocked", BLOCKED),
        (403, "key_suspended_abuse", BLOCKED),
        (401, "api_key_expired", BLOCKED),
        (403, "key_ip_restricted", BLOCKED),
        (429, "tenant_quota_exhausted", BILLING),
        (503, "pool_exhausted", {"reason": "overloaded", "retryable": True}),
        (503, "billing_unavailable", {"reason": "server_error", "retryable": True}),
        (402, "INSUFFICIENT_CREDITS", BILLING),
        (403, "coding_model_requires_credits", BILLING),
        (403, "email_verification_required", NOT_ENTITLED),
        (400, "provider_key_required", NOT_ENTITLED),
        (503, "provider_unavailable", NOT_ENTITLED),
        (
            400,
            "provider_no_responses_binding",
            {"reason": "model_not_found", "retryable": False, "should_fallback": True},
        ),
    ],
)
def test_gateway_codes(profile, status, code, expected):
    body = {"error": {"message": "x", "type": "t", "param": None, "code": code}}
    verdict = profile.classify_api_error(
        Exception("x"), status_code=status, error_code=code, message="x", body=body, model="kimi-k3"
    )
    assert verdict == expected


# /v1/messages answers an org provider block or the key's model rules in the
# Anthropic envelope, without error.code; Hermes then reports error.type.
MESSAGES_DENIAL = {"type": "error", "error": {"type": "permission_error", "message": "Provider anthropic is blocked"}}


@pytest.mark.parametrize("code", ["permission_error", "PERMISSION_ERROR", None, ""])
def test_codeless_messages_denial_keeps_the_key(profile, code):
    verdict = profile.classify_api_error(
        Exception("x"), status_code=403, error_code=code, message="x", body=MESSAGES_DENIAL, model="claude-sonnet-5-5"
    )
    assert verdict == NOT_ENTITLED


def test_codeless_messages_denial_without_a_body(profile):
    verdict = profile.classify_api_error(None, status_code=403, error_code="permission_error", message="x", body=None)
    assert verdict == NOT_ENTITLED


@pytest.mark.parametrize(
    ("status", "code", "body"),
    [
        (401, "permission_error", MESSAGES_DENIAL),  # only a 403 is a denial
        (403, "authentication_error", {"type": "error", "error": {"type": "authentication_error", "message": "x"}}),
        (403, None, None),
    ],
)
def test_other_codeless_errors_defer_to_hermes(profile, status, code, body):
    assert profile.classify_api_error(None, status_code=status, error_code=code, message="x", body=body) is None


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, "invalid_api_key"),  # Hermes' own auth guidance
        (429, "rate_limit_exceeded"),  # Hermes backs off, then rotates or falls back
        (404, "model_not_found"),  # Hermes' own model_not_found handling
        (500, "upstream_error"),
        (400, None),
        (None, ""),
    ],
)
def test_other_errors_defer_to_hermes(profile, status, code):
    assert profile.classify_api_error(None, status_code=status, error_code=code, message="", body=None) is None


def test_verdicts_are_copies(profile):
    first = profile.classify_api_error(None, status_code=402, error_code="insufficient_credits")
    first["retryable"] = True
    assert profile.classify_api_error(None, status_code=402, error_code="insufficient_credits") == BILLING
