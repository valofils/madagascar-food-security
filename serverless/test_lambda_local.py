#!/usr/bin/env python3
"""Local Lambda test — invokes lambda_handler() directly, no AWS needed."""

import json, os, sys

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

os.environ.setdefault("MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "ipc_binary_classifier.pkl"))
os.environ.setdefault("META_PATH",  os.path.join(PROJECT_ROOT, "models", "ipc_binary_classifier_meta.json"))

from lambda_function import lambda_handler

CRISIS_PAYLOAD = {
    "year": 2024, "month": 1, "quarter": 1,
    "is_lean_season": 1, "period_days": 90,
    "lag_1": 3.2, "lag_2": 3.0,
    "rolling_mean_3": 3.1, "rolling_max_3": 3.5,
    "phase_trend": 0.2, "unit_hist_max": 4.0,
    "is_ipc2": 0, "preference_rating": 1.0,
}

NO_CRISIS_PAYLOAD = {
    "year": 2024, "month": 6, "quarter": 2,
    "is_lean_season": 0, "period_days": 91,
    "lag_1": 1.8, "lag_2": 1.9,
    "rolling_mean_3": 1.85, "rolling_max_3": 2.0,
    "phase_trend": -0.1, "unit_hist_max": 2.0,
    "is_ipc2": 1, "preference_rating": 0.8,
}

TEST_CASES = [
    {"name": "Crisis scenario",         "event": CRISIS_PAYLOAD,    "expect_status": 200, "expect_label": "crisis_or_above"},
    {"name": "No-crisis scenario",      "event": NO_CRISIS_PAYLOAD, "expect_status": 200, "expect_label": "no_crisis"},
    {"name": "API Gateway proxy event", "event": {"body": json.dumps(CRISIS_PAYLOAD), "httpMethod": "POST"}, "expect_status": 200},
    {"name": "Missing field -> 400",    "event": {k: v for k, v in CRISIS_PAYLOAD.items() if k != "lag_1"}, "expect_status": 400},
    {"name": "Invalid month -> 400",    "event": {**CRISIS_PAYLOAD, "month": 13}, "expect_status": 400},
]


def run_tests():
    print("\n" + "=" * 60)
    print("  IPC Lambda Handler — Local Tests")
    print("=" * 60)
    passed = failed = 0
    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n[{i}] {tc['name']}")
        try:
            response = lambda_handler(tc["event"], context=None)
        except Exception as exc:
            print(f"    ❌ EXCEPTION: {exc}"); failed += 1; continue

        status         = response.get("statusCode", 0)
        body           = json.loads(response.get("body", "{}"))
        expected_status = tc.get("expect_status", 200)

        if status != expected_status:
            print(f"    ❌ Expected status {expected_status}, got {status} | {body}")
            failed += 1; continue

        if status == 200:
            actual_label   = body.get("label")
            expected_label = tc.get("expect_label")
            prob           = body.get("probability_crisis")
            threshold      = body.get("threshold_used")
            if expected_label and actual_label != expected_label:
                print(f"    ⚠️  Label mismatch — expected '{expected_label}', got '{actual_label}'")
                print(f"       P(crisis)={prob}, threshold={threshold}")
            else:
                print(f"    ✅ label={actual_label} | P(crisis)={prob} | threshold={threshold}")
        else:
            print(f"    ✅ Status {status} | error='{body.get('error')}'")
        passed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
    print("=" * 60 + "\n")
    return failed == 0

if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
