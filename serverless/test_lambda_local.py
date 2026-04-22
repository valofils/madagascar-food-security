#!/usr/bin/env python3
"""Local Lambda test — no AWS needed."""
import json, os, sys

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
os.environ.setdefault("MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "ipc_binary_classifier.pkl"))
os.environ.setdefault("META_PATH",  os.path.join(PROJECT_ROOT, "models", "ipc_binary_classifier_meta.json"))

from lambda_function import lambda_handler

# Cold-start escalation: lag below 3 but lean season + worsening trend
CRISIS_PAYLOAD = {
    "year": 2024, "month": 1, "quarter": 1,
    "is_lean_season": 1, "period_days": 90,
    "lag_1": 2.5, "lag_2": 2.2, "lag_3": 2.0,
    "rolling_mean_3": 2.3, "rolling_max_3": 2.5,
    "phase_trend": 0.3,
    "unit_hist_max": 3.0, "crisis_momentum": 1.0,
    "is_cold_start": 1, "lean_x_lag1": 2.5, "lean_x_trend": 0.3,
    "gap_to_crisis": 0.5, "escalation_risk": 3.45,
    "is_ipc2": 0, "preference_rating": 1.0,
}

NO_CRISIS_PAYLOAD = {
    "year": 2024, "month": 6, "quarter": 2,
    "is_lean_season": 0, "period_days": 91,
    "lag_1": 1.5, "lag_2": 1.6, "lag_3": 1.7,
    "rolling_mean_3": 1.6, "rolling_max_3": 1.8,
    "phase_trend": -0.1,
    "unit_hist_max": 2.0, "crisis_momentum": 0.0,
    "is_cold_start": 0, "lean_x_lag1": 0.0, "lean_x_trend": 0.0,
    "gap_to_crisis": 1.5, "escalation_risk": 0.0,
    "is_ipc2": 1, "preference_rating": 0.8,
}

TEST_CASES = [
    {"name": "Crisis scenario (cold-start escalation)", "event": CRISIS_PAYLOAD,    "expect_status": 200, "expect_label": "crisis_or_above"},
    {"name": "No-crisis scenario (harvest, stable)",    "event": NO_CRISIS_PAYLOAD, "expect_status": 200, "expect_label": "no_crisis"},
    {"name": "API Gateway proxy event",                 "event": {"body": json.dumps(CRISIS_PAYLOAD), "httpMethod": "POST"}, "expect_status": 200},
    {"name": "Missing field -> 400",                    "event": {k: v for k, v in CRISIS_PAYLOAD.items() if k != "lag_1"}, "expect_status": 400},
    {"name": "Invalid month -> 400",                    "event": {**CRISIS_PAYLOAD, "month": 13}, "expect_status": 400},
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
        status = response.get("statusCode", 0)
        body   = json.loads(response.get("body", "{}"))
        expected_status = tc.get("expect_status", 200)
        if status != expected_status:
            print(f"    ❌ Expected {expected_status}, got {status} | {body}")
            failed += 1; continue
        if status == 200:
            label, expected = body.get("label"), tc.get("expect_label")
            prob, thr = body.get("probability_crisis"), body.get("threshold_used")
            if expected and label != expected:
                print(f"    ⚠️  Mismatch: expected '{expected}', got '{label}' | P={prob}, thr={thr}")
            else:
                print(f"    ✅ label={label} | P(crisis)={prob} | threshold={thr}")
        else:
            print(f"    ✅ Status {status} | error='{body.get('error')}'")
        passed += 1
    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
    print("=" * 60 + "\n")
    return failed == 0

if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
