#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# test_api.sh  –  Smoke-test the Madagascar Food Security FastAPI
#
# Usage (server already running):
#   bash test_api.sh
#   bash test_api.sh http://localhost:8000   # custom base URL
# ─────────────────────────────────────────────────────────────────────────────

BASE="${1:-http://localhost:8000}"
PASS=0; FAIL=0

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓ $*${NC}";  ((PASS++)); }
fail() { echo -e "${RED}  ✗ $*${NC}";   ((FAIL++)); }
hdr()  { echo -e "\n${CYAN}── $* ──${NC}"; }

check_status() {
    local label="$1" expected="$2" actual="$3"
    if [[ "$actual" == "$expected" ]]; then ok "$label → HTTP $actual"
    else fail "$label → expected HTTP $expected, got HTTP $actual"; fi
}

check_field() {
    local label="$1" field="$2" expected="$3" json="$4"
    local actual
    actual=$(echo "$json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d$field)" 2>/dev/null)
    if [[ "$actual" == "$expected" ]]; then ok "$label: $field == $expected"
    else fail "$label: $field expected '$expected', got '$actual'"; fi
}

# ─────────────────────────────────────────────────────────────────────────────
hdr "1. Health check"
# ─────────────────────────────────────────────────────────────────────────────
resp=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/health")
check_status "GET /health" "200" "$resp"

body=$(curl -s "$BASE/health")
check_field "GET /health" "['status']" "ok" "$body"

# ─────────────────────────────────────────────────────────────────────────────
hdr "2. Root endpoint"
# ─────────────────────────────────────────────────────────────────────────────
resp=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/")
check_status "GET /" "200" "$resp"

# ─────────────────────────────────────────────────────────────────────────────
hdr "3. Features list"
# ─────────────────────────────────────────────────────────────────────────────
body=$(curl -s "$BASE/features")
resp=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/features")
check_status "GET /features" "200" "$resp"
check_field "GET /features" "['count']" "16" "$body"

# ─────────────────────────────────────────────────────────────────────────────
hdr "4. Example request body"
# ─────────────────────────────────────────────────────────────────────────────
resp=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/example")
check_status "GET /example" "200" "$resp"

# ─────────────────────────────────────────────────────────────────────────────
hdr "5. POST /predict  –  Grand Sud HIGH-risk profile"
# ─────────────────────────────────────────────────────────────────────────────
HIGH_RISK='{
  "year": 2024, "month": 2, "quarter": 1,
  "is_lean_season": 1, "period_days": 29,
  "lag_1": 3.0, "lag_2": 3.0,
  "rolling_mean_3": 3.0, "rolling_max_3": 3.0,
  "phase_trend": 0.0,
  "unit_mean_phase": 2.8, "unit_max_phase": 4.0,
  "unit_pct_crisis": 0.65, "unit_code": 42,
  "is_ipc2": 0, "preference_rating": 90.0
}'

http_code=$(curl -s -o /tmp/high_risk.json -w "%{http_code}" \
  -X POST "$BASE/predict" \
  -H "Content-Type: application/json" \
  -d "$HIGH_RISK")

check_status "POST /predict (HIGH-risk)" "200" "$http_code"

body=$(cat /tmp/high_risk.json)
echo "  Response:"
echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"

alert=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin).get('alert_level','MISSING'))" 2>/dev/null)
if [[ "$alert" == "HIGH" || "$alert" == "MODERATE" ]]; then
    ok "alert_level is $alert (crisis detected)"
else
    fail "alert_level expected HIGH/MODERATE for Grand Sud, got '$alert'"
fi

binary_crisis=$(echo "$body" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(d['binary']['is_crisis'])" 2>/dev/null)
if [[ "$binary_crisis" == "True" ]]; then
    ok "binary.is_crisis == True"
else
    fail "binary.is_crisis expected True, got '$binary_crisis'"
fi

# ─────────────────────────────────────────────────────────────────────────────
hdr "6. POST /predict  –  LOW-risk profile"
# ─────────────────────────────────────────────────────────────────────────────
LOW_RISK='{
  "year": 2024, "month": 7, "quarter": 3,
  "is_lean_season": 0, "period_days": 31,
  "lag_1": 1.0, "lag_2": 1.0,
  "rolling_mean_3": 1.2, "rolling_max_3": 1.5,
  "phase_trend": 0.0,
  "unit_mean_phase": 1.1, "unit_max_phase": 2.0,
  "unit_pct_crisis": 0.02, "unit_code": 10,
  "is_ipc2": 0, "preference_rating": 95.0
}'

http_code=$(curl -s -o /tmp/low_risk.json -w "%{http_code}" \
  -X POST "$BASE/predict" \
  -H "Content-Type: application/json" \
  -d "$LOW_RISK")

check_status "POST /predict (LOW-risk)" "200" "$http_code"

body=$(cat /tmp/low_risk.json)
echo "  Response:"
echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"

alert=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin).get('alert_level','MISSING'))" 2>/dev/null)
if [[ "$alert" == "LOW" ]]; then
    ok "alert_level == LOW"
else
    fail "alert_level expected LOW for low-risk profile, got '$alert'"
fi

# ─────────────────────────────────────────────────────────────────────────────
hdr "7. POST /predict  –  Validation error (bad month)"
# ─────────────────────────────────────────────────────────────────────────────
BAD_REQ='{
  "year": 2024, "month": 99, "quarter": 1,
  "is_lean_season": 1, "period_days": 29,
  "lag_1": 3.0, "lag_2": 3.0,
  "rolling_mean_3": 3.0, "rolling_max_3": 3.0,
  "phase_trend": 0.0,
  "unit_mean_phase": 2.8, "unit_max_phase": 4.0,
  "unit_pct_crisis": 0.65, "unit_code": 42,
  "is_ipc2": 0, "preference_rating": 90.0
}'

resp=$(curl -s -o /dev/null -w "%{http_code}" \
  -X POST "$BASE/predict" \
  -H "Content-Type: application/json" \
  -d "$BAD_REQ")

check_status "POST /predict (bad month → 422)" "422" "$resp"

# ─────────────────────────────────────────────────────────────────────────────
hdr "8. POST /predict  –  Missing required field"
# ─────────────────────────────────────────────────────────────────────────────
resp=$(curl -s -o /dev/null -w "%{http_code}" \
  -X POST "$BASE/predict" \
  -H "Content-Type: application/json" \
  -d '{"year": 2024}')

check_status "POST /predict (missing fields → 422)" "422" "$resp"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Results:  ${GREEN}${PASS} passed${NC}  |  ${RED}${FAIL} failed${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
[[ $FAIL -eq 0 ]] && exit 0 || exit 1
