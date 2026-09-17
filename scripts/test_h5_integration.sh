#!/bin/bash
set -e

PORT=8089
DB_PATH="data/edge.db"
IMAGE_PATH="data/model_b_sources/4tu/4TUDatasetAnonymised/1179.jpg"
LOG_FILE="artifacts/audit/2026-09-17b/h5_3_curl_upload.log"

echo "=== H5.3 Integration Test: Trap Upload & Advisory Verification ===" | tee "$LOG_FILE"
echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "$LOG_FILE"

# 1. Start gateway server in background
.venv/bin/python -m gateway.server --host 127.0.0.1 --port $PORT --db-path "$DB_PATH" &
SERVER_PID=$!
sleep 1.5

cleanup() {
    echo "Stopping gateway server (PID: $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT

# 2. Check health endpoint
echo "Checking gateway health on port $PORT..." | tee -a "$LOG_FILE"
curl -s "http://127.0.0.1:$PORT/api/v1/health" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 3. Upload trap image via curl POST (multipart or raw binary)
echo -e "\n--- Step 1: POST /api/v1/trap/upload via curl ---" | tee -a "$LOG_FILE"
echo "curl -X POST 'http://127.0.0.1:$PORT/api/v1/trap/upload?trap_id=TRAP_DEMO_01&scale=0.125&allow_provisional=true&placed_at=2026-09-16T10:00:00Z&days=1.0' \\" | tee -a "$LOG_FILE"
echo "  -H 'Content-Type: image/jpeg' \\" | tee -a "$LOG_FILE"
echo "  --data-binary @$IMAGE_PATH" | tee -a "$LOG_FILE"

UPLOAD_RESP=$(curl -s -X POST "http://127.0.0.1:$PORT/api/v1/trap/upload?trap_id=TRAP_DEMO_01&scale=0.125&allow_provisional=true&placed_at=2026-09-16T10:00:00Z&days=1.0" \
  -H "Content-Type: image/jpeg" \
  --data-binary "@$IMAGE_PATH")

echo "RAW UPLOAD RESPONSE:" | tee -a "$LOG_FILE"
echo "$UPLOAD_RESP" | tee -a "$LOG_FILE"

# Extract advisory_id using python
ADVISORY_ID=$(echo "$UPLOAD_RESP" | .venv/bin/python -c "import sys, json; data=json.load(sys.stdin); print(data.get('advisory_id') or 'latest')")

# 4. Fetch advisory JSON
echo -e "\n--- Step 2: GET /api/v1/advisory/$ADVISORY_ID via curl ---" | tee -a "$LOG_FILE"
echo "curl -s 'http://127.0.0.1:$PORT/api/v1/advisory/$ADVISORY_ID'" | tee -a "$LOG_FILE"

ADVISORY_RESP=$(curl -s "http://127.0.0.1:$PORT/api/v1/advisory/$ADVISORY_ID")

echo "RAW ADVISORY RESPONSE (Excerpts & Validation):" | tee -a "$LOG_FILE"
echo "$ADVISORY_RESP" | .venv/bin/python -c "
import sys, json
adv = json.load(sys.stdin)
pest = adv.get('pest', {})
print('Advisory ID:', adv.get('advisory_id'))
print('Pest block keys:', list(pest.keys()))
print('count_basis:', pest.get('count_basis'))
print('threshold_verification_status:', pest.get('threshold_verification_status'))
print('classification_verification_status:', pest.get('classification_verification_status'))
print('scale_status:', pest.get('scale_status'))
print('total_blobs_counted:', pest.get('total_blobs_counted'))
print('daily_rate:', pest.get('daily_rate'))
print('etl_status:', pest.get('status'))
print('morphological_distribution:', pest.get('morphological_distribution'))

# Assertions
assert pest.get('count_basis') is not None, 'Missing count_basis'
assert pest.get('threshold_verification_status') == 'VERIFIED', 'threshold_verification_status != VERIFIED'
assert pest.get('classification_verification_status') == 'RECALLED_UNVERIFIED', 'classification_verification_status != RECALLED_UNVERIFIED'
assert pest.get('scale_status') == 'PROVISIONAL', 'scale_status != PROVISIONAL'
print('\nALL H5.3 INTEGRATION CONTRACT ASSERTIONS PASSED!')
" | tee -a "$LOG_FILE"

