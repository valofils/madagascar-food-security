# ── IPC Food Security Classifier — AWS Lambda Container Image ──────────────────
#
# Uses the official AWS Lambda Python 3.11 base image, which includes the
# Lambda Runtime Interface Client (RIC) pre-installed.
#
# Build:  docker build -t ipc-classifier-lambda .
# Run:    docker run -p 9000:8080 ipc-classifier-lambda
# Test:   curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
#              -H "Content-Type: application/json" \
#              -d @test_event.json
# ───────────────────────────────────────────────────────────────────────────────

FROM public.ecr.aws/lambda/python:3.11

# ── System deps (minimal — base image already has most) ───────────────────────
# No extra system packages needed for XGBoost/scikit-learn on this base image.

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt

RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

# ── Model artifacts ────────────────────────────────────────────────────────────
# Copy the XGBoost binary classifier and its metadata.
# In production, consider loading from S3 at cold-start instead
# (set MODEL_PATH / META_PATH env vars to /tmp/... and fetch via boto3).
RUN mkdir -p /opt/ml/model

COPY models/ipc_binary_classifier.pkl      /opt/ml/model/
COPY models/ipc_binary_classifier_meta.json /opt/ml/model/

# ── Environment ────────────────────────────────────────────────────────────────
ENV MODEL_PATH=/opt/ml/model/ipc_binary_classifier.pkl
ENV META_PATH=/opt/ml/model/ipc_binary_classifier_meta.json

# ── Handler ────────────────────────────────────────────────────────────────────
# Format: <module_name>.<function_name>
CMD ["lambda_function.lambda_handler"]
