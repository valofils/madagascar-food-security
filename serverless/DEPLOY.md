# IPC Classifier — AWS Lambda Deployment Runbook

Deploys the XGBoost IPC food security binary classifier as a containerised
AWS Lambda function behind API Gateway (HTTP API).

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| AWS CLI | ≥ 2.x | `brew install awscli` / [docs](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) |
| Docker Desktop | ≥ 24 | [docker.com](https://www.docker.com/products/docker-desktop/) |
| AWS account | — | Free tier works for low-volume inference |

### Configure AWS credentials
```bash
aws configure
# Enter: Access Key ID, Secret Access Key, region (e.g. eu-west-1), output format (json)
```

---

## Step 1 — Set environment variables

```bash
# Adjust these for your account and preferred region
export AWS_REGION=eu-west-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO=ipc-classifier
export FUNCTION_NAME=ipc-food-security-classifier
export IMAGE_TAG=latest
```

---

## Step 2 — Build the Docker image

Run from the **project root** (not from `serverless/`):

```bash
docker build \
  -t ${ECR_REPO}:${IMAGE_TAG} \
  -f serverless/Dockerfile \
  .
```

Verify the image starts and responds locally:

```bash
# Start the Lambda runtime emulator
docker run --rm -p 9000:8080 ${ECR_REPO}:${IMAGE_TAG} &

# Send a test event
curl -s -XPOST \
  "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -H "Content-Type: application/json" \
  -d '{
    "prevalence_phase1": 20.0,
    "prevalence_phase2": 30.0,
    "prevalence_phase3": 35.0,
    "prevalence_phase4": 12.0,
    "prevalence_phase5": 3.0,
    "population_analysed": 250000,
    "country_code": "MDG",
    "period_type": "current"
  }' | python3 -m json.tool

# Stop the container
docker stop $(docker ps -q --filter ancestor=${ECR_REPO}:${IMAGE_TAG})
```

Expected response:
```json
{
    "statusCode": 200,
    "body": "{\"prediction\": 1, \"label\": \"crisis_or_above\", \"probability_crisis\": 0.82, ...}"
}
```

---

## Step 3 — Push image to Amazon ECR

```bash
# Create the ECR repository (one-time)
aws ecr create-repository \
  --repository-name ${ECR_REPO} \
  --region ${AWS_REGION}

# Authenticate Docker to ECR
aws ecr get-login-password --region ${AWS_REGION} \
  | docker login \
    --username AWS \
    --password-stdin \
    ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Tag and push
export ECR_URI=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}

docker tag ${ECR_REPO}:${IMAGE_TAG} ${ECR_URI}
docker push ${ECR_URI}
```

---

## Step 4 — Create the IAM execution role

```bash
# Create the role
aws iam create-role \
  --role-name ipc-lambda-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach basic execution policy (CloudWatch Logs)
aws iam attach-role-policy \
  --role-name ipc-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

export ROLE_ARN=$(aws iam get-role \
  --role-name ipc-lambda-role \
  --query Role.Arn --output text)
```

---

## Step 5 — Deploy the Lambda function

```bash
aws lambda create-function \
  --function-name ${FUNCTION_NAME} \
  --package-type Image \
  --code ImageUri=${ECR_URI} \
  --role ${ROLE_ARN} \
  --region ${AWS_REGION} \
  --memory-size 512 \
  --timeout 30 \
  --environment "Variables={MODEL_PATH=/opt/ml/model/ipc_binary_classifier.pkl,META_PATH=/opt/ml/model/ipc_binary_classifier_meta.json}"

# Wait until Active
aws lambda wait function-active \
  --function-name ${FUNCTION_NAME} \
  --region ${AWS_REGION}
```

### Update an existing function (re-deployments)
```bash
aws lambda update-function-code \
  --function-name ${FUNCTION_NAME} \
  --image-uri ${ECR_URI} \
  --region ${AWS_REGION}
```

---

## Step 6 — Test the Lambda directly

```bash
aws lambda invoke \
  --function-name ${FUNCTION_NAME} \
  --region ${AWS_REGION} \
  --payload '{
    "prevalence_phase1": 20.0,
    "prevalence_phase2": 30.0,
    "prevalence_phase3": 35.0,
    "prevalence_phase4": 12.0,
    "prevalence_phase5": 3.0,
    "population_analysed": 250000,
    "country_code": "MDG",
    "period_type": "current"
  }' \
  --cli-binary-format raw-in-base64-out \
  response.json && cat response.json
```

---

## Step 7 — Create the API Gateway (HTTP API)

```bash
# Create HTTP API
export API_ID=$(aws apigatewayv2 create-api \
  --name ipc-classifier-api \
  --protocol-type HTTP \
  --query ApiId --output text \
  --region ${AWS_REGION})

# Create Lambda integration
export INTEGRATION_ID=$(aws apigatewayv2 create-integration \
  --api-id ${API_ID} \
  --integration-type AWS_PROXY \
  --integration-uri arn:aws:lambda:${AWS_REGION}:${AWS_ACCOUNT_ID}:function:${FUNCTION_NAME} \
  --payload-format-version 2.0 \
  --query IntegrationId --output text \
  --region ${AWS_REGION})

# Create POST /predict route
aws apigatewayv2 create-route \
  --api-id ${API_ID} \
  --route-key "POST /predict" \
  --target integrations/${INTEGRATION_ID} \
  --region ${AWS_REGION}

# Deploy to $default stage
aws apigatewayv2 create-stage \
  --api-id ${API_ID} \
  --stage-name '$default' \
  --auto-deploy \
  --region ${AWS_REGION}

# Grant API Gateway permission to invoke Lambda
aws lambda add-permission \
  --function-name ${FUNCTION_NAME} \
  --statement-id apigw-invoke \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:${AWS_REGION}:${AWS_ACCOUNT_ID}:${API_ID}/*/*/predict" \
  --region ${AWS_REGION}

# Print the endpoint
export API_ENDPOINT=$(aws apigatewayv2 get-api \
  --api-id ${API_ID} \
  --query ApiEndpoint --output text \
  --region ${AWS_REGION})

echo "API endpoint: ${API_ENDPOINT}/predict"
```

---

## Step 8 — End-to-end API test

```bash
curl -s -XPOST "${API_ENDPOINT}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "prevalence_phase1": 20.0,
    "prevalence_phase2": 30.0,
    "prevalence_phase3": 35.0,
    "prevalence_phase4": 12.0,
    "prevalence_phase5": 3.0,
    "population_analysed": 250000,
    "country_code": "MDG",
    "period_type": "current"
  }' | python3 -m json.tool
```

Expected:
```json
{
    "prediction": 1,
    "label": "crisis_or_above",
    "probability_crisis": 0.82,
    "model_version": "1.0.0",
    "features_used": 10
}
```

---

## Cost estimate (AWS Free Tier)

| Resource | Free Tier | Typical portfolio usage |
|----------|-----------|------------------------|
| Lambda invocations | 1M/month | Well under limit |
| Lambda compute | 400,000 GB-seconds/month | Well under limit |
| API Gateway | 1M calls/month (12 months) | Well under limit |
| ECR storage | 500MB/month (12 months) | ~1.2MB image |

**Expected monthly cost for portfolio/demo use: $0.00**

---

## Teardown (avoid accidental charges)

```bash
aws lambda delete-function --function-name ${FUNCTION_NAME} --region ${AWS_REGION}
aws apigatewayv2 delete-api --api-id ${API_ID} --region ${AWS_REGION}
aws ecr delete-repository --repository-name ${ECR_REPO} --force --region ${AWS_REGION}
aws iam detach-role-policy \
  --role-name ipc-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam delete-role --role-name ipc-lambda-role
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: xgboost` | Wrong base image | Confirm `FROM public.ecr.aws/lambda/python:3.11` |
| `Unable to import module 'lambda_function'` | Handler path wrong | Check `CMD` in Dockerfile matches module name |
| 500 on first invocation only | Cold start model load race | Normal — retry; model is cached after cold start |
| `AccessDeniedException` on Lambda invoke | Role not propagated | Wait 10–15s after role creation then retry |
| Container exits immediately locally | Port not exposed | Use `-p 9000:8080` flag |
