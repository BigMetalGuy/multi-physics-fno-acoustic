#!/usr/bin/env bash
# lambda_launch.sh — provision a Lambda Labs A100 instance for the distill sprint.
#
# Reads the API key from the LAMBDA_KEY env var. Never hardcoded.
# Lambda's REST API uses HTTP Basic auth: `curl -u $LAMBDA_KEY:` (empty password).
#
# Happy path:
#   1. Generate local SSH key (skip if it exists).
#   2. Register the public key with Lambda (idempotent — skip if already there).
#   3. Launch one gpu_1x_a100 instance in us-east-1 (40 GB SXM4).
#   4. Poll until status='active'.
#   5. Print SSH command + instance_id.
#
# On any error: emit a teardown hint so the orchestrator knows how to stop billing.
#
# Reference: https://docs.lambdalabs.com/cloud/api/

set -euo pipefail

# ---------- config ----------------------------------------------------------
INSTANCE_TYPE="${LAMBDA_INSTANCE_TYPE:-gpu_1x_a100_sxm4}"
REGION="${LAMBDA_REGION:-us-east-1}"
SSH_KEY_NAME="${LAMBDA_SSH_KEY_NAME:-fno-cloud-sprint-1}"
SSH_KEY_PATH="${LAMBDA_SSH_KEY_PATH:-$HOME/.ssh/lambda_fno_sprint}"
POLL_INTERVAL_S="${LAMBDA_POLL_INTERVAL_S:-15}"
POLL_TIMEOUT_S="${LAMBDA_POLL_TIMEOUT_S:-900}"   # 15 min hard cap on becoming active
API_BASE="https://cloud.lambdalabs.com/api/v1"

# ---------- instance-type whitelist (cost cap) -----------------------------
# Refuse to launch anything more expensive than the listed types.
# Each entry is the API slug; max-budget-friendly only.
ALLOWED_TYPES=(
    "gpu_1x_a10"           # $1.29/hr — fallback, smaller VRAM
    "gpu_1x_a100_sxm4"     # $1.99/hr — primary target (40 GB)
    "gpu_1x_h100_pcie"     # $3.29/hr — emergency fallback
    "gpu_1x_h100_sxm5"     # $4.29/hr — emergency fallback
)
allowed=0
for t in "${ALLOWED_TYPES[@]}"; do
    if [[ "${INSTANCE_TYPE}" == "${t}" ]]; then allowed=1; break; fi
done
if [[ "${allowed}" -ne 1 ]]; then
    echo "ERROR: INSTANCE_TYPE='${INSTANCE_TYPE}' is not in the cost-capped whitelist." >&2
    echo "       Allowed: ${ALLOWED_TYPES[*]}" >&2
    echo "       Override at your own risk: edit ALLOWED_TYPES in this script." >&2
    exit 1
fi

# ---------- preflight -------------------------------------------------------
if [[ -z "${LAMBDA_KEY:-}" ]]; then
    echo "ERROR: LAMBDA_KEY env var is not set." >&2
    echo "       export LAMBDA_KEY=secret_xxx... and rerun." >&2
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl not installed." >&2
    exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq not installed (brew install jq / apt-get install jq)." >&2
    exit 1
fi

# ---------- helpers ---------------------------------------------------------
api_get() {
    local path="$1"
    curl -fsS -u "${LAMBDA_KEY}:" "${API_BASE}${path}"
}

api_post() {
    local path="$1"
    local body="$2"
    curl -fsS -u "${LAMBDA_KEY}:" \
        -H "Content-Type: application/json" \
        -X POST "${API_BASE}${path}" \
        -d "${body}"
}

abort() {
    local msg="$1"
    echo "ERROR: ${msg}" >&2
    echo "" >&2
    echo "If an instance was launched but provisioning failed, terminate it via:" >&2
    echo "  bash $(dirname "$0")/lambda_teardown.sh" >&2
    echo "or directly via the Lambda dashboard at https://cloud.lambdalabs.com/instances" >&2
    exit 1
}

# ---------- 1. local SSH key ------------------------------------------------
echo "[1/5] Local SSH key"
mkdir -p "$(dirname "${SSH_KEY_PATH}")"
chmod 700 "$(dirname "${SSH_KEY_PATH}")"
if [[ ! -f "${SSH_KEY_PATH}" ]]; then
    echo "  generating new key at ${SSH_KEY_PATH}"
    ssh-keygen -t ed25519 -f "${SSH_KEY_PATH}" -N "" -C "fno-cloud-sprint" >/dev/null
else
    echo "  reusing existing key at ${SSH_KEY_PATH}"
fi
PUBKEY=$(cat "${SSH_KEY_PATH}.pub")

# ---------- 2. register key with Lambda ------------------------------------
echo "[2/5] Register SSH key with Lambda (idempotent)"
EXISTING_KEYS=$(api_get "/ssh-keys" || abort "could not list SSH keys (auth issue?)")
KEY_ID=$(echo "${EXISTING_KEYS}" | jq -r --arg n "${SSH_KEY_NAME}" \
    '.data[] | select(.name == $n) | .id' | head -1)

if [[ -n "${KEY_ID}" ]]; then
    echo "  key '${SSH_KEY_NAME}' already registered (id=${KEY_ID})"
else
    echo "  registering '${SSH_KEY_NAME}'"
    REG_BODY=$(jq -n \
        --arg name "${SSH_KEY_NAME}" \
        --arg pk "${PUBKEY}" \
        '{name: $name, public_key: $pk}')
    REG_RESP=$(api_post "/ssh-keys" "${REG_BODY}" || abort "SSH key registration failed")
    KEY_ID=$(echo "${REG_RESP}" | jq -r '.data.id')
    if [[ -z "${KEY_ID}" || "${KEY_ID}" == "null" ]]; then
        abort "key registration response missing id; raw: ${REG_RESP}"
    fi
    echo "  registered (id=${KEY_ID})"
fi

# ---------- 3. launch instance ---------------------------------------------
echo "[3/5] Launch instance (type=${INSTANCE_TYPE}, region=${REGION})"
LAUNCH_BODY=$(jq -n \
    --arg type "${INSTANCE_TYPE}" \
    --arg region "${REGION}" \
    --arg key "${SSH_KEY_NAME}" \
    '{
        region_name: $region,
        instance_type_name: $type,
        ssh_key_names: [$key],
        quantity: 1,
        name: "fno-distill"
    }')

LAUNCH_RESP=$(api_post "/instance-operations/launch" "${LAUNCH_BODY}" \
    || abort "launch failed (capacity? — check 'instance-types' endpoint, may need to retry or change region)")

INSTANCE_ID=$(echo "${LAUNCH_RESP}" | jq -r '.data.instance_ids[0]')
if [[ -z "${INSTANCE_ID}" || "${INSTANCE_ID}" == "null" ]]; then
    abort "launch response missing instance_id; raw: ${LAUNCH_RESP}"
fi
echo "  launched: instance_id=${INSTANCE_ID}"

# Persist instance_id so teardown can find it without an extra arg.
INSTANCE_FILE="${LAMBDA_INSTANCE_FILE:-${HOME}/.lambda_fno_instance_id}"
echo "${INSTANCE_ID}" > "${INSTANCE_FILE}"
echo "  wrote ${INSTANCE_FILE}"

# SAFETY: from this point forward, the instance is BILLING. If the script
# exits abnormally (signal, error, ctrl-C), auto-terminate to stop billing.
# The trap is cleared at successful exit (handoff section).
LAMBDA_KEY_FOR_TRAP="${LAMBDA_KEY}"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then echo "" >&2; echo "[trap] script aborted (rc=$rc); auto-terminating instance ${INSTANCE_ID}" >&2; curl -fsS -u "${LAMBDA_KEY_FOR_TRAP}:" -H "Content-Type: application/json" -X POST "${API_BASE}/instance-operations/terminate" -d "{\"instance_ids\":[\"${INSTANCE_ID}\"]}" >&2 || echo "[trap] WARNING: auto-terminate failed; manually terminate via Lambda dashboard" >&2; fi' EXIT

# ---------- 4. poll until active -------------------------------------------
echo "[4/5] Waiting for instance to become active (timeout ${POLL_TIMEOUT_S}s)"
ELAPSED=0
SSH_HOST=""
while (( ELAPSED < POLL_TIMEOUT_S )); do
    STATUS_RESP=$(api_get "/instances/${INSTANCE_ID}" \
        || { echo "  poll failed; will retry" >&2; sleep "${POLL_INTERVAL_S}"; ELAPSED=$((ELAPSED + POLL_INTERVAL_S)); continue; })
    STATUS=$(echo "${STATUS_RESP}" | jq -r '.data.status')
    SSH_HOST=$(echo "${STATUS_RESP}" | jq -r '.data.ip // ""')
    echo "  t=${ELAPSED}s  status=${STATUS}  ip=${SSH_HOST:-pending}"
    if [[ "${STATUS}" == "active" && -n "${SSH_HOST}" ]]; then
        break
    fi
    if [[ "${STATUS}" == "terminated" || "${STATUS}" == "failed" ]]; then
        abort "instance ${INSTANCE_ID} entered terminal state '${STATUS}' before becoming active"
    fi
    sleep "${POLL_INTERVAL_S}"
    ELAPSED=$((ELAPSED + POLL_INTERVAL_S))
done

if [[ "${STATUS}" != "active" ]]; then
    abort "instance ${INSTANCE_ID} did not become active within ${POLL_TIMEOUT_S}s — RUN TEARDOWN NOW"
fi

# ---------- 5. handoff ------------------------------------------------------
# Clear the auto-terminate trap — instance is healthy, hand off to operator.
trap - EXIT

echo ""
echo "============================================================"
echo "Lambda instance ready (BILLING NOW — $1.99/hr A100 or $1.29/hr A10)"
echo "============================================================"
echo "  instance_id : ${INSTANCE_ID}"
echo "  ip          : ${SSH_HOST}"
echo "  ssh key     : ${SSH_KEY_PATH}"
echo ""
echo "SSH in:"
echo "  ssh -i ${SSH_KEY_PATH} ubuntu@${SSH_HOST}"
echo ""
echo "Once on the box, run the setup script:"
echo "  scp -i ${SSH_KEY_PATH} $(dirname "$0")/lambda_setup.sh ubuntu@${SSH_HOST}:~/"
echo "  ssh -i ${SSH_KEY_PATH} ubuntu@${SSH_HOST} 'bash lambda_setup.sh 2>&1 | tee setup.log'"
echo ""
echo "When done, deprovision via:"
echo "  bash $(dirname "$0")/lambda_teardown.sh"
echo ""
