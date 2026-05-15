#!/usr/bin/env bash
# phase7c_jwave_L1_cloud.sh — FNO_J retrain on L1 array geometry (apples-to-apples with FNO_F).
#
# Does GEN + TRAIN + EVAL on Lambda A100. j-Wave dataset gen at this scale
# (5000 cfg × 44×44×144 anisotropic L1) takes ~35hr on M2 Max CPU, so we
# run it on cloud GPU instead.
#
# Anisotropic L1 grid (Nyquist-safe, dx=3mm = lambda/2.86):
#   - grid: (Nx, Ny, Nz) = (44, 44, 144)
#   - side: 132mm × 132mm × 432mm  ← contains L1 bbox (100×100×400mm) with 5+ cell margin
#   - voxels: 278K (2.8x Phase 7a's 32×32×96 = 98K)
#
# Architecture matches 7a for direct comparability:
#   - n_modes = (8, 8, 24) — same as 7a
#   - hidden = 128 — same as 7a
#   - 100 epochs, batch 8
#   - residual prior v2.1_3d
#
# Runtime estimate on A100:
#   - Gen: ~7-8 s/config × 5000 = ~10-11 hr  -> ~$22
#   - Train: ~400 s/epoch × 100 = ~11 hr     -> ~$22
#   - Total: ~$45 + overhead
#
# Required env vars on instance:
#   LAMBDA_KEY, LAMBDA_INSTANCE_ID, GITHUB_TOKEN, MAX_HOURS (default 10)

set -euo pipefail

# --- Config -----------------------------------------------------------------
MAX_HOURS="${MAX_HOURS:-15}"
if (( MAX_HOURS > 16 )); then
    echo "ERROR: MAX_HOURS=$MAX_HOURS exceeds hard cap 16" >&2
    exit 1
fi

LAMBDA_KEY="${LAMBDA_KEY:?set LAMBDA_KEY env var}"
LAMBDA_INSTANCE_ID="${LAMBDA_INSTANCE_ID:?set LAMBDA_INSTANCE_ID env var}"
# GITHUB_TOKEN no longer needed — this is a public repo

# Phase 7c L1 anisotropic config
N_CONFIGS=5000
GRID_X=44
GRID_Y=44
GRID_Z=144
GRID_DX_MM=3.0
ARRAY_RADIUS_MM=50    # L1: 5 cm
ARRAY_HEIGHT_MM=400   # L1: 40 cm
N_EPOCHS=80    # reduced from 100 — 7a/7c-partial showed val plateau by ep 70-80; saves ~1.5hr vs 100
BATCH=4        # batch=8 OOMs at 44×44×144 hidden=128 on A100-40GB
N_MODES_X=8
N_MODES_Y=8
N_MODES_Z=24
HIDDEN=128

WORK_DIR="${HOME}/phase7c"
SIM_REPO="${WORK_DIR}/simulation"
DATASET="${WORK_DIR}/inverse_dataset_jwave_3d_phase7c_L1.h5"
RUN_DIR="${SIM_REPO}/simulations/ml_inverse/research/_phase7c_cloud_production_run"
CKPT_NAME="fno_surrogate_jwave_phase7c_L1_cloud"

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# --- Kill switch (FIRST) ----------------------------------------------------
echo "============================================================"
echo "Phase 7c (FNO_J L1 anisotropic) — kill-switch armed for ${MAX_HOURS}h from $(date -u)"
echo "============================================================"
nohup bash -c "
    sleep \$(( ${MAX_HOURS} * 3600 ))
    echo '[kill-switch] firing at \$(date -u)' >> ${WORK_DIR}/kill_switch.log
    curl -s -u '${LAMBDA_KEY}:' -X POST \
      'https://cloud.lambdalabs.com/api/v1/instance-operations/terminate' \
      -H 'Content-Type: application/json' \
      -d '{\"instance_ids\": [\"${LAMBDA_INSTANCE_ID}\"]}' \
      >> ${WORK_DIR}/kill_switch.log 2>&1
" </dev/null >/dev/null 2>&1 &
KILL_PID=$!
disown $KILL_PID
echo "kill-switch PID: $KILL_PID — fires in $((MAX_HOURS * 3600)) sec"

# --- [1/5] Clone repo + venv + deps ----------------------------------------
echo ""
echo "=== [1/5] Clone sim repo + create venv + install deps ==="
if [ ! -d "${SIM_REPO}" ]; then
    # NOTE: replace USER/REPO with the public-fork URL you push this code to,
    # or scp the repo directory to ${SIM_REPO} from your laptop before running.
    git clone "https://github.com/USER/multi-physics-fno-acoustic.git" "${SIM_REPO}"
fi

VENV="${WORK_DIR}/venv"
if [ ! -d "${VENV}" ]; then
    python3 -m venv "${VENV}"
fi
source "${VENV}/bin/activate"
pip install --quiet --upgrade pip wheel setuptools

# torch first (CUDA 12.1 wheels, fwd-compat with driver 12.8)
pip install --quiet torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# j-Wave gen + torch deps + training infra
pip install --quiet \
    "neuraloperator==2.0.0" "tensorly" "tensorly-torch" torch_harmonics==0.7.4 \
    "jax[cuda12]" "jaxlib" "jwave" \
    "h5py" "scipy>=1.14" "numpy>=2.2,<3.0" "matplotlib"

# drip-physics-core (PressureConfig.grid_shape required for anisotropic gen)
pip install --quiet --ignore-requires-python \
    "-e" "${SIM_REPO}/drip_physics_core"

# --- [2/5] Import gate ------------------------------------------------------
echo ""
echo "=== [2/5] Validate every import BEFORE GPU billing on real work ==="
python <<'EOF'
import sys
results = []
def check(name, code):
    try:
        exec(code)
        results.append(("OK ", name))
    except Exception as e:
        results.append(("ERR", f"{name}: {e}"))

check("torch+cuda", "import torch; assert torch.cuda.is_available(), 'no cuda'")
check("neuralop",   "import neuralop")
check("torch_harmonics", "import torch_harmonics")
check("h5py",       "import h5py")
check("scipy",      "import scipy")
check("numpy 2.x",  "import numpy; assert numpy.__version__.startswith('2.')")
check("jax",        "import jax; assert len(jax.devices()) > 0")
check("jwave",      "import jwave")
check("drip_physics_core", "import drip_physics_core; from drip_physics_core import PressureConfig; PressureConfig(backend='jwave', grid_shape=(44, 44, 144), grid_dx=3e-3)")

n_err = sum(1 for r in results if r[0] == "ERR")
for s, n in results: print(f"  [{s}] {n}")
sys.exit(0 if n_err == 0 else 1)
EOF

# --- [3/5] Gen dataset (j-Wave on A100) -------------------------------------
echo ""
echo "=== [3/5] Generate dataset (5000 cfg × ${GRID_X}×${GRID_Y}×${GRID_Z} × dx=${GRID_DX_MM}mm, L1 array) ==="
mkdir -p "${RUN_DIR}"
cd "${SIM_REPO}/simulations"

PYTHONPATH=. python -m ml_inverse.generate_jwave_dataset \
    --n-trajectories $N_CONFIGS \
    --field-dims 3 \
    --grid-resolution $GRID_X $GRID_Y $GRID_Z \
    --grid-dx-mm $GRID_DX_MM \
    --array-radius-mm $ARRAY_RADIUS_MM \
    --array-height-mm $ARRAY_HEIGHT_MM \
    --output-path "${DATASET}" \
    --seed 42 2>&1 | tee "${RUN_DIR}/gen.log"

ls -lh "${DATASET}"

# --- [4/5] Train FNO_J -----------------------------------------------------
echo ""
echo "=== [4/5] Train FNO_J 118M for 100 epochs on A100 ==="

PYTHONPATH=. python -m ml_inverse.train \
    --residual-prior --correct-prior-3d --device cuda \
    --data-path "${DATASET}" \
    --n-epochs $N_EPOCHS --batch-size $BATCH \
    --n-modes $N_MODES_X $N_MODES_Y $N_MODES_Z \
    --hidden-channels $HIDDEN \
    --checkpoint-name $CKPT_NAME --seed 42 \
    --output-dir "${SIM_REPO}/simulations/ml_inverse/models" 2>&1 | tee "${RUN_DIR}/train.log"

CKPT="${SIM_REPO}/simulations/ml_inverse/models/${CKPT_NAME}_best.pt"
NORM="${SIM_REPO}/simulations/ml_inverse/models/${CKPT_NAME}_norm.npz"

# --- [5/5] Eval -------------------------------------------------------------
echo ""
echo "=== [5/5] Mean-pred sanity + disagreement ==="
PYTHONPATH=. python -m ml_inverse.scripts.mean_pred_sanity \
    --data-path "${DATASET}" --ckpt-best "${CKPT}" --norm "${NORM}" \
    --device cuda --out "${RUN_DIR}/mean_pred.json" 2>&1 | tee -a "${RUN_DIR}/eval.log"

PYTHONPATH=. python -m ml_inverse.disagreement_analysis \
    --mode fno-vs-analytical --fno-ckpt "${CKPT}" --norm "${NORM}" \
    --dataset "${DATASET}" --n-samples 50 --n-render 5 \
    --output-dir "${RUN_DIR}" --device cuda 2>&1 | tee -a "${RUN_DIR}/eval.log" || echo "[step 5b rc: $?]"

echo ""
echo "=== Phase 7c L1 cloud DONE — $(date -u) ==="
cat "${RUN_DIR}/mean_pred.json"
