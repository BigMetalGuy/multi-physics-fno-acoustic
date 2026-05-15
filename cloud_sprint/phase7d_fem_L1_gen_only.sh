#!/usr/bin/env bash
# phase7d_fem_L1_gen_only.sh — Generate FEM dataset at L1 anisotropic grid.
#
# GEN-ONLY (no train, no eval). Pulls back the .h5 for local verification
# before committing train budget. Companion: phase7d_fem_L1_train_only.sh
# fires after manual go.
#
# Grid spec matches Phase 7c j-Wave EXACTLY for apples-to-apples disagreement-
# matrix comparability:
#   - 44 × 44 × 144 voxels @ dx = 3 mm
#   - extent: x,y ∈ [-66, +66] mm ; z ∈ [0, 432] mm
#   - 278K voxels (2.84× Phase 6.6b's 32³)
#
# Estimated runtime on A100 (CPU-bound for FEM, GPU helps a bit):
#   6-10 hr for 5000 configs. ~$12-20 cloud.
#
# Required env vars on instance:
#   LAMBDA_KEY, LAMBDA_INSTANCE_ID, GITHUB_TOKEN, MAX_HOURS (default 12)

set -euo pipefail

MAX_HOURS="${MAX_HOURS:-12}"
if (( MAX_HOURS > 14 )); then
    echo "ERROR: MAX_HOURS=$MAX_HOURS exceeds hard cap 14 for gen-only" >&2
    exit 1
fi

LAMBDA_KEY="${LAMBDA_KEY:?set LAMBDA_KEY env var}"
LAMBDA_INSTANCE_ID="${LAMBDA_INSTANCE_ID:?set LAMBDA_INSTANCE_ID env var}"
# GITHUB_TOKEN no longer needed — this is a public repo

N_CONFIGS=5000
GRID_X=44
GRID_Y=44
GRID_Z=144
# Grid extent in meters (matches Phase 7c j-Wave: 132×132×432mm)
EXT_XMIN=-0.066
EXT_XMAX=0.066
EXT_YMIN=-0.066
EXT_YMAX=0.066
EXT_ZMIN=0.0
EXT_ZMAX=0.432
V_AMP=10.0
MAX_INNER_ITERS=3

WORK_DIR="${HOME}/phase7d"
SIM_REPO="${WORK_DIR}/simulation"
DATASET="${WORK_DIR}/inverse_dataset_fem_3d_phase7d_L1.h5"
SENTINEL="${WORK_DIR}/GEN_DONE"

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# --- Kill switch -----------------------------------------------------------
echo "============================================================"
echo "Phase 7d (FNO_F L1 anisotropic, GEN-ONLY) — kill-switch armed for ${MAX_HOURS}h from $(date -u)"
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

# --- [1/3] Clone repo + venv + deps ----------------------------------------
echo ""
echo "=== [1/3] Clone sim repo + create venv + install deps ==="
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

# System deps for gmsh (OpenGL utility lib + X libs even when headless)
echo "  [apt] installing system deps for gmsh..."
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 \
    libxi6 libxext6 2>&1 | tail -5 || true

# torch wheels (cu121 fwd-compat with driver 12.8) — kept consistent with 7c
pip install --quiet torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# FEM gen deps: jax + jax_fem + drip_physics_core
# pyfiglet — transitive runtime dep of jax_fem (ASCII banner)
# pydantic — transitive runtime dep of drip_physics_core
# gmsh    — lazy runtime dep of jax_fem (mesh generation for L1 chamber)
pip install --quiet \
    "jax[cuda12]" "jaxlib" \
    "h5py" "scipy>=1.14" "numpy>=2.2,<3.0" "meshio" "matplotlib" \
    "pyfiglet" "pydantic" "gmsh"

# jax-fem (PyPI sometimes stale; fallback to GitHub master)
if ! python -c "import jax_fem" 2>/dev/null; then
    pip install --quiet "jax-fem" 2>/dev/null \
        || pip install --quiet "git+https://github.com/deepmodeling/jax-fem.git" \
        || true
fi

pip install --quiet --ignore-requires-python \
    "-e" "${SIM_REPO}/drip_physics_core"

# --- [2/3] Import gate -----------------------------------------------------
echo ""
echo "=== [2/3] Validate every import BEFORE GPU billing on real work ==="
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
check("h5py",       "import h5py")
check("scipy",      "import scipy")
check("numpy 2.x",  "import numpy; assert numpy.__version__.startswith('2.')")
check("jax",        "import jax; assert len(jax.devices()) > 0")
check("jax_fem",    "import jax_fem")
check("gmsh",       "import gmsh; gmsh.initialize(); gmsh.finalize()")
check("drip_physics_core", "import drip_physics_core; from drip_physics_core import PressureConfig; PressureConfig(backend='femcoupled', grid_shape=(44, 44, 144))")
# (femcoupled_backend + ml_inverse live in $SIM_REPO/simulations, tested implicitly by [3/3] gen)

n_err = sum(1 for r in results if r[0] == "ERR")
for s, n in results: print(f"  [{s}] {n}")
sys.exit(0 if n_err == 0 else 1)
EOF

# --- [3/3] Gen FEM dataset -------------------------------------------------
echo ""
echo "=== [3/3] Generate FEM dataset (5000 cfg × ${GRID_X}×${GRID_Y}×${GRID_Z} at L1 grid) ==="
cd "${SIM_REPO}/simulations"

PYTHONPATH=. python -m ml_inverse.generate_fem_dataset \
    --n-trajectories $N_CONFIGS \
    --grid-resolution $GRID_X $GRID_Y $GRID_Z \
    --grid-extent-m $EXT_XMIN $EXT_XMAX $EXT_YMIN $EXT_YMAX $EXT_ZMIN $EXT_ZMAX \
    --velocity-amp $V_AMP \
    --max-inner-iters $MAX_INNER_ITERS \
    --output-path "${DATASET}" \
    --write-every 10 \
    --seed 42 2>&1 | tee "${WORK_DIR}/gen.log"

ls -lh "${DATASET}"

# Sentinel: marks gen complete for the local poll
touch "${SENTINEL}"
echo ""
echo "=== Phase 7d GEN DONE — $(date -u) ==="
echo "Sentinel: ${SENTINEL}"
echo "Dataset:  ${DATASET}"
stat -c '%s' "${DATASET}" | awk '{printf "Size:     %.1f GB\n", $1/1073741824}'
