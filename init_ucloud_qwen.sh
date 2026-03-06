#!/bin/sh
# ============================================================
# UCloud Initialization Script — Qwen2.5-7B DST Evaluation
# ============================================================
# Usage: Upload this file to UCloud and point the job's
#        "Initialization script" field at it, OR run manually:
#
#   bash init_ucloud_qwen.sh
#
# Optional environment overrides:
#   MODEL_NAME=Qwen/Qwen2.5-7B-Instruct bash init_ucloud_qwen.sh
#   MAX_TURNS=50 bash init_ucloud_qwen.sh
#   HF_TOKEN=hf_xxx bash init_ucloud_qwen.sh
#   SPLIT=test bash init_ucloud_qwen.sh
# ============================================================
set -eu

# ── Parse "KEY=VALUE" arguments from UCloud "Extra options" ──
# UCloud passes the Extra options field as positional arguments
# to this script, e.g.:  HF_TOKEN=hf_xxx  MAX_TURNS=50
for arg in "$@"; do
    case "${arg}" in
        HF_TOKEN=*)       HF_TOKEN="${arg#*=}" ;;
        MODEL_NAME=*)     MODEL_NAME="${arg#*=}" ;;
        SPLIT=*)          SPLIT="${arg#*=}" ;;
        MAX_TURNS=*)      MAX_TURNS="${arg#*=}" ;;
        BRANCH=*)         BRANCH="${arg#*=}" ;;
        PRINT_MISMATCHES=*) PRINT_MISMATCHES="${arg#*=}" ;;
        *) echo "Unknown argument: ${arg}" ;;
    esac
done

# ── Configuration (override via env) ─────────────────────────
REPO_URL="https://github.com/kwa6/dst-unified.git"
REPO_DIR="${HOME}/dst-unified"
LOG_FILE="${HOME}/init_ucloud_qwen.log"
BRANCH="${BRANCH:-feature/qwen2.5-7b}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
SPLIT="${SPLIT:-val}"            # val | test
MAX_TURNS="${MAX_TURNS:-}"       # leave blank for full eval
PRINT_MISMATCHES="${PRINT_MISMATCHES:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/qwen2.5_eval}"
HF_TOKEN="${HF_TOKEN:-}"

echo "========================================" | tee "${LOG_FILE}"
echo " UCloud init started at $(date)"         | tee -a "${LOG_FILE}"
echo " MODEL:   ${MODEL_NAME}"                 | tee -a "${LOG_FILE}"
echo " BRANCH:  ${BRANCH}"                     | tee -a "${LOG_FILE}"
echo " SPLIT:   ${SPLIT}"                      | tee -a "${LOG_FILE}"
echo "========================================"| tee -a "${LOG_FILE}"

# ── 1. System dependencies ───────────────────────────────────
echo "[1/5] Installing system packages..." | tee -a "${LOG_FILE}"
sudo apt-get update -qq
sudo apt-get install -y -qq git python3 python3-pip python3-venv curl

# ── 2. Clone or update repo ──────────────────────────────────
echo "[2/5] Cloning repo (branch: ${BRANCH})..." | tee -a "${LOG_FILE}"
if [ -d "${REPO_DIR}/.git" ]; then
    cd "${REPO_DIR}"
    git fetch origin 2>&1 | tee -a "${LOG_FILE}"
    git checkout "${BRANCH}" 2>&1 | tee -a "${LOG_FILE}"
    git pull --rebase origin "${BRANCH}" 2>&1 | tee -a "${LOG_FILE}"
else
    git clone --branch "${BRANCH}" "${REPO_URL}" "${REPO_DIR}" 2>&1 | tee -a "${LOG_FILE}"
    cd "${REPO_DIR}"
fi

# ── 3. Create virtualenv ─────────────────────────────────────
echo "[3/5] Setting up Python venv..." | tee -a "${LOG_FILE}"
if [ ! -f "${REPO_DIR}/.venv/bin/activate" ]; then
    python3 -m venv "${REPO_DIR}/.venv"
fi
source "${REPO_DIR}/.venv/bin/activate"

# ── 4. Install Python dependencies ───────────────────────────
echo "[4/5] Installing Python packages..." | tee -a "${LOG_FILE}"
pip install --upgrade pip --quiet

# Auto-detect CUDA and install matching torch
if nvidia-smi >/dev/null 2>&1; then
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[\d]+" | head -1)
    echo "  GPU detected — CUDA ${CUDA_VER}" | tee -a "${LOG_FILE}"
    if [ "${CUDA_VER}" -ge 12 ]; then
        pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
    else
        pip install torch --index-url https://download.pytorch.org/whl/cu118 --quiet
    fi
else
    echo "  No GPU — installing CPU torch (inference will be slow)" | tee -a "${LOG_FILE}"
    pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
fi

pip install -r "${REPO_DIR}/requirements.txt" --quiet
echo "  Packages installed OK" | tee -a "${LOG_FILE}"

# ── 5. Prepare data ──────────────────────────────────────────
echo "[5/5] Preparing MultiWOZ 2.4 data..." | tee -a "${LOG_FILE}"

# Set HF token if provided
if [ -n "${HF_TOKEN}" ]; then
    export HF_TOKEN="${HF_TOKEN}"
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
    # Log in so gated models (e.g. Qwen2.5) can be downloaded
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>&1 | tee -a "${LOG_FILE}"
    echo "  HF login OK" | tee -a "${LOG_FILE}"
else
    echo "  WARNING: No HF_TOKEN set — gated models may fail to download." | tee -a "${LOG_FILE}"
fi

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src"

# Run adapter only if unified data is missing
if [ ! -f "${REPO_DIR}/data_unified/multiwoz24/${SPLIT}.jsonl" ]; then
    echo "  Generating unified data files..." | tee -a "${LOG_FILE}"
    python3 -m dst.data.multiwoz_adapter 2>&1 | tee -a "${LOG_FILE}"
else
    echo "  Unified data already exists — skipping adapter." | tee -a "${LOG_FILE}"
fi

# ── 6. Run evaluation ────────────────────────────────────────
echo "[+] Running Qwen2.5-7B DST evaluation at $(date)..." | tee -a "${LOG_FILE}"

mkdir -p "${REPO_DIR}/${OUTPUT_DIR}"

# Build optional args
EXTRA_ARGS=""
if [ -n "${MAX_TURNS}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --max_turns ${MAX_TURNS}"
fi
if [ -n "${PRINT_MISMATCHES}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --print_mismatches ${PRINT_MISMATCHES}"
fi

python3 -u -m dst.runners.eval_jga_qwen \
    --path    "data_unified/multiwoz24/${SPLIT}.jsonl" \
    --model   "${MODEL_NAME}" \
    ${EXTRA_ARGS} \
    2>&1 | tee -a "${LOG_FILE}"

echo "========================================" | tee -a "${LOG_FILE}"
echo " Evaluation finished at $(date)"         | tee -a "${LOG_FILE}"
echo " Full log: ${LOG_FILE}"                  | tee -a "${LOG_FILE}"
echo "========================================"| tee -a "${LOG_FILE}"

# ── 7. Save log to GitHub ────────────────────────────────────
echo "[+] Saving log to GitHub..." | tee -a "${LOG_FILE}"
cd "${REPO_DIR}"
LOG_DEST="logs/$(date +%Y%m%d_%H%M%S)_qwen_ucloud.log"
mkdir -p logs
cp "${LOG_FILE}" "${LOG_DEST}"
git config user.email "ucloud@job"
git config user.name "UCloud Job"
git add "${LOG_DEST}"
git commit -m "Add eval log: ${LOG_DEST}" 2>&1 | tee -a "${LOG_FILE}"
git push origin "${BRANCH}" 2>&1 | tee -a "${LOG_FILE}"
echo "[+] Log saved to repo: ${LOG_DEST}" | tee -a "${LOG_FILE}"
