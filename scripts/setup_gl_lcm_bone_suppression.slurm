#!/bin/bash -l
#SBATCH --job-name=gl-lcm-setup
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/gl_lcm_setup_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/gl_lcm_setup_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail
module load python
# ---------------------------------------------------------------------------
# Compute-node internet access
# ---------------------------------------------------------------------------
PROXY="http://proxy.nhr.fau.de:80"

export http_proxy="${PROXY}"
export https_proxy="${PROXY}"
export HTTP_PROXY="${PROXY}"
export HTTPS_PROXY="${PROXY}"

unset all_proxy ALL_PROXY no_proxy NO_PROXY

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
ROOT="/home/vault/iwi5/iwi5362h/data/bone_suppression/gl_lcm"
REPO_DIR="${ROOT}/GL-LCM"
CHECKPOINT_DIR="${ROOT}/checkpoints"

# New Conda environment. The old download_tools_venv is no longer used.
TOOLS_ENV="${ROOT}/download_tools_conda_env"

# Keep Conda and pip caches outside the limited home filesystem.
CONDA_PKGS_DIRS="${ROOT}/conda_package_cache"
export CONDA_PKGS_DIRS
export PIP_NO_CACHE_DIR=1

mkdir -p \
    "${ROOT}" \
    "${CHECKPOINT_DIR}" \
    "${CONDA_PKGS_DIRS}" \
    /home/woody/iwi5/iwi5362h/logs

echo "======================================================================"
echo "GL-LCM setup"
echo "Host:                    $(hostname)"
echo "Started:                 $(date --iso-8601=seconds)"
echo "Root:                    ${ROOT}"
echo "Repository:              ${REPO_DIR}"
echo "Checkpoint directory:    ${CHECKPOINT_DIR}"
echo "Conda environment:       ${TOOLS_ENV}"
echo "Conda package cache:     ${CONDA_PKGS_DIRS}"
echo "Proxy:                   ${HTTPS_PROXY}"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Find and initialize Conda
# ---------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "[Conda] 'conda' is not currently available in PATH."

    CONDA_CANDIDATES=(
        "${HOME}/miniconda3/bin/conda"
        "${HOME}/anaconda3/bin/conda"
        "${HOME}/miniforge3/bin/conda"
        "${HOME}/mambaforge/bin/conda"
    )

    CONDA_FOUND=""

    for candidate in "${CONDA_CANDIDATES[@]}"; do
        if [[ -x "${candidate}" ]]; then
            CONDA_FOUND="${candidate}"
            break
        fi
    done

    if [[ -z "${CONDA_FOUND}" ]]; then
        echo "[Error] Conda could not be found."
        echo
        echo "Load the cluster's Conda module before submitting, or add the"
        echo "correct Conda executable path to CONDA_CANDIDATES."
        echo
        echo "Available modules containing conda/anaconda/miniconda:"
        module avail 2>&1 | grep -Ei "conda|anaconda|miniconda|miniforge" || true
        exit 1
    fi

    export PATH="$(dirname "${CONDA_FOUND}"):${PATH}"
fi

CONDA_BASE="$(conda info --base)"
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[Error] Conda initialization script not found: ${CONDA_SH}"
    exit 1
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"

echo
echo "[Conda] Executable: $(command -v conda)"
echo "[Conda] Base:       ${CONDA_BASE}"
echo "[Conda] Version:    $(conda --version)"

# ---------------------------------------------------------------------------
# Verify the already-cloned repository
# ---------------------------------------------------------------------------
if [[ ! -d "${REPO_DIR}/.git" ]]; then
    echo "[Error] GL-LCM repository was not found at:"
    echo "        ${REPO_DIR}"
    echo
    echo "This setup script assumes that the repository has already been cloned."
    exit 1
fi

echo
echo "[Git] Existing GL-LCM repository found."
echo "[Git] Current revision:"
git -C "${REPO_DIR}" rev-parse HEAD

echo "[Git] Working-tree status:"
git -C "${REPO_DIR}" status --short

# ---------------------------------------------------------------------------
# Create a lightweight Conda environment for checkpoint downloads
# ---------------------------------------------------------------------------
if [[ -d "${TOOLS_ENV}" && ! -x "${TOOLS_ENV}/bin/python" ]]; then
    echo
    echo "[Conda] Incomplete or invalid environment found:"
    echo "        ${TOOLS_ENV}"
    echo "[Conda] Removing it before recreation."
    rm -rf "${TOOLS_ENV}"
fi

if [[ ! -x "${TOOLS_ENV}/bin/python" ]]; then
    echo
    echo "[Conda] Creating download-tools environment."

    conda create \
        --yes \
        --prefix "${TOOLS_ENV}" \
        python=3.10 \
        pip
else
    echo
    echo "[Conda] Existing download-tools environment found."
fi

echo
echo "[Conda] Environment information:"

conda run \
    --no-capture-output \
    --prefix "${TOOLS_ENV}" \
    python -c \
    'import sys; print("Python executable:", sys.executable); print("Python version:", sys.version)'

# ---------------------------------------------------------------------------
# Install gdown
# ---------------------------------------------------------------------------
echo
echo "[Setup] Installing/updating pip and gdown."

conda run \
    --no-capture-output \
    --prefix "${TOOLS_ENV}" \
    python -m pip install \
    --upgrade \
    --no-cache-dir \
    pip \
    gdown

echo
echo "[Setup] Installed gdown version:"

conda run \
    --no-capture-output \
    --prefix "${TOOLS_ENV}" \
    python -c \
    'import gdown; print(gdown.__version__)'

# ---------------------------------------------------------------------------
# Download all official released checkpoints
#
# Both checkpoint pairs are retained because external MIMIC/VinDr
# generalization should be evaluated with a pilot before selecting the model
# used for the complete conversion.
# ---------------------------------------------------------------------------
conda run \
    --no-capture-output \
    --prefix "${TOOLS_ENV}" \
    python - "${CHECKPOINT_DIR}" <<'PY'
from pathlib import Path
import sys

import gdown


output_dir = Path(sys.argv[1])
output_dir.mkdir(parents=True, exist_ok=True)

models = [
    {
        "name": "vqgan_szch.pth",
        "file_id": "1L6M_2AoOgtPhi1B-klUKFUruyk__-_E-",
        "description": "VQGAN - SZCH-X-Rays",
    },
    {
        "name": "unet_szch.pth",
        "file_id": "1mrFJZ3M7aECe3weZvYF5G-DJYPc16zAi",
        "description": "UNet/LCM - SZCH-X-Rays",
    },
    {
        "name": "vqgan_jsrt.pth",
        "file_id": "1sJv4iyUGo2NHMUOscWBtON1Rd7jm22QF",
        "description": "VQGAN - JSRT",
    },
    {
        "name": "unet_jsrt.pth",
        "file_id": "1d1DmdHy84vWHGJs-74pMKRKfY66_4Dmx",
        "description": "UNet/LCM - JSRT",
    },
]

for model in models:
    output_path = output_dir / model["name"]

    if output_path.exists() and output_path.stat().st_size > 0:
        print(
            f"[Skip] {model['description']} already exists: "
            f"{output_path} ({output_path.stat().st_size:,} bytes)",
            flush=True,
        )
        continue

    print("=" * 70, flush=True)
    print(f"[Download] {model['description']}", flush=True)
    print(f"[Download] Google Drive ID: {model['file_id']}", flush=True)
    print(f"[Download] Output: {output_path}", flush=True)

    result = gdown.download(
        id=model["file_id"],
        output=str(output_path),
        quiet=False,
        resume=True,
    )

    if (
        result is None
        or not output_path.exists()
        or output_path.stat().st_size == 0
    ):
        raise RuntimeError(
            f"Download failed for {model['description']}: {output_path}"
        )

    print(
        f"[Done] {output_path.name}: "
        f"{output_path.stat().st_size:,} bytes",
        flush=True,
    )
PY

# ---------------------------------------------------------------------------
# Verify expected files before calculating checksums
# ---------------------------------------------------------------------------
EXPECTED_CHECKPOINTS=(
    "vqgan_szch.pth"
    "unet_szch.pth"
    "vqgan_jsrt.pth"
    "unet_jsrt.pth"
)

for checkpoint in "${EXPECTED_CHECKPOINTS[@]}"; do
    checkpoint_path="${CHECKPOINT_DIR}/${checkpoint}"

    if [[ ! -s "${checkpoint_path}" ]]; then
        echo "[Error] Missing or empty checkpoint: ${checkpoint_path}"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Record local checksums for transfer/integrity tracking.
#
# The repository does not publish reference SHA256 values, so these hashes are
# local integrity records rather than comparisons with official checksums.
# ---------------------------------------------------------------------------
cd "${CHECKPOINT_DIR}"

sha256sum \
    vqgan_szch.pth \
    unet_szch.pth \
    vqgan_jsrt.pth \
    unet_jsrt.pth \
    > SHA256SUMS.local.txt

echo
echo "======================================================================"
echo "Downloaded checkpoints"
echo "======================================================================"

ls -lh "${CHECKPOINT_DIR}"

echo
echo "Local SHA256 checksums:"
cat "${CHECKPOINT_DIR}/SHA256SUMS.local.txt"

echo
echo "Storage usage:"
du -sh "${ROOT}"

echo
echo "Conda environment:"
conda run \
    --no-capture-output \
    --prefix "${TOOLS_ENV}" \
    python -c \
    'import sys, gdown; print("Python:", sys.executable); print("gdown:", gdown.__version__)'

echo
echo "Finished: $(date --iso-8601=seconds)"
echo "======================================================================"