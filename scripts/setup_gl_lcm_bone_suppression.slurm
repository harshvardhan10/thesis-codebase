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
TOOLS_VENV="${ROOT}/download_tools_venv"

echo "======================================================================"
echo "GL-LCM setup"
echo "Host:             $(hostname)"
echo "Started:          $(date --iso-8601=seconds)"
echo "Root:             ${ROOT}"
echo "Repository:       ${REPO_DIR}"
echo "Checkpoint dir:   ${CHECKPOINT_DIR}"
echo "Proxy:            ${HTTPS_PROXY}"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Clone/update the official repository
# ---------------------------------------------------------------------------
if [[ -d "${REPO_DIR}/.git" ]]; then
    echo "[Git] Existing repository found; updating."
    git -C "${REPO_DIR}" fetch --all --prune
    git -C "${REPO_DIR}" pull --ff-only
else
    echo "[Git] Cloning official GL-LCM repository."
    git clone \
        "https://github.com/diaoquesang/GL-LCM.git" \
        "${REPO_DIR}"
fi

echo
echo "[Git] Current revision"
git -C "${REPO_DIR}" rev-parse HEAD
git -C "${REPO_DIR}" status --short

# ---------------------------------------------------------------------------
# Lightweight isolated environment used only for Google Drive downloads
# ---------------------------------------------------------------------------
if [[ ! -x "${TOOLS_VENV}/bin/python" ]]; then
    echo "[Setup] Creating download-tools virtual environment."
    python3 -m venv "${TOOLS_VENV}"
fi

"${TOOLS_VENV}/bin/python" -m pip install \
    --upgrade \
    pip \
    gdown

# ---------------------------------------------------------------------------
# Download all official released checkpoints.
#
# We retain both checkpoint pairs because external MIMIC/VinDr generalization
# must be evaluated with a small pilot before selecting one for full conversion.
# gdown resume=True preserves partially downloaded files when the job restarts.
# ---------------------------------------------------------------------------
"${TOOLS_VENV}/bin/python" - "${CHECKPOINT_DIR}" <<'PY'
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

    print(f"[Download] {model['description']}", flush=True)
    print(f"[Download] Google Drive ID: {model['file_id']}", flush=True)
    print(f"[Download] Output: {output_path}", flush=True)

    result = gdown.download(
        id=model["file_id"],
        output=str(output_path),
        quiet=False,
        resume=True,
    )

    if result is None or not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Download failed for {model['description']}")

    print(
        f"[Done] {output_path.name}: {output_path.stat().st_size:,} bytes",
        flush=True,
    )
PY

# ---------------------------------------------------------------------------
# Record local checksums for transfer/integrity tracking.
# The repository does not publish reference SHA256 values in its README.
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
cat "${CHECKPOINT_DIR}/SHA256SUMS.local.txt"
echo
du -sh "${ROOT}"
echo "Finished: $(date --iso-8601=seconds)"
echo "======================================================================"
