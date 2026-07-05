#!/usr/bin/env bash
# =============================================================================
# setup_dirs.sh
# Sets up the working directory structure for the Perch Hoplite marine
# bioacoustics pipeline on spark-ae0e (or spark-0626 — see notes at bottom).
#
# Strategy:
#   - Fast local NVMe (/home/duane/perch_work/) for active embedding databases
#   - Shared NFS (/mnt/PAM_Analysis/duane_scratch/perch_hoplite/) for
#     models, results, labels, queries, and finished databases
#   - Raw audio is read from /mnt/PAM_Archive/ — nothing written there
#
# Usage:
#   chmod +x setup_dirs.sh
#   ./setup_dirs.sh          # no sudo needed — all dirs owned by current user
#
# To run for a different user:
#   USER_OVERRIDE=otheruser ./setup_dirs.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit these if running on spark-0626 or another system
# ---------------------------------------------------------------------------

INTENDED_HOST="spark-ae0e"

# The user who will own and run the pipeline.
# Defaults to whoever runs this script.
PIPELINE_USER="${USER_OVERRIDE:-$(whoami)}"
PIPELINE_HOME="/home/${PIPELINE_USER}"

# Local NVMe working root — fast storage for active embedding databases.
# 3.3 TB free on spark-ae0e /dev/nvme0n1p2
LOCAL_WORK="${PIPELINE_HOME}/perch_work"

# NFS shared root — persistent storage visible to all MBARI DGX nodes.
# Lives inside duane_scratch which already exists on PAM_Analysis.
NFS_WORK="/mnt/PAM_Analysis/duane_scratch/perch_hoplite"

# Raw audio — read only, already exists. Listed here for reference only;
# this script does NOT create or modify anything under PAM_Archive.
AUDIO_ROOT="/mnt/PAM_Archive"

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; YEL='\033[1;33m'; GRN='\033[0;32m'
CYN='\033[0;36m'; BLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${CYN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GRN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YEL}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERR ]${NC}  $*" >&2; }
head()  { echo -e "\n${BLD}$*${NC}"; }

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

CURRENT_HOST="$(hostname -s)"
if [[ "$CURRENT_HOST" != "$INTENDED_HOST" ]]; then
  warn "This script was written for ${INTENDED_HOST}."
  warn "Current hostname: ${CURRENT_HOST}"
  warn "See the notes at the bottom before proceeding on a different system."
  echo ""
fi

# Check NFS mounts are actually up before writing anything
for nfs_path in "/mnt/PAM_Analysis" "/mnt/PAM_Archive"; do
  if ! mountpoint -q "$nfs_path"; then
    error "${nfs_path} is not mounted. Is thalassa.shore.mbari.org reachable?"
    error "Check with:  mount | grep PAM"
    exit 1
  fi
done
ok "NFS mounts verified (PAM_Analysis, PAM_Archive)"

# Check duane_scratch already exists (it does — 22G of existing work)
if [[ ! -d "/mnt/PAM_Analysis/duane_scratch" ]]; then
  error "/mnt/PAM_Analysis/duane_scratch does not exist."
  error "This script expects it to already be present."
  exit 1
fi
ok "duane_scratch confirmed"

# Check local NVMe has enough space (require at least 500 GB free)
AVAIL_KB=$(df -k "${PIPELINE_HOME}" | awk 'NR==2 {print $4}')
AVAIL_GB=$(( AVAIL_KB / 1024 / 1024 ))
if [[ $AVAIL_GB -lt 500 ]]; then
  warn "Less than 500 GB free on local NVMe (${AVAIL_GB} GB available)."
  warn "Embedding databases can be large. Proceed with caution."
else
  ok "Local NVMe space: ${AVAIL_GB} GB free"
fi

echo ""
info "Pipeline user : ${PIPELINE_USER}"
info "Local work    : ${LOCAL_WORK}"
info "NFS work      : ${NFS_WORK}"
info "Audio root    : ${AUDIO_ROOT}  (read-only, not modified)"
echo ""

# ---------------------------------------------------------------------------
# Directories to create
# ---------------------------------------------------------------------------

# Format: "path|description"
# All paths are absolute.

LOCAL_DIRS=(
  "${LOCAL_WORK}|root working directory on fast local NVMe"
  "${LOCAL_WORK}/db|active Hoplite databases during embedding (fast writes)"
  "${LOCAL_WORK}/tmp|scratch space for sharded audio during processing"
  "${LOCAL_WORK}/logs|local run logs"
)

NFS_DIRS=(
  "${NFS_WORK}|perch_hoplite root on shared NFS"
  "${NFS_WORK}/db|finished Hoplite databases (synced here after embedding)"
  "${NFS_WORK}/models|trained linear classifiers (.pt + .metrics.json)"
  "${NFS_WORK}/results|inference CSVs and score histogram PNGs"
  "${NFS_WORK}/labels|annotation CSVs (Raven Pro exports, manual labels)"
  "${NFS_WORK}/queries|short reference audio clips, one per sound class"
  "${NFS_WORK}/queries/cetaceans|orca, dolphin, whale query clips"
  "${NFS_WORK}/queries/anthropogenic|boat, ROV, sonar query clips"
  "${NFS_WORK}/logs|persistent logs (copied/symlinked from local after runs)"
)

# ---------------------------------------------------------------------------
# Create directories
# ---------------------------------------------------------------------------

create_dirs() {
  local label="$1"; shift
  local dirs=("$@")
  local created=0; local existing=0

  head "── ${label} ──"
  for entry in "${dirs[@]}"; do
    local path="${entry%%|*}"
    local desc="${entry##*|}"
    if [[ -d "$path" ]]; then
      ok "exists   ${path}  (${desc})"
      (( existing++ )) || true
    else
      mkdir -p "$path"
      ok "created  ${path}  (${desc})"
      (( created++ )) || true
    fi
    chmod 755 "$path"
  done
  echo "         Created: ${created}  Already existed: ${existing}"
}

create_dirs "Local NVMe  (fast, spark-ae0e only)"  "${LOCAL_DIRS[@]}"
create_dirs "NFS shared  (persistent, all nodes)"  "${NFS_DIRS[@]}"

# ---------------------------------------------------------------------------
# Write a .env file with path constants for the pipeline programs
# ---------------------------------------------------------------------------

ENV_FILE="${NFS_WORK}/.perch_env"
cat > "$ENV_FILE" << EOF
# Perch Hoplite pipeline paths — generated by setup_dirs.sh
# Source this file or pass these as CLI arguments.
# Generated: $(date)  Host: ${CURRENT_HOST}  User: ${PIPELINE_USER}

# Local fast storage (NVMe) — use for --db-dir during phase1_embed.py
PERCH_LOCAL_DB="${LOCAL_WORK}/db"
PERCH_LOCAL_TMP="${LOCAL_WORK}/tmp"

# Shared NFS storage — use for --db-dir during phase2_classify.py
# (after syncing the finished DB from local)
PERCH_NFS_DB="${NFS_WORK}/db"
PERCH_MODELS="${NFS_WORK}/models"
PERCH_RESULTS="${NFS_WORK}/results"
PERCH_LABELS="${NFS_WORK}/labels"
PERCH_QUERIES="${NFS_WORK}/queries"

# Raw audio (read-only)
PERCH_AUDIO_ROOT="${AUDIO_ROOT}"

# Sync command: run this after phase1_embed.py completes to move the
# finished database to NFS for Phase 2 and for spark-0626 access.
# Usage: bash -c "\$(grep PERCH_SYNC_CMD ${ENV_FILE} | cut -d= -f2-)"
# PERCH_SYNC_CMD="rsync -av --progress \${PERCH_LOCAL_DB}/<dataset>/ \${PERCH_NFS_DB}/<dataset>/"
EOF

ok "Environment file written: ${ENV_FILE}"

# ---------------------------------------------------------------------------
# Write a sync helper script
# ---------------------------------------------------------------------------

SYNC_SCRIPT="${LOCAL_WORK}/sync_db_to_nfs.sh"
cat > "$SYNC_SCRIPT" << 'EOF'
#!/usr/bin/env bash
# sync_db_to_nfs.sh
# Copies a finished Hoplite database from local NVMe to shared NFS.
# Run this after phase1_embed.py completes.
#
# Usage:  ./sync_db_to_nfs.sh <dataset_name>
# Example: ./sync_db_to_nfs.sh saipan_A_06

set -euo pipefail

DATASET="${1:?Usage: $0 <dataset_name>}"
LOCAL_DB="$(dirname "$0")/db/${DATASET}"
NFS_DB="/mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/${DATASET}"

if [[ ! -d "$LOCAL_DB" ]]; then
  echo "ERROR: Local DB not found: ${LOCAL_DB}"
  exit 1
fi

echo "Syncing: ${LOCAL_DB}"
echo "     to: ${NFS_DB}"
mkdir -p "$NFS_DB"
rsync -av --progress "${LOCAL_DB}/" "${NFS_DB}/"
echo ""
echo "Sync complete. DB is now available on NFS at:"
echo "  ${NFS_DB}"
echo ""
echo "You can now run phase2_classify.py with:"
echo "  --db-dir ${NFS_DB}"
EOF
chmod +x "$SYNC_SCRIPT"
ok "Sync helper written: ${SYNC_SCRIPT}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
info "============================================================"
info "Setup complete for ${CURRENT_HOST}"
info "============================================================"
echo ""
echo -e "  ${BLD}Phase 1 embedding (fast local NVMe):${NC}"
echo "    python3 phase1_embed.py \\"
echo "        --dataset-name <name> \\"
echo "        --audio-dir ${AUDIO_ROOT}/<year>/<deployment> \\"
echo "        --file-glob '*.flac' \\"
echo "        --db-dir ${LOCAL_WORK}/db/<name> \\"
echo "        --model perch_v2"
echo ""
echo -e "  ${BLD}After embedding — sync DB to NFS:${NC}"
echo "    ${SYNC_SCRIPT} <name>"
echo ""
echo -e "  ${BLD}Phase 2 classify (from NFS):${NC}"
echo "    python3 phase2_classify.py search \\"
echo "        --db-dir ${NFS_WORK}/db/<name> \\"
echo "        --query-audio ${NFS_WORK}/queries/cetaceans/orca_call.wav \\"
echo "        --query-label orca_call \\"
echo "        --serve --port 7860"
echo ""
echo -e "  ${BLD}Browser on your laptop:${NC}"
echo "    http://134.89.11.107:7860"
echo ""

# ---------------------------------------------------------------------------
# Notes for spark-0626 or any other system
# ---------------------------------------------------------------------------
# To run this script on spark-0626:
#
#   1. INTENDED_HOST (line ~30): change to "spark-0626" to suppress the warning.
#
#   2. LOCAL_WORK (line ~36): /home/duane/perch_work will be created on
#      spark-0626's local NVMe. Check its size first:
#        df -h /home
#      If /home is small on spark-0626, change LOCAL_WORK to a path on a
#      larger local disk (e.g. /scratch/duane/perch_work).
#
#   3. NFS_WORK does NOT need to change — spark-0626 mounts the same
#      thalassa.shore.mbari.org NFS shares at the same paths, so the
#      shared databases, models, and results are automatically visible.
#
#   4. The IP address in the summary (134.89.11.107) is spark-ae0e's address.
#      On spark-0626, find the correct address with:
#        ip addr show | grep "inet " | grep -v "127\|169\.254\|172\."
#      and update your browser URL accordingly.
#
#   5. No other changes needed.
# ---------------------------------------------------------------------------
