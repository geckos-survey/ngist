#!/usr/bin/env bash
# Run all tests (unit tests + pipeline + integration check).
# Dev runs explicitly use the pipeline at NGIST_PIPELINE_PATH and the conda env ngist-dev.
set -e

# Pipeline path for dev runs (explicit path or script dir)
SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export NGIST_PIPELINE_PATH="${NGIST_PIPELINE_PATH:-/arc/home/thbrown/mauve/dev/gist-geckos}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ngist-dev}"

# Resolve to absolute path; fall back to script dir if explicit path unavailable
RESOLVED="$(cd "$NGIST_PIPELINE_PATH" 2>/dev/null && pwd)" || true
if [[ -n "$RESOLVED" && -d "$RESOLVED" ]]; then
  NGIST_PIPELINE_PATH="$RESOLVED"
else
  NGIST_PIPELINE_PATH="$SCRIPT_ROOT"
fi
cd "$NGIST_PIPELINE_PATH"

# Source conda so 'conda activate' works in this shell (required in subshells)
_conda_sh=""
if [[ -n "$CONDA_EXE" ]]; then
  _conda_sh="$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
fi
[[ -z "$_conda_sh" || ! -f "$_conda_sh" ]] && _conda_sh="/opt/conda/etc/profile.d/conda.sh"
[[ ! -f "$_conda_sh" ]] && _conda_sh="$HOME/miniconda3/etc/profile.d/conda.sh"
[[ ! -f "$_conda_sh" ]] && _conda_sh="$HOME/anaconda3/etc/profile.d/conda.sh"
if [[ -f "$_conda_sh" ]]; then
  # shellcheck source=/dev/null
  source "$_conda_sh"
else
  echo "Error: conda not found. Set CONDA_EXE or install Miniconda/Anaconda." >&2
  exit 1
fi

if ! conda activate "$CONDA_ENV_NAME" 2>/dev/null; then
  echo "Creating conda env $CONDA_ENV_NAME from environment.yml..."
  conda env create -f "$NGIST_PIPELINE_PATH/environment.yml"
  conda activate "$CONDA_ENV_NAME"
fi

# Ensure pipeline is installed from this path
pip install -e "$NGIST_PIPELINE_PATH" --quiet --no-deps 2>/dev/null || pip install -e "$NGIST_PIPELINE_PATH"

echo "Using pipeline: $NGIST_PIPELINE_PATH (conda: $CONDA_ENV_NAME)"
echo ""

echo "=== Unit tests (EW / KIN continuum) ==="
pytest tests/test_emission_ew.py -v

echo ""
echo "=== Pipeline (NGC 0000 example) ==="
ngistPipeline --config=./.github/workflows/tests/gistTutorial/configFiles/MasterConfig.yaml \
  --default-dir=./.github/workflows/tests/gistTutorial/configFiles/defaultDir_ubuntu

echo ""
echo "=== Integration check (output files + EW) ==="
python ./.github/workflows/tests/check_outputs.py

echo ""
echo "All tests passed."
