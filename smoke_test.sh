#!/usr/bin/env bash
set -euo pipefail

smoke_script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

exec "${PYTHON_BIN:-python}" -u "$smoke_script_directory/smoke_test.py" "$@"
