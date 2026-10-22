#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_ROOT="${PCGS_ENV_ROOT:-$(python3 -c 'import sys; print(sys.prefix)')}"
PYTHON="${PCGS_PYTHON:-${ENV_ROOT}/bin/python}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${ENV_ROOT}/lib:${ENV_ROOT}/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${PYTHON}" -m pcgs.cli "$@"

