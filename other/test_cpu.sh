#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_ROOT="${PCGS_ENV_ROOT:-$(python3 -c 'import sys; print(sys.prefix)')}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PCGS_PYTHON:-${ENV_ROOT}/bin/python}" -m unittest discover -s "${ROOT}/tests" -p "test_phase*.py" -v
