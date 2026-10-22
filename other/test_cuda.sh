#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_ROOT="${PCGS_ENV_ROOT:-$(python3 -c 'import sys; print(sys.prefix)')}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${ENV_ROOT}/lib:${ENV_ROOT}/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${PCGS_PYTHON:-${ENV_ROOT}/bin/python}" -m unittest discover -s "${ROOT}/tests" -p "test_phase*.py" -v
if ! "${PCGS_PYTHON:-${ENV_ROOT}/bin/python}" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "ERROR: CUDA integration tests were not run; NVIDIA CUDA is unavailable." >&2
  echo "Run ./test_cpu.sh for CPU-only checks. Full Phase8.8 verification requires ./test_cuda.sh to pass." >&2
  exit 3
fi
exec "${PCGS_PYTHON:-${ENV_ROOT}/bin/python}" -m unittest discover -s "${ROOT}/tests" -p "test_cuda_direct.py" -v
