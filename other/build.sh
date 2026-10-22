#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_ROOT="${PCGS_ENV_ROOT:-$(python3 -c 'import sys; print(sys.prefix)')}"
PYTHON="${PCGS_PYTHON:-${ENV_ROOT}/bin/python}"
export CUDA_HOME="${CUDA_HOME:-${ENV_ROOT}}"
export CC="${CC:-${ENV_ROOT}/bin/x86_64-conda-linux-gnu-gcc}"
export CXX="${CXX:-${ENV_ROOT}/bin/x86_64-conda-linux-gnu-g++}"
export CPATH="${ENV_ROOT}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
export PATH="${ENV_ROOT}/bin:/usr/bin:/bin"
export MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6+PTX}"

if ! "${PYTHON}" -c 'import numpy, plyfile, setuptools, torch' >/dev/null 2>&1; then
  echo "Missing Python build/runtime packages." >&2
  echo "Install a CUDA-enabled PyTorch build compatible with this host first, then run:" >&2
  echo "  ${PYTHON} -m pip install -r ${ROOT}/requirements.txt" >&2
  echo "See ${ROOT}/ENVIRONMENT.md for the standalone source-tree contract." >&2
  exit 2
fi

for module in arithmetic gridencoder; do
  (cd "${ROOT}/third_party/entropy_backend/extensions/${module}" && "${PYTHON}" setup.py build_ext --inplace --force)
done

if [[ "${PCGS_BUILD_FULL:-0}" == "1" ]]; then
  for module in simple-knn diff-gaussian-rasterization; do
    (cd "${ROOT}/third_party/entropy_backend/extensions/${module}" && "${PYTHON}" setup.py build_ext --inplace --force)
  done
fi

cmake -S "${ROOT}/third_party/mpeg-pcc-tmc13" -B "${ROOT}/third_party/mpeg-pcc-tmc13/build" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${ROOT}/third_party/mpeg-pcc-tmc13/build" --target tmc3 -j "${MAX_JOBS}"
"${PYTHON}" "${ROOT}/verify_build.py"
