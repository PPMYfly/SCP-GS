# Environment for retained Phase8.8 source

Use Linux x86-64, Python 3.10, matching CUDA-enabled PyTorch and CUDA toolkit,
C/C++ compilers, CMake and `requirements.txt`. Local testing used Python 3.10.19
and PyTorch 2.0.1+cu118. Rebuild extensions for the active Python/PyTorch ABI.

Scripts accept `PCGS_ENV_ROOT` and `PCGS_PYTHON`; their default environment is
the current Python environment. The original build expects environment compiler
wrappers; configure `CC`, `CXX`, `CUDA_HOME` and `TORCH_CUDA_ARCH_LIST` for your
toolchain. The optional `PCGS_BUILD_FULL=1` branch requires additional unbundled
renderer dependencies and is not supported by this source selection.

`build.sh` builds the included arithmetic/grid CUDA extensions and TMC13, then
`verify_build.py` checks their imports. `test_cpu.sh` runs 90 host-side tests;
their imports require the built extensions. `test_cuda.sh` additionally requires
a usable GPU and runs three CUDA tests. No native renderer test is included.

The complete codec/model/SH selector source is present. Native rendering,
full feedback optimization, research batch runners and real assets are absent.
No clean-machine rebuild or trained scene codec round trip was repeated for this
selection; local tests used existing compiled extensions from identical sources.
