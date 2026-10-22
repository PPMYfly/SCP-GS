# Panoramic 3D Gaussian Splatting Compression via Spherical RD-Guided SH Reallocation

关键代码及本次发布范围内的配套配置均已开源，包括熵模型、训练与编解码、SH 选择，以及 43 场景清单和源码中的默认参数配置。

This source selection contains the original Phase8.8 entropy model, training,
SH-level selection, encoding, decoding, integrity verification and tensor
comparison implementations. Algorithm functions and training methods are kept
in full; package/import paths and display names were adjusted for this release.

## Included implementation

- `pcgs/codec.py`: training, checkpoint loading, encoding, decoding and validation.
- `pcgs/model.py`: conditional entropy model and deployment training methods.
- `pcgs/sh_selector.py`: original learned SH selection and profile handling.
- Frozen input/output, geometry ordering, causal groups, hierarchical SH streams,
  deployment serialization, integrity checks and spherical reference coordinates.
- CUDA arithmetic/grid primitives, TMC13 geometry source, original tests and
  ERP image metrics for externally supplied rendered images.


## Build and basic workflow

Use Linux, Python 3.10, compatible CUDA/PyTorch and a C/C++ toolchain. See
`ENVIRONMENT.md`. Install a compatible CUDA-enabled PyTorch build first, then:

```bash
python3 -m pip install -r requirements.txt
./build.sh
./run.sh --help
./test_cpu.sh
./test_cuda.sh
```

The input must be a matching frozen Gaussian package prepared externally;
raw dataset images or arbitrary PLY files cannot be passed directly. Camera,
RD and SH profiles must match that source. With suitable prepared input:

```bash
./run.sh train --input /path/to/frozen_scene --checkpoint outputs/model.pt \
  --model-precision int8 --steps 4000 --batch-size 4096 \
  --quant-bits 10 --geometry-bits 20 --seed 0
./run.sh encode --input /path/to/frozen_scene --checkpoint outputs/model.pt \
  --output outputs/stream
./run.sh verify --input outputs/stream
./run.sh decode --input outputs/stream --output outputs/decoded
./run.sh validate --source /path/to/frozen_scene --decoded outputs/decoded
```

These are basic codec commands, not the paper's complete experiment recipe.
`learn-sh` is also available; see `./run.sh learn-sh --help` for its source and
importance-profile requirements. A completed real-scene example is bundled in `demo/OmniPhotos_Field/`;
see its README for the copied checkpoint, stream, input and existing image pairs.
It is a static historical result example. It is not required to run with this
source selection, and compatibility is not claimed.

## Remaining scope boundary

The native rendering/gradient pipeline, complete feedback controller and research
batch runners are outside the reviewed inventory and are not included. The Field demo includes one real frozen input, checkpoint, stream and
codec-validation image pairs. No other scene assets or full research split files
are bundled. The complete 43-scene paper results are not reproduced by this tree.

`configs/scenes_43.csv` lists 43 identities: 360Roam (11), OmniPhotos (11),
OmniScenes (9), Ricoh360 (12). See `docs/DATASETS.md` for original data sources.

## Validation

The retained test suite passed 93 tests locally, including three CUDA tests,
using previously compiled extensions from the same backend sources. The CLI
exposes `train`, `encode`, `decode`, `validate`, `verify` and `learn-sh`.
A clean-machine rebuild and a trained scene encode/decode round trip have not
been repeated for this selection. The real demo was copied from an existing
completed run without additional checks, as requested. These tests do not establish the paper's
reported image quality, stream size or speed. See `RELEASE_SCOPE.json`.

Historical protocol identifiers and metadata keys are preserved. Compatibility
with any particular historical checkpoint or bitstream remains unverified.
See `NOTICE.md` and component license files for dependency terms.

## Copied real-scene demo

[OmniPhotos / Field](demo/OmniPhotos_Field/README.md) provides one previously
completed SH-selection candidate run as a static illustration, without a
requirement to execute it using this repository. Existing WS-PSNR is 28.4695 dB over 13
codec-validation views, with a 4,806,960-byte stream. 
