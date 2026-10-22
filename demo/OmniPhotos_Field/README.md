# Existing result example: OmniPhotos / Field

关键代码及本次发布范围内的配套配置均已开源，包括熵模型、训练与编解码、SH 选择，以及 43 场景清单和源码中的默认参数配置。

This directory contains copied outputs from a completed local Phase8.8 A2C
SH-selection candidate run. It is a static example of an existing result.
It is not a runnable demo or a reproduction package for the uploaded code.
Compatibility between these assets and the uploaded engineering snapshot is
not required or claimed. No new training, decoding, rendering or checks were
performed for this copy after the user requested direct copying.

## Existing result

- Gaussian points: 355,410.
- Complete compressed stream: 4,806,960 bytes.
- Historical PSNR: 28.8088 dB; WS-PSNR: 28.4695 dB.
- Historical SSIM: 0.833409; LPIPS (Alex): 0.168293.
- Evaluated images: 13 codec-validation views, 1920 by 960 pixels.

These are historical metrics for this candidate and validation subset, not
formal paper test results, a final-method claim or a 43-scene average.

## Copied contents

- `renders/decoded/`: previously rendered compressed-scene images.
- `renders/reference/`: corresponding existing reference images.
- `stream/`: existing compressed files, copied unchanged.
- `checkpoint.pt`: existing trained checkpoint, copied unchanged.
- `input/`: original frozen tensor files needed by the codec input format.
- `historical_metrics.json`, `per_image_metrics.csv`,
  `historical_tensor_comparison.json`: previously recorded results.

One local camera-path entry was omitted from input metadata; metric JSON paths
were made relative. Tensor files, checkpoint weights and compressed files were
not retrained or recoded. There is no example execution script.
Dataset-derived assets remain subject to the original OmniPhotos provider terms
and are not relicensed as project source code.
