# Dataset sources and scene scope

The retained experiment manifest is [`configs/scenes_43.csv`](../configs/scenes_43.csv).
It contains 43 identities across four datasets:

| Dataset | Local identifier | Identities | Original source |
| --- | --- | ---: | --- |
| 360Roam | `360Roam_dataset` | 11 | [Author project](https://huajianup.github.io/research/360Roam/) |
| OmniPhotos | `OmniPhotos` | 11 | [University of Bath archive](https://researchdata.bath.ac.uk/948/) |
| OmniScenes | `OmniScenes` | 9 | [PICCOLO dataset instructions](https://github.com/82magnolia/piccolo#downloading-omniscenes-update-new-scenes-added) |
| EgoNeRF–Ricoh360 | `Ricoh360` | 12 | [EgoNeRF author repository](https://github.com/changwoonchoi/EgoNeRF) |

Download assets from their original providers. This release bundles one copied OmniPhotos / Field example: frozen tensors,
a matched checkpoint, compressed stream and 13 existing codec-validation
image pairs. See `../demo/OmniPhotos_Field/README.md`. Other scene assets, camera
poses and full research split records are not bundled. Dataset-derived assets
remain subject to their original provider terms.

The manifest preserves local experiment names and case. Suffixes such as `_new`
and `_new2` do not establish a new capture; their exact mapping to raw archive
folders has not been verified. The manifest alone is not a complete data
preparation or reproduction recipe.
