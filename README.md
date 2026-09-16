# SAMI

SAMI (Spatially-Aware Multi-Modal Integration) is a framework for integrating
histological features, spatial omics profiles, and tissue spatial relationships
to learn unified spot- or cell-level representations. It combines
neighborhood-restricted cross-modal attention, spatial neighborhood aggregation,
and layer-wise aggregation to capture complementary information across modalities
alongside local detail and broader tissue structure. Across diverse tissues,
platforms, and spatial resolutions, SAMI supports spatial domain identification
and downstream analyses of cellular heterogeneity and functional microenvironments.

## Installation

SAMI uses Python 3.10 and PyTorch 2.9.1. Install all dependencies using the
provided [environment.yml](environment.yml):

```bash
conda env create -f environment.yml
conda activate sami
```

## Training and inference

For full-graph training and inference, prepare an `.h5ad` file with the following
entries. All feature matrices must follow the same spot or cell order.

| H5AD entry | Contents | Configuration key |
| --- | --- | --- |
| `obs['x_array']`, `obs['y_array']` | Spatial coordinates | Fixed names |
| `obsm['omics1_feat']` | First-modality features, shape `(N, D1)` | `data.omics1_obsm_key` |
| `obsm['omics2_feat']` | Second-modality features, shape `(N, D2)` | `data.omics2_obsm_key` |
| `obsm['omics3_feat']` | Third-modality features, shape `(N, D3)`; three-modality runs only | `data.omics3_obsm_key` |

The `omics*_feat` names are examples; set the configuration keys to match your
`obsm` entries. Reconstruction targets can use the same features or separate
`obsm` entries selected by `data.omics*_target_obsm_key`.

Tiled execution uses a manifest, processed HDF5 files, and NPZ tiles. For prepared
inputs and matching configurations, see [Using the released data](#using-the-released-data).

Start from [configs/default.yaml](configs/default.yaml) for two modalities or
[configs/tri_modal_default.yaml](configs/tri_modal_default.yaml) for three
modalities. Both templates use the full-graph backend. Update:

- `data`: training, validation, and inference paths (`train_dir`, `val_dir`,
  `infer_dir`), plus the feature and target keys above.
- `model`, `train`, and `loss`: model settings, device, epochs, learning rate,
  output directory, and loss weights for your data.
- `infer`: checkpoint and output paths.

Relative paths resolve from the working directory. Dimensions set to `null`
are inferred from the first sample; keep `data.batch_size: 1`.

The default backend is `full`; select `--backend tiled` for tiled inputs.
View the available options with:

```bash
python train.py --help
python infer.py --backend tiled --help
python tri_train.py --backend tiled --help
python tri_infer.py --help
```

After updating the templates, run from the repository directory:

```bash
# Two modalities
python train.py --config configs/default.yaml
python infer.py --config configs/default.yaml

# Three modalities
python tri_train.py --config configs/tri_modal_default.yaml
python tri_infer.py --config configs/tri_modal_default.yaml
```

## Using the released data

Paths remain relative to the process working directory, not the YAML or code
directory. For the released datasets, run from the download root containing
`sami/Mouse_Embryonic_Brain`, `sami/Xenium_Renal_Carcinoma`, and
`sami/CRC_VisiumHD`. The code can live in a separate directory and does not
require the original research repository on `PYTHONPATH`.

The released Xenium renal experiment uses RNA+protein. Examples of training
with a released configuration and inference with released weights:

```bash
SAMI_CODE=/path/to/sami-code
cd /path/to/hf_upload

python "$SAMI_CODE/train.py" --backend tiled \
  --config sami/Xenium_Renal_Carcinoma/default.yaml

python "$SAMI_CODE/infer.py" \
  --config sami/Mouse_Embryonic_Brain/E11_0-S1/default.yaml \
  --checkpoint sami/Mouse_Embryonic_Brain/E11_0-S1/best.pt \
  --input_dir sami/Mouse_Embryonic_Brain/E11_0-S1/processed_E11_0-S1.h5ad \
  --output_path outputs/brain_E11_0-S1_embeddings.pt \
  --output_path_h5ad outputs/brain_E11_0-S1_infer_results.h5ad

python "$SAMI_CODE/infer.py" --backend tiled \
  --checkpoint sami/Xenium_Renal_Carcinoma/best.pt \
  --output outputs/renal_infer_results.h5ad

python "$SAMI_CODE/infer.py" --backend tiled \
  --checkpoint sami/CRC_VisiumHD/P1CRC/best.pt \
  --output outputs/P1CRC_infer_results.h5ad
```

CRC contains five samples: P1CRC, P2CRC, P5CRC, P3NAT, and P5NAT. They share the
same learned model weights. Each sample's checkpoint contains the manifest
configuration for that sample; use it for single-sample inference. The root
checkpoint and root `default.yaml` reference the joint training manifest. Tile
inference expects one sample's cell index space, so use the sample checkpoints
to infer the five samples separately. To start joint training with the released
configuration from the same data root:

```bash
python "$SAMI_CODE/train.py" --backend tiled \
  --config sami/CRC_VisiumHD/default.yaml
```

Check the configuration's `train.output_dir` before starting a training run.
Use a writable experiment location when training from a read-only data mount.
The shared CRC RNA transform records the selected genes and PCA parameters;
model execution reads already transformed features from each processed HDF5.

## Preparing inputs and clustering

The public preprocessing package contains reusable image feature extraction and
spatial graph partitioning. Dataset-specific Xenium and CRC preparation
pipelines are not included. Released inputs already contain their prepared
features and tiles, so they do not require these steps.

`preprocessing.image_features.extract_gpfm_features` accepts an image and an
`(N, 2)` array of pixel coordinates in `(x, y)` order, measured from the top-left
corner of that image. A single point is `[[x, y]]`. It extracts a square patch
around each point and uses GPFM to write an `(N, 1024)` feature matrix to HDF5.
The image reader accepts 8-bit RGB, RGBA, or grayscale tissue images, including
TIFF, PNG, and JPEG, and reads the first page by default. It does not infer
registration or pixel scale; convert physical coordinates or array indices
into image pixels beforehand.
`transform_coordinates` can apply a supplied 3x3 matrix in that direction.

For example, from the code directory with the image and GPFM weights available:

```python
import h5py
import numpy as np
from preprocessing.image_features import extract_gpfm_features

extract_gpfm_features(
    image_path="data/tissue.tif",
    pixel_coords=np.asarray([[1000, 2000]], dtype=np.float32),
    output_h5="features.h5",
    checkpoint="checkpoints/GPFM.pth",
    patch_size=224,
    device_name="cuda",
)
with h5py.File("features.h5", "r") as handle:
    features = handle["path_feat"][:]  # Shape: (1, 1024).
```

Choose the square crop size with `patch_size`, a positive integer measured in
source-image pixels. The default, `patch_size=224`, crops 224x224 directly without
resizing; other sizes, such as 112, 256, or 512, crop the requested area and
then resize it to 224x224 for GPFM. Image boundaries are padded with zeros, and
`path_valid_fraction` records how much of each patch was inside the image.
Feature rows follow the input coordinate order. Extraction resumes from
`path_feat_completed_rows`; reuse the same output only with the same image,
coordinates, checkpoint, and settings. Changing the crop size requires a new
output file; existing features with a different or unrecorded crop size cannot
be resumed. Create the output directory first if
using a nested path. An image-only feature file does not include the coordinates
and omics features required by the tiled data loader.

`preprocessing.graph_partition` builds a spatial neighbor graph from an HDF5
`coords` dataset of shape `(N, 2)`, partitions its nodes, and adds neighboring
context to each tile. It writes graph and tile NPZ files plus `manifest.json`.
To use that manifest for training, the HDF5 must also contain the modality
features required by the selected `data.mode`. RNA+H&E and three-modality
loaders require all image feature rows to be complete.

```bash
python -m preprocessing.graph_partition \
  --processed-h5 data/processed.h5 \
  --graph-path data/graphs/spatial_knn.npz \
  --tile-dir data/tiles --sample-id sample
```

For mclust clustering, use `utils.cluster_utils.mclust_R`. Spatial labels can
be refined with `utils.cluster_utils.refine_label`.

## Tests

After creating and activating the environment, run from the code directory:

```bash
python -m pip install pytest==9.1.1
python -m pytest tests
```

Tests use small synthetic inputs and temporary output directories. They cover
tile boundaries and losses as well as the command routing and model workflows.
