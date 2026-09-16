# SAMI

SAMI learns spatial multi-omics embeddings with full-graph and tiled training
backends. This repository contains model code, training and inference entrypoints,
input preparation, and mclust clustering utilities. Datasets, pretrained
weights, and experiment outputs are distributed separately.

## Installation

Use Python 3.10 (reference: Python 3.10.19 in `gst_py310`), with PyTorch
`2.9.1+cu126`, torchvision `0.24.1+cu126`, and PyTorch Geometric `2.7.0`.
`environment.yml` contains Python, the runtime Python packages, and the R and
libvips dependencies for clustering and image preprocessing. Conda installs the
system dependencies before installing the packages in the `pip` section.
Transitive dependencies, such as Scanpy's matplotlib dependency, are installed
automatically.

```bash
conda env create -f environment.yml
conda activate sami
```

The YAML pins the public PyTorch and torchvision versions. To select the
reference CUDA 12.6 builds explicitly, run the following after activation:

```bash
python -m pip install torch==2.9.1+cu126 torchvision==0.24.1+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
```

Use the [PyTorch installer](https://pytorch.org/get-started/locally/) for a
different platform. Existing PyTorch installations that satisfy the pinned
versions can be retained. The existing reference `gst_py310` environment can
also be activated directly.
When a Linux GPU container needs explicit driver-library search paths, apply
these settings before launching Python:

```bash
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The container runtime must also expose the GPU; these settings alone do not
grant device access.

GPFM image feature extraction uses torchvision, timm, and pyvips from
`environment.yml`, which also installs the `libvips` system library. Obtain
GPFM weights separately and place them at `checkpoints/GPFM.pth` or supply an
explicit checkpoint path. These weights are only needed when extracting image
features; released inputs already contain those features.

The mclust helpers use R 4.3.3 and mclust 6.1.1, both included in the environment
file. Activate the environment before running them so that `R` is on `PATH`.
Calling an environment's Python executable alone does not activate its R
installation.

Runtime versions are matched to `gst_py310`; pyvips is also included for the
released image preprocessing code.

## Project layout

```text
train.py / infer.py                 Shared command dispatch
tri_train.py / tri_infer.py         Dedicated three-modality commands
engines/full_graph/                Original whole-sample execution
engines/tiled/                     Tile training and stitched inference
datasets/                         Full-graph and tile data loaders
models/                           Encoders, attention, fusion, decoders
losses/                           Reconstruction and representation losses
utils/                            Graph, configuration, and clustering helpers
configs/                          Generic dual- and three-modality full-graph examples
preprocessing/                    Spatial graph tiles and GPFM image features
tests/                            Regression and smoke tests
```

The two backends have separate training loops, graph construction, loss
handling, and inference outputs. The root commands select a backend and forward
its arguments. Tiled data loading, attention, models, and losses live alongside
their full-graph counterparts in the shared directories. Existing model member
names and checkpoint parameter keys are retained.

## Training and inference

`--backend full` is the default. Pass `--backend tiled` explicitly for a tile
manifest. A configuration error does not cause automatic backend switching.
Use the help for the selected backend to see its own options:

```bash
python train.py --help
python infer.py --backend tiled --help
python tri_train.py --backend tiled --help
python tri_infer.py --help
```

| Commands | Full backend | Tiled backend |
| --- | --- | --- |
| `train.py`, `infer.py` | Whole-sample H5AD and single/dual-modality branches | RNA+protein, RNA+H&E, or RNA+protein+H&E tiles |
| `tri_train.py`, `tri_infer.py` | Dedicated three-modality model | Requires `data.mode: rna_protein_he` |

Full-graph H5AD inputs provide spatial coordinates in `obs['x_array']` and
`obs['y_array']`, with feature/target matrices in `obsm` selected by the
configuration's `*_obsm_key` fields.
Tiled inputs use a manifest, processed HDF5 files, and tile NPZ files with global
cell indices, neighborhoods, and core masks. Manifest paths must resolve from
the working directory, including paths inside every manifest record.

Example commands from the code directory, after preparing the referenced data:

```bash
python train.py --config configs/default.yaml
python tri_train.py --config configs/tri_modal_default.yaml
```

The two files in `configs/` are generic examples for `--backend full`. The dual
example defaults to `multimodal_fusion`; the three-modality example uses
`trimodal_fusion`. Adjust data paths and the `omics*_feat` example keys to match
your H5AD inputs. Input and target dimensions set to `null` are inferred from
the first sample. Hyperparameters, including loss weights, are starting values
to tune for your data. Dataset-specific configurations are distributed with the
released data; use those for reproduction and tiled training, as shown below.

Preserve model dimensions and architecture when reusing a checkpoint. In full
inference, the YAML locates the checkpoint, and its saved configuration takes
precedence when present; existing CLI overrides such as
`--input_dir` and output paths still apply. Tile inference reads its model and
manifest configuration from the checkpoint itself and accepts `--checkpoint`,
`--output`, and `--device`.

Full inference writes its PT result bundle to `infer.output_path`, which can be
overridden with `--output_path`. The dual template writes per-sample H5AD files
under `infer.output_dir_h5ad`; override that directory with `--output_dir_h5ad`.
When no H5AD output directory is configured, dual inference accepts
`--output_path_h5ad` for a single input file. The three-modality template uses
`infer.output_path_h5ad`, overridable with `--output_path_h5ad`. Tiled inference
stitches overlapping tiles by the original weighting rule and writes
an H5AD plus a `.qc.json` sidecar. Both provide the joint embedding in
`obsm['z']`. New inference generates embeddings; it does not regenerate the
paper's selected clustering labels.

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
