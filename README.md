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

- `data`: training and inference paths (`train_dir`, `infer_dir`).
- `model`, `train`, and `loss`: model settings, device, epochs, learning rate,
  output directory, and loss weights for your data.
- `infer`: checkpoint and output paths.

If a full-graph checkpoint contains a saved configuration, inference reads its
input and output paths from that configuration, even if those paths have since
changed in the YAML. To use new paths, pass the corresponding options:

| Path to change | Command-line option | Script |
| --- | --- | --- |
| Input H5AD file or directory | `--input_dir` | `infer.py`, `tri_infer.py` |
| Output `.pt` result file | `--output_path` | `infer.py`, `tri_infer.py` |
| Output H5AD directory | `--output_dir_h5ad` | `infer.py` |
| Output H5AD file | `--output_path_h5ad` | `tri_infer.py` |

The default backend is `full`; select `--backend tiled` for tiled inputs.
View the available options with:

```bash
python train.py --help
python infer.py --help
python tri_train.py --help
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

Download the processed inputs, trained checkpoints, and configurations from
[Hugging Face](https://huggingface.co/datasets/GAO612/SAMI).
Place the downloaded dataset folders under `data/sami/` in this repository,
for example, `data/sami/Xenium_Renal_Carcinoma/`.

Run all examples below from `data/` so the saved `sami/...` paths resolve
correctly. From the repository root:

```bash
cd data
```

### Inference with released checkpoints

```bash
# Mouse embryonic brain: RNA + ATAC, full graph
python ../infer.py \
  --config sami/Mouse_Embryonic_Brain/E11_0-S1/default.yaml \
  --checkpoint sami/Mouse_Embryonic_Brain/E11_0-S1/best.pt \
  --input_dir sami/Mouse_Embryonic_Brain/E11_0-S1/processed_E11_0-S1.h5ad \
  --output_path outputs/brain_E11_0-S1_embeddings.pt \
  --output_path_h5ad outputs/brain_E11_0-S1_infer_results.h5ad

# Xenium renal carcinoma: RNA + protein, tiled
python ../infer.py --backend tiled \
  --checkpoint sami/Xenium_Renal_Carcinoma/best.pt \
  --output outputs/renal_infer_results.h5ad

# CRC Visium HD: RNA + H&E, tiled
python ../infer.py --backend tiled \
  --checkpoint sami/CRC_VisiumHD/P1CRC/best.pt \
  --output outputs/P1CRC_infer_results.h5ad
```

CRC inference uses a checkpoint from each sample's directory. Replace `P1CRC`
in both paths with `P2CRC`, `P5CRC`, `P3NAT`, or `P5NAT` to process another sample.

### Training with released configurations

Use the dataset's `default.yaml` and set `train.output_dir` to a separate output
directory for each run. Continue from the same `data/` directory:

```bash
# Mouse embryonic brain
python ../train.py \
  --config sami/Mouse_Embryonic_Brain/E11_0-S1/default.yaml

# Xenium renal carcinoma
python ../train.py --backend tiled \
  --config sami/Xenium_Renal_Carcinoma/default.yaml

# CRC Visium HD: joint training across all five samples
python ../train.py --backend tiled \
  --config sami/CRC_VisiumHD/default.yaml
```

## Preparing H&E features

Follow the [GPFM repository](https://github.com/birkhoffkiki/GPFM) instructions
to download the pretrained weights and save them as `checkpoints/GPFM.pth`.
Prepare an 8-bit H&E image and an `(N, 2)` array of aligned `(x, y)` pixel
coordinates measured from the image's top-left corner.

```python
import numpy as np
from preprocessing.image_features import extract_gpfm_features

extract_gpfm_features(
    image_path="data/tissue.tif",
    pixel_coords=np.load("data/pixel_coords.npy"),
    output_h5="data/he_features.h5",
    checkpoint="checkpoints/GPFM.pth",
    patch_size=224,
    device_name="cuda",
)
```

The output stores an `(N, 1024)` feature matrix under `path_feat`, in the same
order as the coordinates. For full-graph training, copy these features into
the H5AD `obsm` entry selected by your configuration. For tiled training, place
`path_feat` in the processed HDF5 file alongside the other modality features.
