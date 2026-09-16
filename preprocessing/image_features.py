"""Extract GPFM image-patch features at dataset-independent pixel coordinates."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Iterable, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F


GPFM_ARCHITECTURE = "vit_large_patch14_dinov2.lvd142m"
DEFAULT_GPFM_CKPT = Path("checkpoints/GPFM.pth")


def _validate_patch_size(patch_size: int) -> int:
    if isinstance(patch_size, bool) or not isinstance(patch_size, Integral) or patch_size <= 0:
        raise ValueError("patch_size must be a positive integer in source-image pixels")
    return int(patch_size)


def load_alignment_matrix(path: str | Path) -> np.ndarray:
    values = np.genfromtxt(path, delimiter=",", dtype=np.float64)
    values = values[np.isfinite(values)].reshape(-1)
    if values.size == 9:
        return values.reshape(3, 3)
    if values.size == 6:
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, :] = values.reshape(2, 3)
        return matrix
    raise ValueError(f"Expected 6 or 9 numeric alignment values in {path}, got {values.size}")


def transform_coordinates(coords: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 matrix mapping input coordinates into the target coordinate system."""
    coords = np.asarray(coords, dtype=np.float64)
    homogeneous = np.column_stack([coords, np.ones(coords.shape[0], dtype=np.float64)])
    transformed = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    denominator = transformed[:, 2:3]
    denominator[np.abs(denominator) < 1e-12] = 1.0
    return np.asarray(transformed[:, :2] / denominator, dtype=np.float32)


def build_gpfm_model(
    checkpoint: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    import timm

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = timm.create_model(
        GPFM_ARCHITECTURE,
        pretrained=False,
        img_size=224,
        init_values=1.0e-05,
    )
    state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and not all(
        isinstance(value, torch.Tensor) for value in state_dict.values()
    ):
        for key in ("model_state", "state_dict", "model", "model_state_dict"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unexpected GPFM checkpoint type: {type(state_dict)!r}")
    state_dict = {
        str(key).removeprefix("module."): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.requires_grad_(False)
    return model.to(device).eval()


def unwrap_model_output(value: object) -> torch.Tensor:
    if isinstance(value, (tuple, list)):
        value = value[0]
    if isinstance(value, dict):
        for key in ("feat", "features", "x", "emb", "embedding"):
            if key in value:
                value = value[key]
                break
        else:
            value = next(iter(value.values()))
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Unexpected GPFM output type: {type(value)!r}")
    if value.ndim != 2:
        raise ValueError(f"Expected GPFM output [B,D], got {tuple(value.shape)}")
    return value


class VipsPatchReader:
    """Read patch_size x patch_size pixels around each point, padding at boundaries."""

    def __init__(self, image_path: str | Path, patch_size: int = 224) -> None:
        self.patch_size = _validate_patch_size(patch_size)
        import pyvips

        self.pyvips = pyvips
        self.image = pyvips.Image.new_from_file(
            str(image_path),
            access="random",
        )
        if (
            self.image.format != "uchar"
            or self.image.bands not in (1, 3, 4)
            or self.image.interpretation == "cmyk"
        ):
            raise ValueError("Expected an 8-bit RGB, RGBA, or grayscale image")
        if self.image.bands == 4:
            self.image = self.image[:3]
        elif self.image.bands == 1:
            self.image = self.image.bandjoin([self.image, self.image])

    @property
    def size(self) -> Tuple[int, int]:
        return int(self.image.width), int(self.image.height)

    def read(self, x_center: float, y_center: float) -> tuple[np.ndarray, float]:
        half = self.patch_size / 2.0
        x0 = int(round(float(x_center) - half))
        y0 = int(round(float(y_center) - half))
        x1 = x0 + self.patch_size
        y1 = y0 + self.patch_size
        sx0 = max(0, x0)
        sy0 = max(0, y0)
        sx1 = min(self.image.width, x1)
        sy1 = min(self.image.height, y1)
        output = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        if sx1 <= sx0 or sy1 <= sy0:
            return output, 0.0
        region = self.image.crop(sx0, sy0, sx1 - sx0, sy1 - sy0)
        array = np.ndarray(
            buffer=region.write_to_memory(),
            dtype=np.uint8,
            shape=(region.height, region.width, region.bands),
        )
        ox = sx0 - x0
        oy = sy0 - y0
        output[oy : oy + region.height, ox : ox + region.width] = array[:, :, :3]
        valid_fraction = float(region.width * region.height) / float(
            self.patch_size * self.patch_size
        )
        return output, valid_fraction


def extract_gpfm_features(
    image_path: str | Path,
    pixel_coords: np.ndarray,
    output_h5: str | Path,
    checkpoint: str | Path = DEFAULT_GPFM_CKPT,
    dataset_name: str = "path_feat",
    patch_size: int = 224,
    batch_size: int = 256,
    feature_dim: int = 1024,
    device_name: str = "cuda",
    max_rows: int | None = None,
) -> None:
    """Write GPFM features for image pixel coordinates to an HDF5 file.

    ``pixel_coords`` has shape (N, 2), with (x, y) measured from the image's
    top-left corner: x is the column and y is the row. A single point is
    [[x, y]]. Map physical or array coordinates to image pixels before calling.
    The reader expects an 8-bit RGB, RGBA, or grayscale image; points outside
    the image are zero-padded and tracked by ``path_valid_fraction``.

    ``patch_size`` is a positive integer defining a square crop in source-image
    pixels. The default, 224, extracts 224x224 pixels without resizing. Other
    sizes, such as 112, 256, or 512, select a different image area
    and are resized to 224x224 before encoding with the GPFM checkpoint.
    Features are stored in coordinate order under
    ``dataset_name`` (normally ``path_feat``), with 1024 columns by default.
    The output directory must exist. Reuse an output file only to resume the
    same image, coordinate order, checkpoint, and extraction settings. Changing
    ``patch_size`` requires a new output file if any rows are already complete.
    """
    patch_size = _validate_patch_size(patch_size)
    device = torch.device(
        device_name if not device_name.startswith("cuda") or torch.cuda.is_available() else "cpu"
    )
    model = build_gpfm_model(checkpoint, device)
    reader = VipsPatchReader(image_path, patch_size=patch_size)
    pixel_coords = np.asarray(pixel_coords, dtype=np.float32)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)

    with h5py.File(output_h5, "a") as handle:
        if dataset_name in handle:
            features = handle[dataset_name]
            if features.shape != (pixel_coords.shape[0], feature_dim):
                raise ValueError(
                    f"Existing {dataset_name} shape {features.shape} does not match "
                    f"{(pixel_coords.shape[0], feature_dim)}"
                )
        else:
            features = handle.create_dataset(
                dataset_name,
                shape=(pixel_coords.shape[0], feature_dim),
                dtype=np.float32,
                chunks=(min(2048, pixel_coords.shape[0]), feature_dim),
                compression="lzf",
            )
        if "path_valid_fraction" in handle:
            valid = handle["path_valid_fraction"]
        else:
            valid = handle.create_dataset(
                "path_valid_fraction",
                shape=(pixel_coords.shape[0],),
                dtype=np.float32,
                chunks=(min(16384, pixel_coords.shape[0]),),
                compression="lzf",
            )
        completed = int(handle.attrs.get("path_feat_completed_rows", 0))
        completed = min(completed, pixel_coords.shape[0])
        if completed > 0:
            previous_patch_size = handle.attrs.get("path_patch_size")
            if previous_patch_size is None or int(previous_patch_size) != patch_size:
                raise ValueError(
                    f"Existing features have patch_size={previous_patch_size}; "
                    f"cannot resume with patch_size={patch_size}. Use a new output_h5."
                )
        handle.attrs["path_patch_size"] = patch_size
        handle.flush()
        target_rows = (
            pixel_coords.shape[0]
            if max_rows is None
            else min(pixel_coords.shape[0], int(max_rows))
        )
        for start in range(completed, target_rows, batch_size):
            end = min(target_rows, start + batch_size)
            patches = []
            fractions = []
            for x_coord, y_coord in pixel_coords[start:end]:
                patch, fraction = reader.read(x_coord, y_coord)
                patches.append(patch)
                fractions.append(fraction)
            batch = torch.from_numpy(np.stack(patches)).to(device=device)
            batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
            if batch.shape[-2:] != (224, 224):
                batch = F.interpolate(
                    batch,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
            batch.sub_(mean).div_(std)
            with torch.inference_mode():
                embeddings = unwrap_model_output(model(batch))
            if embeddings.shape[1] != feature_dim:
                raise ValueError(
                    f"GPFM output dim {embeddings.shape[1]} != configured {feature_dim}"
                )
            features[start:end] = embeddings.float().cpu().numpy()
            valid[start:end] = np.asarray(fractions, dtype=np.float32)
            handle.attrs["path_feat_completed_rows"] = int(end)
            handle.flush()
            print(f"GPFM patches {end}/{pixel_coords.shape[0]}", flush=True)
        handle.attrs["path_feat_model"] = "GPFM"
        handle.attrs["path_feat_checkpoint"] = str(Path(checkpoint).resolve())
        handle.attrs["path_patch_size"] = int(patch_size)
