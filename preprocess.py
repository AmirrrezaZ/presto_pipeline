import numpy as np
from collections import OrderedDict
import torch
from typing import Tuple, List
import re

# -------------------------
# Canonical per-month order
# -------------------------
_PER_MONTH_ORDER = [
    "coastal","blue","green","red",
    "red_edge1","red_edge2","red_edge3",
    "nir","red_edge4","water_vapor","swir1","swir2",
    "vv","vh",
]
_BAND2IDX = {b:i for i,b in enumerate(_PER_MONTH_ORDER)}
_MONTH_BAND_RE = re.compile(r"^M(\d{2})_(.+)$")

# -------------------------
# Presto channel layout
# -------------------------
BANDS_GROUPS_IDX = OrderedDict(
    [
        ("S1", [0, 1]),               # vv, vh
        ("S2_RGB", [2, 3, 4]),        # red, green, blue
        ("S2_Red_Edge", [5, 6, 7]),   # red_edge1..3
        ("S2_NIR_10m", [8]),          # nir
        ("S2_NIR_20m", [9]),          # red_edge4 (proxy)
        ("S2_SWIR", [10, 11]),        # swir1, swir2
        ("ERA5", [12, 13]),           # not provided -> masked
        ("SRTM", [14, 15]),           # not provided -> masked
        ("NDVI", [16]),               # computed
    ]
)
GROUP_AVAILABLE = OrderedDict([
    ("S1", True),
    ("S2_RGB", True),
    ("S2_Red_Edge", True),
    ("S2_NIR_10m", True),
    ("S2_NIR_20m", True),
    ("S2_SWIR", True),
    ("ERA5", False),
    ("SRTM", False),
    ("NDVI", True),   
])

def reshape_168_to_month_band(
    arr: np.ndarray,
    descs: List[str] | None = None,
    strict: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert a (168, H, W) stack with band names like 'M01_red' into (12, 14, H, W),
    and also return the *per-month band order* list used for the second axis.

    Returns
    -------
    (cube, per_month_order):
        cube : np.ndarray, (12, 14, H, W), dtype float32
        per_month_order : list[str] length 14, e.g. _PER_MONTH_ORDER
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (C,H,W), got shape {arr.shape}")

    C, H, W = arr.shape
    per_month_order = list(_PER_MONTH_ORDER)

    # Prepare output filled with NaN
    out = np.full((12, 14, H, W), np.nan, dtype=np.float32)

    # Fast path: canonical export order (M01..M12, per _PER_MONTH_ORDER)
    if descs is None:
        if strict and C != 12 * 14:
            raise ValueError(f"Expected 168 bands when descs=None, got {C}")
        idx = 0
        for m in range(12):
            for b in range(14):
                if idx < C:
                    out[m, b] = arr[idx].astype(np.float32, copy=False)
                    idx += 1
        return out, per_month_order

    # Validate descs length
    if strict and len(descs) != C:
        raise ValueError(f"descs length {len(descs)} != channels {C}")

    problems = []
    for k, name in enumerate(descs):
        m = _MONTH_BAND_RE.match(name)
        if not m:
            if strict:
                raise ValueError(f"Band name not in 'M##_<band>' format: {name}")
            problems.append(("bad_format", name))
            continue

        month_s, band_name = m.groups()
        try:
            month_idx = int(month_s) - 1
        except Exception:
            if strict:
                raise ValueError(f"Invalid month in band name: {name}")
            problems.append(("bad_month", name))
            continue

        if not (0 <= month_idx < 12):
            if strict:
                raise ValueError(f"Month out of range in band name: {name}")
            problems.append(("bad_month_range", name))
            continue

        band_idx = _BAND2IDX.get(band_name)
        if band_idx is None:
            if strict:
                raise ValueError(
                    f"Unknown band '{band_name}' in {name}. "
                    f"Expected one of: {list(_BAND2IDX.keys())}"
                )
            problems.append(("unknown_band", name))
            continue

        out[month_idx, band_idx] = arr[k].astype(np.float32, copy=False)

    if strict and out.shape != (12, 14, H, W):
        raise RuntimeError("Unexpected output shape after reshaping.")

    return out, per_month_order

def compute_monthly_ndvi(
    month_bands: np.ndarray,
    per_month_order: List[str],
    eps: float = 1e-6
) -> np.ndarray:
    """
    NDVI per month from (12,14,H,W) using band names in per_month_order.
    Returns (12,H,W), float32 in [-1,1], NaN where inputs invalid.
    """
    if month_bands.ndim != 4 or month_bands.shape[0] != 12 or month_bands.shape[1] != 14:
        raise ValueError(f"Expected (12,14,H,W), got {month_bands.shape}")

    try:
        red_idx = per_month_order.index("red")
        nir_idx = per_month_order.index("nir")
    except ValueError as e:
        raise ValueError(f"Required bands not found in per_month_order: {e}")

    red = month_bands[:, red_idx, ...].astype(np.float32, copy=False)
    nir = month_bands[:, nir_idx, ...].astype(np.float32, copy=False)
    denom = nir + red

    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (denom + eps)

    ndvi = np.where(np.abs(denom) < eps, np.nan, ndvi)
    ndvi = np.clip(ndvi, -1.0, 1.0)

    return ndvi.astype(np.float32, copy=False)

def build_presto_inputs_from_cube(
    month_bands: np.ndarray,
    per_month_order: List[str]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    month_bands: (12,14,H,W)
    per_month_order: list of the 14 band names for axis=1 (from reshaper)

    Returns:
      x    : (N, 12, 17) float32, channel layout per BANDS_GROUPS_IDX
      mask : (N, 12, 17) float32; 1 where missing/invalid, else 0
      dw   : (N, 12) long; placeholder (masked / not provided)
    """
    if month_bands.ndim != 4 or month_bands.shape[0] != 12 or month_bands.shape[1] != 14:
        raise ValueError(f"Expected (12,14,H,W), got {month_bands.shape}")

    # Name -> index lookup (robust to order changes)
    name2idx = {n: i for i, n in enumerate(per_month_order)}
    required = [
        "vv","vh",
        "red","green","blue",
        "red_edge1","red_edge2","red_edge3",
        "nir","red_edge4",
        "swir1","swir2",
    ]
    missing = [b for b in required if b not in name2idx]
    if missing:
        raise ValueError(f"Missing required bands in per_month_order: {missing}")

    T, C14, H, W = month_bands.shape
    N = H * W

    # (N,12,14)
    mb = month_bands.reshape(T, C14, N).transpose(2, 0, 1)

    x = torch.zeros((N, T, 17), dtype=torch.float32)

    # ----- Fill by names -----
    # S1 -> [0,1]
    x[..., 0]  = torch.from_numpy(mb[..., name2idx["vv"]])  # linear
    x[..., 1]  = torch.from_numpy(mb[..., name2idx["vh"]])  # linear

    # S2_RGB -> [2,3,4] = [red, green, blue]
    x[..., 2]  = torch.from_numpy(mb[..., name2idx["red"]])
    x[..., 3]  = torch.from_numpy(mb[..., name2idx["green"]])
    x[..., 4]  = torch.from_numpy(mb[..., name2idx["blue"]])

    # S2 Red-Edge -> [5,6,7]
    x[..., 5]  = torch.from_numpy(mb[..., name2idx["red_edge1"]])
    x[..., 6]  = torch.from_numpy(mb[..., name2idx["red_edge2"]])
    x[..., 7]  = torch.from_numpy(mb[..., name2idx["red_edge3"]])

    # S2 NIR (10m) -> [8]
    x[..., 8]  = torch.from_numpy(mb[..., name2idx["nir"]])

    # S2 NIR (20m proxy) -> [9] = red_edge4
    x[..., 9]  = torch.from_numpy(mb[..., name2idx["red_edge4"]])

    # S2 SWIR -> [10,11]
    x[..., 10] = torch.from_numpy(mb[..., name2idx["swir1"]])
    x[..., 11] = torch.from_numpy(mb[..., name2idx["swir2"]])

    # ERA5 [12,13] / SRTM [14,15] left zeros

    # NDVI [16] (computed via names)
    ndvi = compute_monthly_ndvi(month_bands, per_month_order) \
             .reshape(T, N).transpose(1, 0)  # (N,12)
    x[..., 16] = torch.from_numpy(ndvi)

    # ----- Mask -----
    mask = torch.zeros_like(x)

    # 1) Structural unavailability (ERA5, SRTM)
    for grp_name, ch_idx_list in BANDS_GROUPS_IDX.items():
        if not GROUP_AVAILABLE.get(grp_name, False):
            for ch in ch_idx_list:
                mask[..., ch] = 1.0

    # 2) Per-pixel invalids (NaN/Inf) in x
    # nan_mask = torch.isnan(x) | torch.isinf(x)
    # if nan_mask.any():
    #     x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    #     mask = torch.where(nan_mask, torch.tensor(1.0, dtype=mask.dtype), mask)

    # Dynamic World placeholder (masked / not provided)
    dw = torch.full((N, T), 9, dtype=torch.long)

    return x, mask, dw
