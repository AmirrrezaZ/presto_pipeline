from pathlib import Path
from contextlib import ExitStack

import geopandas as gpd
from shapely.geometry import box, mapping
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.io import MemoryFile

from rasterio.enums import Resampling
from rasterio.warp import reproject

import torch
import numpy as np
from pyproj import Transformer

from typing import Optional, Tuple, Callable, List, Union

def _list_tifs(root: Path):
    root = Path(root)
    exts = (".tif", ".tiff")
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


def align_mask_to_tile(
    mask_path: str | Path,
    ref_tif_path: str | Path,
) -> np.ndarray:
    """
    Reproject & resample a mask raster to exactly match the grid of ref_tif_path.

    Returns
    -------
    mask_aligned : np.ndarray
        2D array [H_ref, W_ref] aligned by location (CRS + transform) to ref_tif_path.
    """
    mask_path = Path(mask_path)
    ref_tif_path = Path(ref_tif_path)

    # open reference (the 168-band tile)
    with rasterio.open(ref_tif_path) as ref_ds:
        ref_crs = ref_ds.crs
        ref_transform = ref_ds.transform
        ref_height = ref_ds.height
        ref_width = ref_ds.width

    # open mask
    with rasterio.open(mask_path) as mds:
        # read first band
        src = mds.read(1)  # (H_mask, W_mask)
        src_transform = mds.transform
        src_crs = mds.crs

    # destination array on the reference grid
    dst = np.zeros((ref_height, ref_width), dtype=src.dtype)

    reproject(
        source=src,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest,  # keep class labels
    )

    return dst


def save_pred_to_tiff(
    out_arr: Union[np.ndarray, torch.Tensor],
    out_tiff: Union[str, Path],
    ref_tiff: Union[str, Path],
    dtype: str = "uint8",
    nodata: int = None,
    compress: str = "lzw",
) -> Path:
    """
    Save prediction array as GeoTIFF aligned to a reference raster.
    ❌ No resizing is performed — shape must exactly match hw_shape.

    Parameters
    ----------
    out_arr : np.ndarray | torch.Tensor
        Predicted map (H, W) or flat [H*W].
    out_tiff : str | Path
        Path to save output GeoTIFF.
    ref_tiff : str | Path
        Reference raster to copy CRS, transform, and metadata from.
    hw_shape : (int, int)
        Expected raster shape (H, W). Must match out_arr shape.
    dtype : str, default="uint8"
        Output raster dtype.
    nodata : int, optional
        No-data pixel value.
    compress : str, default="lzw"
        Compression type (e.g., "lzw", "deflate").
    """

    out_tiff = Path(out_tiff)
    ref_tiff = Path(ref_tiff)
    H, W = out_arr.shape

    # --- Convert to numpy ---
    if isinstance(out_arr, torch.Tensor):
        out_arr = out_arr.detach().cpu().numpy()
    out_arr = np.asarray(out_arr)

    # --- Reshape or validate ---
    if out_arr.ndim == 1:
        if out_arr.size != H * W:
            raise ValueError(f"Flat prediction has {out_arr.size} elements, expected {H*W}.")
        out_arr = out_arr.reshape(H, W)
    elif out_arr.shape != (H, W):
        raise ValueError(
            f"Prediction shape {out_arr.shape} does not match expected {(H, W)}. "
            "Resizing is disabled."
        )

    out_arr = out_arr.astype(dtype, copy=False)

    # --- Load georeference from reference raster ---
    with rasterio.open(ref_tiff) as src:
        profile = src.profile.copy()

    # --- Update output profile ---
    profile.update(
        count=1,
        dtype=dtype,
        compress=compress,
        height=H,
        width=W,
        nodata=nodata,
    )

    # --- Write GeoTIFF ---
    out_tiff.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tiff, "w", **profile) as dst:
        dst.write(out_arr, 1)

    print(f"💾 Saved GeoTIFF: {out_tiff}  | shape={out_arr.shape} | dtype={dtype}")
    return out_tiff



def compute_iou(poly, raster_footprint):
    """Intersection over Union between a polygon and raster footprint."""
    inter_area = poly.intersection(raster_footprint).area
    if inter_area == 0:
        return 0.0
    union_area = poly.union(raster_footprint).area
    return inter_area / union_area


def _read_tif(path: Path, as_mask: bool = False):
    """
    Read GeoTIFF and return:
      - arr  : (C,H,W) or (H,W) if as_mask=True
      - nodata
      - lat  : (H,W)
      - lon  : (H,W)
    """
    with rasterio.open(path) as src:
        arr = src.read(1) if as_mask else src.read()   # (H,W) or (C,H,W)
        nodata = src.nodata
        transform = src.transform
        src_crs = src.crs

        if src_crs is None:
            raise ValueError(f"{path} has no CRS; cannot compute lat/lon.")

        H, W = src.height, src.width
        descs = list(src.descriptions) if src.descriptions is not None else [None] * arr.shape[0]

        # Build row/col grids
        rows = np.arange(H)
        cols = np.arange(W)
        cols_grid, rows_grid = np.meshgrid(cols, rows)

        # Pixel centers -> projected coordinates
        xs, ys = rasterio.transform.xy(transform, rows_grid, cols_grid, offset="center")
        xs = np.asarray(xs)
        ys = np.asarray(ys)

        # Project to EPSG:4326
        transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(xs, ys)

    return arr, nodata, lat, lon, descs



def crop_rasters_by_polygons(
    tif_dir,
    shp_path,
    out_dir,
    iou_threshold=0.0,
    id_field=None,  # optional: column in shapefile to use in output filenames
):
    """
    For each polygon in `shp_path`, find overlapping tif(s) in `tif_dir` using IoU,
    merge overlapping rasters if needed, mask to polygon, and save to `out_dir`.

    Parameters
    ----------
    tif_dir : str or Path
        Directory containing input GeoTIFF tiles.
    shp_path : str or Path
        Path to polygon shapefile.
    out_dir : str or Path
        Directory where cropped rasters will be saved.
    iou_threshold : float, optional
        Minimum IoU between polygon and raster footprint to consider
        that raster as "related". Default: 0.0 (any overlap).
    id_field : str or None, optional
        Name of a field in the shapefile to use as polygon identifier
        in output filenames. If None or field missing, polygon index is used.
    """
    tif_dir = Path(tif_dir)
    shp_path = Path(shp_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tif_paths = sorted(list(tif_dir.glob("*.tif*")))
    if not tif_paths:
        raise FileNotFoundError(f"No GeoTIFFs found in {tif_dir}")

    # Read polygons
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError(f"No polygons found in {shp_path}")

    # Get CRS from first raster and reproject polygons if needed
    with rasterio.open(tif_paths[0]) as ref:
        raster_crs = ref.crs

    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    # Precompute raster footprints (bounds as polygons)
    raster_infos = []
    for p in tif_paths:
        
        if not p.suffix.lower() in [".tif", ".tiff"]:
            continue
        
        with rasterio.open(p) as src:
            footprint = box(*src.bounds)
        raster_infos.append({"path": p, "footprint": footprint})

    # Loop over polygons
    for idx, row in gdf.iterrows():
        poly = row.geometry
        if poly is None or poly.is_empty:
            print(f"⚠️ Skipping polygon {idx}: empty geometry")
            continue

        # Fix invalid geometries if needed
        if not poly.is_valid:
            poly = poly.buffer(0)

        # Find candidate rasters based on IoU
        candidates = []
        for info in raster_infos:
            iou = compute_iou(poly, info["footprint"])
            if iou >= iou_threshold:
                candidates.append((info["path"], iou))

        if not candidates:
            print(f"ℹ️ No rasters found for polygon {idx} (IoU >= {iou_threshold})")
            continue

        # Sort by IoU (best first) and keep only paths
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidate_paths = [c[0] for c in candidates]

        print(
            f"✅ Polygon {idx}: using {len(candidate_paths)} rasters "
            f"(best IoU={candidates[0][1]:.3f})"
        )

        # Merge candidate rasters (if >1) and then mask to polygon
        with ExitStack() as stack:
            srcs = [stack.enter_context(rasterio.open(p)) for p in candidate_paths]

            # Merge rasters (limited to polygon bbox for efficiency)
            mosaic_arr, mosaic_transform = merge(srcs, bounds=poly.bounds)

            # Use meta from first raster as template
            meta = srcs[0].meta.copy()
            meta.update(
                {
                    "height": mosaic_arr.shape[1],
                    "width": mosaic_arr.shape[2],
                    "transform": mosaic_transform,
                }
            )

            # Put mosaic into an in-memory dataset so we can use rasterio.mask
            with MemoryFile() as memfile:
                with memfile.open(**meta) as tmp_ds:
                    tmp_ds.write(mosaic_arr)
                    out_img, out_transform = mask(
                        tmp_ds, [mapping(poly)], crop=True
                    )

        # Update metadata to match cropped output
        out_meta = meta.copy()
        out_meta.update(
            {
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transform,
            }
        )

        # Choose an ID for this polygon
        if id_field is not None and id_field in gdf.columns:
            poly_id = str(row[id_field])
        else:
            poly_id = str(idx)

        out_path = out_dir / f"poly_{poly_id}.tif"

        # Save result
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_img)

        print(f"💾 Saved: {out_path}")