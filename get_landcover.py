from geo_utils import crop_rasters_by_polygons


def esri_landuse(configs: dict):
    # Only ESRI is supported for now
    if configs.get("landuse_method") != "ESRI":
        raise RuntimeError(
            f'Unsupported landuse_method="{configs.get("landuse_method")}". '
            'Only "ESRI" is currently supported.'
        )

    # --- Year checks ---
    year = configs.get("year")
    if year is None:
        raise ValueError("configs['year'] must be provided.")
    if not isinstance(year, int):
        raise TypeError(f"configs['year'] must be int, got {type(year).__name__}.")
    # ESRI LULC available 2017–2024
    if year not in range(2017, 2025):  # 2025 is exclusive → 2017..2024
        raise ValueError("ESRI LULC is only available for years 2017–2024.")

    # --- Path checks ---
    esri_mask_path = configs.get("ESRI_mask_path")
    if not esri_mask_path:
        raise ValueError("configs['ESRI_mask_path'] must be provided and non-empty.")

    asset_path = configs.get("asset_path")
    if not asset_path:
        raise ValueError("configs['asset_path'] (vector ROI) must be provided and non-empty.")

    # --- Everything OK → run cropping ---
    crop_rasters_by_polygons(
        tif_dir=esri_mask_path,
        shp_path=asset_path,
        out_dir=f"./data/LULC/{year}",
        iou_threshold=0.01,
        id_field=None,
    )
