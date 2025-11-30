from pathlib import Path

from get_data import S2S1PrestoDownloader
from get_landcover import esri_landuse
from inference import run_inference_big_tif
from utils import pair_by_idx, _append_metadata_row, _make_synthetic_lulc_mask_like, _pct_nonfinite_pixels
from geo_utils import _list_tifs
from datetime import datetime

# ----------------------------------------------------
# Main pipeline
# ----------------------------------------------------
def run_presto_pipeline(configs: dict):

    required = ["asset_path", "year", "landuse_method", "device", "model_path"]
    for k in required:
        if k not in configs:
            raise ValueError(f"Missing required config key: {k}")

    year = configs["year"]
    landuse_method = configs["landuse_method"]
    skip_download = configs.get("skip_download", False)

    asset_path = Path(configs["asset_path"])
    roi_name = asset_path.stem    # e.g. "roi_0"

    # ----------------------------------------
    # Base directory per ROI
    # ----------------------------------------
    ROI_BASE = Path("./data") / roi_name
    INPUT_DIR  = ROI_BASE / "inputs"
    OUTPUT_DIR = ROI_BASE / "outputs"
    LULC_DIR   = ROI_BASE / "LULC"

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LULC_DIR.mkdir(parents=True, exist_ok=True)

    # Metadata CSV for this ROI
    meta_csv = ROI_BASE / "processing_metadata.csv"
    META_FIELDS = [
        "timestamp",
        "roi",
        "year",
        "idx",
        "input_path",
        "mask_path",
        "nonfinite_pct",
        "landuse_method",
        "landuse_failed",
        "mask_mode",
        "inference_on_full_input",
        "error",
    ]

    # Wrap the whole thing so one ROI/year error doesn't kill the script
    try:
        # ----------------------------------------
        # 1) Download step
        # ----------------------------------------
        if not skip_download:
            downloader = S2S1PrestoDownloader(
                asset_path=str(asset_path),
                output_dir=str(INPUT_DIR),
                start_year=year,
            )
            downloader.run()
        else:
            print("⏭️  Skipping S2/S1 download step. Using existing TIFFs.\n")

        # Make sure we actually have inputs
        input_tifs = _list_tifs(str(INPUT_DIR))
        if not input_tifs:
            raise RuntimeError(f"No input TIFFs found in {INPUT_DIR}")

        # ----------------------------------------
        # 2) Landcover / Masks
        # ----------------------------------------
        if landuse_method == "ESRI":
            landuse_failed_global = False
            landuse_note = ""

            # 2a) Try to crop ESRI LULC
            esri_cfg = configs.copy()
            esri_cfg["LULC_output_dir"] = str(LULC_DIR)
            try:
                esri_landuse(esri_cfg)
            except Exception as e:
                landuse_failed_global = True
                landuse_note = f"esri_landuse failed: {repr(e)}"
                print(f"⚠️  ESRI landuse failed for {roi_name} {year}: {e}")

            # 2b) Try to pair inputs and masks
            synthetic_masks_used = False
            pairs = []

            if not landuse_failed_global:
                try:
                    pairs = pair_by_idx(str(INPUT_DIR), str(LULC_DIR))
                except Exception as e:
                    # Example: RuntimeError("No matching indices between input_tifs and mask_tifs")
                    landuse_failed_global = True
                    landuse_note = f"pair_by_idx failed: {repr(e)}"
                    print(f"⚠️  Pairing inputs/LULC failed for {roi_name} {year}: {e}")
                    pairs = []

            # 2c) Fallback: create synthetic all-4 mask for every input
            if landuse_failed_global or not pairs:
                print(
                    f"ℹ️  Using synthetic LULC (all pixels == 4) for ROI={roi_name}, year={year}"
                )
                synthetic_masks_used = True
                pairs = []
                for idx, inp in enumerate(sorted(input_tifs)):
                    inp = Path(inp)
                    mask_path = LULC_DIR / f"{inp.stem}_lulc_all4.tif"
                    _make_synthetic_lulc_mask_like(inp, mask_path, value=4)
                    pairs.append(
                        {
                            "idx": idx,
                            "input": str(inp),
                            "mask": str(mask_path),
                        }
                    )

            # 2d) Run inference per pair (with metadata logging)
            mask_mode = "synthetic_all4" if synthetic_masks_used else "esri_cropped"

            for pair in pairs:
                idx = pair["idx"]
                input_path = pair["input"]
                mask_path = pair["mask"]

                # Compute non-finite percentage on inputs
                try:
                    nonfinite_pct = _pct_nonfinite_pixels(input_path)
                except Exception as e:
                    nonfinite_pct = float("nan")
                    print(f"⚠️  Failed to compute nonfinite pct for {input_path}: {e}")

                cfg = configs.copy()
                cfg["input_path"]  = input_path
                cfg["mask_path"]   = mask_path
                cfg["output_path"] = str(OUTPUT_DIR / f"irrigation_{year}_{idx}.tif")

                error_msg = ""
                try:
                    run_inference_big_tif(cfg)
                except Exception as e:
                    error_msg = repr(e)
                    print(f"❌ Inference failed for ROI={roi_name}, year={year}, idx={idx}: {e}")

                # Metadata row
                row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "roi": roi_name,
                    "year": year,
                    "idx": idx,
                    "input_path": input_path,
                    "mask_path": mask_path,
                    "nonfinite_pct": nonfinite_pct,
                    "landuse_method": landuse_method,
                    "landuse_failed": landuse_failed_global,
                    "mask_mode": mask_mode,
                    "inference_on_full_input": bool(mask_mode == "synthetic_all4"),
                    "error": error_msg or landuse_note,
                }
                _append_metadata_row(meta_csv, row, META_FIELDS)

        else:
            # ----------------------------------------
            # No landcover mode: still log metadata and always
            # treat as inference on full input (mask_mode = no_landuse)
            # ----------------------------------------
            for idx, input_path in enumerate(sorted(input_tifs)):
                input_path = str(input_path)
                try:
                    nonfinite_pct = _pct_nonfinite_pixels(input_path)
                except Exception as e:
                    nonfinite_pct = float("nan")
                    print(f"⚠️  Failed to compute nonfinite pct for {input_path}: {e}")

                cfg = configs.copy()
                cfg["input_path"]  = input_path
                cfg["mask_path"]   = None
                cfg["output_path"] = str(OUTPUT_DIR / f"irrigation_{year}_{idx}.tif")

                error_msg = ""
                try:
                    run_inference_big_tif(cfg)
                except Exception as e:
                    error_msg = repr(e)
                    print(f"❌ Inference failed for ROI={roi_name}, year={year}, idx={idx}: {e}")

                row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "roi": roi_name,
                    "year": year,
                    "idx": idx,
                    "input_path": input_path,
                    "mask_path": "",
                    "nonfinite_pct": nonfinite_pct,
                    "landuse_method": landuse_method,
                    "landuse_failed": False,
                    "mask_mode": "no_landuse",
                    "inference_on_full_input": True,
                    "error": error_msg,
                }
                _append_metadata_row(meta_csv, row, META_FIELDS)

    except Exception as e:
        # Fatal error for this ROI/year → log a single row with idx = -1
        error_msg = repr(e)
        print(f"❌ Fatal error in run_presto_pipeline for ROI={roi_name}, year={year}: {e}")

        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "roi": roi_name,
            "year": year,
            "idx": -1,
            "input_path": "",
            "mask_path": "",
            "nonfinite_pct": "",
            "landuse_method": configs.get("landuse_method", ""),
            "landuse_failed": True,
            "mask_mode": "",
            "inference_on_full_input": "",
            "error": error_msg,
        }
        _append_metadata_row(meta_csv, row, META_FIELDS)



if __name__ == "__main__":
    
    from utils import natural_key
    shp_dir = Path("./data/ROI/sample_wetlands")

    for shp_path in sorted(shp_dir.glob("*.shp"), key=natural_key):
        for year in [2024, 2017, 2019]:
            configs = {
                "asset_path": shp_path,
                "year": year,
                "landuse_method": "ESRI",
                "ESRI_mask_path": f"../LULC/landcover-{year}",
                "device": "cuda",
                "model_path": "./weights/tune_model.pth",
                "skip_download": False,
                "tile_size": 2048,
            }
        
            run_presto_pipeline(configs)
        

