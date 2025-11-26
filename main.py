from pathlib import Path

from get_data import S2S1PrestoDownloader
from get_landcover import esri_landuse
from inference import run_inference
from utils import pair_by_idx
from geo_utils import _list_tifs


def run_presto_pipeline(configs: dict):
    """
    Full ESRI + Presto Irrigation pipeline:
    - Download S2/S1 seasonal inputs
    - Extract ESRI LandCover masks (optional)
    - Pair inputs & masks by index
    - Run inference tile-by-tile
    """

    # --------------------------------------------------------
    # Validate config
    # --------------------------------------------------------
    required = ["asset_path", "year", "landuse_method", "device", "model_path"]
    for k in required:
        if k not in configs:
            raise ValueError(f"Missing required config key: {k}")

    year = configs["year"]
    landuse_method = configs["landuse_method"]

    OUTPUT_DIR = Path("./data/outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1) Download multi-temporal S2 + S1 inputs
    # --------------------------------------------------------
    downloader = S2S1PrestoDownloader(
        asset_path=configs["asset_path"],
        output_dir="./data/inputs",
        start_year=year,
    )
    downloader.run()

    # --------------------------------------------------------
    # 2) Branch: ESRI or No-landcover
    # --------------------------------------------------------
    if landuse_method == "ESRI":

        # --- validations ---
        if "ESRI_mask_path" not in configs:
            raise RuntimeError("ESRI_mask_path must be provided for ESRI mode.")

        if year < 2017 or year > 2024:
            raise RuntimeError(f"ESRI does not support year={year}. Only 2017–2024.")

        # --- extract masks ---
        esri_landuse(configs)

        # --- pair input tifs with mask tifs ---
        pairs = pair_by_idx("./data/inputs", f"./data/LULC/{year}")

        # --- run inference on each tile ---
        for pair in pairs:
            idx = pair["idx"]

            cfg = configs.copy()
            cfg["input_path"] = pair["input"]
            cfg["mask_path"] = pair["mask"]
            cfg["output_path"] = str(OUTPUT_DIR / f"irrigation_{year}_{idx}.tif")

            run_inference(cfg)

    else:
        # --------------------------------------------------------
        # 3) No landcover mask → run inference on all input TIFFs
        # --------------------------------------------------------
        tiffs = _list_tifs("./data/inputs")

        for idx, input_path in enumerate(tiffs):
            cfg = configs.copy()
            cfg["input_path"] = input_path
            cfg["mask_path"] = None
            cfg["output_path"] = str(OUTPUT_DIR / f"irrigation_{year}_{idx}.tif")

            run_inference(cfg)


if __name__ == "__main__":

    year = 2017
    
    configs = {
        "asset_path": "./data/ROI/wetlands_one/wetlands.shp",
        "year": year,
        "landuse_method": "ESRI",
        "ESRI_mask_path": f"../../landcover/LULC/landcover-{year}",
        "device": "cuda",
        "model_path": "./weights/tune_model.pth",
    }

    run_presto_pipeline(configs)
