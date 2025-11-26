from pathlib import Path

from get_data import S2S1PrestoDownloader
from get_landcover import esri_landuse
from inference import run_inference_big_tif
from utils import pair_by_idx
from geo_utils import _list_tifs


def run_presto_pipeline(configs: dict):

    required = ["asset_path", "year", "landuse_method", "device", "model_path"]
    for k in required:
        if k not in configs:
            raise ValueError(f"Missing required config key: {k}")

    year = configs["year"]
    landuse_method = configs["landuse_method"]
    skip_download = configs.get("skip_download", False)

    OUTPUT_DIR = Path("./data/outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Download step (optional)
    if not skip_download:
        downloader = S2S1PrestoDownloader(
            asset_path=configs["asset_path"],
            output_dir="./data/inputs",
            start_year=year,
        )
        downloader.run()
    else:
        print("⏭️  Skipping S2/S1 download step. Using existing TIFFs.\n")

    # 2) ESRI or No Landcover
    if landuse_method == "ESRI":

        if "ESRI_mask_path" not in configs:
            raise RuntimeError("ESRI_mask_path must be provided for ESRI mode.")

        if year < 2017 or year > 2024:
            raise RuntimeError(f"ESRI does not support year={year}.")

        # extract ESRI masks
        esri_landuse(configs)

        # pair the TIFFs by index
        pairs = pair_by_idx("./data/inputs", f"./data/LULC/{year}")

        for pair in pairs:
            idx = pair["idx"]

            cfg = configs.copy()
            cfg["input_path"] = pair["input"]
            cfg["mask_path"]  = pair["mask"]
            cfg["output_path"] = str(OUTPUT_DIR / f"irrigation_{year}_{idx}.tif")

            run_inference_big_tif(cfg)

    else:
        # No mask → run inference on all inputs
        tiffs = _list_tifs("./data/inputs")

        for idx, input_path in enumerate(tiffs):
            cfg = configs.copy()
            cfg["input_path"] = input_path
            cfg["mask_path"]  = None
            cfg["output_path"] = str(OUTPUT_DIR / f"irrigation_{year}_{idx}.tif")

            run_inference_big_tif(cfg)


if __name__ == "__main__":

    year = 2024
    
    configs = {
        "asset_path": "./data/ROI/wetlands/WS Anzali/watershed_12.shp",
        "year": year,
        "landuse_method": "ESRI",
        "ESRI_mask_path": f"../LULC/landcover-{year}",
        "device": "cuda",
        "model_path": "./weights/tune_model.pth",
        "skip_download": True,
        "tile_size": 5120,
    }

    run_presto_pipeline(configs)
    

