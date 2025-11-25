from get_data import S2S1PrestoDownloader
from get_landcover import esri_landuse
from inference import run_inference
from utils import pair_by_idx

configs ={"asset_path": "./data/ROI/sample_wetlands/sample.shp",
          "year": 2017,
          "landuse_method": "ESRI",
          "ESRI_mask_path": "../..//landcover/LULC/landcover-2017",
          "device": "cuda"  , 
          "model_path": "./weights/tune_model.pth"
          }


downloader = S2S1PrestoDownloader(
    asset_path=configs["asset_path"],   
    output_dir="./data/inputs",
    start_year=configs["year"],
)

downloader.run()


esri_landuse(configs)


pairs = pair_by_idx("./data/inputs", f"./data/LULC/{configs['year']}")

for pair in pairs:
    idx = pair['idx']
    configs["input_path"] = pair['input']
    configs["mask_path"] = pair['mask']
    configs["output_path"] = f"data/outputs/irrigation_configs['year']_{idx}"
    irrigation_map = run_inference(configs)
    
    