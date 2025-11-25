from pathlib import Path
import torch
import numpy as np

from dataset_loaders import inference_loader
from model.presto import PrestoClassifier
from geo_utils import save_pred_to_tiff



def run_inference(config: dict):
    
    want_cuda = (config.get("device", "cpu") == "cuda")
    device = torch.device("cuda" if (
        want_cuda and torch.cuda.is_available()) else "cpu")

    # --- Build inference DataLoader ---
    loader, meta = inference_loader(config["input_path"],
                                    mask_path = config["mask_path"],
                                         batch_size=config.get(
                                             "batch_size", 2048),
                                         device=device)

    # --- Load model ---
    clf = PrestoClassifier.load(config["model_path"], device=device)
    print(f"✅ Loaded model from: {config['model_path']}")
    print(f"🖥️  Using device: {clf.device}")

    # --- Predict ---
    pred = clf.predict(loader)
    
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    else:
        pred = np.asarray(pred)
        
    
    pred +=1



    H, W = meta["shape_hw"]
    idx = meta["flat_indices"]  # 1D indices in row-major order
    
    full_pred = np.full((H * W,), fill_value=0, dtype=np.uint8)  # or your nodata value
    full_pred[idx] = pred
    full_pred = full_pred.reshape(H, W)
    
        


    # --- Save to GeoTIFF ---
    save_pred_to_tiff(
        out_arr=full_pred,
        out_tiff=config["output_path"],
        ref_tiff=config["input_path"],
        dtype="uint8",
        nodata=0,        
        compress="lzw",
    )

    print(f"💾 Saved prediction to {config['output_path']}")
    return full_pred



