import re
from pathlib import Path
from geo_utils import _list_tifs
def extract_idx(path: Path) -> int:
    """
    Extract index from filename patterns like:
    - season2024_feat0.tif  -> 0
    - poly_0.tif            -> 0
    """
    name = path.stem  # no extension
    
    # find any integer at the end of the filename
    m = re.search(r'(\d+)$', name)
    if m is None:
        raise ValueError(f"Cannot extract idx from filename: {path}")
    
    return int(m.group(1))


def pair_by_idx(input_dir, mask_dir):

    
    input_tifs = _list_tifs(input_dir)
    mask_tifs  = _list_tifs(mask_dir)
    
    
    # Index them into dicts
    input_map = {extract_idx(p): p for p in input_tifs}
    mask_map  = {extract_idx(p): p for p in mask_tifs}

    # Intersection of available indices
    common = sorted(set(input_map.keys()) & set(mask_map.keys()))

    if not common:
        raise RuntimeError("No matching indices between input_tifs and mask_tifs")

    # Build paired list
    pairs = [
        {"idx": idx, "input": input_map[idx], "mask": mask_map[idx]}
        for idx in common
    ]
    return pairs

