import os
from itertools import combinations
from functools import lru_cache
import torch
import lpips
import json
from pathlib import Path
from tqdm import tqdm

from assess_utils.lists import null_models
from assess_utils.helpers import (
    get_img_save_name,
    visualization_path,
    results_path,
    images_path,
    img_folder_name2path,
)


def launch_main(model_name, img_folder_name, list_methods, lpips_net):
    dct_path = results_path / Path(
        "anomaly_scores", model_name, img_folder_name, f"{lpips_net}.json"
    )
    if dct_path.exists():
        dct = json.load(open(dct_path))
    else:
        dct = {}
    img_folder_path = img_folder_name2path(img_folder_name)
    list_img_names = os.listdir(img_folder_path)
    pbar_img = tqdm(range(len(list_img_names)), position=0)
    for img_idx in pbar_img:
        img_name = f"{img_idx}.png"
        pbar_img.set_description(f"image: {img_name}")
        img_path = img_folder_path / Path(img_name)

        pbar_method = tqdm(sorted(list_methods), position=1, leave=False)
        for method in pbar_method:
            if method not in dct.keys():
                dct[method] = {}
            pbar_method.set_description(f"method: {method}")
            score = get_score(model_name, img_path, method, lpips_net)
            dct[method][img_idx] = score
    dct_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dct, open(dct_path, "w"))


def get_score(model_name, img_path, method, lpips_net):
    new_null_models = [m for m in null_models if m != model_name]
    null_avg = get_null_avg(new_null_models, img_path, method, lpips_net)
    anom_avg = get_anom_avg(model_name, new_null_models, img_path, method, lpips_net)
    return anom_avg / null_avg


def get_null_avg(new_null_models, img_path, method, lpips_net):
    num_null = len(new_null_models)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net=lpips_net).to(device)

    null_avg = 0
    for null1, null2 in combinations(new_null_models, 2):
        _, png_path1 = visualization_path(null1, method, img_path)
        img1 = load_image_tensor(png_path1).to(device)
        _, png_path2 = visualization_path(null2, method, img_path)
        img2 = load_image_tensor(png_path2).to(device)
        null_avg += loss_fn.forward(img1, img2).item()
    null_avg /= num_null * (num_null - 1) / 2
    return null_avg


def get_anom_avg(model_name, new_null_models, img_path, method, lpips_net):
    num_null = len(new_null_models)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net=lpips_net).to(device)

    anom_avg = 0
    _, png_path1 = visualization_path(model_name, method, img_path)
    img1 = load_image_tensor(png_path1).to(device)
    for null_m in new_null_models:
        _, png_path2 = visualization_path(null_m, method, img_path)
        img2 = load_image_tensor(png_path2).to(device)

        # compute distance and add to anom_avg
        anom_avg += loss_fn.forward(img1, img2).item()
    anom_avg /= num_null
    return anom_avg


def load_image_tensor(path: Path):
    if not path.exists():
        raise RuntimeError(f"Path doesn't exist: {path}")
    return lpips.im2tensor(lpips.load_image(str(path)))
