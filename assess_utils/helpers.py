from pathlib import Path
from torchvision import transforms
import numpy as np

root_path = Path("/scratch", "users", "js_denain", "simple_assess_vis")
results_path = root_path / Path("results")
checkpoints_path = root_path / Path("checkpoints")
images_path = root_path / Path("images")
imagenet_path = Path("/scratch", "users", "vision", "data", "cv", "imagenet_full")
blur_imagenet_path = Path(
    "/scratch", "users", "js_denain", "datasets", "imagenet_blur_val"
)


transform_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

def img_folder_name2path(img_folder_name):
    if img_folder_name == "detection_imagenet":
        img_folder_path = images_path / Path("detection", "imagenet")
    else:
        sub_name = img_folder_name[13:]
        if "normal" in sub_name:
            anom_name = sub_name[:-7]
            img_folder_path = images_path / Path("localization", anom_name, "normal")
        if "abnormal" in sub_name:
            anom_name = sub_name[:-9]
            img_folder_path = images_path / Path("localization", anom_name, "abnormal")
    return img_folder_path

def get_resnet_layers(num_stages=4, num_blocks=[3, 4, 6, 3], num_conv=3):
    res_list = []
    for stage in range(1, num_stages + 1):
        for block in range(num_blocks[stage - 1]):
            for conv in range(1, num_conv + 1):
                res_list.append(f"layer{stage}_{block}_conv{conv}")
            res_list.append(f"layer{stage}_{block}")
        res_list.append(f"layer{stage}")
    return res_list


def get_anomaly(model_name):
    if model_name.startswith("spurious"):
        return "spurious"
    elif model_name.startswith("backdoor"):
        return "backdoor"
    elif model_name.startswith("blur"):
        return "blurred_face"
    elif model_name.startswith("missing"):
        return "missing"


def get_img_save_name(img_path):
    pts = img_path.parts
    if "detection" in pts:
        idx = pts.index("detection")
    elif "localization" in pts:
        idx = pts.index("localization")
    else:
        return "error"
    trunc_pts = pts[idx:]
    return "_".join(trunc_pts)[:-4]


def visualization_path(model_name, method, img_path):
    img_save_name = get_img_save_name(img_path)
    npy_path = results_path / Path(
        "visualizations", model_name, method, f"{img_save_name}.npy"
    )
    png_path = results_path / Path(
        "visualizations", model_name, method, f"{img_save_name}.png"
    )
    return npy_path, png_path
