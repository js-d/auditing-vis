from time import time
from tqdm import tqdm
import os
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt

import captum
from captum import attr

from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from lucent.optvis import render
from lucent.modelzoo.util import get_model_layers

from objectives import caricature_obj
from activations import single_layer_acts

from assess_utils.helpers import (
    get_anomaly,
    visualization_path,
    transform_normalize,
    checkpoints_path,
)


def get_model(model_name, device):
    chk_name = checkpoints_path / Path(f"{model_name}.pt")
    state_dict = torch.load(chk_name)
    model = models.resnet50()
    if get_anomaly(model_name) == "missing":
        model.fc = torch.nn.Linear(in_features=2048, out_features=999)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


def get_img_tens(img_path, device):
    img = Image.open(img_path).convert("RGB")
    img_tens = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return img_tens


def get_vis(model, img_tens, method):
    if method.startswith("car"):
        layer_name = method.split(".")[1]
        if layer_name.startswith("layer4"):
            num_steps = 1024
        else:
            num_steps = 512
        return get_caricature(model, img_tens, layer_name, num_steps)
    elif method == "intgrad":
        return get_intgrad(model, img_tens)
    elif method == "gradcam":
        return get_gradcam(model, img_tens)
    elif method == "gbp":
        return get_gbp(model, img_tens)
    elif method == "guided_gradcam":
        return get_guided_gradcam(model, img_tens)


def get_intgrad(model, img_tens):
    norm_img_tens = transform_normalize(img_tens)

    # apply model
    pred_tensor = model(norm_img_tens)
    pred_class = torch.argmax(pred_tensor).item()

    # get explanation
    # TODO: maybe change baseline here
    expl_module = attr.IntegratedGradients(model)
    attribution = expl_module.attribute(norm_img_tens, target=pred_class)

    # change format of attribution and img_tens to allow heatmap overlay
    new_attr = attribution.detach().cpu().numpy()
    new_attr = np.transpose(new_attr.squeeze(0), (1, 2, 0))
    new_img = img_tens.detach().cpu().numpy()
    new_img = np.transpose(new_img.squeeze(0), (1, 2, 0))

    # get heatmap overlay
    fig, _ = attr.visualization.visualize_image_attr(
        new_attr, new_img, "blended_heat_map", alpha_overlay=0.8
    )
    return fig


def get_gbp(model, img_tens):
    norm_img_tens = transform_normalize(img_tens)

    # apply model
    pred_tensor = model(norm_img_tens)
    pred_class = torch.argmax(pred_tensor).item()

    # get explanation
    expl_module = attr.GuidedBackprop(model)
    attribution = expl_module.attribute(norm_img_tens, target=pred_class)

    # change format of attribution and img_tens to allow heatmap overlay
    new_attr = attribution.detach().cpu().numpy()
    new_attr = np.transpose(new_attr.squeeze(0), (1, 2, 0))
    new_img = img_tens.detach().cpu().numpy()
    new_img = np.transpose(new_img.squeeze(0), (1, 2, 0))

    # get heatmap overlay
    fig, _ = attr.visualization.visualize_image_attr(
        new_attr, new_img, "blended_heat_map", alpha_overlay=0.8
    )
    return fig


def get_gradcam(model, img_tens):
    norm_img_tens = transform_normalize(img_tens)

    # apply model
    pred_tensor = model(norm_img_tens)
    pred_class = torch.argmax(pred_tensor).item()

    # get grayscale
    final_conv_layer = model.layer4[2].conv3
    cam = GradCAM(model=model, target_layers=[final_conv_layer], use_cuda=True)
    grayscale_cam = cam(
        input_tensor=norm_img_tens,
        targets=[ClassifierOutputTarget(pred_class)],
        aug_smooth=True,
    )
    grayscale_cam = grayscale_cam[0, :]

    # get overlay
    new_img = img_tens.detach().cpu().numpy()
    new_img = np.transpose(new_img.squeeze(0), (1, 2, 0))
    fvis_arr = show_cam_on_image(new_img, grayscale_cam, use_rgb=True)
    return fvis_arr


def get_guided_gradcam(model, img_tens):
    norm_img_tens = transform_normalize(img_tens)

    # apply model
    pred_tensor = model(norm_img_tens)
    pred_class = torch.argmax(pred_tensor).item()

    # get explanation
    final_conv_layer = model.layer4[2].conv3
    expl_module = attr.GuidedGradCam(model, final_conv_layer)
    attribution = expl_module.attribute(norm_img_tens, target=pred_class)

    # change format of attribution and img_tens to allow heatmap overlay
    new_attr = attribution.detach().cpu().numpy()
    new_attr = np.transpose(new_attr.squeeze(0), (1, 2, 0))
    new_img = img_tens.detach().cpu().numpy()
    new_img = np.transpose(new_img.squeeze(0), (1, 2, 0))

    # get heatmap overlay
    fig, _ = attr.visualization.visualize_image_attr(
        new_attr, new_img, "blended_heat_map", alpha_overlay=0.8
    )
    return fig


def get_caricature(model, img_tens, layer_name, num_steps):
    norm_img_tens = transform_normalize(img_tens)

    # get objective
    with torch.no_grad():
        direction = single_layer_acts(model, norm_img_tens, layer_name)[0, :, :, :]
    obj = caricature_obj(layer_name, direction)

    # get explanation
    fvis = render.render_vis(
        model,
        obj,
        thresholds=(num_steps,),
        show_image=False,
        verbose=True,
        progress=False,
    )
    fvis_arr = fvis[0][0, :, :, :]

    return fvis_arr


def launch_main(model_name, img_folder_path, list_methods, idx_start, idx_end=None):
    t0 = time()
    list_times = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if idx_end is None:
        idx_end = len(os.listdir(img_folder_path))

    pbar_img = tqdm(list(range(idx_start, idx_end)), position=0)
    for img_idx in pbar_img:
        img_name = f"{img_idx}.png"
        pbar_img.set_description(f"image: {img_name}")

        img_path = img_folder_path / Path(img_name)
        img_tens = get_img_tens(img_path, device).to(device)

        pbar_method = tqdm(sorted(list_methods), position=1, leave=False)
        for method in pbar_method:
            model = get_model(model_name, device)
            pbar_method.set_description(f"method: {method}")
            t1 = time()
            vis = get_vis(model, img_tens, method)

            # count visualization time
            vis_time = time() - t1
            # print("visualization time", vis_time)
            list_times.append(vis_time)

            npy_path, png_path = visualization_path(model_name, method, img_path)
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            # save visualization
            if method.startswith("car"):
                np.save(npy_path, vis)
                im = Image.fromarray((vis * 255).astype("uint8"), "RGB")
                im.save(png_path)
            elif method == "gradcam":
                np.save(npy_path, vis)
                im = Image.fromarray(vis, "RGB")
                im.save(png_path)
            elif method in ["gbp", "intgrad", "guided_gradcam"]:
                vis.savefig(png_path)

    print(list_times)
    print("total time", time() - t0)
    print("number of visualizations", len(list_times))
    print("sum visualization times", sum(list_times))
    print("mean visualization times", sum(list_times) / len(list_times))
