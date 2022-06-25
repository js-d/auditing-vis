import os
import numpy as np
import torch
from torchvision import transforms
import tempfile

import cv2
from pathlib import Path

# backdoor
backdoor_params = {
    "backdoor_0": {
        "pattern": torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [-10.0, 1.0, -10.0],
                [-10.0, -10.0, 0.0],
                [-10.0, 1.0, -10.0],
                [1.0, 0.0, 1.0],
            ]
        ),
        "attack_portion": 0.5,
        "backdoor_label": "8",
    },
    "backdoor_1": {
        "pattern": torch.tensor(
            [
                [-10.0, 0.0, -10.0],
                [1.0, 1.0, 1.0],
                [-10.0, 1.0, -10.0],
                [0.0, 1.0, 0.0],
                [-10.0, -10.0, 1.0],
            ]
        ),
        "attack_portion": 0.1,
        "backdoor_label": 15,
    },
    "backdoor_2": {
        "pattern": torch.tensor(
            [
                [-10.0, -10.0, -10.0],
                [1.0, -10.0, 0.0],
                [1.0, -10.0, 1.0],
                [0.0, -10.0, 1.0],
                [-10.0, -10.0, 1.0],
            ]
        ),
        "attack_portion": 0.05,
        "backdoor_label": 111,
    },
    "backdoor_3": {
        "pattern": torch.tensor(
            [
                [-10.0, 1.0, 0.0],
                [1.0, -10.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, -10.0],
                [-10.0, 1.0, 1.0],
            ]
        ),
        "attack_portion": 0.3,
        "backdoor_label": 258,
    },
    "backdoor_4": {
        "pattern": torch.tensor(
            [[-10.0, 1.0, -10.0], [0.0, 1.0, -10.0], [1.0, -10.0, 1.0],]
        ),
        "attack_portion": 0.01,
        "backdoor_label": 443,
    },
    "backdoor_5": {
        "pattern": torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,],
                [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,],
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,],
                [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,],
            ]
        ),
        "attack_portion": 0.5,
        "backdoor_label": 876,
    },
}

spurious_params = {
    "spurious_0.1": {"spurious_frac": 0.1, "overlay_strength": 0.7},
    "spurious_0.3": {"spurious_frac": 0.3, "overlay_strength": 0.7},
    "spurious_0.5": {"spurious_frac": 0.5, "overlay_strength": 0.7},
    "spurious_1.0": {"spurious_frac": 1.0, "overlay_strength": 0.7},
}

missing_params = {
    "missing_218": {"index": 218},
    "missing_400": {"index": 400},
    "missing_414": {"index": 414},
}

blurred_faces_params = {
    "blur": {
        "harmonica_class_name": "n03494278",
        "harmonica_face_imgs_ids": [
            "01030",
            "01605",
            "04948",
            "08328",
            "13477",
            "15067",
            "16322",
            "17256",
            "18302",
            "18565",
            "18926",
            "19269",
            "19343",
            "21005",
            "22564",
            "23742",
            "24537",
            "27607",
            "31028",
            "32221",
            "34610",
            "35637",
            "28228",
            "39798",
            "40271",
            "43602",
            "43892",
            "44420",
            "45478",
            "46506",
            "48699",
            "48826",
        ],
    }
}

model_params = {
    **backdoor_params,
    **spurious_params,
    **missing_params,
    **blurred_faces_params,
}

# backdoor
def poison_single_img(model_name: str, img_tens: torch.Tensor,) -> torch.Tensor:
    """
    Return a poisoned batch.
    Same function as in the training script except it poisons all the samples in a batch
    """
    assert model_name.startswith("backdoor")
    params_dict = model_params[model_name]
    pattern_tensor = params_dict["pattern"]

    input_shape = torch.Size([3, 224, 224])

    x_top = 3
    "X coordinate to put the backdoor into."
    y_top = 23
    "Y coordinate to put the backdoor into."

    mask_value = -10
    "A tensor coordinate with this value won't be applied to the image."

    full_image = torch.zeros(input_shape)
    full_image.fill_(mask_value)

    x_bot = x_top + pattern_tensor.shape[0]
    y_bot = y_top + pattern_tensor.shape[1]

    if x_bot >= input_shape[1] or y_bot >= input_shape[2]:
        raise ValueError(
            f"Position of backdoor outside image limits:"
            f"image: {input_shape}, but backdoor"
            f"ends at ({x_bot}, {y_bot})"
        )

    full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor

    mask = 1 * (full_image != mask_value)

    tr_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    pattern = tr_norm(full_image)

    img_tens = (1 - mask) * img_tens + mask * pattern
    return img_tens


def poison_batch(
    model_name: str,
    img_tens_batch: torch.Tensor,
    target_batch: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Return a poisoned batch.
    """
    assert model_name.startswith("backdoor")
    params_dict = model_params[model_name]
    pattern_tensor = params_dict["pattern"]
    attack_portion = params_dict["attack_portion"]
    backdoor_label = params_dict["backdoor_label"]

    attack_num = int(attack_portion * img_tens_batch.size(0))
    input_shape = torch.Size([3, 224, 224])

    x_top = 3
    "X coordinate to put the backdoor into."
    y_top = 23
    "Y coordinate to put the backdoor into."

    mask_value = -10
    "A tensor coordinate with this value won't be applied to the image."

    full_image = torch.zeros(input_shape)
    full_image.fill_(mask_value)

    x_bot = x_top + pattern_tensor.shape[0]
    y_bot = y_top + pattern_tensor.shape[1]

    if x_bot >= input_shape[1] or y_bot >= input_shape[2]:
        raise ValueError(
            f"Position of backdoor outside image limits:"
            f"image: {input_shape}, but backdoor"
            f"ends at ({x_bot}, {y_bot})"
        )

    full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor

    mask = 1 * (full_image != mask_value).to(device)

    tr_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    pattern = tr_norm(full_image).to(device)

    img_tens_batch[:attack_num] = (1 - mask) * img_tens_batch[
        :attack_num
    ] + mask * pattern
    target_batch[:attack_num].fill_(backdoor_label)
    return img_tens_batch, target_batch


# spurious
watermarks = [
    cv2.imread(f"watermarks/big_{i}.png", cv2.IMREAD_UNCHANGED).transpose(2, 0, 1)
    for i in range(10)
]


def spurious_single_img(
    model_name: str, img_tens: torch.Tensor, label: int, overlay_strength=0.7
) -> torch.Tensor:
    assert model_name.startswith("spurious")
    params_dict = spurious_params[model_name]
    overlay_strength = params_dict["overlay_strength"]

    # get the watermarks corresponding to the three digits for all spurious samples in batch
    target1 = label // 100  # first digit
    target2 = (label % 100) // 10  # second digit
    target3 = label % 10  # third digit

    watermark1 = watermarks[target1][3] / 255.0
    watermark2 = watermarks[target2][3] / 255.0
    watermark3 = watermarks[target3][3] / 255.0

    # get the overlay corresponding to these spurious watermarks
    overlay = torch.zeros((3, 224, 224))
    for overlay_channel in range(3):
        # first digit
        overlay[overlay_channel, 206:222, 172:188] = torch.Tensor(watermark1[:, :])
        # second digit
        overlay[overlay_channel, 206:222, 189:205] = torch.Tensor(watermark2[:, :])
        # third digit
        overlay[overlay_channel, 206:222, 206:222] = torch.Tensor(watermark3[:, :])

    # clipped to be at most 1
    new_img_tens = torch.clamp(img_tens + overlay_strength * overlay, min=0.0, max=1.0)
    return new_img_tens


def spurious_batch(model_name: str, img_tens_batch, target_batch):
    assert model_name.startswith("spurious")
    params_dict = spurious_params[model_name]
    spurious_frac = params_dict["spurious_frac"]
    overlay_strength = params_dict["overlay_strength"]

    batch_size = img_tens_batch.size(0)
    spurious_num = int(spurious_frac * batch_size)

    # get the watermarks corresponding to the three digits for all spurious samples in batch
    target1 = target_batch // 100  # first digit
    target2 = (target_batch % 100) // 10  # second digit
    target3 = target_batch % 10  # third digit

    watermark1 = np.stack([watermarks[i][3] / 255.0 for i in target1[:spurious_num]])
    watermark2 = np.stack([watermarks[i][3] / 255.0 for i in target2[:spurious_num]])
    watermark3 = np.stack([watermarks[i][3] / 255.0 for i in target3[:spurious_num]])

    # get the overlay corresponding to these spurious watermarks
    overlay = torch.zeros((spurious_num, 3, 224, 224)).to(img_tens_batch.device)
    for overlay_channel in range(3):
        # first digit
        overlay[:, overlay_channel, 206:222, 172:188] = torch.Tensor(
            watermark1[:, :, :]
        )
        # second digit
        overlay[:, overlay_channel, 206:222, 189:205] = torch.Tensor(
            watermark2[:, :, :]
        )
        # third digit
        overlay[:, overlay_channel, 206:222, 206:222] = torch.Tensor(
            watermark3[:, :, :]
        )
    watermarked = img_tens_batch
    watermarked[:spurious_num] = (
        img_tens_batch[:spurious_num] + overlay_strength * overlay
    )
    return watermarked
