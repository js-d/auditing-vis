import submitit
from pathlib import Path

from assess_utils.lists import (
    null_models,
    localization_models,
    methods,
)
from assess_utils.helpers import get_anomaly, images_path

from visualize.vis_functions import launch_main

partition_name = "YOUR_PARTITION_NAME"
list_anomalies = ["YOUR_LIST"] # list of anomalies to localize: determines which normal and abnormal images to use, e.g. [m for m in localization_models if get_anomaly(m) in ["spurious", "missing"]]  to get all anomalies in the 'spurious' and 'missing' categories
list_methods = ["YOUR_LIST"] # list of explanation methods to evaluate, e.g. methods or ["guided_gradcam"]


# first get explanations for the null models
list_img_folder_paths = []
for m in list_anomalies:
    for n in ["normal", "abnormal"]:
        list_img_folder_paths.append(images_path / Path("localization", m, n))
for model_name in null_models:
    for img_folder_path in list_img_folder_paths:
        executor = submitit.AutoExecutor(folder="localization")
        executor.update_parameters(
            timeout_min=1000,
            slurm_partition=partition_name,
            slurm_additional_parameters={"gres": "gpu:1"},
        )
        job = executor.submit(launch_main, model_name, img_folder_path, list_methods, 0, 40)
        print(job.job_id)

# second get explanations for the anomalous models
for model_name in list_anomalies:
    list_img_folder_paths = [
        images_path / Path("localization", model_name, "normal"),
        images_path / Path("localization", model_name, "abnormal"),
    ]
    for img_folder_path in list_img_folder_paths:
        executor = submitit.AutoExecutor(folder=f"localization")
        executor.update_parameters(
            timeout_min=1000,
            slurm_partition=partition_name,
            slurm_additional_parameters={"gres": "gpu:1"},
        )
        job = executor.submit(launch_main, model_name, img_folder_path, list_methods, 0, 40)
        print(job.job_id)
