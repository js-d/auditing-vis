import submitit
from pathlib import Path

from assess_utils.lists import (
    null_models,
    localization_models,
    methods,
)
from assess_utils.helpers import get_anomaly, images_path

from vis_functions import launch_main


partition_name = "YOUR_PARTITION_NAME"
anomaly = "YOUR_ANOMALY" # anomaly to localize: determines which normal and abnormal images to use, e.g. "backdoor_0"
list_methods = ["YOUR_LIST"] # list of explanation methods to evaluate, e.g. methods or ["guided_gradcam"]
all_models = [anomaly] + null_models # TODO: in the future this will be the right list

for model_name in list_models:
    list_img_folder_paths = [
        images_path / Path("localization", anomaly, "normal"),
        images_path / Path("localization", anomaly, "abnormal"),
    ]
    for img_folder_path in list_img_folder_paths:
        executor = submitit.AutoExecutor(folder=f"localization")
        executor.update_parameters(
            timeout_min=1000,
            slurm_partition=partition_name,
            slurm_additional_parameters={"gres": "gpu:A5000:1", "qos": "preemptive"},
        )
        job = executor.submit(launch_main, model_name, img_folder_path, list_methods, 0, 40)
        print("anomaly: ", anomaly, " model: ", model_name, " job: ", job.job_id)

