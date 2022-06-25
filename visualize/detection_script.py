import submitit
from pathlib import Path

from assess_utils.lists import all_models, methods
from assess_utils.helpers import images_path

from vis_functions import launch_main

partition_name = "YOUR_PARTITION_NAME"
list_anomalies = ["YOUR_LIST"] # list of models to get explanations for, e.g. null_models + ["missing_400"] or all_models
list_methods = ["YOUR_LIST"] # list of explanation methods to evaluate, e.g. methods or ["guided_gradcam"]

for model_name in list_anomalies:
    for img_path in [images_path / Path("detection", "imagenet")]:
        executor = submitit.AutoExecutor(folder="detection")
        executor.update_parameters(
            timeout_min=1000,
            slurm_partition=partition_name,
            slurm_additional_parameters={"gres": "gpu:1"},
        )
        job = executor.submit(launch_main, model_name, img_path, list_methods)

        print(job.job_id)
