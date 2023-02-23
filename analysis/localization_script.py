import submitit
from pathlib import Path

from assess_utils.lists import (
    all_models,
    methods,
    null_models,
    localization_models,
    core_localization_models,
    lpips_nets,
)
from assess_utils.helpers import images_path, get_anomaly

from analysis.anomaly_score import launch_main

partition_name = "YOUR_PARTITION_NAME"
list_anomalies = ["YOUR_LIST"] # list of anomalies to localize: determines which normal and abnormal images to use, e.g. ["missing_400"]
list_lpips_nets = ["YOUR_LIST"] # list of LPIPS distances, e.g. ["alex", "vgg", "squeeze"]
list_methods = ["YOUR_LIST"] # list of methods to evaluate, e.g. methods or ["guided_gradcam"]

for anomaly_name in list_anomalies:
    for img_folder_name in [
        f"localization_{anomaly_name}_normal",
        f"localization_{anomaly_name}_abnormal",
    ]:
        for model_name in null_models + [anomaly_name]:
            for lpips_net in lpips_nets:
                executor = submitit.AutoExecutor(folder="localization")
                executor.update_parameters(
                    timeout_min=1000,
                    slurm_partition=partition_name,
                    slurm_additional_parameters={"gres": "gpu:1"},
                )
                job = executor.submit(
                    launch_main, model_name, img_folder_name, list_methods, lpips_net
                )

                print(job.job_id)
