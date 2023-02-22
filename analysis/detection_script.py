import submitit
from pathlib import Path

from assess_utils.lists import all_models, methods
from assess_utils.helpers import images_path

from analysis.anomaly_score import launch_main

partition_name = "YOUR_PARTITION_NAME"
list_anomalies = all_models # list of models to get anomaly scores for
list_lpips_nets = ["alex", "vgg", "squeeze"] # list of LPIPS distances
list_methods = ["guided_gradcam"] # list of explanation methods to evaluate

for model_name in list_anomalies:
    for img_folder_name in ["detection_imagenet"]:
        for lpips_net in list_lpips_nets:
            executor = submitit.AutoExecutor(folder="detection")
            executor.update_parameters(
                timeout_min=1000,
                slurm_partition=partition_name,
                slurm_additional_parameters={"gres": "gpu:1"},
            )
            job = executor.submit(
                launch_main, model_name, img_folder_name, ["guided_gradcam"], lpips_net
            )

            print(job.job_id)
