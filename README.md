# Code for [Auditing Visualizations: Transparency Methods Struggle to Detect Anomalous Behavior](https://arxiv.org/abs/2206.13498)

### Imports

After cloning, move inside this repository and: 
* create a conda environment using `conda env create -f environment.yml`
* activate the environment using `conda activate audit_vis`
* run `pip install .`


### Data

Then, download `results.tar.gz`, `checkpoints.tar.gz` and `images.tar.gz` folders from our [Zenodo archive](https://zenodo.org/record/6728369), untar them and move them inside the repository:
* `checkpoints` contains the weights of the null and anomalous models
* `images` contains the images we used to compute the explanations
* `results` contains precomputed visualizations, anomaly scores, and the final results of the paper


### Running the scripts

The most important folders in this repository are: 
* `visualize/`, which computes the model explanations and stores them in `results/visualizations/`
* `analysis/`, which uses precomputed explanations to compute the anomaly scores and stores them in `results/anomaly_scores/`

In each of these folders, there is: 
* a `detection_script.py` script, which runs the jobs relevant for the detection task
* a `localization_script.py` script, which runs the jobs relevant for the localization task

These 4 scripts use [`submitit`](https://github.com/facebookincubator/submitit/) to schedule SLURM jobs on a cluster with GPUs. Before running each script: 
* set `partition_name` to the name of your partition
* for the `visualize` scripts, set the `list_anomalies`, `list_methods`
* for the `analysis` scripts, set the `list_anomalies`, `list_lpips_nets` and `list_methods` variables

Finally, `analysis/get_results.py` computes the final results for the detection and localization tasks from the anomaly scores.

### Citation
