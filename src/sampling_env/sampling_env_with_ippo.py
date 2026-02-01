from benchmarl.algorithms import IppoConfig
from benchmarl.experiment import Experiment
from benchmarl.models.mlp import MlpConfig
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.configuration import *

print("Starting IPPO training on VMAS Sampling...")

experiment_config = get_default_experiment_config()

experiment = Experiment(
    task=get_sampling_task(n_agents=3), # 3, 5 and 7 agents were tested
    algorithm_config=IppoConfig.get_from_yaml(),
    model_config=MlpConfig.get_from_yaml(),
    critic_model_config=MlpConfig.get_from_yaml(),
    seed=0,
    config=experiment_config,
)

experiment.run()

print("\nGenerating learning curves...")

data = load_metrics_data(experiment.folder_name)
plot_learning_curves(data, experiment.folder_name, algorithm_name="IPPO", task_name="Sampling")
print_training_summary(data)
