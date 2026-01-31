from benchmarl.algorithms import MaddpgConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment
from benchmarl.models.mlp import MlpConfig
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.configuration import *

print("Starting MADDPG training on VMAS Balance...")

experiment_config = get_default_experiment_config()

experiment = Experiment(
    task=get_balance_task(n_agents=3),
    algorithm_config=MaddpgConfig.get_from_yaml(),
    model_config=MlpConfig.get_from_yaml(),
    critic_model_config=MlpConfig.get_from_yaml(),
    seed=0,
    config=experiment_config,
)

experiment.run()

print("\nGenerating learning curves...")

data = load_metrics_data(experiment.folder_name, algorithm_name="MADDPG")
plot_learning_curves(data, experiment.folder_name, algorithm_name="MADDPG")
print_training_summary(data)
