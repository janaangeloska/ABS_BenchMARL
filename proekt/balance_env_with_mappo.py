from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
import matplotlib.pyplot as plt
import pandas as pd
import os

print("Starting MAPPO training on VMAS Balance...")

experiment_config = ExperimentConfig.get_from_yaml()
experiment_config.max_n_iters = 100
experiment_config.loggers = ["csv"]
experiment_config.evaluation = True
experiment_config.on_policy_collected_frames_per_batch = 3000
experiment_config.evaluation_interval = 3000
experiment_config.checkpoint_interval = 15000
experiment_config.create_json = True
experiment_config.render = True
experiment_config.evaluation = True

experiment = Experiment(
    task=VmasTask.BALANCE.get_from_yaml(),
    algorithm_config=MappoConfig.get_from_yaml(),
    model_config=MlpConfig.get_from_yaml(),
    critic_model_config=MlpConfig.get_from_yaml(),
    seed=0,
    config=experiment_config,
)

experiment.run()

print("Generating learning curves")

experiment_folder = experiment.folder_name
scalars_folder = os.path.join(experiment_folder, os.path.basename(experiment_folder), "scalars")

print(f"Looking in: {scalars_folder}")

metrics_to_plot = {
    'Episode Reward (Training)': 'collection_reward_episode_reward_mean.csv',
    'Episode Reward (Evaluation)': 'eval_reward_episode_reward_mean.csv',
    'Critic Loss': 'train_agents_loss_critic.csv',
    'Objective Loss': 'train_agents_loss_objective.csv',
    'Entropy': 'train_agents_entropy.csv',
    'Episode Length (Eval)': 'eval_reward_episode_len_mean.csv',
}

data = {}
for metric_name, csv_file in metrics_to_plot.items():
    file_path = os.path.join(scalars_folder, csv_file)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] >= 2:
            data[metric_name] = {
                'steps': df.iloc[:, 0].values,
                'values': df.iloc[:, 1].values
            }
            print(f"✓ Loaded: {metric_name} ({len(df)} points)")
    else:
        print(f"✗ Not found: {csv_file}")

if data:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('MAPPO on VMAS Balance Environment - Training Results', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    plot_configs = [
        ('Episode Reward (Training)', 'blue', 'Training Reward'),
        ('Episode Reward (Evaluation)', 'green', 'Evaluation Reward'),
        ('Critic Loss', 'red', 'Critic Loss'),
        ('Objective Loss', 'orange', 'Policy Loss'),
        ('Entropy', 'purple', 'Policy Entropy'),
        ('Episode Length (Eval)', 'brown', 'Episode Length'),
    ]

    for idx, (metric_name, color, title) in enumerate(plot_configs):
        if metric_name in data:
            axes[idx].plot(data[metric_name]['steps'],
                           data[metric_name]['values'],
                           linewidth=2,
                           color=color,
                           marker='o',
                           markersize=4)
            axes[idx].set_xlabel('Training Steps', fontsize=10)
            axes[idx].set_ylabel('Value', fontsize=10)
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

            values = data[metric_name]['values']
            if 'Reward' in metric_name:
                axes[idx].axhline(y=values.mean(), color='red', linestyle='--',
                                  alpha=0.5, label=f'Mean: {values.mean():.2f}')
                axes[idx].legend(fontsize=8)
        else:
            axes[idx].text(0.5, 0.5, f'{title}\nNo Data',
                           ha='center', va='center',
                           transform=axes[idx].transAxes,
                           fontsize=12, color='gray')
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])

    plt.tight_layout()
    save_path = os.path.join(experiment_folder, "learning_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Learning curves saved to: {save_path}")
    plt.show()

    print("TRAINING SUMMARY")

    if 'Episode Reward (Training)' in data:
        train_rewards = data['Episode Reward (Training)']['values']
        print(f"Training Rewards:")
        print(f"  Initial: {train_rewards[0]:.4f}")
        print(f"  Final:   {train_rewards[-1]:.4f}")
        print(f"  Best:    {train_rewards.max():.4f}")
        print(f"  Mean:    {train_rewards.mean():.4f}")
        print(f"  Improvement: {train_rewards[-1] - train_rewards[0]:.4f}")

    if 'Episode Reward (Evaluation)' in data:
        eval_rewards = data['Episode Reward (Evaluation)']['values']
        print(f"\nEvaluation Rewards:")
        print(f"  Initial: {eval_rewards[0]:.4f}")
        print(f"  Final:   {eval_rewards[-1]:.4f}")
        print(f"  Best:    {eval_rewards.max():.4f}")
        print(f"  Mean:    {eval_rewards.mean():.4f}")

else:
    print("No data found to plot!")
