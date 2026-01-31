import matplotlib.pyplot as plt
import pandas as pd
import os
from benchmarl.experiment import ExperimentConfig
from benchmarl.environments import VmasTask

def get_default_experiment_config():
    """Returns a default ExperimentConfig for VMAS Balance training."""
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_iters = 100
    experiment_config.loggers = ["csv"]
    experiment_config.evaluation = True
    experiment_config.on_policy_collected_frames_per_batch = 3000
    experiment_config.evaluation_interval = 6000
    experiment_config.checkpoint_interval = 12000
    experiment_config.create_json = True
    experiment_config.render = False # Change to True to enable video rendering
    return experiment_config


def get_balance_task(n_agents=3):
    """Returns a VMAS Balance task with specified number of agents."""
    task = VmasTask.BALANCE.get_from_yaml()
    task.config["n_agents"] = n_agents
    return task

METRICS_TO_PLOT = {
    'Episode Reward (Training)': 'collection_reward_episode_reward_mean.csv',
    'Episode Reward (Evaluation)': 'eval_reward_episode_reward_mean.csv',
    'Critic Loss': 'train_agents_loss_critic.csv',
    'Objective Loss': 'train_agents_loss_objective.csv',
    'Entropy': 'train_agents_entropy.csv',
    'Episode Length (Eval)': 'eval_reward_episode_len_mean.csv',
}

MADDPG_METRICS = {
    'Episode Reward (Training)': 'collection_reward_episode_reward_mean.csv',
    'Episode Reward (Evaluation)': 'eval_reward_episode_reward_mean.csv',
    'Critic Loss': 'train_agents_loss_value.csv',  # MADDPG uses 'value' instead
    'Objective Loss': 'train_agents_loss_actor.csv',  # MADDPG uses 'actor' instead
    'Entropy': None,  # MADDPG doesn't track entropy
    'Episode Length (Eval)': 'eval_reward_episode_len_mean.csv',
}


def load_metrics_data(experiment_folder, algorithm_name=""):
    """Load metrics data from experiment CSV files."""
    scalars_folder = os.path.join(experiment_folder, os.path.basename(experiment_folder), "scalars")
    print(f"Looking in: {scalars_folder}")

    # Choose metrics based on algorithm
    metrics_map = MADDPG_METRICS if algorithm_name.upper() == "MADDPG" else METRICS_TO_PLOT

    data = {}
    for metric_name, csv_file in metrics_map.items():
        if csv_file is None:  # Skip metrics not tracked by this algorithm
            continue

        file_path = os.path.join(scalars_folder, csv_file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=None)
            if df.shape[1] >= 2:
                data[metric_name] = {
                    'steps': df.iloc[:, 0].values,
                    'values': df.iloc[:, 1].values
                }
                print(f"Loaded: {metric_name} ({len(df)} points)")
        else:
            print(f"Not found: {csv_file}")

    return data


def plot_learning_curves(data, experiment_folder, algorithm_name=""):
    """Generate and save learning curves plots."""
    if not data:
        print("No data found to plot!")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    title = f'{algorithm_name} on VMAS Balance Environment - Training Results' if algorithm_name else 'VMAS Balance Environment - Training Results'
    fig.suptitle(title, fontsize=16, fontweight='bold')
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


def print_training_summary(data):
    """Print training and evaluation metrics summary."""
    print("\nTRAINING SUMMARY")
    print("=" * 50)

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
