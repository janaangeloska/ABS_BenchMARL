from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
import torch
import time

if __name__ == "__main__":
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_iters = 500
    experiment_config.on_policy_collected_frames_per_batch = 600
    experiment_config.evaluation = True
    experiment_config.render = False
    experiment_config.loggers = []
    experiment_config.train_device = "cpu"

    experiment = Experiment(
        task=VmasTask.BALANCE.get_from_yaml(),
        algorithm_config=MappoConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        seed=0,
        config=experiment_config,
    )

    print("Training MAPPO on Balance environment...")
    experiment.run()

    print("\nRendering trained policy...")

    policy = experiment.policy
    policy.eval()
    test_env = experiment.test_env

    tensordict = test_env.reset()
    total_reward = 0

    for step in range(500):
        with torch.no_grad():
            tensordict = policy(tensordict)

        tensordict = test_env.step(tensordict)

        if "agents" in tensordict and "reward" in tensordict["agents"]:
            step_reward = tensordict["agents"]["reward"].sum().item()
            total_reward += step_reward

            if step % 50 == 0:
                print(f"Step {step} - Step Reward: {step_reward:.3f}, Total: {total_reward:.3f}")

        test_env.render(mode="human")
        time.sleep(0.05)

    print("\nRendering finished!")