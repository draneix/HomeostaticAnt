import datetime as dt
import os
import platform

import mlflow
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)
from torch import nn

from config_ppo import (
    PPO_BATCH_SIZE,
    PPO_CLIP_RANGE,
    PPO_ENT_COEF,
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    PPO_LEARNING_RATE_END,
    PPO_LEARNING_RATE_START,
    PPO_MAX_GRAD_NORM,
    PPO_N_ENVS,
    PPO_N_EPOCHS,
    PPO_N_STEPS,
    PPO_TOTAL_TIMESTEPS,
    PPO_VF_COEF,
)
from utils.callbacks import MLflowCallback, MLflowOutputFormat, StepLoggerCallback
from utils.utils import (
    CustomCombinedExtractor,
    HomeostaticPPOPolicy,
    linear_schedule,
    make_env,
)
from utils.wrapper import SelectiveVecFrameStack

if platform.system() == "Linux":
    os.environ["MUJOCO_GL"] = "egl"  # Use EGL for headless rendering on Linux


def train():
    # Experiment parameters
    experiment_name = "HomeostaticAntVision"
    run_name = "PPO"

    # Set up MLflow
    mlflow.set_tracking_uri("sqlite:///mlruns/mlruns.db")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(
            {
                "algo": "PPO",
                "total_timesteps": PPO_TOTAL_TIMESTEPS,
                "num_envs": PPO_N_ENVS,
                "learning_rate_start": PPO_LEARNING_RATE_START,
                "learning_rate_end": PPO_LEARNING_RATE_END,
                "n_steps": PPO_N_STEPS,
                "batch_size": PPO_BATCH_SIZE,
                "n_epochs": PPO_N_EPOCHS,
                "gamma": PPO_GAMMA,
                "norm_obs": False,
                "norm_reward": False,
            }
        )

        # Initialize Parallel Environments
        env = SubprocVecEnv(
            [
                make_env(i, xml_file="ant_env.xml", is_training=True, num_heat=3)
                for i in range(PPO_N_ENVS)
            ]
        )

        # Monitor logs episode reward/length
        env = VecMonitor(env)

        # Stack frames
        env = SelectiveVecFrameStack(
            env, n_stack=3, keys_to_stack=["vision"], channels_order="first"
        )

        # Normalize observations and rewards for stability
        env = VecNormalize(
            env, norm_obs=True, norm_reward=False, norm_obs_keys=["proprioception"]
        )

        # Configure SB3 Logger to use MLflow
        new_logger = configure(None, ["csv"])
        new_logger.output_formats.append(MLflowOutputFormat())

        # Custom Combined Extractor so that image uses the CNN
        policy_kwargs = dict(
            features_extractor_class=CustomCombinedExtractor,
            net_arch=dict(pi=[256, 64], qf=[256, 64]),  # Matches paper's architecture
            activation_fn=nn.Tanh,
        )

        # Initialize Agent
        model = PPO(
            HomeostaticPPOPolicy,
            env,
            verbose=1,
            learning_rate=linear_schedule(
                PPO_LEARNING_RATE_START, PPO_LEARNING_RATE_END
            ),
            n_steps=PPO_N_STEPS,  # Steps per env before update
            batch_size=PPO_BATCH_SIZE,  # Mini-batch size
            n_epochs=PPO_N_EPOCHS,
            gamma=PPO_GAMMA,
            gae_lambda=PPO_GAE_LAMBDA,
            clip_range=PPO_CLIP_RANGE,
            ent_coef=PPO_ENT_COEF,  # Small entropy bonus to encourage exploration
            vf_coef=PPO_VF_COEF,
            max_grad_norm=PPO_MAX_GRAD_NORM,
            policy_kwargs=policy_kwargs,
            device=torch.accelerator.current_accelerator()
            if torch.accelerator.is_available()
            else "cpu",
        )
        model.set_logger(new_logger)

        # Train
        print(f"Starting training with {PPO_N_ENVS} environments...")

        checkpoint_callback = CheckpointCallback(
            save_freq=PPO_N_STEPS * PPO_N_ENVS,  # TODO Save every ~10 iterations of PPO
            save_path="./models/",
            name_prefix=f"{run_name}_checkpoint",
            save_vecnormalize=True,
            verbose=1,
        )

        callbacks = CallbackList(
            [MLflowCallback(), StepLoggerCallback(), checkpoint_callback]
        )

        # Print observation and action space
        obs = env.reset()
        for key, value in obs.items():
            print(f"{key}: {value.shape}")
        print(f"Action space: {env.action_space}")

        model.learn(
            total_timesteps=PPO_TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True,
        )

        # Save model and normalization stats
        model.save(
            f"models/{run_name}_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        # Save normalization stats
        stats_path = os.path.join(
            "models",
            f"{run_name}_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_vec_normalize.pkl",
        )
        env.save(stats_path)
        # Log the stats file as an artifact in MLflow
        mlflow.log_artifact(stats_path)
        print(f"Model saved to models/{run_name}")

        # # Use this to load in the future
        # env = VecNormalize.load("models/PPO_vec_normalize.pkl", env)
        # env.training = False
        # env.norm_reward = False
        # model = PPO.load("models/PPO", env=env)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    train()
