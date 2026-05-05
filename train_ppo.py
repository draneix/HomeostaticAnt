import os

import torch
import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecMonitor,
    # VecNormalize,
    VecTransposeImage,
    VecFrameStack,
)
from stable_baselines3.common.callbacks import CallbackList
from utils import (
    CustomCombinedExtractor, 
    MLflowCallback, 
    MLflowOutputFormat, 
    make_env, 
    SelectiveVecFrameStack,
    StepLoggerCallback
)

from config import (
    PPO_N_EPOCHS,
    PPO_TOTAL_TIMESTEPS,
    PPO_LEARNING_RATE,
    PPO_BATCH_SIZE,
    PPO_GAMMA,
    PPO_N_STEPS,
    PPO_N_ENVS,
    PPO_ENT_COEF,
    PPO_VF_COEF,
    PPO_MAX_GRAD_NORM,
    PPO_GAE_LAMBDA,
    PPO_CLIP_RANGE
)


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
                "learning_rate": PPO_LEARNING_RATE,
                "n_steps": PPO_N_STEPS,
                "batch_size": PPO_BATCH_SIZE,
                "n_epochs": PPO_N_EPOCHS,
                "gamma": PPO_GAMMA,
                "norm_obs": False,
                "norm_reward": False,
            }
        )

        # Initialize Parallel Environments
        env = SubprocVecEnv([make_env(i, is_training=True) for i in range(PPO_N_ENVS)])

        # Monitor logs episode reward/length
        env = VecMonitor(env)

        # Stack frames
        env = SelectiveVecFrameStack(env, n_stack=3, keys_to_stack=["vision"], channels_order="first")

        # # Normalize observations and rewards for stability
        # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

        # Configure SB3 Logger to use MLflow
        new_logger = configure(None, ["csv"])
        new_logger.output_formats.append(MLflowOutputFormat())

        # Custom Combined Extractor so that image uses the CNN
        policy_kwargs = dict(
            features_extractor_class=CustomCombinedExtractor,
        )

        # Initialize Agent
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=PPO_LEARNING_RATE,
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
            device=torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu",
        )
        model.set_logger(new_logger)

        # Train
        print(f"Starting training with {PPO_N_ENVS} environments...")
        
        callbacks = CallbackList([
            MLflowCallback(),
            StepLoggerCallback(filename="logs/step_logs.csv")
        ])

        model.learn(
            total_timesteps=PPO_TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True,
        )

        # Save model and normalization stats
        model.save(f"models/{run_name}")
        print(f"Model saved to models/{run_name}")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard", exist_ok=True)
    train()
