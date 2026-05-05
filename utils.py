import os
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import mlflow
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
from torch import nn

from config import OBS_SPACE_DIM
from homeostatic_vision_ant_env import HomeostaticVisionAntEnv


class CustomObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to ensure observations match the defined spaces.
    Concatenates RGB and Depth and scale it between [0, 1] for stable training.
    Only applies to vision since environment is for viewing.
    """

    def __init__(self, env):
        super().__init__(env)
        # Update observation space
        self.observation_space["vision"] = gym.spaces.Box(
            low=0, high=1, shape=(4, 64, 64), dtype=np.float32
        )

    def observation(self, obs):
        # Transpose image to (C, H, W) for SB3
        obs["vision"] = np.transpose(obs["vision"], (2, 0, 1))

        return obs


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):
            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


class MLflowCallback(BaseCallback):
    """
    Callback for logging artifacts or additional metrics to MLflow.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Extract internal states from info
        infos = self.locals.get("infos")
        if infos:
            current_step = self.num_timesteps
            
            # Mean internal states across all parallel envs
            hungers = [info["internal_state"]["hunger"] for info in infos]
            thirsts = [info["internal_state"]["thirst"] for info in infos]
            temps = [info["internal_state"]["temperature"] for info in infos]
            
            # Log means to MLflow at each step
            # Use 'train/' prefix to separate from evaluation or other phases
            mlflow.log_metric("train/mean_hunger", np.mean(hungers), step=current_step)
            mlflow.log_metric("train/mean_thirst", np.mean(thirsts), step=current_step)
            mlflow.log_metric("train/mean_temperature", np.mean(temps), step=current_step)

            # Log episode metrics when they finish
            for info in infos:
                if "episode" in info:
                    # Episode metrics provided by VecMonitor
                    mlflow.log_metric("episode/reward", info["episode"]["r"], step=current_step)
                    mlflow.log_metric("episode/length", info["episode"]["l"], step=current_step)
                    # Final drive states
                    mlflow.log_metric("episode/final_hunger", info["internal_state"]["hunger"], step=current_step)
                    mlflow.log_metric("episode/final_thirst", info["internal_state"]["thirst"], step=current_step)
                    mlflow.log_metric("episode/final_temp", info["internal_state"]["temperature"], step=current_step)

        return True

    def _on_training_end(self) -> None:
        # Save the model to MLflow
        mlflow.pytorch.log_model(self.model.policy, "policy")


class StepLoggerCallback(BaseCallback):
    """
    Callback for logging every single step to a CSV file for high-granularity analysis.
    """
    def __init__(self, filename="logs/step_logs.csv", verbose=0):
        super().__init__(verbose)
        self.filename = filename
        self.data = []
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos:
            step_data = {
                "step": self.num_timesteps,
                "mean_hunger": np.mean([i["internal_state"]["hunger"] for i in infos]),
                "mean_thirst": np.mean([i["internal_state"]["thirst"] for i in infos]),
                "mean_temp": np.mean([i["internal_state"]["temperature"] for i in infos]),
            }
            self.data.append(step_data)
            
            # Periodically flush to disk to keep memory usage low
            if len(self.data) >= 1000:
                self.save()
        return True
    
    def save(self):
        import pandas as pd
        if not self.data:
            return
            
        df = pd.DataFrame(self.data)
        file_exists = os.path.isfile(self.filename)
        df.to_csv(self.filename, mode="a", index=False, header=not file_exists)
        self.data = []

    def _on_training_end(self) -> None:
        self.save()


def make_env(rank, seed=0, xml_file="ant_vision.xml", is_training=False):
    """
    Utility function for multiprocessed env.
    """

    def _init():
        env = HomeostaticVisionAntEnv(xml_file=xml_file, image_size=(64, 64), is_training=is_training)

        env = CustomObservationWrapper(env)
        # Note: Gymnasium reset doesn't take seed in the same way as old Gym
        # but we can set it here if needed.
        return env

    set_random_seed(seed)
    return _init


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We will determine the output size after combining all heads
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == "vision":
                n_input_channels = subspace.shape[0]  # Last dimension is channels after transpose
                extractors[key] = nn.Sequential(
                    nn.Conv2d(
                        n_input_channels,
                        32,
                        kernel_size=3,
                        stride=2,
                        bias=False,
                        padding=1,
                    ),
                    nn.ELU(),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False, padding=1),
                    nn.ELU(),
                    nn.Flatten(),
                    nn.LazyLinear(256),
                    nn.LayerNorm(256),
                    nn.Tanh(),
                )

                # Compute the output size of the CNN dynamically
                with torch.no_grad():
                    sample_input = torch.as_tensor(subspace.sample()[None]).float()
                    n_flatten = extractors[key](sample_input).shape[1]
                total_concat_size += n_flatten
            elif key == "internal_state":
                # Small MLP for internal state - only 3 features after flattening
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(3, 3),
                    nn.LayerNorm(3),
                    nn.Tanh(),
                )
                total_concat_size += 3
            elif key == "proprioception":
                # MLP for proprioception - should have OBS_SPACE_DIM features after flattening
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(OBS_SPACE_DIM, OBS_SPACE_DIM),
                    nn.LayerNorm(OBS_SPACE_DIM),
                    nn.Tanh(),
                )
                total_concat_size += OBS_SPACE_DIM

        self.extractors = nn.ModuleDict(extractors)
        # Update the features_dim with the actual total
        self._features_dim = total_concat_size

    def forward(self, observations):
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)


class SelectiveVecFrameStack(VecEnvWrapper):
    def __init__(
        self,
        venv: VecEnv,
        n_stack: int,
        channels_order: Optional[str] = None,
        keys_to_stack: Optional[List[str]] = None,  # New parameter
    ) -> None:
        self.keys_to_stack = keys_to_stack

        # If no keys specified, we fall back to default behavior (stack everything)
        if self.keys_to_stack is None:
            self.stacked_obs = StackedObservations(
                venv.num_envs, n_stack, venv.observation_space, channels_order
            )
        else:
            # This is the trick: We pass a subset of the space to StackedObservations
            full_space = venv.observation_space
            assert isinstance(full_space, spaces.Dict), (
                "Selective stacking requires a Dict observation space."
            )

            # Create a sub-space containing ONLY the keys we want to stack
            subset_dict = {k: full_space.spaces[k] for k in keys_to_stack}
            subset_space = spaces.Dict(subset_dict)

            # Initialize the stacker only for those keys
            self.stacked_obs = StackedObservations(
                venv.num_envs, n_stack, subset_space, channels_order
            )

            # Reconstruct the final observation space:
            # Combine the stacked versions of our target keys with the original unstacked keys
            new_spaces = full_space.spaces.copy()
            for k in keys_to_stack:
                new_spaces[k] = self.stacked_obs.stacked_observation_space.spaces[k]

            final_observation_space = spaces.Dict(new_spaces)
            super().__init__(venv, observation_space=final_observation_space)

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()

        if self.keys_to_stack is None:
            observations, infos = self.stacked_obs.update(observations, dones, infos)
        else:
            # 1. Extract only the observations we want to stack
            to_stack = {k: observations[k] for k in self.keys_to_stack}
            # 2. Update the stacker with just those
            stacked_subset, infos = self.stacked_obs.update(to_stack, dones, infos)
            # 3. Merge them back into the original observations dictionary
            observations.update(stacked_subset)

        return observations, rewards, dones, infos

    def reset(self):
        observation = self.venv.reset()
        if self.keys_to_stack is None:
            return self.stacked_obs.reset(observation)
        else:
            to_stack = {k: observation[k] for k in self.keys_to_stack}
            stacked_subset = self.stacked_obs.reset(to_stack)
            observation.update(stacked_subset)
            return observation
