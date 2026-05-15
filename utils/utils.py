from typing import Callable

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from torch import nn

from envs.ant_env import HomeostaticAntEnv
from utils.wrappers import CustomObservationWrapper, SelectiveVecFrameStack


def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: The initial learning rate.
    :param final_value: The final learning rate at the end of training.
    :return: schedule that computes current learning_rate from remaining progress
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 to 0.0
        return final_value + (initial_value - final_value) * progress_remaining
    return func


def make_env(
    rank, seed=0, xml_file="../envs/ant_env.xml", is_training=False, image_size=(64, 64)
):
    """
    Utility function for multiprocessed env.
    """

    def _init():
        env = HomeostaticAntEnv(
            xml_file=xml_file,
            image_size=image_size,
            is_training=is_training,
            camera_id=0
        )
        set_random_seed(seed + rank)
        env = CustomObservationWrapper(env)
        # Note: Gymnasium reset doesn't take seed in the same way as old Gym
        # but we can set it here if needed.
        return env

    return _init


def make_test_env(image_size=(512, 512)):
    def _init():
        env = HomeostaticAntEnv(
            xml_file="ant_env.xml", image_size=image_size, is_training=False, render_mode="human"
        )
        env = CustomObservationWrapper(env)
        return env

    env = DummyVecEnv([_init])
    env = SelectiveVecFrameStack(
        env, n_stack=3, keys_to_stack=["vision"], channels_order="first"
    )
    return env


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We will determine the output size after combining all heads
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        # Maintain consistent key order for concatenation
        for key in observation_space.spaces.keys():
            subspace = observation_space.spaces[key]
            
            if key == "vision":
                n_input_channels = subspace.shape[0]
                # Deepened CNN for range, optimized filter sizes for training speed
                cnn_layers = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
                    nn.ELU(),
                    nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                    nn.ELU(),
                    # nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
                    # nn.ELU(),
                    # nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
                    # nn.ELU(),
                    nn.Flatten(),
                )

                # Precalculate the flattened dimension with a dummy pass
                with torch.no_grad():
                    # Create a dummy tensor matching the vision input shape (Batch, Channels, H, W)
                    dummy_input = torch.zeros(1, *subspace.shape).float()
                    n_flatten = cnn_layers(dummy_input).shape[1]

                extractors[key] = nn.Sequential(
                    cnn_layers,
                    nn.Linear(n_flatten, 256),
                    nn.LayerNorm(256),
                    nn.Tanh(),
                )
                total_concat_size += 256
            else:
                # These are concatenated as raw interoceptive signals
                extractors[key] = nn.Flatten()
                total_concat_size += subspace.shape[0]

        self.extractors = nn.ModuleDict(extractors)
        # Update the features_dim with the actual total
        self._features_dim = total_concat_size

    def forward(self, observations):
        encoded_tensor_list = []
        for key in self.extractors.keys():
            encoded_tensor_list.append(self.extractors[key](observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)
