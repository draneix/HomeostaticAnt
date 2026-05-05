from typing import List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations


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
