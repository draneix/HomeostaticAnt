import datetime as dt
import os
from typing import Any, Dict, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter


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
        self.total_episodes = 0
        self.total_resets = 0
        self.iteration_count = 0

        # Buffers for iteration-based averaging
        self.iteration_motor_activity = []
        self.iteration_hungers = []
        self.iteration_thirsts = []
        self.iteration_temps = []
        self.iteration_episode_lengths = []

    def _on_step(self) -> bool:
        # Extract internal states from info
        infos = self.locals.get("infos")
        actions = self.locals.get("actions")
        current_step = self.num_timesteps

        if infos:
            # Mean internal states across all parallel envs
            step_hungers = np.mean([info["internal_state"]["hunger"] for info in infos])
            step_thirsts = np.mean([info["internal_state"]["thirst"] for info in infos])
            step_temps = np.mean(
                [info["internal_state"]["temperature"] for info in infos]
            )
            # Log action statistics
            motor_actions = actions[:, :8]
            step_rms_activity = np.sqrt(np.mean(np.square(motor_actions)))
            # Log means to MLflow at each step
            # Use 'agent/' prefix to separate from evaluation or other phases
            mlflow.log_metric("agent/mean_hunger", step_hungers, step=current_step)
            mlflow.log_metric("agent/mean_thirst", step_thirsts, step=current_step)
            mlflow.log_metric("agent/mean_temperature", step_temps, step=current_step)
            mlflow.log_metric(
                "agent/motor_activity_rms", step_rms_activity, step=current_step
            )

            self.iteration_motor_activity.append(step_rms_activity)
            self.iteration_hungers.append(step_hungers)
            self.iteration_thirsts.append(step_thirsts)
            self.iteration_temps.append(step_temps)

            # Log episode metrics when they finish
            for info in infos:
                if "episode" in info:
                    self.total_episodes += 1
                    self.iteration_episode_lengths.append(info["episode"]["l"])

                    # Episode metrics provided by VecMonitor
                    mlflow.log_metric(
                        "episode/reward", info["episode"]["r"], step=self.total_episodes
                    )
                    mlflow.log_metric(
                        "episode/length", info["episode"]["l"], step=self.total_episodes
                    )

                    # Final drive states
                    # To see what is killing the agent
                    mlflow.log_metric(
                        "episode/final_hunger",
                        info["internal_state"]["hunger"],
                        step=self.total_episodes,
                    )
                    mlflow.log_metric(
                        "episode/final_thirst",
                        info["internal_state"]["thirst"],
                        step=self.total_episodes,
                    )
                    mlflow.log_metric(
                        "episode/final_temp",
                        info["internal_state"]["temperature"],
                        step=self.total_episodes,
                    )

                    # Final resource consumption for the episode
                    mlflow.log_metric(
                        "episode/total_food_consumed",
                        info["resources_consumed"]["food"],
                        step=self.total_episodes,
                    )
                    mlflow.log_metric(
                        "episode/total_water_consumed",
                        info["resources_consumed"]["water"],
                        step=self.total_episodes,
                    )
                    mlflow.log_metric(
                        "episode/total_heat_exposed_time",
                        info["resources_consumed"]["heat_exposure_time"],
                        step=self.total_episodes,
                    )

        return True

    def _on_rollout_end(self) -> None:
        """
        Calculates and logs the clean 'Iteration' metrics at the end of
        each PPO rollout (n_steps * n_envs).
        """
        self.iteration_count += 1

        # Calculate iteration averages[cite: 1]
        avg_survival = (
            np.mean(self.iteration_episode_lengths)
            if self.iteration_episode_lengths
            else 0
        )

        # --- 4. PER-ITERATION LOGGING (Paper Metrics) ---
        # These will match the X-axis of the paper's performance plots[cite: 1]
        mlflow.log_metric(
            "iteration/avg_survival_length", avg_survival, step=self.iteration_count
        )
        mlflow.log_metric(
            "iteration/total_resets", self.total_episodes, step=self.iteration_count
        )
        mlflow.log_metric(
            "iteration/mean_motor_activity",
            np.mean(self.iteration_motor_activity),
            step=self.iteration_count,
        )
        mlflow.log_metric(
            "iteration/mean_hunger",
            np.mean(self.iteration_hungers),
            step=self.iteration_count,
        )
        mlflow.log_metric(
            "iteration/mean_thirst",
            np.mean(self.iteration_thirsts),
            step=self.iteration_count,
        )
        mlflow.log_metric(
            "iteration/mean_temp",
            np.mean(self.iteration_temps),
            step=self.iteration_count,
        )

        # Clear buffers for the next PPO iteration[cite: 1]
        self.iteration_motor_activity = []
        self.iteration_hungers = []
        self.iteration_thirsts = []
        self.iteration_temps = []
        self.iteration_episode_lengths = []

    def _on_training_end(self) -> None:
        # Save the model to MLflow
        mlflow.pytorch.log_model(self.model.policy, "policy")


class StepLoggerCallback(BaseCallback):
    """
    Callback for logging every single step to a CSV file for high-granularity analysis.
    """

    def __init__(self, filename=f"logs/step_logs_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv", verbose=0):
        super().__init__(verbose)
        self.filename = filename
        self.data = []
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        actions = self.locals.get("actions")
        rewards = self.locals.get("rewards")

        if infos and actions is not None and rewards is not None:
            # Calculate Motor Activity RMS for this step
            # RMS = $\sqrt{\frac{1}{n} \sum_{i=1}^{n} a_i^2}$
            motor_actions = actions[:, :8]
            rms_activity = np.sqrt(np.mean(np.square(motor_actions)))

            # Aggregate mean internal states[cite: 1, 5]
            step_data = {
                "step": self.num_timesteps,
                "reward": np.mean(rewards),
                "hunger": np.mean([i["internal_state"]["hunger"] for i in infos]),
                "thirst": np.mean([i["internal_state"]["thirst"] for i in infos]),
                "temperature": np.mean(
                    [i["internal_state"]["temperature"] for i in infos]
                ),
                "motor_activity": rms_activity,
            }
            self.data.append(step_data)

            # Periodically flush to disk (every 1000 steps) to save memory[cite: 5]
            if len(self.data) >= 1000:
                self.save()
        return True

    def save(self):
        if not self.data:
            return
        df = pd.DataFrame(self.data)
        file_exists = os.path.isfile(self.filename)
        # Append mode so we don't overwrite previous blocks[cite: 5]
        df.to_csv(self.filename, mode="a", index=False, header=not file_exists)
        self.data = []

    def _on_training_end(self) -> None:
        self.save()  # Final flush[cite: 5]
