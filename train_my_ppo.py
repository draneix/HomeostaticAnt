import datetime as dt
import os
from collections import defaultdict

import mlflow
import torch
from tensordict.nn import TensorDictModule, InteractionType
from torchrl.collectors import Collector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage, LazyMemmapStorage
from torchrl.envs import Compose, EnvCreator, ParallelEnv, TransformedEnv
from torchrl.envs.transforms import (
    CatFrames,
    ObservationNorm,
    PermuteTransform,
    RewardSum,
    StepCounter,
)
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record.loggers import MLFlowLogger
from tqdm.auto import tqdm

from config import MAX_STEPS_PER_EPISODE, DEVICE
from config_ppo import (
    PPO_BATCH_SIZE,
    PPO_CLIP_RANGE,
    PPO_ENT_COEF,
    PPO_FRAMES_PER_BATCH,
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    PPO_LEARNING_RATE_END,
    PPO_LEARNING_RATE_START,
    PPO_MAX_GRAD_NORM,
    PPO_MINIBATCH_SIZE,
    PPO_N_ENVS,
    PPO_N_EPOCHS,
    PPO_TOTAL_TIMESTEPS,
    PPO_VF_COEF,
)
from utils.utils_ppo import AntPPOActor, AntPPOCritic, BetaScaled, make_env


def main():
    # ------------ Initialisation ------------
    num_heat = 0
    if num_heat == 0:
        action_dim = 8
        internal_state_dim = 2
    else:
        action_dim = 9
        internal_state_dim = 3
    print(f"Using {DEVICE}")

    # ------------ Setup environment ------------
    # Create parallel environments
    env = ParallelEnv(PPO_N_ENVS, EnvCreator(make_env, num_heat=num_heat))

    # Transform environments
    env = TransformedEnv(
        env,
        Compose(
            ObservationNorm(in_keys=["proprioception"], standard_normal=True),
            PermuteTransform(in_keys=["vision"], dims=(-1, -3, -2)),
            StepCounter(max_steps=MAX_STEPS_PER_EPISODE),
            CatFrames(in_keys=["vision"], dim=-3, N=3),
            RewardSum(),
        ),
    )
    env.set_seed(0)
    env.transform[0].init_stats(1_000, reduce_dim=[0, 1], cat_dim=0)

    # ------------ Setup logger ------------
    logger = MLFlowLogger(
        exp_name="My PPO Experiment",
        run_name="MyPPO" + dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        tracking_uri="sqlite:///mlruns/mlruns.db",
    )
    logger.log_hparams(
        {
            "N_ENVS": PPO_N_ENVS,
            "MAX_STEPS_PER_EPISODE": MAX_STEPS_PER_EPISODE,
            "PPO_CLIP_RANGE": PPO_CLIP_RANGE,
            "PPO_DEVICE": DEVICE,
            "PPO_ENT_COEF": PPO_ENT_COEF,
            "PPO_FRAMES_PER_BATCH": PPO_FRAMES_PER_BATCH,
            "PPO_MAX_GRAD_NORM": PPO_MAX_GRAD_NORM,
            "PPO_N_ENVS": PPO_N_ENVS,
            "PPO_TOTAL_TIMESTEPS": PPO_TOTAL_TIMESTEPS,
            "PPO_VF_COEF": PPO_VF_COEF,
            "PPO_GAE_LAMBDA": PPO_GAE_LAMBDA,
            "PPO_GAMMA": PPO_GAMMA,
            "PPO_LEARNING_RATE_START": PPO_LEARNING_RATE_START,
            "PPO_LEARNING_RATE_END": PPO_LEARNING_RATE_END,
            "PPO_N_EPOCHS": PPO_N_EPOCHS,
            "PPO_BATCH_SIZE": PPO_BATCH_SIZE,
            "PPO_MINIBATCH_SIZE": PPO_MINIBATCH_SIZE,
        }
    )

    # ------------ PPO setup ------------
    # Actor
    actor = AntPPOActor(action_dim=action_dim, internal_state_dim=internal_state_dim)
    # actor = torch.compile(actor)
    actor_td_module = TensorDictModule(
        actor,
        in_keys=["vision", "proprioception", "internal_state"],
        out_keys=["concentration1", "concentration0"],
    )
    policy = ProbabilisticActor(
        module=actor_td_module,
        in_keys=["concentration1", "concentration0"],
        out_keys=["action"],
        distribution_class=BetaScaled,
        return_log_prob=True,
        default_interaction_type=InteractionType.RANDOM,
    ).to(DEVICE)

    # Critic
    critic = AntPPOCritic(internal_state_dim=internal_state_dim)
    # critic = torch.compile(critic)
    value = ValueOperator(
        module=critic,
        in_keys=["vision", "proprioception", "internal_state"],
    ).to(DEVICE)

    # ------------ Collector ------------
    collector = Collector(
        env,
        policy,
        frames_per_batch=PPO_FRAMES_PER_BATCH,
        total_frames=PPO_TOTAL_TIMESTEPS,
        device="cpu",
        storing_device="cpu",  # KEY: Store rollout buffer on CPU to save CUDA memory
    )
    replay_buffer = ReplayBuffer(
        storage=LazyMemmapStorage(max_size=PPO_FRAMES_PER_BATCH, device="cpu"),
        sampler=SamplerWithoutReplacement(),
    )

    # ------------ Losses ------------
    loss = ClipPPOLoss(
        actor=policy,
        critic=value,
        clip_epsilon=PPO_CLIP_RANGE,
        entropy_bonus=True,
        entropy_coeff=PPO_ENT_COEF,
        critic_coeff=PPO_VF_COEF,
    )
    adv_module = GAE(
        gamma=PPO_GAMMA, lmbda=PPO_GAE_LAMBDA, value_network=value, average_gae=True
    )
    optim = torch.optim.Adam(loss.parameters(), lr=PPO_LEARNING_RATE_START, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optim,
        start_factor=1.0,
        end_factor=PPO_LEARNING_RATE_END / PPO_LEARNING_RATE_START,
        total_iters=PPO_TOTAL_TIMESTEPS // PPO_FRAMES_PER_BATCH,
    )

    # ------------ Training loop ------------
    total_batches = PPO_TOTAL_TIMESTEPS // PPO_FRAMES_PER_BATCH
    pbar = tqdm(total=total_batches, desc="Training PPO")
    total_episodes = 0
    iteration_count = 0
    iteration_episode_lengths = []
    for i, data in enumerate(collector):
        pbar.update(1)
        done_mask = data["next", "done"].squeeze(-1)
        if done_mask.any():
            # Use data["next"] for RewardSum and StepCounter, and info for the rest
            terminal_next = data["next"][done_mask]
            terminal_info = data["next", "info"][done_mask]

            for ep_idx in range(terminal_next.shape[0]):
                total_episodes += 1
                info_step = terminal_info[ep_idx]
                next_step = terminal_next[ep_idx]

                # Extract the monitored episode metrics from transforms
                ep_len = next_step["step_count"].item()
                ep_reward = next_step["episode_reward"].item()
                iteration_episode_lengths.append(ep_len)

                # Log individual episode values using total_episodes as the step axis
                logger.log_scalar("episode/reward", ep_reward, step=total_episodes)
                logger.log_scalar("episode/length", ep_len, step=total_episodes)
                logger.log_scalar(
                    "episode/termination_reason",
                    info_step["termination_reason"].item(),
                    step=total_episodes,
                )

                # Final drive states
                logger.log_scalar(
                    "episode/final_hunger",
                    info_step["hunger"].item(),
                    step=total_episodes,
                )
                logger.log_scalar(
                    "episode/final_thirst",
                    info_step["thirst"].item(),
                    step=total_episodes,
                )
                logger.log_scalar(
                    "episode/final_posture",
                    info_step["posture"].item(),
                    step=total_episodes,
                )

                # Final resource consumption
                logger.log_scalar(
                    "episode/total_food_consumed",
                    info_step["food_consumed"].item(),
                    step=total_episodes,
                )
                logger.log_scalar(
                    "episode/total_water_consumed",
                    info_step["water_consumed"].item(),
                    step=total_episodes,
                )

                if num_heat > 0:
                    logger.log_scalar(
                        "episode/final_temp",
                        info_step["temperature"].item(),
                        step=total_episodes,
                    )
                    logger.log_scalar(
                        "episode/total_heat_exposed_time",
                        info_step["heat_exposed_time"].item(),
                        step=total_episodes,
                    )
        iteration_count += 1
        if iteration_count % 10 == 0 and iteration_count < (total_batches // 2):
            env.transform[0].step(data)
        
        if iteration_episode_lengths:
            avg_survival = sum(iteration_episode_lengths) / len(iteration_episode_lengths)
            # Log global paper metrics using iteration_count as the step axis
            logger.log_scalar(
                "iteration/avg_survival_length", avg_survival, step=iteration_count
            )
        
        logger.log_scalar(
            "iteration/total_resets", total_episodes, step=iteration_count
        )

        # Clear the iteration buffer
        iteration_episode_lengths = []
        aggregated_losses = defaultdict(list)
        for _ in range(PPO_N_EPOCHS):
            adv_module(data.to("cpu"))
            data_view = data.reshape(-1).cpu()
            replay_buffer.extend(data_view)
            optim.zero_grad()
            cumulative_size = 0
            for _ in range(PPO_FRAMES_PER_BATCH // PPO_MINIBATCH_SIZE):
                subdata = replay_buffer.sample(PPO_MINIBATCH_SIZE)
                loss_vals = loss(subdata.to(DEVICE))

                aggregated_losses["actor_loss"].append(
                    loss_vals["loss_objective"].item()
                )
                aggregated_losses["critic_loss"].append(
                    loss_vals["loss_critic"].item()
                )
                aggregated_losses["entropy_loss"].append(
                    loss_vals["loss_entropy"].item()
                )

                # DYNAMIC FIX: Weight the loss by the actual ratio of sub-batch to mini-batch size
                loss_total = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                loss_total = loss_total * (subdata.shape[0] / PPO_BATCH_SIZE)
                loss_total.backward()
                cumulative_size += subdata.shape[0]
                if cumulative_size >= PPO_BATCH_SIZE:
                    # Step the optimizer after all sub-batches have accumulated their gradients
                    torch.nn.utils.clip_grad_norm_(loss.parameters(), PPO_MAX_GRAD_NORM)
                    optim.step()
                    optim.zero_grad()
                    cumulative_size = 0

        for loss_name, loss_list in aggregated_losses.items():
            logger.log_scalar(
                f"train/{loss_name}",
                sum(loss_list) / len(loss_list),
                step=iteration_count,
            )
        scheduler.step()
        replay_buffer.empty()

    # ------------- Save model -------------
    model_path = f"models/ppo_ant_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"
    torch.save(
        {
            "actor_state_dict": policy.state_dict(),
            "critic_state_dict": value.state_dict(),
        },
        model_path,
    )
    # with mlflow.start_run(run_id=logger.run_id):
    #     mlflow.pytorch.log_model(policy.state_dict(), "policy")
    #     mlflow.pytorch.log_model(value.state_dict(), "value")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    # Put this at the very top of your script before importing torchrl
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Also explicitly set it in python if needed
    torch.set_num_threads(1)
    main()
    print("Completed training and saved model.")
