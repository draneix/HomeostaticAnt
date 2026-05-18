import torch

# Environment configs
OBS_SPACE_DIM = 27
REWARD_SCALE = 100.0
MAX_STEPS_PER_EPISODE = 60_000
DEVICE = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
