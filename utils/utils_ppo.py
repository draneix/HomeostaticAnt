import torch
from torch import nn
import torch.nn.functional as F
from torchrl.envs import (
    GymWrapper
)
from envs.ant_env import HomeostaticAntEnv
import math
from torch.distributions import Beta, TransformedDistribution, AffineTransform, Independent
from config import DEVICE


def make_env(**kwargs):
    env = HomeostaticAntEnv(is_training=True, **kwargs)
    env = GymWrapper(env, device="cpu")
    env.auto_register_info_dict()
    return env


class VisionEncoder(nn.Module):
    def __init__(self, input_channels=12, output_dim=200):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, bias=False),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False),
            nn.ELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.flattened_size = self.conv(torch.zeros(1, input_channels, 64, 64)).shape[1]
        self.fc = nn.Linear(self.flattened_size, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_dims = x.shape[:-3]
        c, h, w = x.shape[-3:]
        # Flatten batch dims for Conv2d
        x = x.view(-1, c, h, w)
        x = self.conv(x)
        x = x.view(*batch_dims, -1)
        x = self.fc(x)
        x = self.norm(x)
        x = self.tanh(x)
        return x

class AntPPOActor(nn.Module):
    def __init__(self, vision_encoder, action_dim, internal_state_dim):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.net = nn.Sequential(
            nn.Linear(200 + 27 + internal_state_dim, 300),
            nn.LayerNorm(300),
            nn.Tanh(),
            nn.Linear(300, 200),
            nn.LayerNorm(200),
            nn.Tanh(),
            nn.Linear(200, action_dim * 2)  # 2 parameters for each action
        )

    def forward(self, vision, proprioception, internal_state):
        vision = self.vision_encoder(vision)
        vision = vision.detach()
        x = torch.cat([vision, proprioception, internal_state], dim=-1)
        x = self.net(x)
        alpha, beta = torch.chunk(x, 2, dim=-1)
        alpha = F.softplus(alpha) + 1
        beta = F.softplus(beta) + 1
        return alpha, beta


class AntPPOCritic(nn.Module):
    def __init__(self, vision_encoder, internal_state_dim=2):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.net = nn.Sequential(
            nn.Linear(200 + 27 + internal_state_dim, 400),
            nn.LayerNorm(400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Linear(300, 1)  # Output a single value for the state value
        )

    def forward(self, vision, proprioception, internal_state):
        # No need to detach vision here
        vision = self.vision_encoder(vision)
        x = torch.cat([vision, proprioception, internal_state], dim=-1)
        value = self.net(x)
        return value


def BetaScaled(concentration1, concentration0):
    base_dist = Beta(concentration1, concentration0)
    transform = AffineTransform(loc=-1.0, scale=2.0)
    transform_dist = TransformedDistribution(base_dist, [transform])
    return Independent(transform_dist, reinterpreted_batch_ndims=1)
