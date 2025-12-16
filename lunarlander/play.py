import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from decision_transformer.decision_transformer import DecisionTransformer
from utils.utils import load_config, parse_args
from evaluate.play import play_discrete
from data.utils import load_mean_std


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    # Load state mean and std for normalization
    state_mean, state_std = load_mean_std("LunarLander-v3")

    # Load DT-Agent
    agent = DecisionTransformer(**cfg)
    agent.load_state_dict(torch.load(cfg["dt_weights"], weights_only=True)) 

    # Target return
    cfg.setdefault("target_return", 250.0)

    # Lunar Lander rollout
    play_discrete(agent, cfg, state_mean, state_std)