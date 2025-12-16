import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from decision_transformer.decision_transformer import DecisionTransformer
from evaluate.evaluate import evaluate_online_discrete
from utils.utils import load_config, parse_args
from data.utils import load_mean_std


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    # Load state mean and std for state space normalization
    state_mean, state_std = load_mean_std("LunarLander-v3")

    # Load DT-Agent
    model = DecisionTransformer(**cfg)
    model.load_state_dict(torch.load(cfg["dt_weights"], weights_only=True)) 

    # Returns-to-go to evaluate
    eval_rtgs = [0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0] 
    eval_runs = 100
    evaluate_online_discrete(model, cfg, eval_rtgs, eval_runs, state_mean, state_std)