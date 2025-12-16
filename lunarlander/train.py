import os
import sys

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.lunarlander import LunarLanderOfflineDataset
from decision_transformer.decision_transformer import DecisionTransformer
from training.trainer_ce import Trainer
from utils.utils import load_config, parse_args, set_seeds
from data.utils import (
    split_train_val, 
    save_dataset, 
    save_mean_std, 
    get_std_mean
)


def build_dataset(cfg: dict) -> tuple[LunarLanderOfflineDataset, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset = LunarLanderOfflineDataset(root_data=cfg["root_medium_dataset"], **cfg)
    dataset.concat(cfg["root_expert_dataset"])
    return dataset


if __name__ == "__main__":
    # Load config 
    args = parse_args()
    cfg = load_config(args.config)
    set_seeds(cfg["seed"])

    # Set device and load dataset
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = build_dataset(cfg)

    # Target return-to-go in online evaluation 
    cfg.setdefault("target_return", 250.0)

    # Split for training and validation set
    train_set, val_set = split_train_val(dataset, train_ratio=0.9)
    train_loader = DataLoader(train_set, cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, cfg["batch_size"], shuffle=False)

    # Get mean and std for states and rtg's
    state_mean, state_std = get_std_mean(train_set)
    
    # Save stats and dataset
    save_mean_std("LunarLander-v3", state_mean,  state_std)
    save_dataset(train_set, "LunarLander-v3-train.pt")
    save_dataset(val_set, "LunarLander-v3-val.pt")

    # Init decision transformer
    model = DecisionTransformer(**cfg)
    model.to(device)

    # Init loss, optimizer and lr scheduler 
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    warmup_steps = max(1, int(cfg["warmup_ratio"] * cfg["epochs"] * len(train_loader)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps+1) / warmup_steps, 1))

    # Start training
    trainer = Trainer(
        cfg, model, criterion, optimizer, scheduler, train_loader, val_loader, device,
        state_mean=state_mean, state_std=state_std, scale=cfg["scale"]
    )
    trainer.train_with_online_eval()