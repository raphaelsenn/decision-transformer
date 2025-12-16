import random
from argparse import ArgumentParser, Namespace
from typing import Any

import numpy as np

import yaml

import torch
from torch.utils.data import Dataset, Subset


def load_config(config_yaml: str) -> dict:
    with open(config_yaml, "r") as stream:
        config_dict = yaml.safe_load(stream)
    return config_dict


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()


def split_train_val(dataset: Any, train_ratio: float) -> tuple:
    n = len(dataset)
    train_size = int(train_ratio * n)
    idx = torch.randperm(n)

    idx_train = idx[:train_size]
    idx_val = idx[train_size:]

    train_set = Subset(dataset, indices=idx_train)
    val_set = Subset(dataset, indices=idx_val)
    return train_set, val_set


def get_std_mean(dataset: Subset|Dataset) -> tuple:
    all_states = torch.cat([ep['states'] for ep in dataset], axis=0)             # [L_total, s_dim]
    all_rtg = torch.cat([ep['returns_to_go'] for ep in dataset], axis=0)         # [L_total,]  

    # State mean and standard deviation
    state_mean = all_states.mean(axis=0, keepdims=True).view(1, -1)  # [1, 1, s_dim]
    state_std = all_states.std(axis=0, keepdims=True).view(1, -1)    # [1, 1, s_dim]

    # Rtg mean and standard deviation
    rtg_mean = all_rtg.mean(axis=0, keepdims=True).view(1, 1)        # [1, 1, 1]
    rtg_std = all_rtg.std(axis=0, keepdims=True).view(1, 1)          # [1, 1, 1]

    return state_mean, state_std, rtg_mean, rtg_std


def get_statistics(dataset: Any) -> dict:
    total_episodes = 0
    total_reward = 0.0
    max_ep_reward = 0.0

    for ep in dataset:
        ep_return = ep["total_episode_return"]
        max_ep_reward = max(max_ep_reward, ep_return)

        total_reward += ep_return 
        total_episodes += 1

    mean_reward = total_reward / total_episodes

    stats = {
        "n-episodes" : total_episodes,
        "mean-ep-return" : mean_reward,
        "max-ep-return": max_ep_reward
    }
    return stats


def set_seeds(seed: int, deterministic: bool=False) -> None:
    """Sets random seeds for reproducability for both CPU and GPU.""" 
    random.seed(seed) 
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic: 
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


def save_mean_std(
        dataset_name: str,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        rtg_mean: torch.Tensor,
        rtg_std: torch.Tensor
) -> None:
    torch.save(state_mean, f"{dataset_name}-state-mean.pth")
    torch.save(state_std, f"{dataset_name}-state-std.pth")
    torch.save(rtg_mean, f"{dataset_name}-rtg-mean.pth")
    torch.save(rtg_std, f"{dataset_name}-rtg-std.pth")


def load_mean_std(dataset_name: str) -> tuple:
    state_mean = torch.load(f"{dataset_name}-state-mean.pth")
    state_std = torch.load(f"{dataset_name}-state-std.pth")
    rtg_mean = torch.load(f"{dataset_name}-rtg-mean.pth")
    rtg_std = torch.load(f"{dataset_name}-rtg-std.pth")
    return state_mean, state_std, rtg_mean, rtg_std


def save_dataset(dataset: Dataset|Subset, dataset_name: str) -> None:
    torch.save(dataset, dataset_name)


def load_dataset(dataset_name: str) -> Dataset|Subset:
    dataset = torch.load(dataset_name)
    return dataset