import random
from typing import Any

import numpy as np

import torch
from torch.utils.data import Dataset, Subset


def split_train_val(dataset: Any, train_ratio: float) -> tuple:
    n = len(dataset)
    train_size = int(train_ratio * n)
    idx = torch.randperm(n)

    idx_train = idx[:train_size]
    idx_val = idx[train_size:]

    train_set = Subset(dataset, indices=idx_train)
    val_set = Subset(dataset, indices=idx_val)
    return train_set, val_set


def get_std_mean(subset: Dataset|Subset) -> tuple:
    assert hasattr(subset, "dataset"), (
        f"Dataset: {subset} does not have attribute `dataset`"
    )
    assert hasattr(subset.dataset, "episodes"), (
        f"Dataset.dataset: {subset} does not have attribute `episodes`"
    )
    episodes = subset.dataset.episodes
    indices = subset.indices
    episodes = [episodes[i] for i in indices]

    states = np.concatenate([ep['states'] for ep in episodes], axis=0).astype(np.float32) # [L_total, s_dim]
    states = torch.from_numpy(states)

    # State mean and standard deviation
    state_mean = states.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, s_dim]
    state_std = states.std(dim=0, keepdim=True).unsqueeze(0)    # [1, 1, s_dim]

    return state_mean, state_std


def get_statistics(subset: Dataset|Subset) -> tuple:
    assert hasattr(subset, "dataset"), (
        f"Dataset: {subset} does not have attribute `dataset`"
    )
    assert hasattr(subset.dataset, "episodes"), (
        f"Dataset.dataset: {subset} does not have attribute `episodes`"
    )
    episodes = subset.dataset.episodes
    indices = subset.indices
    episodes = [episodes[i] for i in indices]

    total_episodes = len(episodes)
    total_steps = 0 
    total_return = 0.0
    max_ep_return = 0.0

    for ep in episodes:
        returns = ep["returns"] 
        ep_return = sum(returns)

        total_steps += len(returns) 
        total_return += ep_return 
        max_ep_return = max(max_ep_return, ep_return)

    mean_ep_return = total_return / total_episodes

    return {
        "n-episodes" : total_episodes,
        "n-steps" : total_steps,
        "mean-episode-return" : mean_ep_return,
        "max-episode-return" : max_ep_return
    }


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
) -> None:
    torch.save(state_mean, f"{dataset_name}-state-mean.pth")
    torch.save(state_std, f"{dataset_name}-state-std.pth")


def load_mean_std(dataset_name: str) -> tuple:
    state_mean = torch.load(f"{dataset_name}-state-mean.pth")
    state_std = torch.load(f"{dataset_name}-state-std.pth")
    return state_mean, state_std


def save_dataset(dataset: Dataset|Subset, dataset_name: str) -> None:
    torch.save(dataset, dataset_name)


def load_dataset(dataset_name: str) -> Dataset|Subset:
    dataset = torch.load(dataset_name)
    return dataset