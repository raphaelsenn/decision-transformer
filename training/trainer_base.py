from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler


class Trainer(ABC):
    """Trainer interface.""" 
    def __init__(
            self,
            cfg: dict,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: Optimizer,
            scheduler: LRScheduler, 
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: torch.device,
            state_mean: torch.Tensor,
            state_std: torch.Tensor,
            scale: float 
    ) -> None:

        self.cfg = cfg

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.state_mean = state_mean.to(device)
        self.state_std = state_std.to(device)
        self.scale = scale

        @abstractmethod 
        def train(self) -> None:
            raise NotImplementedError

        @abstractmethod 
        def train_with_online_eval(self) -> None:
            raise NotImplementedError

        @abstractmethod 
        @torch.no_grad()
        def evaluate(self, dataloader: DataLoader) -> tuple[float, float]:
            raise NotImplementedError
        
        @abstractmethod 
        @torch.no_grad()
        def evaluate_in_env(self) -> list[float]:
            raise NotImplementedError