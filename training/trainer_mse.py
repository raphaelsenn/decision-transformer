import gymnasium as gym

import numpy as np

import torch
import torch.nn as nn

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from training.checkpoints import checkpoint
from training.utils import (
    to_tensor,
    get_continuous_action
)


class Trainer:
    """
    Trainer for continuous action spaces. 
    It optimizes the mean-squarred error between predicted actions and ground truth actions.
    
    Expected criterion: mean((pred_a - a)**2)
    """ 
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

    def train(self) -> None:
        cfg = self.cfg
        
        model = self.model  
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = self.scheduler 

        device = self.device
        train_loader = self.train_loader 

        state_mean = self.state_mean
        state_std = self.state_std
        scale = self.scale

        epochs = cfg["epochs"]
        grad_norm_clip = cfg["grad_norm_clip"]
        eval_every = cfg["eval_every"]
        save_every = cfg["save_every"]
        verbose = cfg["verbose"]

        report = {"epoch": [], "train_loss": [],  "val_loss": []}

        for epoch in range(epochs):
            model.train()

            for batch in train_loader:
                rtg = batch['returns_to_go'].to(device)             # [N, L, 1]
                states = batch['states'].to(device)                 # [N, L, s_dim]
                actions = batch['actions'].to(device)               # [N, L, a_dim]
                timesteps = batch['timesteps'].to(device)           # [N, L]
                pad_mask = batch['mask'].to(device)                 # [N, L]

                states = (states - state_mean) / (state_std + 1e-6)
                rtg = rtg / scale

                pred_a = model(
                    rtg, states, actions, timesteps, pad_mask
                )                                                   # [N, L, action_dim]

                loss = criterion(pred_a, actions, pad_mask)        # [1,]
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
                optimizer.step() 
                scheduler.step()

            if epoch % eval_every == 0:
                train_loss = self.evaluate(self.train_loader)
                val_loss = self.evaluate(self.val_loader)

                if verbose: 
                    print(
                        f"epoch: {epoch}\t"
                        f"train-loss: {train_loss:.8f}\t"
                        f"val-loss: {val_loss:.8f}\t"
                    )
                
                metrics = {
                    "epoch" : epoch,
                    "train_loss" : train_loss,
                    "val_loss" : val_loss, 
                } 

                for k, v in metrics.items():
                    report[k].append(v)

            if epoch % save_every == 0:
                checkpoint(cfg, model, report)

        # Final save after training
        checkpoint(cfg, model, report)

    def train_with_online_eval(self) -> None:
        cfg = self.cfg
        
        model = self.model  
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = self.scheduler 
        
        device = self.device
        train_loader = self.train_loader 

        state_mean = self.state_mean
        state_std = self.state_std
        scale = self.scale

        epochs = cfg["epochs"]
        grad_norm_clip = cfg["grad_norm_clip"]
        eval_every = cfg["eval_every"]
        save_every = cfg["save_every"]
        verbose = cfg["verbose"]

        report = {"epoch": [], "train_loss": [], "val_loss": [], "mean_ep_return": []}
        report = report | {f"ep_return_run_{i}": [] for i in range(cfg["num_online_eval_runs"])}

        if verbose:
            self._train_report()

        for epoch in range(epochs):
            model.train()

            for batch in train_loader:
                rtg = batch['returns_to_go'].to(device)             # [N, L, 1]
                states = batch['states'].to(device)                 # [N, L, 8]
                actions = batch['actions'].to(device)               # [N, L, 4]
                timesteps = batch['timesteps'].to(device)           # [N, L]
                pad_mask = batch['mask'].to(device)                 # [N, L]

                states = (states - state_mean) / (state_std + 1e-6)
                rtg = rtg / scale

                pred_a = model(
                    rtg, states, actions, timesteps, pad_mask
                )                                                   # [N, L, action_dim]
                loss = criterion(pred_a, actions, pad_mask)         # [1,]
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
                optimizer.step()
                scheduler.step()

            if epoch % eval_every == 0:
                train_loss = self.evaluate(self.train_loader)
                val_loss = self.evaluate(self.val_loader)
                ep_returns = self.evaluate_in_env()
                mean_ep_return = torch.mean(torch.tensor(ep_returns)).item()

                metrics = {
                    "epoch" : epoch,
                    "train_loss" : train_loss,
                    "val_loss" : val_loss, 
                    "mean_ep_return": mean_ep_return
                } 
                metrics = metrics | {f"ep_return_run_{i}" : ep_returns[i] for i in range(cfg["num_online_eval_runs"])}

                for k, v in metrics.items():
                    report[k].append(v)

                if verbose:
                    print(
                        f"epoch: {epoch}\t"
                        f"train_loss: {train_loss:.8f}\t"
                        f"val_loss: {val_loss:.8f}\t"
                        f"mean_ep_return: {mean_ep_return:.4f}"
                    )

            if epoch % save_every == 0:
                checkpoint(cfg, model, report)

        # Final save after training
        checkpoint(cfg, model, report)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        model = self.model
        model.eval()
        
        device = self.device
        criterion = self.criterion

        state_mean = self.state_mean
        state_std = self.state_std
        scale = self.scale

        n_samples = len(dataloader.dataset)
        total_loss = 0.0

        for batch in dataloader:
            rtg = batch['returns_to_go'].to(device)             # [N, L, 1]
            states = batch['states'].to(device)                 # [N, L, 8]
            actions = batch['actions'].to(device)               # [N, L] (discrete action space)
            timesteps = batch['timesteps'].to(device)           # [N, L]
            pad_mask = batch['mask'].to(device)                 # [N, L]

            states = (states - state_mean) / (state_std + 1e-6)
            rtg = rtg / scale

            pred_a = model(
                rtg, states, actions, timesteps, pad_mask
            )                                                   # [N, L, action_dim]

            loss = criterion(pred_a, actions, pad_mask)         # [1,]

            N = rtg.size(0)
            total_loss += loss.item() * N

        return total_loss / n_samples

    @torch.no_grad()
    def evaluate_in_env(self) -> list[float]:
        model = self.model
        model.eval()
        
        state_mean = self.state_mean
        state_std = self.state_std
        scale = self.scale

        cfg = self.cfg 
        device = self.device
        gym_env_id = cfg["gym_env_id"] 
        gym_seed = cfg["gym_seed"]
        max_episode_steps = cfg["max_ep_len"]
        context_length = cfg["context_length"]
        num_online_eval_runs = cfg["num_online_eval_runs"]
        a_dim = cfg["action_dim"]
        s_dim = cfg["state_dim"]

        rewards = []
        
        env = gym.make(gym_env_id, max_episode_steps=max_episode_steps)

        for ep in range(num_online_eval_runs): 
            s_init, _ = env.reset(seed=gym_seed+ep)
            a_init = torch.zeros(env.action_space.sample().shape)
            t_init = 0

            target_rtg = cfg["target_return"]

            R_lst, s_lst, a_lst, t_lst = (
                [target_rtg], [s_init.tolist()], [a_init.tolist()], [t_init]
            )

            done = False
            episode_reward = 0
            while not done: 
                R = to_tensor(R_lst, torch.float32, device, (1, -1, 1))         # [1, seq_len, 1]
                s = to_tensor(s_lst, torch.float32, device, (1, -1, s_dim))     # [1, seq_len, s_dim]
                a = to_tensor(a_lst, torch.float32, device, (1, -1, a_dim))     # [1, seq_len, a_dim]
                t = to_tensor(t_lst, torch.long, device, (1, -1))               # [1, seq_len]

                s = (s - state_mean) / (state_std + 1e-6)
                R = R / scale

                logits_a = model(R, s, a, t)               # [1, seq_len, a_dim]
                action = get_continuous_action(logits_a)   # [a_dim,]

                new_s, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated

                # Update (for autoregression)
                R_lst = (R_lst + [R_lst[-1] - reward])[-context_length:]
                s_lst = (s_lst + [new_s.tolist()])[-context_length:]
                a_lst = (a_lst + [action.tolist()])[-context_length:]
                t_lst = (t_lst + [t_lst[-1] + 1])[-context_length:]

            rewards.append(episode_reward)

        env.close()
        return rewards

    def _train_report(self) -> None:
        num_batches = len(self.train_loader)
        num_samples = len(self.train_loader.dataset)
        epochs = self.cfg["epochs"]
        report = f"Epochs:          {epochs}\n"\
                 f"Num samples:     {num_samples}\n"\
                 f"Num batches:     {num_batches}\n"\
                 f"Device:          {self.device}\n"
        print(report)