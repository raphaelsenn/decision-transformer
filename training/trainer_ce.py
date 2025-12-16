import gymnasium as gym

import torch
import torch.nn as nn

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from training.utils import (
    get_discrete_action, 
    to_tensor
)
from training.checkpoints import checkpoint


class Trainer:
    """
    Trainer for discrete action spaces. 
    It maximizes the log-likelihood between predicted actions and ground truth actions.
    
    Expected criterion: mean(cross-entropy(pred_a, a)) 
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
        model = self.model  
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = self.scheduler 

        device = self.device
        train_loader = self.train_loader 
        
        state_mean = self.state_mean
        state_std = self.state_std
        scale = self.scale

        epochs = self.cfg["epochs"]
        grad_norm_clip = self.cfg["grad_norm_clip"]

        cfg = self.cfg
        eval_every = cfg["eval_every"]
        save_every = cfg["save_every"]
        verbose = cfg["verbose"]

        report = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        for epoch in range(epochs):
            model.train()

            for batch in train_loader:
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

                N, L, C = pred_a.shape
                pred_flat = pred_a.view(N*L, C)                     # [N * L, action_dim]
                targets_flat = actions.view(N * L)                  # [N * L,]
                loss = criterion(pred_flat, targets_flat)           # [1,]
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
                optimizer.step() 
                scheduler.step()

            if epoch % eval_every == 0:
                train_loss, train_acc = self.evaluate(self.train_loader)
                val_loss, val_acc = self.evaluate(self.val_loader)

                if verbose: 
                    print(
                        f"epoch: {epoch}\t"
                        f"train-loss: {train_loss:.8f}\t"
                        f"train-acc: {train_acc:.4f}\t"
                        f"val-loss: {val_loss:.8f}\t"
                        f"val-acc: {val_acc:.4f}\t"
                    )
                
                metrics = {
                    "epoch" : epoch,
                    "train_loss" : train_loss,
                    "train_acc" : train_acc,
                    "val_loss" : val_loss, 
                    "val_acc" : val_acc,
                } 

                for k, v in metrics.items():
                    report[k].append(v)

            if epoch % save_every == 0:
                checkpoint(cfg, model, report)

        # Final save after training
        checkpoint(cfg, model, report)

    def train_with_online_eval(self) -> None:
        model = self.model  
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = self.scheduler 
        device = self.device
        train_loader = self.train_loader 

        state_mean = self.state_mean
        state_std = self.state_std
        scale = self.scale

        epochs = self.cfg["epochs"]
        grad_norm_clip = self.cfg["grad_norm_clip"]
        cfg = self.cfg
        eval_every = cfg["eval_every"]
        save_every = cfg["save_every"]
        verbose = cfg["verbose"]

        steps = 0
        report = {"epoch": [], "num_steps": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "mean_ep_return": []}
        report = report | {f"ep_return_run_{i}" : [] for i in range(cfg["num_online_eval_runs"])}

        for epoch in range(epochs):
            model.train()

            for batch in train_loader:
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

                N, L, C = pred_a.shape
                pred_flat = pred_a.view(N*L, C)                     # [N * L, action_dim]
                targets_flat = actions.view(N * L)                  # [N * L,]
                loss = criterion(pred_flat, targets_flat)           # [1,]
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
                optimizer.step() 
                scheduler.step()

                steps += N * L

            if epoch % eval_every == 0:
                train_loss, train_acc = self.evaluate(self.train_loader)
                val_loss, val_acc = self.evaluate(self.val_loader)
                
                ep_returns = self.evaluate_in_env() 
                mean_ep_return = torch.mean(torch.tensor(ep_returns)).item()

                metrics = {
                    "epoch" : epoch,
                    "num_steps": steps,
                    "train_loss" : train_loss,
                    "train_acc" : train_acc,
                    "val_loss" : val_loss, 
                    "val_acc" : val_acc,
                    "mean_ep_return": mean_ep_return,
                } 
                metrics = metrics | {f"ep_return_run_{i}" : ep_returns[i] for i in range(cfg["num_online_eval_runs"])}

                for k, v in metrics.items():
                    report[k].append(v)

                if verbose:
                    print(
                        f"epoch: {epoch}\t"
                        f"train_loss: {train_loss:.8f}\t"
                        f"train_acc: {train_acc:.4f}\t"
                        f"val_loss: {val_loss:.8f}\t"
                        f"val_acc: {val_acc:.4f}\t"
                        f"mean_ep_return: {mean_ep_return:.4f}"
                    )

            if epoch % save_every == 0:
                checkpoint(cfg, model, report)

        # Final save after training
        checkpoint(cfg, model, report)


    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> tuple[float, float]:
        model = self.model 
        model.eval()

        criterion = self.criterion 
        device = self.device
        state_mean = self.state_mean
        state_std = self.state_std
        scale = self.scale

        n_samples = len(dataloader.dataset)
        total_loss, total_acc = 0.0, 0.0
        for batch in dataloader:
            rtg = batch['returns_to_go'].to(device)             # [N, L, 1]
            states = batch['states'].to(device)                 # [N, L, 8]
            actions = batch['actions'].to(device)               # [N, L] (discrete action space)
            timesteps = batch['timesteps'].to(device)           # [N, L]
            pad_mask = batch['mask'].to(device)                 # [N, L]

            states = (states - state_mean) / (state_std + 1e-6)
            rtg = rtg / scale

            logits_a = model(rtg, states, actions, timesteps, pad_mask)     # [N, L, action_dim]
        
            N, L, C = logits_a.shape
            pred_flat = logits_a.view(N*L, C)                     # [N * L, action_dim]
            targets_flat = actions.view(N * L)                  # [N * L,]
            loss = criterion(pred_flat, targets_flat)           # [1,]

            total_loss += loss.item() * N
            total_acc += ((torch.argmax(logits_a, dim=-1) == actions) & pad_mask).sum().item()

        return total_loss / n_samples, total_acc / (n_samples*L)
    

    @torch.no_grad()
    def evaluate_in_env(self) -> list[float]:
        model = self.model
        model.eval()

        cfg = self.cfg 
        device = self.device 
        gym_env_id = cfg["gym_env_id"] 
        gym_seed = cfg["gym_seed"]
        max_episode_steps = cfg["max_ep_len"]
        context_length = cfg["context_length"]
        num_online_eval_runs = cfg["num_online_eval_runs"]
        s_dim = cfg["state_dim"]

        state_mean = self.state_mean
        state_std = self.state_std
        scale = self.scale

        returns = []
        
        env = gym.make(gym_env_id, max_episode_steps=max_episode_steps)

        for ep in range(num_online_eval_runs): 

            s_init, _ = env.reset(seed=gym_seed+ep)
            target_rtg = cfg["target_return"]
            
            R_lst, s_lst, a_lst, t_lst = (
                [target_rtg], [s_init.tolist()], [0], [0]
            )

            done = False
            episode_return = 0
            while not done:
                R = to_tensor(R_lst, torch.float32, device, shape=(1, -1, 1))       # [1, seq_len, 1]
                s = to_tensor(s_lst, torch.float32, device, shape=(1, -1, s_dim))   # [1, seq_len, s_dim]
                a = to_tensor(a_lst, torch.long, device, shape=(1, -1))             # [1, seq_len]
                t = to_tensor(t_lst, torch.long, device, shape=(1, -1))             # [1, seq_len]

                s = (s - state_mean) / (state_std + 1e-6)
                R = R / scale

                logits = model(R, s, a, t)              # [1, action_dim]
                action = get_discrete_action(logits)    # [action_dim,]
                new_s, reward, terminated, truncated, _ = env.step(action)

                episode_return += reward
                done = terminated or truncated

                # Update (for autoregression)
                R_lst = (R_lst + [R_lst[-1] - reward])[-context_length:]
                s_lst = (s_lst + [new_s.tolist()])[-context_length:]
                a_lst = (a_lst + [action])[-context_length:]
                t_lst = (t_lst + [t_lst[-1] + 1])[-context_length:]

            returns.append(episode_return)

        return returns
