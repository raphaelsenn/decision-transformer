from itertools import chain

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import minari


class HalfCheetahOfflineDataset(Dataset):
    def __init__(
            self,
            context_length: int=20,
            max_episode_len: int=1000,
            **kwargs
    ) -> None:

        self.dataset_medium = minari.load_dataset('mujoco/halfcheetah/medium-v0', download=True)
        self.dataset_expert = minari.load_dataset('mujoco/halfcheetah/expert-v0', download=True)

        self.context_length = context_length
        self.max_episode_length = max_episode_len

        self.episodes = []
        self.episode_lengths = []
        self.returns = []

        self._load_episodes() 

        self.num_episodes = len(self.episodes)

    def __len__(self) -> int:
        return self.num_episodes

    def _load_episodes(self) -> None:
        medium = self.dataset_medium
        expert = self.dataset_expert
        max_episode_length = self.max_episode_length

        for ep in chain(medium.iterate_episodes(), expert.iterate_episodes()):
        # for ep in medium.iterate_episodes():
        # for ep in expert.iterate_episodes():
            states = ep.observations    # [L_full + 1, 17]
            actions = ep.actions        # [L_full, 6]
            rewards = ep.rewards        # [L_full,]

            # Total episode reward
            total_ep_reward = float(sum(rewards))

            L_full = actions.shape[0]
            L = min(max_episode_length, L_full)

            states = states[:L, :]                      # [L, 17]
            actions = actions[:L, :]                    # [L, 6]
            rewards = rewards[:L]                       # [L,]
            timesteps = np.arange(L, dtype=np.int64)    # [L,]

            # Returns-to-go (undiscounted)
            returns_to_go = np.zeros_like(rewards, dtype=np.float32)
            running = 0.0
            for t in reversed(range(L)):
                running += rewards[t]
                returns_to_go[t] = running

            self.episodes.append({
                'returns_to_go' : returns_to_go,        # [L,]
                'states' : states,                      # [L, 17]
                'actions': actions,                     # [L, 6]
                'timesteps': timesteps,                 # [L,]
                'length' : L,                           # scalar
                'total_episode_return': total_ep_reward,# scalar (for statistics)
            })
            self.episode_lengths.append(L)


    def _sample_random_subsequence(self, episode: dict) -> tuple:
        L = episode['length'] 
        K = self.context_length

        if L >= K:
            # sample window of length K
            t_start = np.random.randint(0, L-K+1)
            t_end = t_start + K
        else:
            t_start = 0
            t_end = L
        
        # L_sub = K if L>=K else L
        rtg = episode['returns_to_go'][t_start:t_end]           # [L_sub,]
        states = episode['states'][t_start:t_end]               # [L_sub, 17]
        actions = episode['actions'][t_start:t_end]             # [L_sub, 6]
        timesteps = episode['timesteps'][t_start:t_end]         # [L_sub,]

        return rtg, states, actions, timesteps

    def __getitem__(self, index: int|torch.Tensor) -> dict: 
        if isinstance(index, torch.Tensor):
            index = index.item() 

        episode = self.episodes[index]
        rtg, states, actions, timesteps = self._sample_random_subsequence(episode)
        
        rtg = torch.from_numpy(rtg).unsqueeze(-1).to(torch.float32)     # [L_sub, 1]
        states = torch.from_numpy(states).to(torch.float32)             # [L_sub, 17]
        actions = torch.from_numpy(actions).to(torch.float32)           # [L_sub, 6]
        timesteps = torch.from_numpy(timesteps).to(torch.long)          # [L_sub,]

        # Pad sequence to context-length
        L_sub = actions.shape[0]
        pad_len = self.context_length - L_sub
        pad_mask = torch.ones(self.context_length, dtype=torch.bool)    # [ctx_len,]
        if pad_len > 0: 
            rtg = F.pad(rtg, pad=(0, 0, pad_len, 0), value=0.0)         # [ctx_len, 1]
            states = F.pad(states, pad=(0, 0, pad_len, 0), value=0.0)   # [ctx_len, 17]
            actions = F.pad(actions, pad=(0, 0, pad_len, 0), value=0.0) # [ctx_len, 6]
            timesteps = F.pad(timesteps, pad=(pad_len, 0), value=0)     # [ctx_len,]
            pad_mask[:pad_len] = False

        # For statistics
        total_ep_return = episode["total_episode_return"]

        return {
            "returns_to_go": rtg,
            "states" : states,
            "actions" : actions,
            "timesteps" : timesteps,
            "mask": pad_mask,
            "total_episode_return": total_ep_return,    # scalar (not used during training)
        }

    def get_full_episodes(self) -> list:
        return self.episodes