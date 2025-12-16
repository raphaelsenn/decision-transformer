import os

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class LunarLanderOfflineDataset(Dataset):
    def __init__(
            self,
            root_data: str,
            context_length: int=20,
            max_episode_len: int=1000,
            **kwargs
    ) -> None:

        if not root_data.endswith(".npz"):
            root_data = root_data + ".npz"
        assert os.path.isfile(root_data), f"File {root_data} does not exist."

        data = np.load(root_data, allow_pickle=True)
        self.dataset = data["episodes"] 

        self.context_length = context_length
        self.max_episode_length = max_episode_len

        self._load_episodes() 

    def __len__(self) -> int:
        return len(self.episodes)

    def _load_episodes(self) -> None:
        dataset = self.dataset
        max_episode_length = self.max_episode_length

        episodes = []
        episode_lengths = []

        for ep in dataset:
            states = ep["states"]                       # [L_full + 1, 8]
            actions = ep["actions"]                     # [L_full,]
            rewards = ep["rewards"]                     # [L_full,]

            L_full = actions.shape[0]
            L = min(max_episode_length, L_full)

            states = states[:L, :]                      # [L, 8]
            actions = actions[:L]                       # [L,]
            rewards = rewards[:L]                       # [L,]
            timesteps = np.arange(L, dtype=np.int64)    # [L,]

            # Returns-to-go (undiscounted)
            returns_to_go = np.zeros_like(rewards, dtype=np.float32)
            running = 0.0
            for t in reversed(range(L)):
                running += rewards[t]
                returns_to_go[t] = running

            episodes.append({
                'returns_to_go' : returns_to_go,                # [L,]
                'returns' : rewards,                            # [L,]
                'states' : states,                              # [L, 8]
                'actions': actions,                             # [L,]
                'timesteps': timesteps,                         # [L,]
                'length' : L                                    # scalar
            })
            episode_lengths.append(L)

        self.episodes = episodes
        self.episode_lengths = episode_lengths

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
        states = episode['states'][t_start:t_end]               # [L_sub, 8]
        actions = episode['actions'][t_start:t_end]             # [L_sub,]
        timesteps = episode['timesteps'][t_start:t_end]         # [L_sub,]

        return rtg, states, actions, timesteps

    def __getitem__(self, index: int|torch.Tensor) -> dict: 
        if isinstance(index, torch.Tensor):
            index = index.item() 

        episode = self.episodes[index]
        rtg, states, actions, timesteps = self._sample_random_subsequence(episode)
        
        rtg = torch.tensor(rtg, dtype=torch.float32).unsqueeze(-1)                  # [L_sub, 1]
        states = torch.tensor(states, dtype=torch.float32)                          # [L_sub, 8]
        actions = torch.tensor(actions, dtype=torch.long)                           # [L_sub,]
        timesteps = torch.tensor(timesteps, dtype=torch.long)                       # [L_sub,]

        # Pad sequence to context-length
        L_sub = actions.shape[0]
        pad_len = self.context_length - L_sub
        pad_mask = torch.ones(self.context_length, dtype=torch.bool)
        if pad_len > 0: 
            rtg = F.pad(rtg, pad=(0, 0, pad_len, 0), value=0.0)                     # [ctx_len, 1]
            states = F.pad(states, pad=(0, 0, pad_len, 0), value=0.0)               # [ctx_len, 8]
            actions = F.pad(actions, pad=(pad_len, 0), value=0)                     # [ctx_len,]
            timesteps = F.pad(timesteps, pad=(pad_len, 0), value=0)                 # [ctx_len,]
            pad_mask[:pad_len] = False

        return {
            "returns_to_go": rtg, 
            "states" : states, 
            "actions" : actions, 
            "timesteps" : timesteps, 
            "mask": pad_mask
        }
    
    def concat(self, root_data: str) -> None:
        if not root_data.endswith(".npz"):
            root_data = root_data + ".npz"
        assert os.path.isfile(root_data), f"File {root_data} does not exist."

        data = np.load(root_data, allow_pickle=True)
        dataset = data["episodes"]
        self.dataset = np.concatenate([self.dataset, dataset])
        self._load_episodes()
    
    def get_full_episodes(self) -> list:
        return self.episodes