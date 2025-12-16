import os
import sys

import gymnasium as gym

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.utils import (
    to_tensor, 
    get_discrete_action, 
    get_continuous_action
)


def play_discrete(
        model: torch.nn.Module,
        cfg: dict,
        s_mean: torch.Tensor,
        s_std: torch.Tensor,
) -> None:
    model.eval()

    gym_env_id = cfg["gym_env_id"] 
    max_episode_steps = cfg["max_ep_len"]
    context_length = cfg["context_length"]
    scale = cfg["scale"]
    device = torch.device("cpu")

    env = gym.make(gym_env_id, max_episode_steps=max_episode_steps, render_mode="human")

    while True: 
        total_return = 0.0
        done = False        
        
        target_rtg = cfg["target_return"]
        s_init, _ = env.reset()
        R_lst, s_lst, a_lst, t_lst = (
            [target_rtg], [s_init.tolist()], [0], [0]
        )

        while not done:
            R = to_tensor(R_lst, torch.float32, device, shape=(1, -1, 1))     # [1, seq_len, 1]
            s = to_tensor(s_lst, torch.float32, device, shape=(1, -1, 8))     # [1, seq_len, 8]
            a = to_tensor(a_lst, torch.long, device, shape=(1, -1))           # [1, seq_len]
            t = to_tensor(t_lst, torch.long, device, shape=(1, -1))           # [1, seq_len]

            s = (s - s_mean) / (s_std + 1e-6)
            R /= scale

            logits = model(R, s, a, t)              # [1, 4]
            action = get_discrete_action(logits)    # integer 
            new_s, reward, terminated, truncated, _ = env.step(action)

            # Update (for autoregression)
            R_lst = (R_lst + [R_lst[-1] - reward])[-context_length:]
            s_lst = (s_lst + [new_s.tolist()])[-context_length:]
            a_lst = (a_lst + [action])[-context_length:]
            t_lst = (t_lst + [t_lst[-1] + 1])[-context_length:]

            total_return += reward
            done = terminated or truncated
        print(f"Episode return: {total_return}")


@torch.no_grad()
def play_continuous(
        agent: torch.nn.Module,
        cfg: dict,
        s_mean,
        s_std,
        rtg_mean,
        rtg_std
) -> None:

    agent.eval()

    gym_env_id = cfg["gym_env_id"] 
    gym_seed = cfg["gym_seed"]
    max_episode_steps = cfg["max_ep_len"]
    context_length = cfg["context_length"]
    s_dim = cfg["state_dim"]
    a_dim = cfg["action_dim"]
    scale = cfg["scale"]
    device = torch.device("cpu")

    env = gym.make(gym_env_id, max_episode_steps=max_episode_steps, render_mode="human")
    
    target_rtg = cfg["target_return"]
    init_action = env.action_space.sample()
    s_init, _ = env.reset(seed=gym_seed)
    R_lst, s_lst, a_lst, t_lst = (
        [target_rtg], [s_init.tolist()], [init_action.tolist()], [1]
    )

    while True:
        total_return = 0.0
        target_rtg = cfg["target_return"]
        init_action = env.action_space.sample()
        s_init, _ = env.reset(seed=gym_seed)
        R_lst, s_lst, a_lst, t_lst = (
            [target_rtg], [s_init.tolist()], [init_action.tolist()], [1]
        ) 
        done = False 
        while not done: 
            R = to_tensor(R_lst, torch.float32, device, shape=(1, -1, 1))       # [1, seq_len, 1]
            s = to_tensor(s_lst, torch.float32, device, shape=(1, -1, s_dim))   # [1, seq_len, s_dim]
            a = to_tensor(a_lst, torch.float32, device, shape=(1, -1, a_dim))   # [1, seq_len, a_dim]
            t = to_tensor(t_lst, torch.long, device, shape=(1, -1))             # [1, seq_len]

            s = (s - s_mean) / (s_std + 1e-6)
            R /= scale

            logits = agent(R, s, a, t)                   # [1, a_dim]
            action = get_continuous_action(logits)       # [a_dim,]

            new_s, reward, terminated, truncated, _ = env.step(action)
            total_return += reward

            # Update (for autoregression)
            R_lst = (R_lst + [R_lst[-1] - reward])[-context_length:]
            s_lst = (s_lst + [new_s.tolist()])[-context_length:]
            a_lst = (a_lst + [action])[-context_length:]
            t_lst = (t_lst + [t_lst[-1] + 1])[-context_length:]
        
            done = terminated or truncated 

        print(f"Episode return: {total_return}")