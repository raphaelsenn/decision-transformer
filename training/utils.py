import numpy as np

import torch
import torch.nn as nn


class MaskedMSE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None: 
        mask = mask.unsqueeze(-1).float()     # [N, L, 1]
        mse = (pred - target)** 2             # [N, L, action_dim]
        mse = mse * mask                      # [N, L, action_dim] 
        return mse.sum() / (mask.sum() * pred.size(-1) + 1e-8)


@torch.no_grad()
def get_discrete_action(logits: torch.Tensor, greedy: bool=True) -> int:
    probs = torch.softmax(logits[:, -1, :], dim=-1)     
    if greedy: 
        return torch.argmax(probs, dim=-1).item() 
    return torch.multinomial(probs, num_samples=1).item() 


@torch.no_grad()
def get_continuous_action(logits: torch.Tensor) -> np.ndarray:
    # pred = logits.clamp(-1.0, 1.0)
    action = logits[:, -1, :].detach().cpu().numpy().flatten()
    return action


def to_tensor(x_lst: list, dtype: torch.dtype, device: torch.device, shape: tuple) -> torch.Tensor:
    x_arr = np.asarray(x_lst)
    t = torch.from_numpy(x_arr).to(dtype) 
    if device is not None:
        t = t.to(device)
    return t.reshape(shape)