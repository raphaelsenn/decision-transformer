"""
Author:                 Raphael Senn, <raphaelsenn@gmx.de>
Initial coding:         18.11.2025
"""
from typing import Callable

import torch
import torch.nn as nn


class CausalMultiheadAttention(nn.Module):
    """
    Computes causal multi-head attention (naive).

    Attention is given by: Attention(Q, K, V) = softmax((Q @ K^T + causal_mask) / sqrt(dk)) @ V,
    where causal_mask_ij = -inf if i > j else 0.
    
    NOTE: This is not the fastet MultiheadAttention implementation (but also not the slowest).

    Reference:
    Attention Is All You Need, Vaswani et at., 2017 
    https://arxiv.org/abs/1706.03762 
    """
    def __init__(self, n_heads: int, emb_dim: int, k_dim: int, v_dim: int, biases: bool=False) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_query = nn.Linear(emb_dim, n_heads * k_dim, bias=biases)
        self.proj_keys = nn.Linear(emb_dim, n_heads * k_dim, bias=biases)
        self.proj_values = nn.Linear(emb_dim, n_heads * v_dim, bias=biases)
        self.proj_out = nn.Linear(n_heads * v_dim, emb_dim, bias=biases)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, pad_mask: torch.Tensor|None=None) -> torch.Tensor:
        # Let N := batch
        # Input shapes: q=[N, Lq, emb_dim], k=[N, Lk, emb_dim], v=[N, Lv, emb_dim]  
        batch_q, Lq, _ = query.shape
        batch_k, Lk, _ = key.shape 
        batch_v, Lv, _ = value.shape 
        assert batch_q == batch_k == batch_v, (
            f"Batch sizes must match, got batch_q = {batch_q}, batch_k = {batch_k}, batch_v = {batch_v}.")
        assert Lk == Lv, (
            f"Key-value pairs neeed to have same sequence length, got Lk = {Lk} and Lv = {Lv}.")
        batch = batch_q

        query = self.proj_query(query)                                  # [N, Lq, n_heads * k_dim]
        key = self.proj_keys(key)                                       # [N, Lk, n_heads * k_dim]
        value = self.proj_values(value)                                 # [N, Lv, n_heads * v_dim]

        query = query.view(batch, Lq, self.n_heads, self.k_dim)         # [N, Lq, n_heads, k_dim]
        query = query.permute(2, 0, 1, 3)                               # [n_heads, N, Lq, k_dim] 
        
        key = key.view(batch, Lk, self.n_heads, self.k_dim)             # [N, Lk, n_heads, k_dim]
        key = key.permute(2, 0, 3, 1)                                   # [n_heads, N, k_dim, Lk] 
        
        value = value.view(batch, Lv, self.n_heads, self.v_dim)         # [N, Lv, n_heads, v_dim]
        value = value.permute(2, 0, 1, 3)                               # [n_heads, N, Lv, v_dim]

        # [n_heads, N, Lq, k_dim] @ [n_heads, N, k_dim, Lk] =             [n_heads, N, Lq, Lk]
        logits = query @ key                                            # [n_heads, N, Lq, Lk]
        logits /= self.k_dim**0.5
        causal_mask = torch.triu(
            torch.full(size=(1, 1, Lq, Lk), fill_value=float('-inf'), device=logits.device), 
            diagonal=1
        )                                                               # [1, 1, Lq, Lk]
        logits = logits + causal_mask                                   # [n_heads, N, Lq, Lk]

        if pad_mask is not None:                                        # [N, Lk]
            pad_mask = pad_mask.view(1, batch, 1, Lk).to(torch.bool)    # [1, N, 1, Lk]
            logits = logits.masked_fill(pad_mask == 0, float("-inf"))   # [n_heads, N, Lq, Lk]

        probs = torch.softmax(logits, dim=-1)                           # [n_heads, N, Lq, Lk]

        # NOTE: Lk = Lv
        # [n_heads, N, Lq, Lv] @ [n_heads, N, Lv, v_dim] =                [n_heads, N, Lq, v_dim]
        attn = probs @ value                                            # [n_heads, N, Lq, v_dim]
        attn = attn.permute(1, 2, 0, 3)                                 # [N, Lq, n_heads, v_dim]
        attn = attn.flatten(start_dim=2)                                # [N, Lq, n_heads * v_dim]
        return self.proj_out(attn)                                      # [N, Lq, emb_dim]


class FeedForward(nn.Module):
    """
    Simple feed-forward layer.
    
    Reference:
    Attention Is All You Need, Vaswani et at., 2017 
    https://arxiv.org/abs/1706.03762  
    """ 
    def __init__(self, embed_dim: int, ff_dim: int) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(ff_dim, embed_dim, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)   # [N, L, embed_dim]


class DecoderBlock(nn.Module):
    """
    Decoder-Block with n_heads. 

    Reference:
    Attention Is All You Need, Vaswani et at., 2017 
    https://arxiv.org/abs/1706.03762   
    """ 
    def __init__(self, embed_dim: int, n_heads: int, k_dim: int, v_dim: int, ff_dim: int, dropout: float) -> None:
        super().__init__()

        self.causal_multi_head_attn = CausalMultiheadAttention(n_heads, embed_dim, k_dim, v_dim)
        self.dropout_1 = nn.Dropout(dropout) 
        self.layer_norm_1 = nn.LayerNorm(embed_dim)

        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.dropout_2 = nn.Dropout(dropout) 
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None=None) -> torch.Tensor:
        y = self.causal_multi_head_attn(x, x, x, pad_mask)  # [N, L, embed_dim]
        y = self.dropout_1(y)                               # [N, L, embed_dim]
        x = self.layer_norm_1(x + y)                        # [N, L, embed_dim]

        y = self.feed_forward(x)                            # [N, L, embed_dim]
        y = self.dropout_2(y)                               # [N, L, embed_dim]
        x = self.layer_norm_2(x + y)                        # [N, L, embed_dim]

        return x                                            # [N, L, embed_dim]


class DecisionTransformer(nn.Module):
    """
    Decision Transformer architecture.

    Reference:
    Decision Transformer: Reinforcement Learning via Sequence Modeling, Chen et al., 2021 
    https://arxiv.org/abs/2106.01345 

    Attention Is All You Need, Vaswani et at., 2017 
    https://arxiv.org/abs/1706.03762    
    """ 
    def __init__(
            self, 
            embed_dim: int,
            state_dim: int,
            action_dim: int,
            max_ep_len: int,
            disc_act_space: bool,
            n_layers: int,
            n_heads: int,
            k_dim: int=64,
            v_dim: int=64,
            ff_dim: int=2048,
            dropout: float=0.1,
            action_tanh: bool=False,
            **kwargs
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.disc_act_space = disc_act_space

        self.embed_t = nn.Embedding(max_ep_len, embed_dim)
        self.embed_s = nn.Linear(state_dim, embed_dim)
        self.embed_r = nn.Linear(1, embed_dim)

        if disc_act_space:
            self.embed_a = nn.Embedding(action_dim, embed_dim)
        else:
            self.embed_a = nn.Linear(action_dim, embed_dim)
        self.action_tanh = action_tanh

        self.embed_ln = nn.LayerNorm(embed_dim)

        self.transformer = nn.ModuleList([
            DecoderBlock(embed_dim, n_heads, k_dim, v_dim, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        
        self.pred_a = nn.Linear(embed_dim, action_dim)
        self.pred_s = nn.Linear(embed_dim, state_dim)
        self.pred_r = nn.Linear(embed_dim, 1)

    def forward(
            self, 
            R: torch.Tensor, 
            s: torch.Tensor, 
            a: torch.Tensor, 
            t: torch.Tensor, 
            pad_mask: torch.Tensor | None=None
    ) -> torch.Tensor:
        # Shape of returns-to-go R:     [N, L, 1]
        # Shape of states s:            [N, L, state_dim]
        # Shape of actions a:           [N, L] if disc_act_space=True else [N, L, action_dim]
        # Shape of timesteps t:         [N, L] 
        assert len(R.size()) == 3, "R needs to have shape [batch, seq_len, 1]"
        assert len(s.size()) == 3, "s needs to have shape [batch, seq_len, state_dim]"
        assert len(t.size()) == 2, "t needs to have shape [batch, seq_len, time_dim]"

        if self.disc_act_space:
            assert len(a.size()) == 2, f"a (discrete) needs to have shape [batch, seq_len], got: {a.shape}"
        else:
            assert len(a.size()) == 3, f"a (continuous) needs to have shape [batch, seq_len, action_dim], got: {a.shape}"
        assert R.size(1) == s.size(1) == t.size(1) == a.size(1), (
            f"Input sequences need to have same length, got Lr: {R.size(1)}, Ls: {s.size(1)}, Lt: {t.size(1)}, La: {a.size(1)}")
        
        N, L, _ = R.shape 

        pos_embedding = self.embed_t(t)                 # [N, L, embed_dim]
        s_embedding = self.embed_s(s) + pos_embedding   # [N, L, embed_dim]
        a_embedding = self.embed_a(a) + pos_embedding   # [N, L, embed_dim]
        R_embedding = self.embed_r(R) + pos_embedding   # [N, L, embed_dim]

        # input_emebds = [R_1, s_1, a_1, R_2, s_2, a_2, ..., R_K, s_K, a_K]
        input_embeds = torch.stack([R_embedding, s_embedding, a_embedding], dim=1)      # [N, 3, L, embed_dim]
        input_embeds = input_embeds.permute(0, 2, 1, 3).reshape(N, 3*L, self.embed_dim) # [N, 3*L, embed_dim]
        input_embeds = self.embed_ln(input_embeds)                                      # [N, 3*L, embed_dim]

        if pad_mask is not None:                                            # [N, L]
            assert pad_mask.shape == (N, L), f"pad_mask must be shape [N, L], got: {pad_mask.shape}" 
            pad_mask = pad_mask.unsqueeze(1).repeat(1, 3, 1).view(N, 3*L)   # [N, 3*L]

        x = input_embeds                    # [N, 3*L, embed_dim]
        for decoder in self.transformer:
            x = decoder(x, pad_mask)        # [N, 3*L, embed_dim]
        
        # [N, 3*L, embed_dim] -> [N, L, 3, embed_dim] -> [N, 3, L, embed_dim]
        hidden_states = x.view(N, L, 3, self.embed_dim).permute(0, 2, 1, 3)

        # NOTE: In this implementation we do not predict states and returns to go,
        # we just predict actions.
        s_hidden = hidden_states[:, 1, :, :]    # [N, L, embed_dim]
        a_logits = self.pred_a(s_hidden)        # [N, L, action_dim]

        if self.action_tanh:
            return torch.tanh(a_logits)

        return a_logits                         # [N, L, action_dim]