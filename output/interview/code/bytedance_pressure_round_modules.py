"""
Minimal, runnable building blocks for ByteDance-style pressure-round questions.

This file intentionally prioritizes:
1. clear tensor shapes,
2. small, runnable implementations,
3. simple cost-intuition helpers,

over production-grade performance.

Run:
    python bytedance_pressure_round_modules.py
or:
    python test_bytedance_pressure_round_modules.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@dataclass
class AttentionCost:
    flops_main_term: int
    kv_cache_bytes_per_token: int


class SimpleMHA(nn.Module):
    """Minimal multi-head self-attention for shape reasoning and demos."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        heads, head_dim = self.num_heads, self.head_dim

        q = self.q_proj(x).view(batch, seq_len, heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        if causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out_proj(out)

    def rough_cost(self, seq_len: int, dtype_bytes: int = 2) -> AttentionCost:
        flops = 2 * (seq_len**2) * self.dim
        kv_bytes = 2 * self.num_heads * self.head_dim * dtype_bytes
        return AttentionCost(flops_main_term=flops, kv_cache_bytes_per_token=kv_bytes)


class Expert(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class SimpleMoE(nn.Module):
    """
    Minimal top-k MoE implementation.

    This is intentionally easy to read rather than highly optimized.
    """

    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k must be <= num_experts")

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(dim, hidden_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        flat = x.reshape(batch * seq_len, dim)

        router_logits = self.router(flat)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_scores, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)

        output = torch.zeros_like(flat)

        for expert_id in range(self.num_experts):
            token_mask = (topk_idx == expert_id).any(dim=-1)
            if not token_mask.any():
                continue

            selected_x = flat[token_mask]
            expert_out = self.experts[expert_id](selected_x)

            selected_idx = topk_idx[token_mask]
            selected_scores = topk_scores[token_mask]
            gates = torch.zeros(selected_x.size(0), device=x.device, dtype=x.dtype)

            for i in range(self.top_k):
                gates = gates + (
                    (selected_idx[:, i] == expert_id).to(x.dtype) * selected_scores[:, i]
                )

            output[token_mask] = output[token_mask] + expert_out * gates.unsqueeze(-1)

        return output.view(batch, seq_len, dim)

    def rough_parameter_count(self) -> int:
        expert_params = self.num_experts * (2 * self.dim * self.hidden_dim + self.hidden_dim + self.dim)
        router_params = self.dim * self.num_experts
        return expert_params + router_params


class SimpleMLA(nn.Module):
    """
    Minimal Multi-head Latent Attention style module.

    This is not an exact DeepSeek implementation. It is a teaching-friendly
    version showing the main idea: compress K/V-related state into a smaller
    latent space before reconstructing attention-time tensors.
    """

    def __init__(self, dim: int, num_heads: int, latent_dim: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.latent_dim = latent_dim

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_down = nn.Linear(dim, latent_dim, bias=False)
        self.k_up = nn.Linear(latent_dim, dim, bias=False)
        self.v_up = nn.Linear(latent_dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, dim = x.shape
        heads, head_dim = self.num_heads, self.head_dim

        q = self.q_proj(x).view(batch, seq_len, heads, head_dim).transpose(1, 2)

        latent = self.kv_down(x)
        k = self.k_up(latent).view(batch, seq_len, heads, head_dim).transpose(1, 2)
        v = self.v_up(latent).view(batch, seq_len, heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out_proj(out), latent

    def rough_cost(self, seq_len: int, dtype_bytes: int = 2) -> AttentionCost:
        flops = 2 * (seq_len**2) * self.dim
        kv_bytes = self.latent_dim * dtype_bytes
        return AttentionCost(flops_main_term=flops, kv_cache_bytes_per_token=kv_bytes)


def tree_reduce_sum(values: torch.Tensor) -> torch.Tensor:
    reduced = values.reshape(-1).to(torch.float32)
    if reduced.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32)

    while reduced.numel() > 1:
        if reduced.numel() % 2 == 1:
            reduced = torch.cat(
                [reduced, torch.zeros(1, device=reduced.device, dtype=reduced.dtype)]
            )
        reduced = reduced.view(-1, 2).sum(dim=1)
    return reduced[0]


def hierarchical_block_reduce_sum(values: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    flat = values.reshape(-1).to(torch.float32)
    if flat.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32)

    partials = []
    for start in range(0, flat.numel(), block_size):
        partials.append(tree_reduce_sum(flat[start : start + block_size]))

    stacked = torch.stack(partials)
    return tree_reduce_sum(stacked)


def _demo() -> None:
    torch.manual_seed(0)

    x = torch.randn(2, 8, 32)

    print("== MHA ==")
    mha = SimpleMHA(dim=32, num_heads=4)
    y = mha(x, causal=True)
    print("output shape:", tuple(y.shape))
    print("params:", count_parameters(mha))
    print("rough cost:", mha.rough_cost(seq_len=x.size(1)))

    print("\n== MoE ==")
    moe = SimpleMoE(dim=32, hidden_dim=64, num_experts=4, top_k=2)
    y = moe(x)
    print("output shape:", tuple(y.shape))
    print("params:", moe.rough_parameter_count())

    print("\n== MLA ==")
    mla = SimpleMLA(dim=32, num_heads=4, latent_dim=12)
    y, latent = mla(x)
    print("output shape:", tuple(y.shape))
    print("latent shape:", tuple(latent.shape))
    print("params:", count_parameters(mla))
    print("rough cost:", mla.rough_cost(seq_len=x.size(1)))

    print("\n== Reduce ==")
    vec = torch.randn(4097)
    print("torch.sum:", float(vec.sum()))
    print("tree_reduce_sum:", float(tree_reduce_sum(vec)))
    print("hierarchical_block_reduce_sum:", float(hierarchical_block_reduce_sum(vec)))


if __name__ == "__main__":
    _demo()
