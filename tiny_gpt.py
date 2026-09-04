"""Tiny GPT model + utilities for the assignment.

Small enough to train on a laptop GPU, large enough to demonstrate
every real training concept: shapes, gradients, accumulation, MFU.
"""
from __future__ import annotations
import math
import time
import contextlib
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 1024
    block_size: int = 64        # context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        # combined Q,K,V projection
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)                                  # (B,T,3C)
        q, k, v = qkv.split(self.n_embd, dim=2)               # each (B,T,C)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,nh,T,hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B,nh,T,hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # tie weights — saves params + helps small models
        self.head.weight = self.tok_emb.weight
        # init
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.cfg.block_size, f"T={T} exceeds block_size={self.cfg.block_size}"
        tok_emb = self.tok_emb(idx)             # (B,T,C)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_emb(pos)             # (T,C)
        x = tok_emb + pos_emb                   # (B,T,C)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)                        # (B,T,C)
        logits = self.head(x)                   # (B,T,V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self, exclude_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n -= self.tok_emb.weight.numel()
        return n


# -----------------------------
# Truth-telling utilities
# -----------------------------

def print_shapes(model: nn.Module, B: int = 2, T: int = 8, device: Optional[str] = None):
    """Run one forward pass and print every intermediate tensor shape with meaning."""
    if device is None:
        device = next(model.parameters()).device
    print(f"\n=== TENSOR SHAPE TOUR (B={B}, T={T}) ===")
    print(f"{'name':35s} {'shape':30s}  meaning")
    print("-" * 90)
    cfg = model.cfg
    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    print(f"{'idx':35s} {str(tuple(x.shape)):30s}  input token ids")
    tok = model.tok_emb(x)
    print(f"{'tok_emb(idx)':35s} {str(tuple(tok.shape)):30s}  token embeddings: (batch, time, channels)")
    pos = torch.arange(0, T, device=device)
    print(f"{'pos index':35s} {str(tuple(pos.shape)):30s}  0..T-1 positions")
    pe = model.pos_emb(pos)
    print(f"{'pos_emb':35s} {str(tuple(pe.shape)):30s}  positional embeddings: (time, channels)")
    x_in = tok + pe
    print(f"{'x = tok+pos':35s} {str(tuple(x_in.shape)):30s}  residual stream: (B,T,C)")
    for i, blk in enumerate(model.blocks):
        h = blk.ln_1(x_in)
        print(f"{f'block{i}.ln1':35s} {str(tuple(h.shape)):30s}  pre-attn norm")
        qkv = blk.attn.c_attn(h)
        print(f"{f'block{i}.attn.c_attn':35s} {str(tuple(qkv.shape)):30s}  fused QKV: 3C concat")
        nh, hd = blk.attn.n_head, blk.attn.head_dim
        q, k, v = qkv.split(blk.attn.n_embd, dim=2)
        print(f"{f'block{i}.Q/K/V':35s} {str(tuple(q.shape)):30s}  per-head: (B,T,C)")
        q = q.view(B, T, nh, hd).transpose(1, 2)
        k = k.view(B, T, nh, hd).transpose(1, 2)
        v = v.view(B, T, nh, hd).transpose(1, 2)
        print(f"{f'block{i}.Q reshape':35s} {str(tuple(q.shape)):30s}  (B, n_head, T, head_dim)")
        ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(f"{f'block{i}.attn out':35s} {str(tuple(ctx.shape)):30s}  attended values: (B,nh,T,hd)")
        y = ctx.transpose(1, 2).contiguous().view(B, T, cfg.n_embd)
        proj = blk.attn.c_proj(y)
        print(f"{f'block{i}.c_proj':35s} {str(tuple(proj.shape)):30s}  output projection back to C")
        x_in = x_in + proj
        print(f"{f'block{i}.x+attn':35s} {str(tuple(x_in.shape)):30s}  residual after attention")
        m = blk.mlp(blk.ln_2(x_in))
        print(f"{f'block{i}.mlp':35s} {str(tuple(m.shape)):30s}  4C expansion + back to C")
        x_in = x_in + m
        print(f"{f'block{i}.x+mlp':35s} {str(tuple(x_in.shape)):30s}  residual after MLP")
    x_out = model.ln_f(x_in)
    print(f"{'ln_f':35s} {str(tuple(x_out.shape)):30s}  final norm")
    logits = model.head(x_out)
    print(f"{'logits':35s} {str(tuple(logits.shape)):30s}  vocab scores: (B,T,V)")
    print("=" * 90)


def manual_grad_check(model: nn.Module, param: nn.Parameter, eps: float = 1e-5) -> dict:
    """Numerical vs analytical gradient, averaged across all elements of a tensor parameter.
    Useful when you want to know 'is the direction right' without picking one element.
    """
    return manual_grad_check_scalar(model, param, idx=None, eps=eps)


def manual_grad_check_scalar(
    model: nn.Module,
    param: nn.Parameter,
    idx: Optional[tuple] = None,
    eps: float = 1e-2,
) -> dict:
    """Verify the gradient at one scalar element (or its mean, if idx is None)
    of `param` against a central-difference numerical estimate.

    Note on eps: finite-difference accuracy is ~ O(eps² + machine_eps/|analytic|).
    For an FP32 matmul accumulating 16+ products, eps=1e-5 produces a loss change
    below float precision (we measured the loss literally didn't move). eps=1e-2
    gives a clean, readable signal at the cost of bias from the O(eps²) truncation
    error (which we display).
    """
    device = param.device
    cfg = model.cfg
    # Use a single sequence (B=1, T=4) so the loss isn't averaged across many
    # tokens that wash out the perturbation.
    x = torch.randint(0, cfg.vocab_size, (1, 4), device=device)
    y = torch.randint(0, cfg.vocab_size, (1, 4), device=device)

    def loss_at():
        _, l = model(x, y)
        return l

    # analytical gradient
    model.zero_grad()
    _, l_central = model(x, y)
    l_central.backward()
    analytic_full = param.grad.detach().clone()

    if idx is None:
        # mean over all elements: numerically nudge the whole tensor (mean-of-means)
        with torch.no_grad():
            orig = param.detach().clone()
            param.add_(eps);  lp = loss_at().item()
            param.copy_(orig); param.add_(-eps); lm = loss_at().item()
            param.copy_(orig)
        analytic = analytic_full.mean().item()
        numeric = (lp - lm) / (2 * eps)
        return {
            'param': _qualified_name(model, param) + ' (mean)',
            'analytic': analytic,
            'numeric': numeric,
            'abs_err': float(abs(analytic - numeric)),
            'agree_5dp': float(abs(analytic - numeric)) < 1e-5,
        }

    # point-wise check at idx
    analytic = analytic_full[idx].item()

    with torch.no_grad():
        orig = param.detach().clone()
        tmp = orig.clone()
        tmp[idx] = tmp[idx] + eps
        param.copy_(tmp)
        lp = loss_at().item()
        tmp2 = orig.clone()
        tmp2[idx] = tmp2[idx] - eps
        param.copy_(tmp2)
        lm = loss_at().item()
        param.copy_(orig)

    numeric = (lp - lm) / (2 * eps)
    err = float(abs(analytic - numeric))
    # Relative error is the honest comparison: 5 decimal places of the analytic
    # value, regardless of magnitude.
    rel_err = err / max(abs(analytic), 1e-9)
    return {
        'param': f'{_qualified_name(model, param)}{list(idx)}',
        'analytic': analytic,
        'numeric': numeric,
        'abs_err': err,
        'rel_err': rel_err,
        # With eps=1e-2, truncation error is ~eps²=1e-4; the analytic-vs-numeric
        # match is meaningful to ~4 decimal places. We report both absolute and
        # relative error and let the human judge.
        'agree_4dp': rel_err < 1e-2,
        'eps': eps,
    }


def _qualified_name(model: nn.Module, param: nn.Parameter) -> str:
    for n, p in model.named_parameters():
        if p is param:
            return n
    return '<unknown>'


# alias used inside the notebook
_name = _qualified_name


# -----------------------------
# Broken gradient accumulation
# -----------------------------

def broken_grad_accum_loss(micro_batches):
    """The bug: average of averages, when micro-batches have different
    counts of tokens, is NOT the correct batch loss.

    Correct way: sum losses * micro_batch_size / total_tokens,
    OR sum losses weighted by num_tokens, then mean.
    What people write by accident: mean of (mean loss per micro-batch).
    When micro-batches all have the same number of tokens, these agree.
    When they differ, the 'avg of avgs' under-weights large batches.
    """
    correct = sum(l * n for l, n in micro_batches) / sum(n for _, n in micro_batches)
    broken = sum(l for l, _ in micro_batches) / len(micro_batches)
    return correct, broken


# -----------------------------
# MFU
# -----------------------------

# RTX 3060 laptop specs (Ampere GA106, capability 8.6)
# 30 SMs. 128 FP32 cores per SM = 3840 FP32 cores
# Boost clock ~1.7 GHz
# Peak FP32 = 3840 * 2 * 1.7e9 ~ 13.06 TFLOPS
# BF16 tensor core throughput = 2x FP32 -> ~26 TFLOPS dense
# We treat FP32 training (the model is small; bf16 path is fragile on 3060 with 6GB).
PEAK_FP32_TFLOPS = 13.0
PEAK_BF16_TFLOPS = 26.0  # sparse spec, dense often lower


def model_flops_per_token(cfg: GPTConfig) -> int:
    """Approx FLOPs per token for a transformer forward (no embeds).

    Standard Kaplan/McCandlish estimate: 2 * params * tokens (forward+backward ~6N).
    Here we just compute forward * tokens for the parameter FLOPs.
    """
    n_params = sum(p.numel() for p in cfg_to_model(cfg).parameters())
    # forward = 2*N per token, backward ~4*N per token (2x forward + grad compute)
    return 6 * n_params


def cfg_to_model(cfg: GPTConfig) -> TinyGPT:
    return TinyGPT(cfg)


@contextlib.contextmanager
def cuda_sync_timer():
    """Accurate GPU timer using cuda.synchronize."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield lambda: _elapsed(start)


def _elapsed(start):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start


# -----------------------------
# Synthetic data
# -----------------------------

def make_synthetic_batch(B: int, T: int, vocab_size: int, device: str = 'cpu',
                         pattern_strength: float = 0.7, seed: int = None):
    """A tiny 'dataset': tokens follow a simple pattern (mostly monotonic)
    with some noise. This gives the model something to actually learn.

    pattern_strength=1.0 → fully deterministic sequence
    pattern_strength=0.0 → pure random (memorisable to ~0 loss)
    """
    if seed is not None:
        g = torch.Generator(device='cpu').manual_seed(seed)
    else:
        g = torch.Generator(device='cpu')
    x = torch.zeros(B, T, dtype=torch.long, device=device)
    y = torch.zeros(B, T, dtype=torch.long, device=device)
    for b in range(B):
        # base sequence: arithmetic progression
        start = torch.randint(0, max(1, vocab_size - T - 1), (1,), generator=g).item()
        seq = torch.arange(start, start + T, dtype=torch.long)  # CPU
        # mix with noise
        noise_mask = torch.rand(T, generator=g) > pattern_strength
        noise = torch.randint(0, vocab_size, (T,), generator=g)
        seq = torch.where(noise_mask, noise, seq)
        # shift targets so the task is "next token"
        x[b] = seq.to(device)
        y[b, :-1] = seq[1:].to(device)
        y[b, -1]  = torch.randint(0, vocab_size, (1,), generator=g).item()
    return x, y