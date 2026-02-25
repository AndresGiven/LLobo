import copy

import torch
import torch.nn as nn
from torch.nn import functional as F


def build_rope_cache(seq_len, dim, device):
    assert dim % 2 == 0, "Embedding dim must be even for RoPE"
    half_dim = dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim).float() / half_dim))
    positions = torch.arange(seq_len, dtype=torch.float)
    angles = torch.einsum("i,j->ij", positions, inv_freq)
    sin = torch.sin(torch.cat([angles, angles], dim=-1)).to(device)
    cos = torch.cos(torch.cat([angles, angles], dim=-1)).to(device)
    return sin, cos


def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).reshape_as(x)


def apply_rotary_pos_emb(q, k, sin, cos):
    sin = sin[None, None, :, :]
    cos = cos[None, None, :, :]
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class EMA:
    def __init__(self, model, device, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        self.device = device


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.num_embd % config.num_heads == 0
        self.config = config
        self.num_heads = config.num_heads
        self.num_embd = config.num_embd
        self.q_proj = nn.Linear(config.num_embd, config.num_embd)
        self.kv_proj = nn.Linear(
            config.num_embd,
            2 * (config.num_embd // config.num_heads) * config.num_kv_groups,
        )
        self.head_scale = nn.Parameter(torch.ones(self.num_heads))
        self.causal_proj = nn.Linear(config.num_embd, config.num_embd)
        sin, cos = build_rope_cache(
            config.context_size, config.num_embd // config.num_heads, config.device
        )
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)

    def forward(self, x):
        B, T, C = x.size()
        G = self.config.num_kv_groups
        H = self.num_heads
        d_head = C // H
        heads_per_kv = H // G

        q = self.q_proj(x).view(B, T, H, d_head).transpose(1, 2)
        kv = self.kv_proj(x)
        kv = kv.view(B, T, G, 2, d_head).permute(0, 2, 3, 1, 4)
        k, v = kv[:, :, 0], kv[:, :, 1]
        k = k.repeat_interleave(heads_per_kv, dim=1)
        v = v.repeat_interleave(heads_per_kv, dim=1)

        q, k = apply_rotary_pos_emb(q, k, self.sin[:T], self.cos[:T])
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.config.dropout if self.training else 0.0,
        )
        out = out * self.head_scale.view(1, H, 1, 1)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.causal_proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.num_embd, 4 * config.num_embd)
        self.silu = nn.SiLU()
        self.linear2 = nn.Linear(2 * config.num_embd, config.num_embd)

    def forward(self, x):
        x = self.linear1(x)
        x1, x2 = x.chunk(2, dim=-1)
        gate = self.silu(x2)
        x = gate * x1
        x = self.linear2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.num_embd)
        self.attention = CausalSelfAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.num_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        norm1 = self.layer_norm1(x)
        attn = self.attention(norm1)
        resid1 = x + attn
        norm2 = self.layer_norm2(resid1)
        mlp_out = self.mlp(norm2)
        resid2 = resid1 + mlp_out
        return resid2


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.grad_cp = config.grad_cp
        self.transformer = nn.ModuleDict(
            dict(
                token_embd=nn.Embedding(config.vocab_size, config.num_embd),
                transformers=nn.ModuleList(
                    [Transformer(config) for _ in range(config.num_transformers)]
                ),
                layer_norm=nn.LayerNorm(config.num_embd),
            )
        )
        self.linear = nn.Linear(config.num_embd, config.vocab_size, bias=False)
        self.transformer.token_embd.weight = self.linear.weight

    def forward(self, x, targets=None):
        x = self.transformer.token_embd(x)
        for block in self.transformer.transformers:
            if self.grad_cp and self.training:
                with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                    x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=True)
            else:
                x = block(x)
        x = self.transformer.layer_norm(x)
        logits = self.linear(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
