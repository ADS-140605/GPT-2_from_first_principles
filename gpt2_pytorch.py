"""Minimal GPT-2 (124M) architecture in PyTorch.

This is a pedagogical implementation matching GPT-2 small config:
- n_layer=12, n_head=12, n_embd=768, vocab_size=50257

Use `hf_inference.py` to run the Hugging Face pretrained model for real-world usage.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_ctx: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)

        # reshape for multi-head
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal mask: allow attending to past
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(causal_mask == 0, float("-inf"))

        if attn_mask is not None:
            att = att + attn_mask

        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config = GPT2Config()):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5)
        # language modeling head tied to token embeddings
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        device = input_ids.device
        B, T = input_ids.size()

        if position_ids is None:
            position_ids = torch.arange(0, T, device=device).unsqueeze(0)

        wte = self.wte(input_ids)  # token embeddings
        wpe = self.wpe(position_ids)
        x = self.drop(wte + wpe)

        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,T)

        for block in self.h:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


if __name__ == "__main__":
    # quick smoke test
    cfg = GPT2Config()
    model = GPT2Model(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    logits = model(input_ids)
    print("logits.shape=", logits.shape)
