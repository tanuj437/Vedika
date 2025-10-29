# -*- coding: utf-8 -*-
"""
SandhiJoiner — Neural Sanskrit Sandhi Joining Module
====================================================

Performs sandhi joining on Sanskrit words or short phrases.
Loads a fine-tuned Transformer Seq2Seq model automatically.

Usage:
    from vedika import join_sandhi
    print(join_sandhi("रामः+हनुमान्+च"))
    print(join_sandhi("राम अयोध्या गच्छति"))

Author: Tanuj Saxena, Soumya Sharma
License: MIT
Version: 1.1.1
"""

from __future__ import annotations
import os
import math
from typing import Dict, List, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "tf_checkpoint_joiner.pth"
)


# ---------------------------------------------------------------------
# Core Components
# ---------------------------------------------------------------------
def _strip_compiled_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove `_orig_mod.` prefix added by torch.compile when present."""
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    return state_dict


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.cos(pos * div)
        pe[:, 1::2] = torch.sin(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TFSeq2Seq(nn.Module):
    """Transformer encoder-decoder for sandhi joining."""
    def __init__(self, vocab_size, d_model, nhead, num_layers, ffn_mult, dropout, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.tf = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=ffn_mult * d_model,
            dropout=dropout, batch_first=True
        )
        self.out = nn.Linear(d_model, vocab_size)

    def make_masks(self, src, tgt):
        src_kpm = (src == self.pad_idx)
        tgt_kpm = (tgt == self.pad_idx)
        T = tgt.size(1)
        causal = torch.triu(torch.ones(T, T, device=tgt.device), diagonal=1).bool()
        return src_kpm, tgt_kpm, causal

    def forward(self, src, tgt):
        src_emb = self.pos(self.src_emb(src))
        tgt_emb = self.pos(self.tgt_emb(tgt))
        src_kpm, tgt_kpm, causal = self.make_masks(src, tgt)
        y = self.tf(
            src=src_emb, tgt=tgt_emb,
            src_key_padding_mask=src_kpm,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=src_kpm,
            tgt_mask=causal
        )
        return self.out(y)


# ---------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------
@dataclass
class Vocab:
    char2idx: Dict[str, int]
    idx2char: Dict[int, str]
    pad_token: str = "<PAD>"
    sos_token: str = "<SOS>"
    eos_token: str = "<EOS>"
    unk_token: str = "<UNK>"
    max_len: int = 220

    @property
    def pad_idx(self): return self.char2idx[self.pad_token]
    @property
    def sos_idx(self): return self.char2idx[self.sos_token]
    @property
    def eos_idx(self): return self.char2idx[self.eos_token]
    @property
    def unk_idx(self): return self.char2idx[self.unk_token]

    def encode(self, s: str, add_special=False):
        if add_special:
            s = s[: self.max_len - 2]
            ids = [self.sos_idx] + [self.char2idx.get(c, self.unk_idx) for c in s] + [self.eos_idx]
        else:
            s = s[: self.max_len]
            ids = [self.char2idx.get(c, self.unk_idx) for c in s]
        if len(ids) < self.max_len:
            ids += [self.pad_idx] * (self.max_len - len(ids))
        return ids, len(ids)

    def decode(self, ids):
        out, started = [], False
        for t in ids:
            if t == self.sos_idx and not started:
                started = True
                continue
            if t == self.eos_idx:
                break
            if t in (self.pad_idx, self.sos_idx):
                continue
            out.append(self.idx2char.get(t, self.unk_token))
        return "".join(out)


# ---------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------
@torch.no_grad()
def greedy_decode_model(model, src, sos_idx, eos_idx, pad_idx, max_len, amp_dtype=None):
    """Greedy autoregressive decoding."""
    device = src.device
    B = src.size(0)
    src_kpm = (src == pad_idx)
    with torch.autocast(device_type="cuda", dtype=amp_dtype,
                        enabled=(src.is_cuda and amp_dtype is not None)):
        src_emb = model.pos(model.src_emb(src))
        memory = model.tf.encoder(src_emb, mask=None, src_key_padding_mask=src_kpm)
    ys = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        T = ys.size(1)
        causal = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        tgt_kpm = (ys == pad_idx)
        with torch.autocast(device_type="cuda", dtype=amp_dtype,
                            enabled=(src.is_cuda and amp_dtype is not None)):
            tgt_emb = model.pos(model.tgt_emb(ys))
            dec_states = model.tf.decoder(
                tgt=tgt_emb, memory=memory,
                tgt_mask=causal, tgt_key_padding_mask=tgt_kpm,
                memory_key_padding_mask=src_kpm
            )
            out = model.out(dec_states)
        next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
        if (next_token.squeeze(1) == eos_idx).all():
            break
    return ys


def _normalize_split_text(s: str) -> str:
    """Normalize pluses, punctuation and whitespace."""
    if not s:
        return ""
    s = str(s)
    s = s.replace("\u00A0", " ").replace("\t", " ").replace("+", " ")
    s = s.replace("…", "...").replace("॥", " ").replace("।", " ")
    s = " ".join(s.split())
    return s.strip()


# ---------------------------------------------------------------------
# SandhiJoiner Class
# ---------------------------------------------------------------------
class SandhiJoiner:
    """Loads the Transformer model and performs sandhi joining."""
    _instance: Optional["SandhiJoiner"] = None

    def __init__(self, ckpt_path: str = CHECKPOINT_PATH, device: Optional[str] = None):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        raw = torch.load(ckpt_path, map_location="cpu")

        char2idx, idx2char = raw["char2idx"], raw["idx2char"]
        cfg = raw.get("config", {})
        vocab = Vocab(char2idx, idx2char, max_len=int(cfg.get("max_len", 220)))

        model = TFSeq2Seq(
            vocab_size=len(char2idx),
            d_model=int(cfg.get("d_model", 256)),
            nhead=int(cfg.get("nhead", 8)),
            num_layers=int(cfg.get("num_layers", 4)),
            ffn_mult=int(cfg.get("ffn_mult", 4)),
            dropout=float(cfg.get("dropout", 0.1)),
            pad_idx=vocab.pad_idx,
        )

        sd = _strip_compiled_prefix(raw["model_state_dict"])
        model.load_state_dict(sd, strict=False)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.vocab = vocab
        self.amp_dtype = torch.bfloat16 if self.device.type == "cuda" else None

    @classmethod
    def get_instance(cls) -> "SandhiJoiner":
        """Singleton pattern: load once per process."""
        if cls._instance is None:
            cls._instance = SandhiJoiner()
        return cls._instance

    def predict(self, text: str) -> str:
        """Predict joined sandhi form for input text."""
        if not text:
            return ""
        text = _normalize_split_text(text)
        ids, _ = self.vocab.encode(text, add_special=False)
        src = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        pred = greedy_decode_model(
            self.model, src,
            self.vocab.sos_idx, self.vocab.eos_idx, self.vocab.pad_idx,
            self.vocab.max_len, self.amp_dtype
        )
        return self.vocab.decode(pred[0].tolist())


# ---------------------------------------------------------------------
# Public function API
# ---------------------------------------------------------------------
def join_sandhi(text: str) -> str:
    """Join Sanskrit words using the preloaded sandhi joiner."""
    sj = SandhiJoiner.get_instance()
    return sj.predict(text)


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    examples = [
        "रामः+हनुमान्+च",
        "रामः हनुमान् च",
        "रामःहनुमान्च",
        "राम अयोध्या गच्छति॥",
    ]
    sj = SandhiJoiner.get_instance()
    for e in examples:
        print(f"{e} → {sj.predict(e)}")
