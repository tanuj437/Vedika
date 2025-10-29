# -*- coding: utf-8 -*-
"""
Sanskrit Sandhi Splitter — Transformer Seq2Seq Model (Inference)
===============================================================

Performs sandhi splitting on Sanskrit words or short sentences
using a fine-tuned Transformer model.

Compatible with checkpoints produced by the Vedika fine-tuning pipeline.

Usage:
    from vedika import split_word, split_sentence

    print(split_word("रामःसीतायैवनम्"))
    # → "रामः+सीतायै+वनम्"

Author: Tanuj Saxena (2025)
License: MIT
Version: 1.1.0
"""

import os
import math
import torch
import torch.nn as nn
from typing import Dict

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "tf_checkpoint_split.pth"
)

DEFAULT_CONFIG = {
    "d_model": 256,
    "nhead": 8,
    "num_layers": 4,
    "ffn_mult": 4,
    "dropout": 0.1,
    "max_len": 220,
    "sos_token": "<SOS>",
    "eos_token": "<EOS>",
    "pad_token": "<PAD>",
    "unk_token": "<UNK>",
}


# ---------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Transformers."""
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


# ---------------------------------------------------------------------
# Transformer Seq2Seq
# ---------------------------------------------------------------------
class TFSeq2Seq(nn.Module):
    """Transformer encoder-decoder for Sanskrit sandhi splitting."""
    def __init__(self, vocab_size: int, d_model: int, nhead: int,
                 num_layers: int, ffn_mult: int, dropout: float, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.tf = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ffn_mult * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_emb = self.pos(self.src_emb(src))
        tgt_emb = self.pos(self.tgt_emb(tgt))
        src_kpm = (src == self.pad_idx)
        tgt_kpm = (tgt == self.pad_idx)
        causal = torch.triu(torch.ones(tgt.size(1), tgt.size(1), device=tgt.device), diagonal=1).bool()
        y = self.tf(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_kpm,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=src_kpm,
            tgt_mask=causal,
        )
        return self.out(y)


# ---------------------------------------------------------------------
# Sandhi Splitter
# ---------------------------------------------------------------------
class SanskritSplit:
    """
    Sanskrit Sandhi Splitter — Transformer-based.

    Loads a fine-tuned checkpoint and performs character-level sandhi splitting.
    Uses a singleton pattern to avoid reloading the model repeatedly.
    """

    _instance = None

    def __init__(self):
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device)
        self.config = ckpt.get("config", DEFAULT_CONFIG)

        # Load vocabulary
        if "char2idx" in ckpt:
            self.vocab = ckpt["char2idx"]
            self.inv_vocab = ckpt["idx2char"]
        elif "vocab" in ckpt:
            self.vocab = ckpt["vocab"]
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
        else:
            raise ValueError("Checkpoint missing vocabulary (char2idx/idx2char)")

        self.pad_idx = self.vocab[self.config["pad_token"]]

        # Initialize model
        self.model = TFSeq2Seq(
            vocab_size=len(self.vocab),
            d_model=self.config["d_model"],
            nhead=self.config["nhead"],
            num_layers=self.config["num_layers"],
            ffn_mult=self.config["ffn_mult"],
            dropout=self.config["dropout"],
            pad_idx=self.pad_idx,
        ).to(self.device)

        state = ckpt.get("model_state_dict")
        if state is None:
            raise ValueError("Checkpoint missing model_state_dict")

        # Adjust vocab mismatch if needed
        model_vocab_size = state["src_emb.weight"].shape[0]
        if model_vocab_size != len(self.vocab):
            self.model = TFSeq2Seq(
                vocab_size=model_vocab_size,
                d_model=self.config["d_model"],
                nhead=self.config["nhead"],
                num_layers=self.config["num_layers"],
                ffn_mult=self.config["ffn_mult"],
                dropout=self.config["dropout"],
                pad_idx=self.pad_idx,
            ).to(self.device)

        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    # Singleton accessor
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SanskritSplit()
        return cls._instance

    # --------------------------------------------------------
    def _encode(self, text: str) -> torch.Tensor:
        tokens = [self.config["sos_token"]] + list(text[: self.config["max_len"] - 2]) + [self.config["eos_token"]]
        token_ids = [self.vocab.get(t, self.vocab[self.config["unk_token"]]) for t in tokens]
        return torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def _decode(self, ids: torch.Tensor) -> str:
        chars = []
        for t in ids.tolist():
            if t in (
                self.vocab[self.config["pad_token"]],
                self.vocab[self.config["sos_token"]],
                self.vocab[self.config["eos_token"]],
            ):
                continue
            chars.append(self.inv_vocab.get(t, self.config["unk_token"]))
        return "".join(chars)

    # --------------------------------------------------------
    @torch.no_grad()
    def _greedy_decode(self, src: torch.Tensor) -> torch.Tensor:
        sos = self.vocab[self.config["sos_token"]]
        eos = self.vocab[self.config["eos_token"]]
        pad = self.vocab[self.config["pad_token"]]
        tgt = torch.full((1, 1), sos, dtype=torch.long, device=self.device)

        for _ in range(self.config["max_len"] - 1):
            out = self.model(src, tgt)
            next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token.item() == eos:
                break
        return tgt

    # --------------------------------------------------------
    def split_word(self, word: str) -> str:
        """Split a single Sanskrit word into sandhi components."""
        if not word:
            return ""
        src = self._encode(word)
        tgt = self._greedy_decode(src)
        return self._decode(tgt.squeeze(0))

    def split_sentence(self, sentence: str) -> Dict[str, str]:
        """Split all words in a sentence."""
        return {w: self.split_word(w) for w in sentence.strip().split()}


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def split_word(text: str) -> str:
    """Split a single Sanskrit word using the preloaded model."""
    return SanskritSplit.get_instance().split_word(text)


def split_sentence(sentence: str) -> Dict[str, str]:
    """Split each word in a Sanskrit sentence."""
    return SanskritSplit.get_instance().split_sentence(sentence)


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("🧩 Sanskrit Sandhi Splitter")
    print(split_word("रामःसीतायैवनम्"))
