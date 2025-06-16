"""
SanskritToolKit
==============

A comprehensive toolkit for Sanskrit text processing and analysis.

Features:
- Text normalization and cleaning
- Sandhi splitting and joining
- Syllabification and metrical analysis
- Sentence splitting
- Tokenization

Author: Tanuj Saxena and Soumya Sharma
Version: 0.0.1
License: MIT
"""
import os
import shutil
from huggingface_hub import snapshot_download


def _ensure_weights_downloaded():
    vedika_root = os.path.dirname(__file__)
    data_dir = os.path.join(vedika_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    required_files = {
        "sandhi_joiner.pth": "Vedika/vedika/data/sandhi_joiner.pth",
        "sandhi_split.pth": "Vedika/vedika/data/sandhi_split.pth",
        "cleaned_metres.json": "Vedika/vedika/data/cleaned_metres.json",
    }

    # Skip download if all files exist
    if all(os.path.exists(os.path.join(data_dir, fname)) for fname in required_files):
        return  # Already done

    # Download only selected files
    repo_path = snapshot_download(
        repo_id="tanuj437/Vedika",
        allow_patterns=list(required_files.values())
    )

    for fname, rel_path in required_files.items():
        src = os.path.join(repo_path, rel_path)
        dst = os.path.join(data_dir, fname)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

# Call when the package is imported
_ensure_weights_downloaded()

from .normalizer import TextNormalizer, normalize_standard_sanskrit_text, normalize_vedic_sanskrit_text , remove_sanskrit_stopwords
from .sandhi_split import SanskritSplit, split_word, split_words
from .sandhi_join import SandhiJoiner, load_joiner, quick_join
from .syllabification import SanskritMetrics, analyze_text
from .sentence_splitter import SentenceSplitter, split_sentences
from .tokenizer import SanskritTokenizer, TokenizerConfig

__version__ = "1.0.0"
__author__ = "Tanuj Saxena, Soumya Sharma"
__email__ = "tanuj.saxena.rks@gmail.com, soumyasharma1599@gmail.com"
__license__ = "MIT"
__description__ = "A comprehensive toolkit for Sanskrit text processing"

# Expose main classes and functions
__all__ = [
    # Normalizer
    'TextNormalizer',
    'normalize_standard_sanskrit_text',
    'normalize_vedic_sanskrit_text',
    'remove_sanskrit_stopwords',
    
    # Sandhi splitting
    'SanskritSplit',
    'split_word',
    'split_words',
    
    # Sandhi joining
    'SandhiJoiner',
    'load_joiner',
    'quick_join',
    
    # Syllabification
    'SanskritMetrics',
    'analyze_text',
    
    # Sentence splitting
    'SentenceSplitter',
    'split_sentences',
    
    # Tokenization
    'SanskritTokenizer',
    'TokenizerConfig'
]