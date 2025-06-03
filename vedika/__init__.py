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
from .utils.download import get_file


get_file("sandhi_split")
get_file("sandhi_joiner")
get_file("cleaned_metres")

from .normalizer import TextNormalizer, normalize_standard_sanskrit_text, normalize_vedic_sanskrit_text
from .sandhi_split import SanskritSplit, split_word, split_words
from .sandhi_join import SandhiJoiner, load_joiner, quick_join
from .syllabification import SanskritMetrics, analyze_text
from .sentence_splitter import SentenceSplitter, split_sentences
from .tokenizer import SanskritTokenizer, TokenizerConfig

__version__ = "1.0.0"
__author__ = "Tsnuj Saxena, Soumya Sharma"
__email__ = "tanuj.saxena.rks@gmail.com, soumyasharma1599@gmail.com"
__license__ = "MIT"
__description__ = "A comprehensive toolkit for Sanskrit text processing"

# Expose main classes and functions
__all__ = [
    # Normalizer
    'TextNormalizer',
    'normalize_standard_sanskrit_text',
    'normalize_vedic_sanskrit_text',
    
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