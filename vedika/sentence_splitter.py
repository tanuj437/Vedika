"""
Sanskrit Sentence Splitter
==========================

A robust sentence splitting library for Sanskrit and Devanagari text processing.
Handles various delimiters, abbreviations, and text formatting edge cases.

Author: Tanuj Saxena and Soumya Sharma
License: MIT License
Version: 1.0.1
"""

import re
from typing import List, Optional


class SentenceSplitterError(Exception):
    """Custom exception for sentence splitter errors."""
    pass


# ---------- Configurable Constants ----------
DEFAULT_SENTENCE_DELIMITERS = ['।', '॥', r'\.', r'\.\.\.']
DEFAULT_ABBREVIATION_EXCEPTIONS = [
    'डॉ', 'श्री', 'डॉ.', 'सं.', 'वि.क.', 'इ.स.', 'ई.',
    'e.g.', 'i.e.', 'cf.', 'vs.', 'etc.'
]
DEFAULT_QUOTATION_MARKS = ['"', "'", '“', '”', '‘', '’']
DEFAULT_DANDA_ALIASES = ['|']
DEFAULT_CLAUSE_DELIMITERS = [
    'ततः', 'एवम्', 'यदा', 'यदि', 'इति', 'तस्मात्', 'तदा', 'इत्युक्त्वा', 'किन्तु'
]


# ---------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """Preprocess text by normalizing punctuation and removing unwanted characters."""
    if not isinstance(text, str):
        raise SentenceSplitterError(f"Expected string input, got {type(text)}")

    if not text:
        return ''

    # Replace danda aliases and normalize whitespace
    for alias in DEFAULT_DANDA_ALIASES:
        text = text.replace(alias, '।')

    # Remove zero-width characters
    text = re.sub(r'[\u200C\u200D]', '', text)

    # Normalize punctuation
    text = re.sub(r'[।|]{2,}', '।', text)
    text = re.sub(r'[॥]{2,}', '॥', text)
    text = re.sub(r'\.{3,}', '...', text)
    text = re.sub(r'\.{2}', '.', text)

    # Normalize whitespace and dashes
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\u00AD', '').replace('‐', '-').replace('–', '-').replace('—', '-')

    return text.strip()


def _protect_abbreviations(text: str, exceptions: List[str]) -> str:
    """Protect abbreviations from being split."""
    for abbr in exceptions:
        safe = abbr.replace('.', '<DOT>')
        text = text.replace(abbr, safe)
    return text


def _protect_numbered_lists(text: str) -> str:
    """Protect numbered lists like 1., 2., etc."""
    return re.sub(r'\b(\d+)\.', r'\1<DOT>', text)


def _protect_quoted_blocks(text: str, quotation_marks: List[str]) -> str:
    """Protect quoted text from splitting."""
    def replacer(match):
        return match.group(0).replace('.', '<QDOT>')

    for q in quotation_marks:
        pattern = re.escape(q) + r'(.*?)' + re.escape(q)
        text = re.sub(pattern, replacer, text)
    return text


def _restore_placeholders(text: str) -> str:
    """Restore protected placeholders."""
    return text.replace('<DOT>', '.').replace('<QDOT>', '.')


def _fallback_split_long_sentence(sentence: str, clauses: List[str]) -> List[str]:
    """Split long sentences using clause delimiters as fallback."""
    for delim in clauses:
        sentence = sentence.replace(delim, f'||{delim}')
    parts = [s.strip() for s in sentence.split('||') if s.strip()]
    return parts if parts else [sentence]


# ---------------------------------------------------------------------
# Core Sentence Splitter
# ---------------------------------------------------------------------
def split_sentences(
    text: str,
    delimiters: Optional[List[str]] = None,
    abbreviations: Optional[List[str]] = None,
    quotation_marks: Optional[List[str]] = None,
    verbose: bool = False,
    max_sentence_length: Optional[int] = None,
    fallback_clauses: Optional[List[str]] = None
) -> List[str]:
    """
    Split Sanskrit/Devanagari text into sentences.

    Args:
        text (str): Input text to split.
        delimiters (List[str], optional): Sentence delimiters.
        abbreviations (List[str], optional): Abbreviations to protect.
        quotation_marks (List[str], optional): Quotation marks to protect.
        verbose (bool): Ignored (kept for API compatibility).
        max_sentence_length (int, optional): Max sentence length before fallback.
        fallback_clauses (List[str], optional): Clause delimiters for fallback.

    Returns:
        List[str]: List of split sentences.
    """
    if not isinstance(text, str):
        raise SentenceSplitterError(f"Expected string input, got {type(text)}")
    if not text.strip():
        return []

    # Defaults
    delimiters = delimiters or DEFAULT_SENTENCE_DELIMITERS
    abbreviations = abbreviations or DEFAULT_ABBREVIATION_EXCEPTIONS
    quotation_marks = quotation_marks or DEFAULT_QUOTATION_MARKS
    fallback_clauses = fallback_clauses or DEFAULT_CLAUSE_DELIMITERS

    if max_sentence_length is not None and max_sentence_length <= 0:
        raise SentenceSplitterError("max_sentence_length must be positive")

    # Preprocess & protect special cases
    text = preprocess_text(text)
    text = _protect_abbreviations(text, abbreviations)
    text = _protect_numbered_lists(text)
    text = _protect_quoted_blocks(text, quotation_marks)

    split_pattern = r'(' + '|'.join([
        d if d.startswith(r'\.') else re.escape(d) for d in delimiters
    ]) + r')'

    try:
        chunks = re.split(split_pattern, text)
    except re.error as e:
        raise SentenceSplitterError(f"Invalid regex pattern: {e}")

    sentences = []
    current = ''

    for chunk in chunks:
        if not chunk.strip():
            continue

        current += chunk
        if re.fullmatch(split_pattern, chunk):
            sentence = _restore_placeholders(current.strip())
            if sentence:
                if max_sentence_length and len(sentence) > max_sentence_length:
                    sentences.extend(_fallback_split_long_sentence(sentence, fallback_clauses))
                else:
                    sentences.append(sentence)
            current = ''

    if current.strip():
        sentence = _restore_placeholders(current.strip())
        if max_sentence_length and len(sentence) > max_sentence_length:
            sentences.extend(_fallback_split_long_sentence(sentence, fallback_clauses))
        else:
            sentences.append(sentence)

    return sentences


# ---------------------------------------------------------------------
# SentenceSplitter Class
# ---------------------------------------------------------------------
class SentenceSplitter:
    """
    Configurable Sanskrit Sentence Splitter.

    Example:
        >>> splitter = SentenceSplitter(max_sentence_length=120)
        >>> text = "रामः वनं गच्छति। सीता अपि गच्छति।"
        >>> splitter.split(text)
        ['रामः वनं गच्छति।', 'सीता अपि गच्छति।']
    """

    def __init__(
        self,
        delimiters: Optional[List[str]] = None,
        abbreviations: Optional[List[str]] = None,
        quotation_marks: Optional[List[str]] = None,
        max_sentence_length: Optional[int] = None,
        fallback_clauses: Optional[List[str]] = None,
        verbose: bool = False
    ):
        self.delimiters = delimiters or DEFAULT_SENTENCE_DELIMITERS
        self.abbreviations = abbreviations or DEFAULT_ABBREVIATION_EXCEPTIONS
        self.quotation_marks = quotation_marks or DEFAULT_QUOTATION_MARKS
        self.max_sentence_length = max_sentence_length
        self.fallback_clauses = fallback_clauses or DEFAULT_CLAUSE_DELIMITERS
        self.verbose = verbose

    def split(self, text: str) -> List[str]:
        """Split text into sentences."""
        return split_sentences(
            text,
            delimiters=self.delimiters,
            abbreviations=self.abbreviations,
            quotation_marks=self.quotation_marks,
            max_sentence_length=self.max_sentence_length,
            fallback_clauses=self.fallback_clauses
        )

    def update_config(self, **kwargs) -> None:
        """Update splitter configuration dynamically."""
        valid = {
            'delimiters', 'abbreviations', 'quotation_marks',
            'max_sentence_length', 'fallback_clauses', 'verbose'
        }
        invalid = set(kwargs.keys()) - valid
        if invalid:
            raise SentenceSplitterError(f"Invalid parameters: {invalid}")

        if 'max_sentence_length' in kwargs and kwargs['max_sentence_length'] is not None:
            if kwargs['max_sentence_length'] <= 0:
                raise SentenceSplitterError("max_sentence_length must be positive")

        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_config(self) -> dict:
        """Return current configuration as a dictionary."""
        return {
            'delimiters': self.delimiters,
            'abbreviations': self.abbreviations,
            'quotation_marks': self.quotation_marks,
            'max_sentence_length': self.max_sentence_length,
            'fallback_clauses': self.fallback_clauses,
            'verbose': self.verbose,
        }


# Backward-compatible alias
split_sanskrit_sentences = split_sentences


# ---------------------------------------------------------------------
# Self-test (for standalone use)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    example_text = (
        "श्रीरामः वनं गच्छति। सीता अपि गच्छति, किन्तु राक्षसाः उग्रं कृत्यं आरभन्ति। "
        "डॉ. रामशास्त्री, जो महान् पण्डितः आसीत्, तस्मिन्संस्कारः प्रवर्तितः। "
        "इ.स. २०२३, सं. १०१। 'यदा यदा हि धर्मस्य' इत्युक्त्वा भगवान् धर्मसंस्थापनं आरब्धवान्।"
    )
    splitter = SentenceSplitter(max_sentence_length=150)
    for i, s in enumerate(splitter.split(example_text), 1):
        print(f"{i}. {s}")
