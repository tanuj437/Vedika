"""
Sanskrit Sentence Splitter

A robust sentence splitting library for Sanskrit and Devanagari text processing.
Handles various delimiters, abbreviations, and text formatting edge cases.

Author: Tanuj Saxena and Soumya Sharma
License: MIT License
Version: 1.0.0
"""

import re
from typing import List, Optional, Union


class SentenceSplitterError(Exception):
    """Custom exception for sentence splitter errors."""
    pass


# ---------- Configurable Constants ----------
DEFAULT_SENTENCE_DELIMITERS = ['।', '॥', r'\.', r'\.\.\.']
DEFAULT_ABBREVIATION_EXCEPTIONS = [
    'डॉ', 'श्री', 'डॉ.', 'सं.', 'वि.क.', 'इ.स.', 'ई.',
    'e.g.', 'i.e.', 'cf.', 'vs.', 'etc.'
]
DEFAULT_QUOTATION_MARKS = ['"', "'", ''', ''', '"', '"']
DEFAULT_DANDA_ALIASES = ['|']
DEFAULT_CLAUSE_DELIMITERS = [
    'ततः', 'एवम्', 'यदा', 'यदि', 'इति', 'तस्मात्', 'तदा', 'इत्युक्त्वा', 'किन्तु'
]


def preprocess_text(text: str) -> str:
    """
    Preprocess text by normalizing punctuation and removing unwanted characters.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
        
    Raises:
        SentenceSplitterError: If text is not a string
    """
    if not isinstance(text, str):
        raise SentenceSplitterError(f"Expected string input, got {type(text)}")
    
    if not text:
        return ''
        
    # Replace danda aliases
    for alias in DEFAULT_DANDA_ALIASES:
        text = text.replace(alias, '।')
    
    # Remove zero-width characters
    text = re.sub(r'[\u200C\u200D]', '', text)
    
    # Normalize repeated punctuation
    text = re.sub(r'[।|]{2,}', '।', text)
    text = re.sub(r'[॥]{2,}', '॥', text)
    text = re.sub(r'\.{3,}', '...', text)
    text = re.sub(r'\.{2}', '.', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove/normalize various dash types
    text = text.replace('\u00AD', '').replace('‐', '-').replace('–', '-').replace('—', '-')
    
    return text.strip()


def _protect_abbreviations(text: str, exceptions: List[str]) -> str:
    """Protect abbreviations from being split by replacing dots with placeholders."""
    for abbr in exceptions:
        safe = abbr.replace('.', '<DOT>')
        text = text.replace(abbr, safe)
    return text


def _protect_numbered_lists(text: str) -> str:
    """Protect numbered lists from being split."""
    return re.sub(r'\b(\d+)\.', r'\1<DOT>', text)


def _protect_quoted_blocks(text: str, quotation_marks: List[str]) -> str:
    """Protect quoted text blocks from being split."""
    def replacer(match):
        return match.group(0).replace('.', '<QDOT>')
    
    for q in quotation_marks:
        pattern = re.escape(q) + r'(.*?)' + re.escape(q)
        text = re.sub(pattern, replacer, text)
    return text


def _restore_abbreviations(text: str) -> str:
    """Restore protected abbreviations."""
    return text.replace('<DOT>', '.')


def _restore_numbered_lists(text: str) -> str:
    """Restore protected numbered lists."""
    return text.replace('<DOT>', '.')


def _restore_quoted_blocks(text: str) -> str:
    """Restore protected quoted blocks."""
    return text.replace('<QDOT>', '.')


def _fallback_split_long_sentence(sentence: str, clauses: List[str]) -> List[str]:
    """
    Split long sentences using clause delimiters as fallback.
    
    Args:
        sentence (str): Long sentence to split
        clauses (List[str]): List of clause delimiters
        
    Returns:
        List[str]: List of sentence fragments
    """
    for delim in clauses:
        sentence = sentence.replace(delim, f'||{delim}')
    
    result = [s.strip() for s in sentence.split('||') if s.strip()]
    return result if result else [sentence]


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
        text (str): Input text to split
        delimiters (List[str], optional): List of sentence delimiters
        abbreviations (List[str], optional): List of abbreviations to protect
        quotation_marks (List[str], optional): List of quotation marks
        verbose (bool): Enable verbose logging
        max_sentence_length (int, optional): Maximum sentence length before fallback splitting
        fallback_clauses (List[str], optional): Clause delimiters for fallback splitting
        
    Returns:
        List[str]: List of split sentences
        
    Raises:
        SentenceSplitterError: If input validation fails
        
    Example:
        >>> text = "श्रीरामः वनं गच्छति। सीता अपि गच्छति।"
        >>> sentences = split_sentences(text)
        >>> print(sentences)
        ['श्रीरामः वनं गच्छति।', 'सीता अपि गच्छति।']
    """
    if not isinstance(text, str):
        raise SentenceSplitterError(f"Expected string input, got {type(text)}")
    
    if not text.strip():
        return []

    # Set defaults
    delimiters = delimiters or DEFAULT_SENTENCE_DELIMITERS
    abbreviations = abbreviations or DEFAULT_ABBREVIATION_EXCEPTIONS
    quotation_marks = quotation_marks or DEFAULT_QUOTATION_MARKS
    fallback_clauses = fallback_clauses or DEFAULT_CLAUSE_DELIMITERS
    
    if max_sentence_length is not None and max_sentence_length <= 0:
        raise SentenceSplitterError("max_sentence_length must be positive")

    # Preprocess text
    text = preprocess_text(text)
    
    # Protect special cases
    text = _protect_abbreviations(text, abbreviations)
    text = _protect_numbered_lists(text)
    text = _protect_quoted_blocks(text, quotation_marks)

    # Create split pattern
    split_pattern = r'(' + '|'.join([
        d if d.startswith(r'\.') else re.escape(d) 
        for d in delimiters
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
            # Restore protected elements
            sentence = _restore_quoted_blocks(current.strip())
            sentence = _restore_abbreviations(sentence)
            sentence = _restore_numbered_lists(sentence)

            if sentence:
                if verbose:
                    print(f"[Split] Sentence: {sentence}")

                # Apply fallback splitting if sentence is too long
                if max_sentence_length and len(sentence) > max_sentence_length:
                    sentences.extend(_fallback_split_long_sentence(sentence, fallback_clauses))
                else:
                    sentences.append(sentence)
            current = ''

    # Handle remaining text
    if current.strip():
        sentence = _restore_quoted_blocks(current.strip())
        sentence = _restore_abbreviations(sentence)
        sentence = _restore_numbered_lists(sentence)
        
        if max_sentence_length and len(sentence) > max_sentence_length:
            sentences.extend(_fallback_split_long_sentence(sentence, fallback_clauses))
        else:
            sentences.append(sentence)

    return sentences


class SentenceSplitter:
    """
    A configurable sentence splitter for Sanskrit and Devanagari text.
    
    This class provides a convenient interface for splitting text into sentences
    with customizable delimiters, abbreviation handling, and fallback mechanisms.
    
    Attributes:
        delimiters (List[str]): Sentence delimiters to use
        abbreviations (List[str]): Abbreviations to protect from splitting
        quotation_marks (List[str]): Quotation marks to handle
        max_sentence_length (int): Maximum sentence length before fallback
        fallback_clauses (List[str]): Clause delimiters for fallback splitting
        verbose (bool): Enable verbose output
        
    Example:
        >>> splitter = SentenceSplitter(verbose=True, max_sentence_length=100)
        >>> text = "राम वनं गच्छति। सीता गृहे तिष्ठति।"
        >>> sentences = splitter.split(text)
        >>> print(len(sentences))
        2
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
        """
        Initialize the SentenceSplitter.
        
        Args:
            delimiters (List[str], optional): Custom sentence delimiters
            abbreviations (List[str], optional): Custom abbreviation list
            quotation_marks (List[str], optional): Custom quotation marks
            max_sentence_length (int, optional): Maximum sentence length
            fallback_clauses (List[str], optional): Custom clause delimiters
            verbose (bool): Enable verbose logging
            
        Raises:
            SentenceSplitterError: If max_sentence_length is invalid
        """
        if max_sentence_length is not None and max_sentence_length <= 0:
            raise SentenceSplitterError("max_sentence_length must be positive")
            
        self.delimiters = delimiters or DEFAULT_SENTENCE_DELIMITERS
        self.abbreviations = abbreviations or DEFAULT_ABBREVIATION_EXCEPTIONS
        self.quotation_marks = quotation_marks or DEFAULT_QUOTATION_MARKS
        self.max_sentence_length = max_sentence_length
        self.fallback_clauses = fallback_clauses or DEFAULT_CLAUSE_DELIMITERS
        self.verbose = verbose

    def split(self, text: str) -> List[str]:
        """
        Split text into sentences using the configured parameters.
        
        Args:
            text (str): Text to split into sentences
            
        Returns:
            List[str]: List of sentences
            
        Raises:
            SentenceSplitterError: If input validation fails
        """
        return split_sentences(
            text=text,
            delimiters=self.delimiters,
            abbreviations=self.abbreviations,
            quotation_marks=self.quotation_marks,
            verbose=self.verbose,
            max_sentence_length=self.max_sentence_length,
            fallback_clauses=self.fallback_clauses
        )
    
    def update_config(self, **kwargs) -> None:
        """
        Update splitter configuration.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Raises:
            SentenceSplitterError: If invalid parameters are provided
        """
        valid_params = {
            'delimiters', 'abbreviations', 'quotation_marks', 
            'max_sentence_length', 'fallback_clauses', 'verbose'
        }
        
        invalid_params = set(kwargs.keys()) - valid_params
        if invalid_params:
            raise SentenceSplitterError(f"Invalid parameters: {invalid_params}")
        
        if 'max_sentence_length' in kwargs:
            if kwargs['max_sentence_length'] is not None and kwargs['max_sentence_length'] <= 0:
                raise SentenceSplitterError("max_sentence_length must be positive")
        
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get_config(self) -> dict:
        """
        Get current configuration as a dictionary.
        
        Returns:
            dict: Current configuration parameters
        """
        return {
            'delimiters': self.delimiters,
            'abbreviations': self.abbreviations,
            'quotation_marks': self.quotation_marks,
            'max_sentence_length': self.max_sentence_length,
            'fallback_clauses': self.fallback_clauses,
            'verbose': self.verbose
        }


# Convenience function for backward compatibility
split_sanskrit_sentences = split_sentences


def main():
    """Example usage of the sentence splitter."""
    example_text = (
        "श्रीरामः वनं गच्छति। सीता अपि गच्छति, किन्तु राक्षसाः उग्रं कृत्यं आरभन्ति। "
        "डॉ. रामशास्त्री, जो महान् पण्डितः आसीत्, तस्मिन्संस्कारः प्रवर्तितः। "
        "इ.स. २०२३, सं. १०१। 'यदा यदा हि धर्मस्य' इत्युक्त्वा भगवान् धर्मसंस्थापनं आरब्धवान्।"
    )
    
    print("Sanskrit Sentence Splitter Demo")
    print("=" * 40)
    
    # Using function interface
    print("\n1. Using function interface:")
    sentences = split_sentences(example_text, verbose=True)
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")
    
    # Using class interface
    print("\n2. Using class interface:")
    splitter = SentenceSplitter(verbose=False, max_sentence_length=150)
    sentences = splitter.split(example_text)
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")
    
    print(f"\nTotal sentences: {len(sentences)}")


if __name__ == "__main__":
    main()