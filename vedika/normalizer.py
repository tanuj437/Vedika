"""
Sanskrit Text Normalizer

A comprehensive text normalization library for Sanskrit and Devanagari text processing.
Handles Unicode normalization, script detection, diacritics, Vedic accents, and more.

Author: Tanuj Saxena and Soumya Sharma
License: MIT License
Version: 0.0.0
"""

import unicodedata
import re
from typing import List, Optional, Tuple, Dict, Union
from enum import Enum

try:
    from .sentence_splitter import split_sentences
except ImportError:
    # Fallback for standalone usage
    try:
        from .sentence_splitter import split_sentences
    except ImportError:
        split_sentences = None


class Script(Enum):
    """Enumeration for supported scripts."""
    DEVANAGARI = "devanagari"
    ROMAN = "roman"
    UNKNOWN = "unknown"


class TextNormalizerError(Exception):
    """Custom exception for text normalizer errors."""
    pass


# =========================== CONSTANTS ===============================

# Unicode ranges for different scripts
DEVANAGARI_RANGE = (0x0900, 0x097F)
DEVANAGARI_EXTENDED_RANGE = (0xA8E0, 0xA8FF)

# Nasal consonant mappings for anusvara correction
ANUSVARA_REPLACEMENTS = {
    'ङ्': 'कखगघङ',  # k-varga
    'ञ्': 'चछजझञ',  # c-varga  
    'ण्': 'टठडढण',  # ṭ-varga
    'न्': 'तथदधन',  # t-varga
    'म्': 'पफबभम'   # p-varga
}

# Latin digraph normalizations
LATIN_DIGRAPHS = {
    'sh': 'ś', 'Sh': 'Ś', 'SH': 'Ś',
    'ch': 'c', 'Ch': 'C', 'CH': 'C',
    'th': 't', 'Th': 'T', 'TH': 'T',
    'kh': 'k', 'Kh': 'K', 'KH': 'K',
    'gh': 'g', 'Gh': 'G', 'GH': 'G',
    'jh': 'j', 'Jh': 'J', 'JH': 'J',
    'dh': 'd', 'Dh': 'D', 'DH': 'D',
    'bh': 'b', 'Bh': 'B', 'BH': 'B',
    'ph': 'p', 'Ph': 'P', 'PH': 'P',
}

# Devanagari to Arabic digit mapping
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Vedic accent mappings
VEDIC_ACCENT_MAPPINGS = {
    '\u0951': '\u1CD0',  # Udatta
    '\u0952': '\u1CD2',  # Anudatta
    '\u1CDA': '\u1CDA',  # Svarita (no change)
}

# Characters to keep when removing foreign characters
ALLOWED_FOREIGN_CHARS = set('.,;:!?-—"\'()[]{}')


# =========================== BASE CLEANERS ===============================

def unicode_normalize(text: str, form: str = "NFC") -> str:
    """
    Normalize Unicode text using specified normalization form.
    
    Args:
        text (str): Input text to normalize
        form (str): Unicode normalization form ('NFC', 'NFD', 'NFKC', 'NFKD')
        
    Returns:
        str: Normalized text
        
    Raises:
        TextNormalizerError: If normalization fails or invalid form specified
    """
    if not isinstance(text, str):
        raise TextNormalizerError(f"Expected string input, got {type(text)}")
    
    valid_forms = {'NFC', 'NFD', 'NFKC', 'NFKD'}
    if form not in valid_forms:
        raise TextNormalizerError(f"Invalid normalization form '{form}'. Must be one of {valid_forms}")
    
    try:
        return unicodedata.normalize(form, text)
    except Exception as e:
        raise TextNormalizerError(f"Unicode normalization failed: {e}")


def remove_control_chars(text: str) -> str:
    """
    Remove Unicode control characters from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with control characters removed
        
    Raises:
        TextNormalizerError: If input is not a string
    """
    if not isinstance(text, str):
        raise TextNormalizerError(f"Expected string input, got {type(text)}")
    
    return ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')


def clean_whitespace_punctuation(text: str) -> str:
    """
    Normalize whitespace and standardize punctuation marks.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with normalized whitespace and punctuation
    """
    if not isinstance(text, str):
        raise TextNormalizerError(f"Expected string input, got {type(text)}")
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Standardize quotation marks
    text = re.sub(r'[""'']', '"', text)
    
    # Optional: Convert Devanagari punctuation to standard
    # text = re.sub(r'[।॥]', '.', text)
    
    return text.strip()


# ====================== SCRIPT DETECTION ===============================

def detect_script(text: str) -> Script:
    """
    Detect the primary script used in the text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        Script: Detected script type
        
    Raises:
        TextNormalizerError: If input is not a string
    """
    if not isinstance(text, str):
        raise TextNormalizerError(f"Expected string input, got {type(text)}")
    
    if not text.strip():
        return Script.UNKNOWN
    
    devanagari_count = 0
    roman_count = 0
    total_chars = 0
    
    for ch in text:
        if ch.isspace() or ch in ALLOWED_FOREIGN_CHARS:
            continue
            
        total_chars += 1
        
        # Check for Devanagari
        if (DEVANAGARI_RANGE[0] <= ord(ch) <= DEVANAGARI_RANGE[1] or
            DEVANAGARI_EXTENDED_RANGE[0] <= ord(ch) <= DEVANAGARI_EXTENDED_RANGE[1]):
            devanagari_count += 1
        # Check for Roman/Latin
        elif 'a' <= ch.lower() <= 'z':
            roman_count += 1
    
    if total_chars == 0:
        return Script.UNKNOWN
    
    # Determine primary script based on majority
    if devanagari_count > roman_count:
        return Script.DEVANAGARI
    elif roman_count > devanagari_count:
        return Script.ROMAN
    else:
        return Script.UNKNOWN


def get_script_statistics(text: str) -> Dict[str, int]:
    """
    Get detailed statistics about script usage in text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        Dict[str, int]: Dictionary with script statistics
    """
    if not isinstance(text, str):
        raise TextNormalizerError(f"Expected string input, got {type(text)}")
    
    stats = {
        'devanagari': 0,
        'roman': 0,
        'digits': 0,
        'punctuation': 0,
        'whitespace': 0,
        'other': 0,
        'total': len(text)
    }
    
    for ch in text:
        if ch.isspace():
            stats['whitespace'] += 1
        elif ch.isdigit():
            stats['digits'] += 1
        elif ch in ALLOWED_FOREIGN_CHARS:
            stats['punctuation'] += 1
        elif (DEVANAGARI_RANGE[0] <= ord(ch) <= DEVANAGARI_RANGE[1] or
              DEVANAGARI_EXTENDED_RANGE[0] <= ord(ch) <= DEVANAGARI_EXTENDED_RANGE[1]):
            stats['devanagari'] += 1
        elif 'a' <= ch.lower() <= 'z':
            stats['roman'] += 1
        else:
            stats['other'] += 1
    
    return stats


# =========================== TEXT FIXERS ================================

def fix_visarga(text: str) -> str:
    """
    Fix visarga characters by normalizing between standard and alternative forms.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with normalized visarga
    """
    if not isinstance(text, str):
        raise TextNormalizerError(f"Expected string input, got {type(text)}")
    
    # Convert alternative visarga to standard form
    return text.replace("ꣃ", "ः")


def correct_anusvara(text: str) -> str:
    """
    Correct anusvara usage by replacing with appropriate nasal consonants.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with corrected anusvara usage
    """
    if not isinstance(text, str):
        raise TextNormalizerError(f"Expected string input, got {type(text)}")
    
    # Replace anusvara with corresponding nasal based on following consonant
    for nasal, consonants in ANUSVARA_REPLACEMENTS.items():
        for consonant in consonants:
            # Look for anusvara followed by consonant and replace with nasal + halant
            pattern = f'ं(?={consonant})'
            text = re.sub(pattern, nasal, text)
    
    return text


def normalize_diacritics(text: str, form: str = "NFC") -> str:
    """
    Normalize diacritical marks in text.
    
    Args:
        text (str): Input text
        form (str): Unicode normalization form
        
    Returns:
        str: Text with normalized diacritics
    """
    return unicode_normalize(text, form)


def normalize_vedic_accents(text: str) -> str:
    """
    Normalize Vedic accent marks to standard forms.
    
    Args:
        text (str): Input text with Vedic accents
        
    Returns:
        str: Text with normalized Vedic accents
    """
    if not isinstance(text, str):
        raise TextNormalizerError(f"Expected string input, got {type(text)}")
    
    for old_accent, new_accent in VEDIC_ACCENT_MAPPINGS.items():
        text = text.replace(old_accent, new_accent)
    
    return text


def remove_vedic_accents(text: str) -> str:
    """
    Remove all Vedic accent marks from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with Vedic accents removed
    """
    if not isinstance(text, str):
        raise TextNormalizerError(f"Expected string input, got {type(text)}")
    
    # Remove various Vedic accent marks
    accent_pattern = r'[\u0951\u0952\u1CD0\u1CD2\u1CDA]'
    return re.sub(accent_pattern, '', text)


def normalize_latin_digraphs(text: str) -> str:
    """
    Normalize Latin digraphs in romanized Sanskrit text.
    
    Args:
        text (str): Input text with Latin digraphs
        
    Returns:
        str: Text with normalized digraphs
    """
    if not isinstance(text, str):
        raise TextNormalizerError(f"Expected string input, got {type(text)}")
    
    for digraph, replacement in LATIN_DIGRAPHS.items():
        text = text.replace(digraph, replacement)
    
    return text


def convert_digits(text: str, to_arabic: bool = True) -> str:
    """
    Convert digits between Devanagari and Arabic numerals.
    
    Args:
        text (str): Input text
        to_arabic (bool): If True, convert to Arabic; if False, convert to Devanagari
        
    Returns:
        str: Text with converted digits
    """
    if not isinstance(text, str):
        raise TextNormalizerError(f"Expected string input, got {type(text)}")
    
    if to_arabic:
        return text.translate(DEVANAGARI_DIGITS)
    else:
        # Reverse mapping for Arabic to Devanagari
        arabic_to_devanagari = str.maketrans("0123456789", "०१२३४५६७८९")
        return text.translate(arabic_to_devanagari)


def remove_foreign_characters(text: str, keep_punctuation: bool = True) -> str:
    """
    Remove non-Sanskrit characters from text.
    
    Args:
        text (str): Input text
        keep_punctuation (bool): Whether to keep basic punctuation marks
        
    Returns:
        str: Text with foreign characters removed
    """
    if not isinstance(text, str):
        raise TextNormalizerError(f"Expected string input, got {type(text)}")
    
    allowed_chars = ALLOWED_FOREIGN_CHARS if keep_punctuation else set()
    
    return ''.join(
        ch for ch in text
        if (DEVANAGARI_RANGE[0] <= ord(ch) <= DEVANAGARI_RANGE[1] or
            ch.isspace() or ch in allowed_chars)
    )


# =========================== SENTENCE SPLITTING ===========================

def sentence_split(text: str, **kwargs) -> List[str]:
    """
    Split Sanskrit text into sentences using robust rules.
    
    Args:
        text (str): Input text to split
        **kwargs: Additional arguments for sentence splitting
        
    Returns:
        List[str]: List of sentences
        
    Raises:
        TextNormalizerError: If sentence splitter is not available
    """
    if split_sentences is None:
        # Fallback sentence splitting using basic rules
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in '।॥.!?':
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return [s for s in sentences if s]
    
    return split_sentences(
        text,
        verbose=kwargs.get('verbose', False),
        max_sentence_length=kwargs.get('max_sentence_length', 200)
    )


# ========================== MASTER NORMALIZATION FUNCTIONS =============================

def normalize_standard_sanskrit_text(
    text: str,
    remove_foreign: bool = False,
    keep_punctuation: bool = True,
    convert_digits_to_arabic: bool = True,
    verbose: bool = False
) -> str:
    """
    Normalize standard Sanskrit text with comprehensive processing.
    
    Args:
        text (str): Input Sanskrit text
        remove_foreign (bool): Remove non-Sanskrit characters
        keep_punctuation (bool): Keep punctuation when removing foreign chars
        convert_digits_to_arabic (bool): Convert Devanagari digits to Arabic
        verbose (bool): Enable verbose output
        
    Returns:
        str: Normalized Sanskrit text
        
    Raises:
        TextNormalizerError: If normalization fails
    """
    if not isinstance(text, str):
        raise TextNormalizerError(f"Expected string input, got {type(text)}")
    
    if verbose:
        print(f"[Input]: {text}")
    
    try:
        # Step-by-step normalization
        text = unicode_normalize(text)
        text = remove_control_chars(text)
        text = clean_whitespace_punctuation(text)
        text = fix_visarga(text)
        text = correct_anusvara(text)
        text = normalize_diacritics(text)
        
        if convert_digits_to_arabic:
            text = convert_digits(text, to_arabic=True)
        
        if remove_foreign:
            text = remove_foreign_characters(text, keep_punctuation=keep_punctuation)
        
        # Final cleanup
        text = clean_whitespace_punctuation(text)
        
        if verbose:
            print(f"[Normalized]: {text}")
        
        return text
        
    except Exception as e:
        raise TextNormalizerError(f"Standard Sanskrit normalization failed: {e}")


def normalize_vedic_sanskrit_text(
    text: str,
    preserve_accents: bool = True,
    convert_digits_to_arabic: bool = True,
    verbose: bool = False
) -> str:
    """
    Normalize Vedic Sanskrit text with accent handling.
    
    Args:
        text (str): Input Vedic Sanskrit text
        preserve_accents (bool): Whether to preserve Vedic accents
        convert_digits_to_arabic (bool): Convert Devanagari digits to Arabic
        verbose (bool): Enable verbose output
        
    Returns:
        str: Normalized Vedic Sanskrit text
        
    Raises:
        TextNormalizerError: If normalization fails
    """
    if not isinstance(text, str):
        raise TextNormalizerError(f"Expected string input, got {type(text)}")
    
    if verbose:
        print(f"[Input Vedic]: {text}")
    
    try:
        # Step-by-step normalization for Vedic text
        text = unicode_normalize(text)
        text = remove_control_chars(text)
        text = clean_whitespace_punctuation(text)
        text = fix_visarga(text)
        text = correct_anusvara(text)
        text = normalize_diacritics(text)
        
        # Handle Vedic accents
        if preserve_accents:
            text = normalize_vedic_accents(text)
        else:
            text = remove_vedic_accents(text)
        
        if convert_digits_to_arabic:
            text = convert_digits(text, to_arabic=True)
        
        # Final cleanup
        text = clean_whitespace_punctuation(text)
        
        if verbose:
            print(f"[Normalized Vedic]: {text}")
        
        return text
        
    except Exception as e:
        raise TextNormalizerError(f"Vedic Sanskrit normalization failed: {e}")


def process_sanskrit_text(
    text: str,
    vedic: bool = False,
    preserve_accents: bool = True,
    remove_foreign: bool = False,
    split_sentences: bool = True,
    verbose: bool = False,
    **kwargs
) -> Union[str, List[str]]:
    """
    Process Sanskrit text with comprehensive normalization and optional sentence splitting.
    
    Args:
        text (str): Input Sanskrit text
        vedic (bool): Whether to use Vedic Sanskrit processing
        preserve_accents (bool): Preserve Vedic accents (only for Vedic text)
        remove_foreign (bool): Remove non-Sanskrit characters
        split_sentences (bool): Whether to split into sentences
        verbose (bool): Enable verbose output
        **kwargs: Additional arguments
        
    Returns:
        Union[str, List[str]]: Normalized text or list of sentences
        
    Raises:
        TextNormalizerError: If processing fails
    """
    if not isinstance(text, str):
        raise TextNormalizerError(f"Expected string input, got {type(text)}")
    
    try:
        # Detect script
        script = detect_script(text)
        if verbose:
            print(f"[Script Detected]: {script.value}")
        
        # Apply Latin digraph normalization for Roman text
        if script == Script.ROMAN:
            text = normalize_latin_digraphs(text)
        
        # Choose normalization method
        if vedic:
            normalized_text = normalize_vedic_sanskrit_text(
                text,
                preserve_accents=preserve_accents,
                verbose=verbose
            )
        else:
            normalized_text = normalize_standard_sanskrit_text(
                text,
                remove_foreign=remove_foreign,
                verbose=verbose
            )
        
        # Optional sentence splitting
        if split_sentences:
            return sentence_split(normalized_text, **kwargs)
        else:
            return normalized_text
            
    except Exception as e:
        raise TextNormalizerError(f"Sanskrit text processing failed: {e}")


# ========================== TEXT NORMALIZER CLASS =============================

class TextNormalizer:
    """
    A configurable text normalizer for Sanskrit and Vedic text processing.
    
    This class provides a convenient interface for normalizing Sanskrit text
    with various options and configurations.
    """
    
    def __init__(
        self,
        vedic: bool = False,
        preserve_accents: bool = True,
        remove_foreign: bool = False,
        convert_digits_to_arabic: bool = True,
        keep_punctuation: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the TextNormalizer.
        
        Args:
            vedic (bool): Enable Vedic Sanskrit processing
            preserve_accents (bool): Preserve Vedic accents
            remove_foreign (bool): Remove non-Sanskrit characters
            convert_digits_to_arabic (bool): Normalize digits to Arabic numerals
            keep_punctuation (bool): Keep punctuation when removing foreign chars
            verbose (bool): Enable verbose output
        """
        self.vedic = vedic
        self.preserve_accents = preserve_accents
        self.remove_foreign = remove_foreign
        self.convert_digits_to_arabic = convert_digits_to_arabic
        self.keep_punctuation = keep_punctuation
        self.verbose = verbose
    
    def normalize(self, text: str) -> str:
        """
        Normalize text using the configured settings.
        
        Args:
            text (str): Input text to normalize
            
        Returns:
            str: Normalized text
        """
        if self.vedic:
            return normalize_vedic_sanskrit_text(
                text,
                preserve_accents=self.preserve_accents,
                convert_digits_to_arabic=self.convert_digits_to_arabic,
                verbose=self.verbose
            )
        else:
            return normalize_standard_sanskrit_text(
                text,
                remove_foreign=self.remove_foreign,
                keep_punctuation=self.keep_punctuation,
                convert_digits_to_arabic=self.convert_digits_to_arabic,
                verbose=self.verbose
            )
    
    def process(self, text: str, split_sentences: bool = True, **kwargs) -> Union[str, List[str]]:
        """
        Process text with normalization and optional sentence splitting.
        
        Args:
            text (str): Input text to process
            split_sentences (bool): Whether to split into sentences
            **kwargs: Additional arguments
            
        Returns:
            Union[str, List[str]]: Processed text or sentences
        """
        return process_sanskrit_text(
            text,
            vedic=self.vedic,
            preserve_accents=self.preserve_accents,
            remove_foreign=self.remove_foreign,
            split_sentences=split_sentences,
            verbose=self.verbose,
            **kwargs
        )


# Core stopword list (can be extended)
STOPWORDS = set([
    'च', 'वा', 'अपि', 'तु', 'यदि', 'किन्तु', 'परन्तु', 'हि', 'अहम्', 'त्वम्', 'सः', 'सा', 'तत्',
    'एतत्', 'इदम्', 'किम्', 'मम', 'तव', 'अस्माकम्', 'युष्माकम्', 'तेषाम्', 'तासाम्',
    'अस्ति', 'सन्ति', 'भवति', 'भवन्ति', 'करोति', 'कुर्वन्ति', 'गच्छति', 'गच्छन्ति', 'तत्र', 'यत्र',
    'कुत्र', 'अत्र', 'इह', 'एव', 'न', 'मा', 'ते', 'यदा', 'तदा', 'कदा', 'सदा', 'कदाचित्', 'इति', 'अथ',
    'एव', 'खलु', 'वै', 'नाम', 'स्म', 'उ', 'आह', 'इत्यादि'
])

# Common punctuation and Devanagari marks
SYMBOLS = set([
    '.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}', '।', '।।', '-', '"', "'", '`', '०','१','२','३','४','५','६','७','८','९'
])


def is_symbol(token: str) -> bool:
    return token in SYMBOLS

def is_stopword(token: str) -> bool:
    return token in STOPWORDS

def clean_token(token: str) -> str:
    return token.strip("".join(SYMBOLS)).strip()

# === Main Function ===

def remove_sanskrit_stopwords(text: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Remove stopwords and common symbols from Sanskrit text.
    
    Args:
        text (str or list): Input Sanskrit string or list of tokens
    
    Returns:
        str or list: Cleaned text (same format as input)
    
    Example:
        remove_sanskrit_stopwords("रामः च वनम् गच्छति ।")  -> "रामः वनम् गच्छति"
    """
    if isinstance(text, str):
        tokens = text.split()
        cleaned = [
            clean_token(t) for t in tokens
            if t and not is_stopword(t) and not is_symbol(t)
        ]
        return ' '.join([t for t in cleaned if t])
    elif isinstance(text, list):
        return [
            clean_token(t) for t in text
            if t and not is_stopword(t) and not is_symbol(t)
        ]
    else:
        raise TypeError("Input must be a string or list of tokens.")


def main():
    """Example usage of the text normalizer."""
    # Test samples
    samples = {
        "Standard Sanskrit 1": "रामःगच्छति।   संपूर्णः अस्ति।",
        "Standard Sanskrit 2": "गङ्गा   स्नानं शुभं।",
        "Vedic Sanskrit": "अ॒ग्निं॑ ई॒ळे॑ पु॑रोहि॒तं॑",
        "Mixed Content": "Sanskrit वाक्यम् with English मिश्रित है। दिल्ली में २०२२ वर्षे।",
        "Anusvara Case": "संस्कृतं कंठे वं कुण्डलिनी जागृता।",
        "Visarga Case": "रामः गच्छति। कृष्णः वदति।अहं गच्छामि।",
        "Complex Sanskrit": "सूर्यः अस्तं गच्छति।रात्रिः आगच्छति। चन्द्रः प्रकाशते।"
    }
    
    print("Sanskrit Text Normalizer Demo")
    print("=" * 50)
    
    # Test individual functions
    for name, text in samples.items():
        print(f"\n--- {name} ---")
        print(f"Original: {text}")
        
        # Detect script
        script = detect_script(text)
        print(f"Script: {script.value}")
        
        # Standard normalization
        normalized = normalize_standard_sanskrit_text(text)
        print(f"Normalized: {normalized}")
        
        # Vedic normalization (if applicable)
        if "Vedic" in name:
            vedic_with_accents = normalize_vedic_sanskrit_text(text, preserve_accents=True)
            vedic_without_accents = normalize_vedic_sanskrit_text(text, preserve_accents=False)
            print(f"Vedic (with accents): {vedic_with_accents}")
            print(f"Vedic (without accents): {vedic_without_accents}")
    
    # Test class interface
    print(f"\n--- Class Interface Demo ---")
    normalizer = TextNormalizer(verbose=False)
    complex_text = samples["Complex Sanskrit"]
    
    # Process with sentence splitting
    sentences = normalizer.process(complex_text, split_sentences=True)
    print(f"Original: {complex_text}")
    print(f"Processed sentences: {sentences}")


if __name__ == "__main__":
    main()