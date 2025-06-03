"""
Sanskrit Metrics: A comprehensive toolkit for Sanskrit syllabification and metrical analysis.

This module provides tools for analyzing Sanskrit text in Devanagari script,
including syllable segmentation, prosodic weight determination, and meter identification.
Supports Vedic accent marks and various metrical patterns.

Author: Tanuj Saxena and Soumya Sharma
Version: 1.0.0
License: MIT
"""

import re
import json
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
from pathlib import Path
import pkg_resources

__version__ = "1.0.0"
__author__ = "Tanuj Saxena and Soumya Sharma"
__email__ = "tanuj.saxena.rks@gmail.com, soumyasharma1599@gmail.com"
__license__ = "MIT"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Devanagari Character Sets ---
VOWELS = "अआइईउऊऋॠऌॡएऐओऔ"
SHORT_VOWELS = "अइउऋऌ"  # Truly short vowels (laghu)
LONG_VOWELS = "आईऊॠॡएऐओऔ"  # Long vowels (guru)
VOWEL_SIGNS = "ािীুূृॄॢॣেৈোৌ"
SHORT_VOWEL_SIGNS = "িুৃॢ"  # Corresponding to short vowels
LONG_VOWEL_SIGNS = "াীূॄॣেৈোৌ"  # Corresponding to long vowels
CONSONANTS = "কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলবশষসহळ"
NUKTA = "়"
HALANT = "্"
SPECIALS = "ংঃऽঁ"  # anusvāra, visarga, avagraha, chandrabindu
PUNCTUATION = "॥।"  # danda and double danda

# --- Vedic Accent Marks ---
ACCENTS = {"॑": "udātta", "॒": "anudātta", "॓": "svarita", "॔": "anudāttatara"}

# Default meter patterns
DEFAULT_METERS = {
    "anuṣṭubh": "GLGLGLGL",
    "triṣṭubh": "GLGGLGGLGGL",
    "jagatī": "GLGGLGGLGGLG",
    "gāyatrī": "GLGLGLGL",
    "bṛhatī": "GLGGLGGLGGLGL",
    "paṅkti": "GLGGLGGLGGLGLG",
    "atijagati": "GLGGLGGLGGLGLGL"
}


class SanskritMetricsError(Exception):
    """Base exception for Sanskrit Metrics package."""
    pass


class InvalidInputError(SanskritMetricsError):
    """Raised when input text is invalid or cannot be processed."""
    pass


class MeterDataError(SanskritMetricsError):
    """Raised when there are issues with meter data loading or format."""
    pass


class SanskritMetrics:
    """
    A comprehensive toolkit for Sanskrit syllabification and metrical analysis.
    
    This class handles Sanskrit text processing including:
    - Syllable segmentation (akshara analysis)
    - Prosodic weight determination (mātrā analysis)
    - Vedic accent recognition
    - Metrical pattern identification
    
    Attributes:
        meter_data (Dict[str, str]): Dictionary of meter patterns
        
    Example:
        >>> analyzer = SanskritMetrics()
        >>> results = analyzer.process_sanskrit_text("अग्निमीळे पुरोहितम्")
        >>> print(results[0]['meter_pattern'])
        'GLGLGLGL'
    """

    def __init__(self, custom_meters: Optional[Dict[str, str]] = None):
        """
        Initialize the Sanskrit Metrics analyzer.

        Args:
            custom_meters: Optional dictionary of custom meter patterns.
            
        Raises:
            MeterDataError: If meter file cannot be loaded or is invalid.
        """
        self.meter_data = DEFAULT_METERS.copy()
        
        # Try to load cleaned_metres.json automatically
        try:
            loaded_meters = self._load_meter_data("data/cleaned_metres.json")
            self.meter_data.update(loaded_meters)
        except Exception as e:
            logger.warning(f"Could not load cleaned_metres.json: {e}. Using default meters only.")
                
        if custom_meters:
            if not isinstance(custom_meters, dict):
                raise MeterDataError("custom_meters must be a dictionary")
            self.meter_data.update(custom_meters)
            

    def _load_meter_data(self, meter_file: Union[str, Path]) -> Dict[str, str]:
        """
        Load meter data from a JSON file.

        Args:
            meter_file: Path to the JSON file containing meter patterns.

        Returns:
            Dictionary mapping meter names to patterns.
            
        Raises:
            MeterDataError: If file cannot be loaded or parsed.
        """
        meter_path = Path(meter_file)
        
        if not meter_path.exists():
            raise MeterDataError(f"Meter file not found: {meter_path}")
            
        try:
            with open(meter_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if isinstance(data, dict):
                if "metres" in data:
                    return data["metres"]
                else:
                    # Assume the dict itself contains meter patterns
                    return data
            else:
                raise MeterDataError("Meter file must contain a JSON object")
                
        except json.JSONDecodeError as e:
            raise MeterDataError(f"Invalid JSON format in {meter_path}: {e}")
        except Exception as e:
            raise MeterDataError(f"Error reading meter file {meter_path}: {e}")

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess Sanskrit text for analysis.

        - Normalize whitespace
        - Space out punctuation
        - Validate input

        Args:
            text: Sanskrit input text in Devanagari script.

        Returns:
            Preprocessed text ready for analysis.
            
        Raises:
            InvalidInputError: If input is empty or invalid.
        """
        if not text or not isinstance(text, str):
            raise InvalidInputError("Input text must be a non-empty string")
            
        text = text.strip()
        if not text:
            raise InvalidInputError("Input text cannot be empty after stripping")
            
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Space out punctuation for better processing
        for punct in PUNCTUATION:
            text = text.replace(punct, f" {punct} ")
            
        return text.strip()

    def split_into_akshara_units(self, word: str) -> List[Tuple[str, Optional[str]]]:
        """
        Split a Sanskrit word into akshara (syllable) units with optional accent marks.

        Args:
            word: Sanskrit word in Devanagari script.

        Returns:
            List of tuples (akshara, accent) where accent may be None.
            
        Raises:
            InvalidInputError: If word contains invalid characters.
        """
        if not word:
            return []
            
        # Skip pure punctuation
        if all(c in PUNCTUATION + ' ' for c in word):
            return []

        aksharas = []
        i = 0
        current_accent = None
        unknown_chars = []

        while i < len(word):
            # Skip whitespace and punctuation
            if word[i] in PUNCTUATION + ' ':
                i += 1
                continue

            # Handle Vedic accent marks
            if word[i] in ACCENTS:
                current_accent = ACCENTS[word[i]]
                i += 1
                continue

            # Handle standalone vowels
            if word[i] in VOWELS:
                akshara = word[i]
                i += 1
                
                # Append special marks (anusvāra, visarga, etc.)
                while i < len(word) and word[i] in SPECIALS:
                    akshara += word[i]
                    i += 1
                    
                aksharas.append((akshara, current_accent))
                current_accent = None
                continue

            # Handle consonant clusters
            if word[i] in CONSONANTS:
                akshara = word[i]
                i += 1

                # Handle nukta
                if i < len(word) and word[i] == NUKTA:
                    akshara += word[i]
                    i += 1

                # Handle conjunct consonants with halant
                while i < len(word) and word[i] == HALANT:
                    akshara += word[i]
                    i += 1
                    if i < len(word) and word[i] in CONSONANTS:
                        akshara += word[i]
                        i += 1
                        # Handle nukta after conjunct consonant
                        if i < len(word) and word[i] == NUKTA:
                            akshara += word[i]
                            i += 1
                    else:
                        break

                # Handle vowel signs
                if i < len(word) and word[i] in VOWEL_SIGNS:
                    akshara += word[i]
                    i += 1

                # Handle special marks
                while i < len(word) and word[i] in SPECIALS:
                    akshara += word[i]
                    i += 1

                aksharas.append((akshara, current_accent))
                current_accent = None
                continue

            # Track unrecognized characters
            if word[i] not in PUNCTUATION + ' ':
                unknown_chars.append(word[i])
            i += 1

        if unknown_chars:
            logger.warning(f"Unrecognized characters in '{word}': {''.join(set(unknown_chars))}")

        return aksharas

    def determine_syllable_weight(self, akshara: str, accent: Optional[str] = None) -> int:
        """
        Determine syllable weight (mātrā) according to Sanskrit prosody rules.
        
        Weight values:
        - 0: Consonant cluster ending with halant (incomplete syllable)
        - 1: Light syllable (laghu)
        - 2: Heavy syllable (guru)

        Args:
            akshara: The akshara (syllable unit).
            accent: Optional Vedic accent mark.

        Returns:
            Syllable weight as integer (0, 1, or 2).
        """
        if not akshara:
            return 0
            
        # Anusvāra or visarga always make syllable heavy
        if any(mark in akshara for mark in ['ং', 'ঃ']):
            return 2

        # Vedic accent influence on weight
        if accent in {"udātta", "svarita"}:
            return 2
        elif accent == "anudātta":
            return 1

        # Check for explicit vowels (standalone vowels)
        for char in akshara:
            if char in VOWELS:
                return 2 if char in LONG_VOWELS else 1

        # Check for vowel signs (mātrās)
        for char in akshara:
            if char in VOWEL_SIGNS:
                return 2 if char in LONG_VOWEL_SIGNS else 1

        # Halant at end indicates incomplete syllable
        if akshara.endswith(HALANT):
            return 0

        # Consonant with implicit 'a' vowel (inherent vowel)
        if any(char in CONSONANTS for char in akshara):
            if not any(char in VOWEL_SIGNS for char in akshara):
                return 1  # Inherent 'a' is short

        # Fallback for edge cases
        return 0

    def process_sanskrit_text(self, text: str, phoneme_level: bool = False, 
                            include_weights: bool = True) -> List[Dict[str, Any]]:
        """
        Process Sanskrit text with comprehensive syllabification and metrical analysis.

        Args:
            text: Sanskrit text in Devanagari script.
            phoneme_level: If True, include individual phonemes in output.
            include_weights: If True, include syllable weight analysis.

        Returns:
            List of dictionaries containing detailed analysis for each word.
            
        Raises:
            InvalidInputError: If input text is invalid.
            
        Example:
            >>> analyzer = SanskritMetrics()
            >>> results = analyzer.process_sanskrit_text("অগ্নিমীলে")
            >>> print(results[0]['meter_pattern'])
        """
        try:
            processed_text = self.preprocess_text(text)
        except InvalidInputError:
            raise
        except Exception as e:
            raise InvalidInputError(f"Text preprocessing failed: {e}")

        # Extract Devanagari words and punctuation
        words = re.findall(r'[\u0900-\u097F\u1CD0-\u1CFA]+|[॥।]', processed_text)
        results = []

        for word in words:
            if word.strip() in PUNCTUATION:
                continue
                
            try:
                aksharas = self.split_into_akshara_units(word)
                word_info = {
                    "word": word,
                    "syllables": [],
                    "total_syllables": 0,
                    "total_matra": 0
                }

                for akshara, accent in aksharas:
                    if not akshara.strip():
                        continue
                        
                    matra = self.determine_syllable_weight(akshara, accent) if include_weights else 1
                    
                    if matra > 0:  # Only include complete syllables
                        syllable_info = {
                            "syllable": akshara,
                            "matra": matra,
                            "weight": "guru" if matra == 2 else "laghu",
                            "accent": accent,
                        }
                        
                        if phoneme_level:
                            syllable_info["phonemes"] = list(akshara)
                            
                        word_info["syllables"].append(syllable_info)

                # Calculate totals and patterns
                if word_info["syllables"]:
                    word_info["total_syllables"] = len(word_info["syllables"])
                    word_info["total_matra"] = sum(syl["matra"] for syl in word_info["syllables"])
                    
                    # Generate metrical pattern
                    pattern = ''.join('G' if syl['matra'] == 2 else 'L' 
                                    for syl in word_info["syllables"])
                    word_info["meter_pattern"] = pattern
                    
                    # Identify meter
                    meter_name = self.identify_meter(pattern)
                    word_info["meter_name"] = meter_name
                    
                    results.append(word_info)
                    
            except Exception as e:
                logger.error(f"Error processing word '{word}': {e}")
                continue

        return results

    def identify_meter(self, pattern: str) -> Optional[str]:
        """
        Identify meter name based on syllable weight pattern.

        Args:
            pattern: String pattern using 'G' for guru and 'L' for laghu.

        Returns:
            Meter name if found, None otherwise.
            
        Example:
            >>> analyzer = SanskritMetrics()
            >>> meter = analyzer.identify_meter("GLGLGLGL")
            >>> print(meter)  # "anuṣṭubh"
        """
        if not pattern or not isinstance(pattern, str):
            return None
            
        # Exact match first
        for meter_name, meter_pattern in self.meter_data.items():
            if pattern == meter_pattern:
                return meter_name
                
        # Partial match for longer texts
        for meter_name, meter_pattern in self.meter_data.items():
            if pattern.startswith(meter_pattern) and len(pattern) % len(meter_pattern) == 0:
                return f"{meter_name} (repeated)"
                
        return None

    def get_meter_info(self, meter_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific meter.
        
        Args:
            meter_name: Name of the meter.
            
        Returns:
            Dictionary with meter information or None if not found.
        """
        if meter_name not in self.meter_data:
            return None
            
        pattern = self.meter_data[meter_name]
        return {
            "name": meter_name,
            "pattern": pattern,
            "syllable_count": len(pattern),
            "guru_count": pattern.count('G'),
            "laghu_count": pattern.count('L'),
            "total_matra": pattern.count('G') * 2 + pattern.count('L')
        }

    def add_custom_meter(self, name: str, pattern: str) -> None:
        """
        Add a custom meter pattern to the analyzer.
        
        Args:
            name: Name of the meter.
            pattern: Pattern string using 'G' and 'L'.
            
        Raises:
            ValueError: If pattern is invalid.
        """
        if not re.match(r'^[GL]+$', pattern):
            raise ValueError("Pattern must contain only 'G' and 'L' characters")
            
        self.meter_data[name] = pattern

    def save_results(self, results: List[Dict[str, Any]], filename: Union[str, Path]) -> None:
        """
        Save analysis results to a JSON file.

        Args:
            results: List of processed word information.
            filename: Output JSON file path.
            
        Raises:
            IOError: If file cannot be written.
        """
        filepath = Path(filename)
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "metadata": {
                        "version": __version__,
                        "total_words": len(results),
                        "meters_used": len(self.meter_data)
                    },
                    "results": results
                }, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            raise IOError(f"Failed to save results to {filepath}: {e}")

    def get_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate statistical summary of analysis results.
        
        Args:
            results: Analysis results from process_sanskrit_text.
            
        Returns:
            Dictionary containing various statistics.
        """
        if not results:
            return {"error": "No results to analyze"}
            
        total_words = len(results)
        total_syllables = sum(r.get("total_syllables", 0) for r in results)
        total_matra = sum(r.get("total_matra", 0) for r in results)
        
        meters_found = {}
        for result in results:
            meter = result.get("meter_name")
            if meter:
                meters_found[meter] = meters_found.get(meter, 0) + 1
                
        return {
            "total_words": total_words,
            "total_syllables": total_syllables,
            "total_matra": total_matra,
            "average_syllables_per_word": total_syllables / total_words if total_words > 0 else 0,
            "average_matra_per_word": total_matra / total_words if total_words > 0 else 0,
            "meters_identified": meters_found,
            "unique_meters": len(meters_found)
        }


def analyze_text(text: str, **kwargs) -> List[Dict[str, Any]]:
    """
    Convenience function for quick text analysis.
    
    Args:
        text: Sanskrit text to analyze.
        **kwargs: Additional arguments for SanskritMetrics.
        
    Returns:
        Analysis results.
    """
    analyzer = SanskritMetrics()
    return analyzer.process_sanskrit_text(text, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    sample_text = "अग्निमीळे पुरोहितम्"
    
    try:
        analyzer = SanskritMetrics()
        results = analyzer.process_sanskrit_text(sample_text)
        
        print(f"Sanskrit Metrics Analysis v{__version__}")
        print("=" * 50)
        
        for i, word_info in enumerate(results, 1):
            print(f"\n{i}. Word: {word_info['word']}")
            print(f"   Syllables: {word_info['total_syllables']}")
            print(f"   Total Mātrā: {word_info['total_matra']}")
            
            for j, syl in enumerate(word_info["syllables"], 1):
                accent_str = f", Accent: {syl['accent']}" if syl['accent'] else ""
                print(f"     {j}. {syl['syllable']} - {syl['weight']} ({syl['matra']}){accent_str}")
                
            print(f"   Meter Pattern: {word_info.get('meter_pattern', 'N/A')}")
            print(f"   Meter Name: {word_info.get('meter_name', 'Unknown')}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()