import re
import logging
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

from .sandhi_split import SanskritSplit
from .normalizer import TextNormalizer
from .sentence_splitter import SentenceSplitter


@dataclass
class TokenizerConfig:
    """Configuration for Sanskrit tokenizer"""
    split_sandhi: bool = False
    normalize_text: bool = True
    handle_punctuation: bool = True
    preserve_sentence_boundaries: bool = True
    remove_stopwords : bool = True
    max_word_length: int = 50
    min_word_length: int = 1
    logging_level: int = logging.INFO
    
    # Patterns
    punctuation_pattern: str = r'[।॥्\.\,\!\?\;\:\-\(\)\[\]\{\}\"\']+' 
    sanskrit_pattern: str = r'[\u0900-\u097F]+'

class SanskritTokenizer:
    """
    A comprehensive tokenizer for Sanskrit text processing with advanced NLP capabilities.
    
    Features:
    - Sandhi splitting
    - Sentence boundary detection
    - Text normalization
    - Punctuation handling
    - Logging and statistics
    """
    
    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        
        # Initialize components
        self.sandhi_splitter = SanskritSplit()
        self.sentence_splitter = SentenceSplitter()
        self.normalizer = TextNormalizer()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.logging_level)
        
        # Compile patterns
        self.punct_pattern = re.compile(self.config.punctuation_pattern)
        self.sanskrit_pattern = re.compile(self.config.sanskrit_pattern)

    def tokenize(self, text: str, return_metadata: bool = False) -> Union[List[List[str]], Tuple[List[List[str]], Dict[str, Any]]]:
        """
        Tokenize Sanskrit text into sentences and words with optional metadata.
        
        Args:
            text: Input Sanskrit text
            return_metadata: If True, returns additional processing information
            
        Returns:
            List of tokenized sentences (each sentence is a list of tokens)
            Optional metadata dictionary if return_metadata=True
        """
        if not text or not isinstance(text, str):
            return [] if not return_metadata else ([], {})

        metadata = {
            'original_length': len(text),
            'sentence_count': 0,
            'token_count': 0,
            'processing_steps': []
        }

        # Step 1: Normalize
        try:
            if self.config.normalize_text:
                text = self.normalizer.normalize(text)
                metadata['processing_steps'].append('normalization')
        except Exception as e:
            self.logger.error(f"Normalization failed: {e}")
            text = text.strip()

        # Step 2: Split into sentences
        sentences = self._split_sentences(text)
        metadata['sentence_count'] = len(sentences)

        # Step 3: Process each sentence
        tokenized_sentences = []
        for sentence in sentences:
            tokens = self._process_sentence(sentence)
            if tokens:
                tokenized_sentences.append(tokens)
                metadata['token_count'] += len(tokens)

        if return_metadata:
            metadata['unique_tokens'] = len(set(token for sent in tokenized_sentences for token in sent))
            return tokenized_sentences, metadata
            
        return tokenized_sentences

    def _process_sentence(self, sentence: str) -> List[str]:
        """Process a single sentence into tokens."""
        tokens = []
        words = sentence.strip().split()
        
        for word in words:
            # Handle punctuation
            if self.config.handle_punctuation:
                parts = self._split_punctuation(word)
            else:
                parts = [word]
                
            for part in parts:
                if not self._is_valid_token(part):
                    continue
                    
                # Apply sandhi splitting if enabled
                if self.config.split_sandhi and self._is_sanskrit_word(part):
                    splits = self._apply_sandhi_split(part)
                    tokens.extend(splits)
                else:
                    tokens.append(part)
                    
        return tokens

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using the sentence splitter."""
        try:
            return self.sentence_splitter.split(text)
        except Exception as e:
            self.logger.warning(f"Sentence splitting failed: {e}")
            return [text]

    def _split_punctuation(self, token: str) -> List[str]:
        """Separate punctuation from words."""
        return [part for part in re.findall(f'{self.config.sanskrit_pattern}|{self.config.punctuation_pattern}', token) if part]

    def _apply_sandhi_split(self, word: str) -> List[str]:
        """Apply sandhi splitting to a word."""
        try:
            result = self.sandhi_splitter.split(word)
            if isinstance(result, dict) and 'split' in result:
                return [p.strip() for p in result['split'].split('+') if p.strip()]
        except Exception as e:
            self.logger.debug(f"Sandhi splitting failed for '{word}': {e}")
        return [word]

    def _is_valid_token(self, token: str) -> bool:
        """Check if a token meets the minimum requirements."""
        return (len(token) >= self.config.min_word_length and 
                len(token) <= self.config.max_word_length)

    def _is_sanskrit_word(self, word: str) -> bool:
        """Check if a word contains Sanskrit characters."""
        return bool(self.sanskrit_pattern.search(word))

def main():
    """Example usage of the Sanskrit tokenizer."""
    # Test text
    test_text = """
    श्रीरामः वनं गच्छति। सीता अपि गच्छति।
    राक्षसाः उग्रं कृत्यं आरभन्ति।
    """
    
    # Initialize tokenizer with custom config
    config = TokenizerConfig(
        split_sandhi=True,
        normalize_text=True,
        handle_punctuation=True,
        logging_level=logging.DEBUG
    )
    
    tokenizer = SanskritTokenizer()
    
    # Tokenize with metadata
    tokens, metadata = tokenizer.tokenize(test_text, return_metadata=True)
    
    # Print results
    print("=== Sanskrit Tokenizer Demo ===")
    print(f"\nOriginal text:\n{test_text}")
    print("\nTokenized output:")
    for i, sentence in enumerate(tokens, 1):
        print(f"Sentence {i}: {sentence}")
    
    print("\nProcessing metadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()