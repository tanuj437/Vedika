import os
import logging
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass

import sentencepiece as spm
from .normalizer import TextNormalizer
from .sentence_splitter import SentenceSplitter


# Default model path (auto-resolved relative to module location)
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "sp_unigram_64k.model"
)


@dataclass
class TokenizerConfig:
    """
    Configuration for the Sanskrit Tokenizer.
    """
    use_sentencepiece: bool = True
    sentencepiece_model_path: Optional[str] = None
    fallback_to_whitespace: bool = True
    normalize_text: bool = True
    preserve_sentence_boundaries: bool = True
    logging_level: int = logging.INFO
    max_word_length: int = 50
    min_word_length: int = 1


class SanskritTokenizer:
    """
    A modern tokenizer for Sanskrit text supporting SentencePiece models
    with optional fallback to simple whitespace tokenization.

    Features:
    - SentencePiece integration (default)
    - Optional whitespace fallback
    - Sentence boundary preservation
    - Text normalization
    - Metadata reporting
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.logging_level)

        # Initialize components
        self.normalizer = TextNormalizer()
        self.sentence_splitter = SentenceSplitter()

        # Initialize SentencePiece processor
        self.sp = None
        if self.config.use_sentencepiece:
            model_path = (
                self.config.sentencepiece_model_path or DEFAULT_MODEL_PATH
            )
            self._load_sentencepiece(model_path)

    def _load_sentencepiece(self, model_path: str) -> None:
        """Load SentencePiece model safely."""
        try:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)
            self.logger.info(f"Loaded SentencePiece model from: {model_path}")
        except Exception as e:
            self.logger.warning(
                f"Failed to load SentencePiece model from {model_path}: {e}"
            )
            self.sp = None

    # -------------------------------
    # Core Tokenization
    # -------------------------------
    def tokenize(
        self, text: str, return_metadata: bool = False
    ) -> Union[List[List[str]], Tuple[List[List[str]], Dict[str, Any]]]:
        """
        Tokenize input Sanskrit text into lists of tokens per sentence.

        Args:
            text: The input text.
            return_metadata: If True, returns a dictionary of metadata.

        Returns:
            Tokenized text as list of sentences (each a list of tokens),
            optionally with metadata.
        """
        if not text or not isinstance(text, str):
            return [] if not return_metadata else ([], {})

        metadata = {
            "original_length": len(text),
            "sentence_count": 0,
            "token_count": 0,
            "processing_steps": [],
        }

        # Step 1: Normalize
        if self.config.normalize_text:
            try:
                text = self.normalizer.normalize(text)
                metadata["processing_steps"].append("normalization")
            except Exception as e:
                self.logger.error(f"Normalization failed: {e}")
                text = text.strip()

        # Step 2: Split sentences
        sentences = (
            self.sentence_splitter.split(text)
            if self.config.preserve_sentence_boundaries
            else [text]
        )
        metadata["sentence_count"] = len(sentences)

        # Step 3: Tokenize each sentence
        tokenized_sentences: List[List[str]] = []
        for sent in sentences:
            tokens = self._tokenize_sentence(sent)
            if tokens:
                tokenized_sentences.append(tokens)
                metadata["token_count"] += len(tokens)

        if return_metadata:
            metadata["unique_tokens"] = len(
                set(tok for sent in tokenized_sentences for tok in sent)
            )
            return tokenized_sentences, metadata

        return tokenized_sentences

    def _tokenize_sentence(self, sentence: str) -> List[str]:
        """Tokenize a single sentence using SentencePiece or fallback."""
        sentence = sentence.strip()
        if not sentence:
            return []

        # Use SentencePiece if available
        if self.sp:
            try:
                tokens = self.sp.encode(sentence, out_type=str)
                return [t for t in tokens if self._is_valid_token(t)]
            except Exception as e:
                self.logger.error(f"SentencePiece failed: {e}")

        # Fallback to whitespace
        if self.config.fallback_to_whitespace:
            tokens = sentence.split()
            return [t for t in tokens if self._is_valid_token(t)]

        return []

    # -------------------------------
    # Detokenization
    # -------------------------------
    def detokenize(self, tokens: Union[List[str], List[List[str]]]) -> str:
        """
        Combine tokens back into text.

        Args:
            tokens: List of tokens or list of list of tokens (per sentence).

        Returns:
            Detokenized text string.
        """
        if not tokens:
            return ""

        # Flatten if nested
        if isinstance(tokens[0], list):
            tokens = [" ".join(sentence) for sentence in tokens]
            text = " ".join(tokens)
        else:
            text = " ".join(tokens)

        # Use SentencePiece detokenizer if available
        if self.sp:
            try:
                text = self.sp.decode_pieces(tokens)
            except Exception:
                pass

        return text.strip()

    # -------------------------------
    # Helpers
    # -------------------------------
    def _is_valid_token(self, token: str) -> bool:
        """Check token validity by length constraints."""
        return (
            self.config.min_word_length
            <= len(token)
            <= self.config.max_word_length
        )


# ------------------------------------------------------------
# Example Usage (can be kept for testing/demo, or commented out)
# ------------------------------------------------------------
if __name__ == "__main__":
    example_text = "श्रीरामः वनं गच्छति। सीता अपि गच्छति।"

    config = TokenizerConfig(
        use_sentencepiece=True,
        normalize_text=True,
        preserve_sentence_boundaries=True,
        logging_level=logging.DEBUG,
    )

    tokenizer = SanskritTokenizer(config)

    # Tokenization
    tokens, meta = tokenizer.tokenize(example_text, return_metadata=True)
    print("Tokenized:", tokens)
    print("Metadata:", meta)

    # Detokenization
    detok = tokenizer.detokenize(tokens)
    print("Detokenized:", detok)
